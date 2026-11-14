#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_w4_opd_bf16.sh
# 功能说明: 单精度在线策略蒸馏 (Single-Precision OPD) 启动预设脚本。
#           使用与多精度实验相同的共享码本学生模型 (parent8, orig2, 安装 2~8 bit)，
#           仅针对单一目标位宽 (OPD_BITS=2 或 4) 采样并由对应量化教师打分更新。
#           默认采用 vLLM BF16 生成后端，且每条提示词采样 N_RESPONSES=2 条轨迹
#           (单步 16*2=32 条轨迹)，与双精度 MOPD 的单步轨迹预算严格对齐。
#
# 启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
#   1) 预览生成的完整训练命令 (不实际启动训练):
#      OPD_BITS=4 bash my_scripts/run_w4_opd_bf16.sh --dry-run
#
#   2) 双卡仅 W2 单精度 OPD 正式训练 (使用 GPU 0,1，离线模式):
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=0,1 OPD_BITS=2 \
#      OUTPUT_DIR=/data/anyprecision_mopd/w2_only_opd \
#      bash my_scripts/run_w4_opd_bf16.sh --run
#
#   3) 双卡仅 W4 单精度 OPD 正式训练 (使用 GPU 2,3，可与上面 W2 任务并发运行):
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=2,3 OPD_BITS=4 \
#      OUTPUT_DIR=/data/anyprecision_mopd/w4_only_opd \
#      bash my_scripts/run_w4_opd_bf16.sh --run
#
#   4) 切换为 HuggingFace 生成后端并运行 2 步冒烟测试:
#      CUDA_VISIBLE_DEVICES=0,1 OPD_BITS=4 ANYPRECISION_ROLLOUT=hf \
#      bash my_scripts/run_w4_opd_bf16.sh --run trainer.total_training_steps=2
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 目标单精度位宽：支持 2 (仅 W2 OPD) 或 4 (仅 W4 OPD)，默认为 4
OPD_BITS="${OPD_BITS:-4}"
case "$OPD_BITS" in
    2) OPD_TEACHER="${W2_TEACHER:-/data/models/Qwen3.5-0.8B-AutoRound-W2-BF16-v2-cal512-seq2048-20261108_051146/Qwen3.5-0.8B-w2g32}" ;;
    4) OPD_TEACHER="${W4_BF16_TEACHER:-/data/models/Qwen3.5-0.8B-int4-AutoRound}" ;;
    *) printf 'OPD_BITS must be 2 or 4.\n' >&2; exit 2 ;;
esac

# 可见物理 GPU 列表，默认使用 4 张卡，可通过环境变量覆盖 (如 0,1 或 2,3)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 学生模型安装的完整码本位宽集合 (保持与多精度实验完全一致的 2~8 bit 结构)
export ANYPRECISION_BITS='[2,3,4,5,6,7,8]'
# 实际参与采样与梯度更新的位宽 (单精度仅包含 OPD_BITS)
export ANYPRECISION_TRAIN_BITS="[$OPD_BITS]"
# Rollout 生成后端：默认使用 vllm 加速，也可设为 hf
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-vllm}"
# 共享码本量化缓存路径 (orig2 种子码本 + 8-bit 256 质心父码本)
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512}"
# 主教师模型路径 (单精度模式下禁用额外教师，底层第二个占位参数复用同一路径即可)
export W4_TEACHER="$OPD_TEACHER"
export W8_TEACHER="$W4_TEACHER"
# 单一精度的损失权重
export PRECISION_LOSS_WEIGHTS='[1]'
# 单精度无需按卡绑定不同位宽，设为 False 让所有 rollout worker 均使用目标位宽
export PRECISION_PER_GPU=False

# 批次与微批次配置：每步 16 条 prompt * 每条采样 2 个回答 = 单步共 32 条轨迹
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export N_RESPONSES="${N_RESPONSES:-2}"
export ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-4}"
export LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-4}"
export TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-1}"
# 实验输出目录
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/parent8_w${OPD_BITS}_opd_bf16_$(date +%Y%m%d_%H%M%S)}"

# 运行模式：默认为 --dry-run，传入 --run 启动真实训练
MODE="${1:---dry-run}"
if (($#)); then shift; fi

# 调用通用 WikiText 实验封装脚本，并传入单精度 OPD 专属 Hydra 覆盖参数：
# - ppo_mini_batch_size 与展开后的总轨迹数对齐，保证每步单次 on-policy 更新
# - n_additional_teachers=0 且关闭 mt_reward_model_1，仅保留单个匹配精度教师
# - 显式向 vLLM 声明支持 [2,4,8] 位宽切换，并使用 FSDP no_shard 复制小规模可训练参数
exec bash "$SCRIPT_DIR/run_wikitext_experiment.sh" "$MODE" \
    "+actor_rollout_ref.model.anyprecision.train_precisions=[$OPD_BITS]" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE * ${N_RESPONSES:-1}))" \
    '++mt_opd.n_additional_teachers=0' \
    "++mt_opd.teacher_domains=[int${OPD_BITS}]" \
    '++mt_reward_model_1.enable=False' \
    '++reward_model.model.quantization_backend=auto_round:torch_zp' \
    '++reward_model.model.dtype=bf16' \
    'actor_rollout_ref.rollout.dtype=bfloat16' \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.quant_method=anyprecision' \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.precisions=[2,4,8]' \
    '++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16' \
    '++actor_rollout_ref.actor.fsdp_config.sharding_strategy=no_shard' \
    'actor_rollout_ref.actor.optim.lr=1e-6' \
    'trainer.total_training_steps=1485' \
    "$@"
