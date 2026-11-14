#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_w2w4_bf16.sh
# 功能说明: 双精度 (W2 + W4) 在线多精度策略蒸馏 (MOPD) 启动预设脚本。
#           学生模型加载共享码本缓存 (parent8, orig2)，每条提示词同时以 W2 和 W4
#           两种位宽生成续写轨迹，并分别路由给 AutoRound W2 BF16 与 W4 BF16 教师
#           进行在线 token 级反向 KL 蒸馏，联合更新共享的 256 维父码本残差 delta。
#
# 启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
#   1) 预览生成的完整训练命令 (不实际启动训练):
#      bash my_scripts/run_w2w4_bf16.sh --dry-run
#
#   2) 四卡启动 W2/W4 双精度 MOPD 正式训练 (默认配置):
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=0,1,2,3 \
#      OUTPUT_DIR=/data/anyprecision_mopd/w2w4_mopd \
#      bash my_scripts/run_w2w4_bf16.sh --run
#
#   3) 双卡运行并自定义 W2:W4 损失权重比例 (例如 2:1):
#      CUDA_VISIBLE_DEVICES=0,1 PRECISION_LOSS_WEIGHTS='[2,1]' \
#      bash my_scripts/run_w2w4_bf16.sh --run
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认使用 4 张物理 GPU；双精度下支持按 GPU 均分 W2 与 W4 生成进程
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 安装完整的 2~8 bit 码本缓存
export ANYPRECISION_BITS='[2,3,4,5,6,7,8]'
# 训练激活位宽集合：同时优化 W2 与 W4
export ANYPRECISION_TRAIN_BITS='[2,4]'
# 默认使用 HuggingFace BF16 生成后端，保证与 BF16 训练前向数值严格一致
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-hf}"
# 共享码本缓存目录 (需包含 lut_2 .. lut_8)
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512}"

# 第一个教师槽位 (环境变量名沿用底层 W4_TEACHER) 绑定 W2 AutoRound BF16 教师
export W4_TEACHER="${W2_TEACHER:-/data/models/Qwen3.5-0.8B-AutoRound-W2-BF16-v2-cal512-seq2048-20261108_051146/Qwen3.5-0.8B-w2g32}"
# 第二个教师槽位 (环境变量名沿用底层 W8_TEACHER) 绑定 W4 AutoRound BF16 教师
export W8_TEACHER="${W4_BF16_TEACHER:-/data/models/Qwen3.5-0.8B-int4-AutoRound}"

# W2 与 W4 的相对损失权重，默认 1:1 等权
export PRECISION_LOSS_WEIGHTS="${PRECISION_LOSS_WEIGHTS:-[1,1]}"
# 各阶段微批次大小配置
export ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-4}"
export LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-4}"
export TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-1}"
# 实验输出目录
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/parent8_w2w4_bf16_$(date +%Y%m%d_%H%M%S)}"

# 运行模式：默认直接 --run，可显式传 --dry-run 仅检查配置
MODE="${1:---run}"
if (($#)); then shift; fi

# 正式运行前检查 W2/W8 码本文件是否齐全，避免误用仅支持 W4..W8 的旧缓存
if [[ "$MODE" != --dry-run && ( ! -f "$ANYPRECISION_CACHE/lut_2/l23.pt" || ! -f "$ANYPRECISION_CACHE/lut_8/l23.pt" ) ]]; then
    printf 'W2/W4 shared cache is missing: %s\n' "$ANYPRECISION_CACHE" >&2
    exit 2
fi

# 启动通用实验封装器，将两个教师均设为 auto_round:torch_zp BF16 后端，并启用 FSDP no_shard
exec bash "$SCRIPT_DIR/run_wikitext_experiment.sh" "$MODE" \
    '+actor_rollout_ref.model.anyprecision.train_precisions=[2,4]' \
    '++mt_opd.teacher_domains=[int2,int4]' \
    '++reward_model.model.quantization_backend=auto_round:torch_zp' \
    '++reward_model.model.dtype=bf16' \
    '++mt_reward_model_1.model.quantization_backend=auto_round:torch_zp' \
    'mt_reward_model_1.model.dtype=bf16' \
    'actor_rollout_ref.rollout.dtype=bfloat16' \
    '++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16' \
    '++actor_rollout_ref.actor.fsdp_config.sharding_strategy=no_shard' \
    "$@"
