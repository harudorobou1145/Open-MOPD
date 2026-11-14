#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_w2w4w8_bf16.sh
# 功能说明: 三精度 (W2 + W4 + W8) 在线多精度策略蒸馏 (MOPD) 启动预设脚本。
#           在同一个共享码本学生模型 (parent8, orig2) 上同时激活 W2、W4、W8 三种
#           位宽，每条提示词展开为 3 条不同位宽的轨迹 (单步 16*3=48 条轨迹)，并由
#           对应的 W2、W4、W8 三个 AutoRound BF16 教师分别监督，从而同时约束父码本
#           本身 (W8) 及其多级分组均值池化子码本 (W4、W2)。
#
# 启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
#   1) 预览生成的完整训练命令 (默认即为 --dry-run):
#      bash my_scripts/run_w2w4w8_bf16.sh --dry-run
#
#   2) 四卡启动 W2/W4/W8 三精度 MOPD 正式训练:
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=0,1,2,3 \
#      OUTPUT_DIR=/data/anyprecision_mopd/w2w4w8_mopd \
#      bash my_scripts/run_w2w4w8_bf16.sh --run
#
#   3) 自定义三精度损失权重与训练步数 (例如跑 10 步验证):
#      CUDA_VISIBLE_DEVICES=0,1,2,3 PRECISION_LOSS_WEIGHTS='[2,1,1]' \
#      bash my_scripts/run_w2w4w8_bf16.sh --run trainer.total_training_steps=10
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认使用 4 张物理 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 安装完整的 2~8 bit 码本缓存
export ANYPRECISION_BITS='[2,3,4,5,6,7,8]'
# 训练激活位宽集合：同时优化 W2、W4、W8
export ANYPRECISION_TRAIN_BITS='[2,4,8]'
# 默认使用 HuggingFace BF16 生成后端
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-hf}"
# 共享码本缓存目录
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-/data/anyprecision_w2w8/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512}"

# 三个精度的 AutoRound BF16 教师模型路径：
# - W4_TEACHER 槽位传入第一个教师 (int2 -> W2 教师)
# - W8_TEACHER 槽位传入第二个教师 (int4 -> W4 教师)
# - W8_BF16_TEACHER 通过 mt_reward_model_2 挂载第三个教师 (int8 -> W8 教师)
export W4_TEACHER="${W2_TEACHER:-/data/models/Qwen3.5-0.8B-AutoRound-W2-BF16-v2-cal512-seq2048-20261108_051146/Qwen3.5-0.8B-w2g32}"
export W8_TEACHER="${W4_BF16_TEACHER:-/data/models/Qwen3.5-0.8B-int4-AutoRound}"
W8_BF16_TEACHER="${W8_BF16_TEACHER:-/data/models/Qwen3.5-0.8B-AutoRound-W8-BF16-cal128-seq2048/Qwen3.5-0.8B-w8g128}"

# W2:W4:W8 的相对损失权重，默认 1:1:1
export PRECISION_LOSS_WEIGHTS="${PRECISION_LOSS_WEIGHTS:-[1,1,1]}"
# 4 张 GPU 无法被 3 种精度整除，因此关闭按卡固定单精度，改为每张卡按批次顺序切换精度
export PRECISION_PER_GPU=False

# 批次与微批次配置 (单步 16 条原始 prompt -> 展开后 48 条轨迹)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-4}"
export LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-4}"
export TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-1}"
# 实验输出目录
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/parent8_w2w4w8_bf16_$(date +%Y%m%d_%H%M%S)}"

# 运行模式：默认为 --dry-run，传入 --run 启动真实训练
MODE="${1:---dry-run}"
if (($#)); then shift; fi

# 启动通用实验封装器，配置 ppo_mini_batch_size=TRAIN_BATCH_SIZE*3 并注册第 3 个教师 mt_reward_model_2
exec bash "$SCRIPT_DIR/run_wikitext_experiment.sh" "$MODE" \
    '+actor_rollout_ref.model.anyprecision.train_precisions=[2,4,8]' \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$((TRAIN_BATCH_SIZE * 3))" \
    '++mt_opd.n_additional_teachers=2' \
    '++mt_opd.teacher_domains=[int2,int4,int8]' \
    "++mt_reward_model_2={enable:True,model:{path:'${W8_BF16_TEACHER}',input_tokenizer:null,use_remove_padding:False,fsdp_config:{param_offload:True},quantized_teacher:True,quantization_backend:'auto_round:torch_zp',dtype:bf16},micro_batch_size_per_gpu:${TEACHER_MICRO_BATCH_SIZE},use_dynamic_bsz:False}" \
    '++reward_model.model.quantization_backend=auto_round:torch_zp' \
    '++reward_model.model.dtype=bf16' \
    '++mt_reward_model_1.model.quantization_backend=auto_round:torch_zp' \
    'mt_reward_model_1.model.dtype=bf16' \
    'actor_rollout_ref.rollout.dtype=bfloat16' \
    '++actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16' \
    '++actor_rollout_ref.actor.fsdp_config.sharding_strategy=no_shard' \
    "$@"
