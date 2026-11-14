#!/usr/bin/env bash
# ==============================================================================
# 脚本名称: run_wikitext_experiment.sh
# 功能说明: WikiText-2 AnyPrecision MOPD 通用实验运行封装器。
#           负责校验 GPU 数量与批次整除关系、配置各阶段微批次大小、创建独立的
#           Ray 临时目录与 TensorBoard 日志目录、记录训练前后环境快照与退出码，
#           并支持单卡最大微批次自动探测 (--tune)。
#           上层预设脚本 (run_w4_opd_bf16.sh / run_w2w4_bf16.sh / run_w2w4w8_bf16.sh)
#           均通过调用本脚本启动，也可直接运行本脚本启动默认的 W4/W8 双精度 MOPD。
#
# 启动示例 (在 /workspace/my_project/Open-MOPD 目录下执行):
#   1) 预览默认 W4/W8 双精度实验命令 (不实际运行):
#      bash my_scripts/run_wikitext_experiment.sh --dry-run
#
#   2) 双卡启动默认 W4/W8 双精度 MOPD 训练 (默认使用 GPU 2,3 与 vLLM):
#      HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
#      CUDA_VISIBLE_DEVICES=2,3 \
#      OUTPUT_DIR=/data/anyprecision_mopd/w4w8_mopd \
#      bash my_scripts/run_wikitext_experiment.sh --run
#
#   3) 单卡自动探测不触发 OOM 的最大微批次大小 (--tune 模式):
#      CUDA_VISIBLE_DEVICES=2 \
#      bash my_scripts/run_wikitext_experiment.sh --tune
#
#   4) 附加任意 Hydra 覆盖参数 (例如只跑 5 步快速验证):
#      CUDA_VISIBLE_DEVICES=2,3 \
#      bash my_scripts/run_wikitext_experiment.sh --run trainer.total_training_steps=5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 训练和 TensorBoard 使用的 Python 解释器，默认使用系统 Python (/usr/bin/python)
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
# 物理 GPU 编号，顺序决定生成进程的编号，默认使用 2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
# 生成后端：vllm（默认）或 hf
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-vllm}"
# 每个生成批次的原始提示词数量（后续按训练精度数与每提示采样数展开）
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
# 遍历数据集的轮数，可通过 trainer.total_training_steps 提前限制训练步数
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# 实验输出目录，默认在 /data 下创建带时间戳的新目录
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/batched_$(date +%Y%m%d_%H%M%S)}"
# TensorBoard 事件目录，始终由 OUTPUT_DIR 派生
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
# Ray 临时目录，默认在 /data 下为每次启动创建独立目录，避免多任务并发冲突
export RAY_TMPDIR="${RAY_TMPDIR:-/data/ap_ray_$(date +%s)_$$}"
# 立即刷新 Python 输出，避免后台日志因缓冲而延迟
export PYTHONUNBUFFERED=1
# 实验名称，默认使用输出目录的末级目录名
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$OUTPUT_DIR")}"

# 每张 GPU 上 actor 单次前向和反向的样本数，增大会增加激活显存
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-8}"
# 生成微批次大小，仅用于 HF 后端，不是 vLLM 的并发上限
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-16}"
# 每张 GPU 单次计算学生对数概率的样本数，默认等于 actor 微批次大小
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-$ACTOR_MICRO_BATCH_SIZE}"
# 每张 GPU 单次教师前向的样本数，增大会增加教师激活显存
TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-2}"

# True 表示教师仅给匹配精度的样本打分；启用教师冲突策略时需要完整打分
PRECISION_TEACHER_SCORING="${PRECISION_TEACHER_SCORING:-True}"
# 默认双精度下 W4 与 W8 学生损失的相对权重（会被归一化至均值为 1）
# 若上层脚本显式设置了 PRECISION_LOSS_WEIGHTS，则优先使用 PRECISION_LOSS_WEIGHTS
W4_LOSS_WEIGHT="${W4_LOSS_WEIGHT:-1}"
W8_LOSS_WEIGHT="${W8_LOSS_WEIGHT:-2}"
# 单卡 --tune 模式的候选微批次大小，使用空格分隔的正整数
MICRO_BATCH_CANDIDATES="${MICRO_BATCH_CANDIDATES:-4 8 16 20 24}"

# 解析运行模式参数：支持 --run / --dry-run / --tune
MODE="${1:---run}"
if (($#)); then shift; fi
case "$MODE" in
    --run|--dry-run|--tune) ;;
    *) printf 'Usage: bash %s [--run|--dry-run|--tune] [Hydra overrides...]\n' "$0"; exit 2 ;;
esac

# 解析并校验可见 GPU 数量（支持 1、2 或 4 张 GPU）
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
# 生成精度分配：auto 在多卡时自动启用按 GPU 绑定生成位宽；也可显式传 True / False 覆盖
PRECISION_PER_GPU="${PRECISION_PER_GPU:-auto}"
if (( GPU_COUNT != 1 && GPU_COUNT != 2 && GPU_COUNT != 4 )); then
    printf 'This launcher supports one, two or four GPUs.\n' >&2
    exit 2
fi
if (( GPU_COUNT > 1 )); then
    (( TRAIN_BATCH_SIZE % GPU_COUNT == 0 )) || { printf 'Global batch must be divisible by GPU count.\n' >&2; exit 2; }
    [[ "$MODE" != --tune ]] || { printf 'Multi-GPU batch tuning is not supported.\n' >&2; exit 2; }
fi
if [[ "$PRECISION_PER_GPU" == auto ]]; then
    PRECISION_PER_GPU=False
    if (( GPU_COUNT > 1 )); then PRECISION_PER_GPU=True; fi
fi

# 校验布尔开关与正整数批次参数
for value in "$PRECISION_PER_GPU" "$PRECISION_TEACHER_SCORING"; do
    [[ "${value,,}" == true || "${value,,}" == false ]] || {
        printf 'Precision switches must be True or False (PRECISION_PER_GPU also accepts auto).\n' >&2
        exit 2
    }
done
for value in "$TRAIN_BATCH_SIZE" "$ACTOR_MICRO_BATCH_SIZE" "$ROLLOUT_MICRO_BATCH_SIZE" \
             "$LOGPROB_MICRO_BATCH_SIZE" "$TEACHER_MICRO_BATCH_SIZE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Batch sizes must be positive integers.\n' >&2; exit 2; }
done
(( TRAIN_BATCH_SIZE % ACTOR_MICRO_BATCH_SIZE == 0 )) || {
    printf 'TRAIN_BATCH_SIZE must be divisible by ACTOR_MICRO_BATCH_SIZE.\n' >&2; exit 2;
}

# 组装并行、微批次与 Ray 参数，调用同目录下的 wikitext_anyprecision_mopd.sh
launch() {
    local mode="$1"
    shift
    bash "${SCRIPT_DIR}/wikitext_anyprecision_mopd.sh" "$mode" -- \
        "trainer.n_gpus_per_node=${GPU_COUNT}" \
        "++trainer.precision_per_gpu=${PRECISION_PER_GPU}" \
        "++mt_opd.precision_teacher_scoring=${PRECISION_TEACHER_SCORING}" \
        "++mt_opd.precision_loss_weights=${PRECISION_LOSS_WEIGHTS:-[${W4_LOSS_WEIGHT},${W8_LOSS_WEIGHT}]}" \
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE}" \
        "actor_rollout_ref.rollout.micro_batch_size=${ROLLOUT_MICRO_BATCH_SIZE}" \
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MICRO_BATCH_SIZE}" \
        "reward_model.micro_batch_size_per_gpu=${TEACHER_MICRO_BATCH_SIZE}" \
        "++mt_reward_model_1.micro_batch_size_per_gpu=${TEACHER_MICRO_BATCH_SIZE}" \
        "trainer.logger=['console','tensorboard']" \
        "+ray_kwargs.ray_init.address=local" \
        "+ray_kwargs.ray_init._temp_dir='${RAY_TMPDIR}'" \
        "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR='${TENSORBOARD_DIR}'" \
        "$@"
}

# --dry-run 模式：仅打印组装后的底层执行命令并退出
if [[ "$MODE" == --dry-run ]]; then
    launch --dry-run "$@"
    exit
fi

# 正式运行前验证 TensorBoard 依赖，初始化输出目录并记录环境、PID、GPU 初始状态与退出码钩子
"$PYTHON_BIN" -c 'from torch.utils.tensorboard import SummaryWriter'
mkdir -p "$OUTPUT_DIR" "$TENSORBOARD_DIR" "$RAY_TMPDIR"
STATUS_FILE="$OUTPUT_DIR/exit_status.txt"
trap 'printf "%s\n" "$?" > "$STATUS_FILE"' EXIT
printf '%s\n' "$$" > "$OUTPUT_DIR/launcher.pid"
printf 'CUDA_VISIBLE_DEVICES=%s\nPYTHON_BIN=%s\nOUTPUT_DIR=%s\nRAY_TMPDIR=%s\n' \
    "$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "$OUTPUT_DIR" "$RAY_TMPDIR" > "$OUTPUT_DIR/environment.txt"
nvidia-smi > "$OUTPUT_DIR/gpu_before.txt"
printf 'Output: %s\nTensorBoard: %s -m tensorboard.main --logdir %q --port 6006\n' \
    "$OUTPUT_DIR" "$PYTHON_BIN" "$TENSORBOARD_DIR"

# --tune 模式：依次尝试候选微批次跑 2 步，自动找到不 OOM 的最大微批次
if [[ "$MODE" == --tune ]]; then
    BASE_OUTPUT="$OUTPUT_DIR"
    BEST=0
    for candidate in ${MICRO_BATCH_CANDIDATES}; do
        [[ "$candidate" =~ ^[1-9][0-9]*$ ]] || exit 2
        export TRAIN_BATCH_SIZE="$candidate"
        ACTOR_MICRO_BATCH_SIZE="$candidate"
        ROLLOUT_MICRO_BATCH_SIZE="$candidate"
        LOGPROB_MICRO_BATCH_SIZE="$candidate"
        export OUTPUT_DIR="${BASE_OUTPUT}/probe_${candidate}"
        export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
        export RAY_TMPDIR="/data/ap_probe_$$_${candidate}"
        mkdir -p "$RAY_TMPDIR" "$TENSORBOARD_DIR"
        if launch --run trainer.total_training_steps=2 trainer.save_freq=-1 "$@" > "$OUTPUT_DIR/train.log" 2>&1; then
            BEST="$candidate"
            printf 'PASS microbatch=%s; log=%s/train.log\n' "$candidate" "$OUTPUT_DIR"
        elif grep -Eiq 'CUDA out of memory|torch.OutOfMemoryError|CUDA error: out of memory' "$OUTPUT_DIR/train.log"; then
            printf 'OOM microbatch=%s; last passing=%s\n' "$candidate" "$BEST"
            break
        else
            tail -n 60 "$OUTPUT_DIR/train.log"
            exit 1
        fi
    done
    printf '%s\n' "$BEST" > "$BASE_OUTPUT/best_microbatch.txt"
    (( BEST > 0 )) || exit 1
    printf 'Probe complete, not a full epoch. Largest passing microbatch=%s.\n' "$BEST"
    exit
fi

# --run 模式：保存批次配置并启动完整训练，同时将标准输出与错误流同步写入 train.log
printf 'batch=%s actor_micro=%s rollout_micro=%s logprob_micro=%s teacher_micro=%s\n' \
    "$TRAIN_BATCH_SIZE" "$ACTOR_MICRO_BATCH_SIZE" "$ROLLOUT_MICRO_BATCH_SIZE" \
    "$LOGPROB_MICRO_BATCH_SIZE" "$TEACHER_MICRO_BATCH_SIZE" | tee "$OUTPUT_DIR/batch_config.txt"
launch --run "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
