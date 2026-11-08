#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 训练和 TensorBoard 使用的 Python 解释器，默认使用系统 Python。
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
# 物理 GPU 编号，顺序决定生成进程的编号，默认使用 2,3。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
# 生成后端：vllm（默认）或 hf。
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-vllm}"
# 每个生成批次的原始提示词数量，每条提示词会展开为 W4 和 W8 两条轨迹。
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
# 遍历数据集的轮数，可通过 trainer.total_training_steps 提前限制训练步数。
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# 实验输出目录，默认在 /data 下创建带时间戳的新目录。
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/batched_$(date +%Y%m%d_%H%M%S)}"
# TensorBoard 事件目录，始终由 OUTPUT_DIR 派生。
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
# Ray 临时目录，默认在 /data 下为每次启动创建独立目录。
export RAY_TMPDIR="${RAY_TMPDIR:-/data/ap_ray_$(date +%s)_$$}"
# 立即刷新 Python 输出，避免后台日志因缓冲而延迟。
export PYTHONUNBUFFERED=1
# 实验名称，默认使用输出目录的末级目录名。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$OUTPUT_DIR")}"
# 每张 GPU 上 actor 单次前向和反向的样本数，增大会增加激活显存。
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-8}"
# 生成微批次大小，仅用于 HF 后端，不是 vLLM 的并发上限。
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-16}"
# 每张 GPU 单次计算学生对数概率的样本数，默认等于 actor 微批次大小。
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-$ACTOR_MICRO_BATCH_SIZE}"
# 每张 GPU 单次教师前向的样本数，增大会增加教师激活显存。
TEACHER_MICRO_BATCH_SIZE="${TEACHER_MICRO_BATCH_SIZE:-2}"
# True 表示教师仅给匹配精度的样本打分；启用教师冲突策略时需要完整打分。
PRECISION_TEACHER_SCORING="${PRECISION_TEACHER_SCORING:-True}"
# W4 学生损失的相对权重，默认为 1；两个权重都必须为正数。
# 两个权重会归一化至均值为 1；设置 W4=1、W8=2 时，实际乘数为 2/3、4/3。
W4_LOSS_WEIGHT="${W4_LOSS_WEIGHT:-1}"
# W8 学生损失的相对权重，默认为 1；设为 2 可得到 W4:W8 为 1:2。
W8_LOSS_WEIGHT="${W8_LOSS_WEIGHT:-2}"
# 单卡 --tune 模式的候选微批次大小，使用空格分隔的正整数。
MICRO_BATCH_CANDIDATES="${MICRO_BATCH_CANDIDATES:-4 8 16 20 24}"
MODE="${1:---run}"
if (($#)); then shift; fi
case "$MODE" in
    --run|--dry-run|--tune) ;;
    *) printf 'Usage: bash %s [--run|--dry-run|--tune] [Hydra overrides...]\n' "$0"; exit 2 ;;
esac
IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_IDS[@]}"
# 生成精度分配：auto 在双卡时自动让两个进程分别生成 W4 和 W8。
# 可显式设置 True 或 False 覆盖自动行为；True 要求使用两张 GPU。
PRECISION_PER_GPU="${PRECISION_PER_GPU:-auto}"
if (( GPU_COUNT > 2 )); then
    printf 'This launcher supports at most two GPUs.\n' >&2
    exit 2
fi
if (( GPU_COUNT == 2 )); then
    (( TRAIN_BATCH_SIZE % 2 == 0 )) || { printf 'Two GPUs require an even global batch.\n' >&2; exit 2; }
    [[ "$MODE" != --tune ]] || { printf 'Two-GPU batch tuning is not supported.\n' >&2; exit 2; }
fi
if [[ "$PRECISION_PER_GPU" == auto ]]; then
    PRECISION_PER_GPU=False
    if (( GPU_COUNT == 2 )); then PRECISION_PER_GPU=True; fi
fi
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

launch() {
    local mode="$1"
    shift
    bash "${SCRIPT_DIR}/wikitext_anyprecision_mopd.sh" "$mode" -- \
        "trainer.n_gpus_per_node=${GPU_COUNT}" \
        "++trainer.precision_per_gpu=${PRECISION_PER_GPU}" \
        "++mt_opd.precision_teacher_scoring=${PRECISION_TEACHER_SCORING}" \
        "++mt_opd.precision_loss_weights=[${W4_LOSS_WEIGHT},${W8_LOSS_WEIGHT}]" \
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

if [[ "$MODE" == --dry-run ]]; then
    launch --dry-run "$@"
    exit
fi
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
printf 'batch=%s actor_micro=%s rollout_micro=%s logprob_micro=%s teacher_micro=%s\n' \
    "$TRAIN_BATCH_SIZE" "$ACTOR_MICRO_BATCH_SIZE" "$ROLLOUT_MICRO_BATCH_SIZE" \
    "$LOGPROB_MICRO_BATCH_SIZE" "$TEACHER_MICRO_BATCH_SIZE" | tee "$OUTPUT_DIR/batch_config.txt"
launch --run "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"