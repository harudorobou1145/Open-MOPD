#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Python interpreter; inherited from the experiment launcher when set.
export PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
# Physical GPU IDs; direct invocation defaults to GPU 2.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Original prompts per batch before W4/W8 expansion; direct default is 8.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# Maximum prompt tokens; longer prompts are filtered out.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
# Maximum generated tokens per trajectory; directly affects rollout cost.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
# Shared W4/W8 calibration cache containing quantization indices and codebooks.
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-${PROJECT_ROOT}/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/calibration_cache/quantized/(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512}"
# Artifact directory; defaults to a fresh timestamped path under /data.
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/wikitext_$(date +%Y%m%d_%H%M%S)}"

# LOCAL_RUN=1 executes training even without --run; otherwise dry-run is allowed.
if [[ " ${*} " == *" --run "* || " ${*} " == *" --exec "* || "${LOCAL_RUN:-0}" == 1 ]]; then
    output_parent="$OUTPUT_DIR"
    while [[ ! -e "$output_parent" ]]; do
        output_parent="$(dirname "$output_parent")"
    done
    if [[ ! -d "$output_parent" || ! -w "$output_parent" ]]; then
        printf 'Output parent is not writable: %s\n' "$output_parent" >&2
        exit 2
    fi
fi

# MODEL_PATH: base student model; default is the local Qwen3.5-0.8B directory.
# W4_TEACHER: local W4 AWQ teacher; W8_TEACHER: local W8 GPTQ teacher.
# TRAIN_FILE: training parquet; VAL_FILE: validation parquet (defaults to TRAIN's standard path).
# Validation is disabled below; the validation file is not the held-out PPL test split.
exec bash "${SCRIPT_DIR}/mt_opd.sh" \
    --model "${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-0.8B}" \
    --teacher "${W4_TEACHER:-${PROJECT_ROOT}/models/Qwen3.5-0.8B-W4A16-AutoRound-AWQ}" \
    --teacher "${W8_TEACHER:-${PROJECT_ROOT}/models/Qwen3.5-0.8B-GPTQ-MLP-W8-cal128-seq2048}" \
    --train "${TRAIN_FILE:-${PROJECT_ROOT}/data/wikitext2_mopd/train.parquet}" \
    --val "${VAL_FILE:-${PROJECT_ROOT}/data/wikitext2_mopd/train.parquet}" \
    --extra trainer.val_before_train=False \
    --extra trainer.test_freq=-1 \
    --extra trainer.save_freq=100 \
    --extra trainer.max_actor_ckpt_to_keep=1 \
    --extra trainer.resume_mode=disable \
    --extra data.dataloader_num_workers=0 \
    --extra data.seed=42 \
    --extra '+data.apply_chat_template_kwargs.chat_template="{{ messages[0][\"content\"] }}"' \
    --extra ++actor_rollout_ref.rollout.log_prob_top_k=4 \
    --extra actor_rollout_ref.actor.use_kl_loss=False \
    --extra algorithm.use_kl_in_reward=False \
    --extra ++mt_reward_model_1.model.quantization_backend=torch \
    --extra mt_reward_model_1.model.dtype=fp16 \
        --extra data.return_multi_modal_inputs=False \
    "$@"