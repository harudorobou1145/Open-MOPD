#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export ANYPRECISION_ROLLOUT=hf
export ANYPRECISION_CACHE="${ANYPRECISION_CACHE:-/data/anyprecision_full_student/cache/quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512}"
export ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-1}"
export LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-1}"
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/full_student_w2w4_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/run_w2w4_bf16.sh" "${@:---dry-run}"