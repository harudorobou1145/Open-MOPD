#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANYPRECISION_ROLLOUT="${ANYPRECISION_ROLLOUT:-hf}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/anyprecision_mopd/prefix_pv_smoke_$(date +%Y%m%d_%H%M%S)}"

MODE="${1:---dry-run}"
if (($#)); then shift; fi

exec bash "$SCRIPT_DIR/run_w2w4_bf16.sh" "$MODE" \
    '+actor_rollout_ref.actor.prefix_pv_enabled=true' \
    '+actor_rollout_ref.actor.prefix_pv_module=0' \
    '+actor_rollout_ref.actor.prefix_pv_lr=0.001' \
    '+actor_rollout_ref.actor.prefix_pv_trust_ratio=0.001' \
    '+actor_rollout_ref.actor.prefix_pv_max_fraction=0.01' \
    'trainer.total_training_steps=3' \
    'trainer.save_freq=3' \
    'trainer.resume_mode=disable' \
    "$@"