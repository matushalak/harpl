#!/usr/bin/env bash
set -euo pipefail

DEVICE="${HARPL_DEVICE:-auto}"

bash bash_scripts/mnist_sprites_hrpl.sh \
    --device "$DEVICE" \
    --seed "${HARPL_SEED:-0}" \
    --experiment_name "${HARPL_EXPERIMENT_NAME:-mnist_sprites_greedy_cts_noiseOnTop0.1_hier_pred_0}" \
    --loss pred \
    --prediction_target enc \
    --pred_lr_mult 10 "$@"
