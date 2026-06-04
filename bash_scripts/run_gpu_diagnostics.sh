#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${HARPL_DEVICE:-cuda}"
EPOCHS="${HARPL_DIAG_EPOCHS:-2}"
NUM_SEQUENCES="${HARPL_DIAG_NUM_SEQUENCES:-16000}"
PREFETCH_FACTOR="${HARPL_PREFETCH_FACTOR:-2}"
VAL_BATCH_SIZE="${HARPL_DIAG_VAL_BATCH_SIZE:-128}"
OFFLINE_BATCH_SIZE="${HARPL_DIAG_OFFLINE_BATCH_SIZE:-128}"

CONFIGS=(
    "128 8"
)

CONFIG_INDEX="${HARPL_DIAG_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
if (( CONFIG_INDEX < 0 || CONFIG_INDEX >= ${#CONFIGS[@]} )); then
    echo "Invalid diagnostic config index: ${CONFIG_INDEX}" >&2
    exit 2
fi

read -r BATCH_SIZE NUM_WORKERS <<< "${CONFIGS[$CONFIG_INDEX]}"
RUN_ID="${SLURM_JOB_ID:-local}_${CONFIG_INDEX}"
EXPERIMENT_NAME="${HARPL_EXPERIMENT_NAME:-gpu_diag_bs${BATCH_SIZE}_nw${NUM_WORKERS}_e${EPOCHS}_${RUN_ID}}"

echo "Running HARPL GPU diagnostic:"
echo "  experiment=${EXPERIMENT_NAME}"
echo "  device=${DEVICE}"
echo "  epochs=${EPOCHS}"
echo "  num_sequences=${NUM_SEQUENCES}"
echo "  batch_size=${BATCH_SIZE}"
echo "  num_workers=${NUM_WORKERS}"
echo "  val_batch_size=${VAL_BATCH_SIZE}"
echo "  offline_batch_size=${OFFLINE_BATCH_SIZE}"
echo "  prefetch_factor=${PREFETCH_FACTOR}"

bash "$SCRIPT_DIR/moving_animals.sh" \
    --device "$DEVICE" \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs "$EPOCHS" \
    --num_sequences "$NUM_SEQUENCES" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --offline_batch_size "$OFFLINE_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --val_size 0.1 \
    --checkpoint_every 0 \
    --offline_task none \
    --offline_epochs 0 \
    --loss lejepa \
    --prediction_target enc \
    --pred_lr_mult 10 \
    "$@"
