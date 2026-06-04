#!/bin/bash

# Run moving animals training in parallel (for local testing or simple parallelization)
# This launches all 8 jobs concurrently using background processes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${HARPL_DEVICE:-cuda}"

echo "Starting 8 parallel training runs..."
echo "  - 3x lejepa (seeds 1-3)"
echo "  - 3x pred (seeds 1-3)"
echo "  - 1x supervised"
echo "  - 1x random"

# lejepa runs (seeds 1-3)
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 1 --experiment_name animals_cts_noiseOnTop0.1_lejepa_1 --loss lejepa --prediction_target enc --pred_lr_mult 10 &
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 2 --experiment_name animals_cts_noiseOnTop0.1_lejepa_2 --loss lejepa --prediction_target enc --pred_lr_mult 10 &
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 3 --experiment_name animals_cts_noiseOnTop0.1_lejepa_3 --loss lejepa --prediction_target enc --pred_lr_mult 10 &

# pred runs (seeds 1-3)
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 1 --experiment_name animals_cts_noiseOnTop0.1_pred_1 --loss pred --prediction_target enc --pred_lr_mult 10 &
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 2 --experiment_name animals_cts_noiseOnTop0.1_pred_2 --loss pred --prediction_target enc --pred_lr_mult 10 &
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 3 --experiment_name animals_cts_noiseOnTop0.1_pred_3 --loss pred --prediction_target enc --pred_lr_mult 10 &

# supervised runs
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 1 --experiment_name animals_cts_noiseOnTop0.1_supervised_1 --loss supervised --lr 1e-3 &
bash "$SCRIPT_DIR/moving_animals.sh" --device "$DEVICE" --seed 1 --experiment_name animals_cts_noiseOnTop0.1_random_1 --loss supervised --freeze &

# Wait for all background jobs to complete
wait
echo "All training runs completed."
