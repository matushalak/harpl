#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${HARPL_DEVICE:-cpu}"

for i in {1..3}; do
    bash "$SCRIPT_DIR/mnist_triplets.sh" --device "$DEVICE" --seed $i --experiment_name mnist_triplets_pred_$i --loss pred --prediction_target enc;
    bash "$SCRIPT_DIR/mnist_triplets.sh" --device "$DEVICE" --seed $i --experiment_name mnist_triplets_inv_sg_$i --loss inv --prediction_target pred --pull_coef 1.0 --push_coef 1.0 --decorr_coef 10.0;
done

bash "$SCRIPT_DIR/mnist_triplets.sh" --device "$DEVICE" --seed $i --experiment_name mnist_triplets_supervised_$i --loss supervised;
bash "$SCRIPT_DIR/mnist_triplets.sh" --device "$DEVICE" --seed $i --experiment_name mnist_triplets_random_$i --loss supervised --freeze;
