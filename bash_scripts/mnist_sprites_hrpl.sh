#!/usr/bin/env bash
set -euo pipefail

read -r -a SPRITEVID_OUTPUT_SIZE_ARGS <<< "${HARPL_SPRITEVID_OUTPUT_SIZE:-64}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python3}"

"${PYTHON_CMD[@]}" -m harpl.scripts.greedy \
    --dataset mnist_sprites \
    --spritevid_max_sprites 10 \
    --spritevid_noise_type gaussian \
    --spritevid_noise_level 0.1 \
    --spritevid_min_scale 0.4 \
    --spritevid_max_scale 1.2 \
    --spritevid_output_size "${SPRITEVID_OUTPUT_SIZE_ARGS[@]}" \
    --sprite_noise_on_top \
    --seq_len 32 \
    --num_sequences "${HARPL_NUM_SEQUENCES:-16000}" \
    --n_areas 6 \
    --area_encoders_kind conv2d \
    --area_enc_bn \
    --area_enc_n_layers 1 \
    --area_enc_kernel_sizes 5,5 5,5 5,5 5,5 5,5 5,5 \
    --area_enc_strides 2,2 2,2 1,1 1,1 1,1 1,1 \
    --area_enc_paddings 2,2 2,2 2,2 2,2 2,2 2,2 \
    --area_enc_dims 32 32 32 32 32 32 \
    --flatten_area_enc_output \
    --area_integrators_kind lstm \
    --area_ctx_dims 512 512 512 512 512 512 \
    --area_predictors_kind mlp \
    --area_pred_hidden_dims 512 512 512 512 512 512 \
    --pred_steps 1 \
    --checkpoint_every "${HARPL_CHECKPOINT_EVERY:-50}" \
    --epochs "${HARPL_EPOCHS:-500}" \
    --lr 3e-4 \
    --batch_size "${HARPL_BATCH_SIZE:-128}" \
    --num_workers "${HARPL_NUM_WORKERS:-8}" \
    --online_task multitask \
    --online_full_spatial_readout \
    --online_input ctx \
    --online_eval_every 5 \
    --val_batch_size "${HARPL_VAL_BATCH_SIZE:-128}" \
    --save_online_readout \
    --offline_task multitask \
    --offline_input ctx \
    --offline_batch_size "${HARPL_OFFLINE_BATCH_SIZE:-128}" \
    --offline_epochs "${HARPL_OFFLINE_EPOCHS:-250}" \
    --save_offline_readout \
    --use_sklearn_regression "$@"
