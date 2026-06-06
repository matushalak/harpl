#!/usr/bin/env bash

read -r -a SPRITEVID_OUTPUT_SIZE_ARGS <<< "${HARPL_SPRITEVID_OUTPUT_SIZE:-64}"

"${PYTHON:-python}" -m harpl.scripts.cli \
    --dataset animals \
    --spritevid_max_sprites 8 \
    --spritevid_noise_type gaussian \
    --spritevid_noise_level 0.1 \
    --spritevid_output_size "${SPRITEVID_OUTPUT_SIZE_ARGS[@]}" \
    --sprite_noise_on_top \
    --seq_len 32 \
    --num_sequences "${HARPL_NUM_SEQUENCES:-16000}" \
    --encoder conv2d \
    --use_bn \
    --enc_n_layers 6 \
    --enc_kernel_size 5,5 5,5 5,5 5,5 5,5 5,5 \
    --enc_stride 2,2 2,2 1,1 1,1 1,1 1,1 \
    --enc_padding 2,2 2,2 2,2 2,2 2,2 2,2 \
    --enc_output_dim 32 \
    --flatten_enc_output \
    --integrator lstm \
    --ctx_dim 512 \
    --predictor mlp \
    --pred_hidden_dim 512 \
    --pred_steps 1 \
    --checkpoint_every "${HARPL_CHECKPOINT_EVERY:-50}" \
    --epochs "${HARPL_EPOCHS:-500}" \
    --use_scheduler \
    --lr 3e-4 \
    --batch_size "${HARPL_BATCH_SIZE:-128}" \
    --num_workers "${HARPL_NUM_WORKERS:-10}" \
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
    --sigreg_lambd_ 0.05 \
    --sigreg_knots 17 \
    --use_sklearn_regression $@
