#!/usr/bin/env bash
set -u
cd /Users/matushalak/Documents/harpl
LOG="logs/mnist_sprites_rpl_mps_screen.log"
STATUS="logs/mnist_sprites_rpl_mps_screen.status"
{
  echo "START $(date)"
  /Users/matushalak/miniforge3/envs/harpl/bin/python -m harpl.scripts.cli \
    --dataset mnist_sprites --spritevid_max_sprites 10 --spritevid_noise_type gaussian --spritevid_noise_level 0.1 \
    --spritevid_min_scale 0.4 --spritevid_max_scale 1.2 --spritevid_output_size 64 --sprite_noise_on_top \
    --seq_len 32 --num_sequences 16000 --encoder conv2d --use_bn --enc_n_layers 6 \
    --enc_kernel_size 5,5 5,5 5,5 5,5 5,5 5,5 --enc_stride 2,2 2,2 1,1 1,1 1,1 1,1 \
    --enc_padding 2,2 2,2 2,2 2,2 2,2 2,2 --enc_output_dim 32 --flatten_enc_output \
    --integrator lstm --ctx_dim 512 --predictor mlp --pred_hidden_dim 512 --pred_steps 1 \
    --checkpoint_every 50 --epochs 500 --use_scheduler --lr 3e-4 --batch_size 64 --num_workers 0 \
    --online_task multitask --online_full_spatial_readout --online_input ctx --online_eval_every 5 \
    --val_batch_size 64 --save_online_readout --offline_task multitask --offline_input ctx \
    --offline_batch_size 64 --offline_epochs 250 --save_offline_readout --sigreg_lambd_ 0.05 --sigreg_knots 17 \
    --use_sklearn_regression --device mps --seed 0 \
    --experiment_name mnist_sprites_cts_noiseOnTop0.1_scale0.4-1.2_pred_0_mps_b64_screen \
    --loss pred --prediction_target enc --pred_lr_mult 10 --logger tensorboard
  code=$?
  echo "EXIT_CODE $code $(date)"
  echo "$code" > "$STATUS"
  exit "$code"
} >> "$LOG" 2>&1
