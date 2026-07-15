#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/Users/matushalak/miniforge3/envs/harpl/bin/python}"
DEVICE="${HARPL_DEVICE:-mps}"
MODEL_RUN="${HARPL_MNIST_RPL_RUN:-mnist_sprites_cts_noiseOnTop0.1_scale0.4-1.2_pred_0_mps_b64_screen}"
MODEL_PATH="${HARPL_MODEL_PATH:-checkpoints/${MODEL_RUN}/model_final.pt}"
READOUT_PATH="${HARPL_READOUT_PATH:-checkpoints/${MODEL_RUN}/online_ctx_readout.pt}"
LOG_DIR="${HARPL_ATTENTION_LOG_DIR:-logs/mnist_bio_attention}"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  -m harpl.scripts.cli_attention
  --model_path "$MODEL_PATH"
  --attention_readout_path "$READOUT_PATH"
  --attention_sprite_source mnist
  --spritevid_max_sprites 10
  --spritevid_output_size 64
  --spritevid_noise_type gaussian
  --spritevid_noise_level 0.1
  --sprite_noise_on_top
  --attention_scale_range 0.4 1.2
  --no-attention_object_recognition_matches_pretraining
  --seq_len 32
  --num_sequences "${HARPL_ATTENTION_NUM_SEQUENCES:-512}"
  --attention_val_sequences "${HARPL_ATTENTION_VAL_SEQUENCES:-128}"
  --attention_test_sequences "${HARPL_ATTENTION_TEST_SEQUENCES:-256}"
  --batch_size "${HARPL_ATTENTION_BATCH_SIZE:-8}"
  --epochs "${HARPL_ATTENTION_EPOCHS:-5}"
  --max_batches "${HARPL_ATTENTION_MAX_BATCHES:-16}"
  --attention_eval_every "${HARPL_ATTENTION_EVAL_EVERY:-0}"
  --attention_report_batches "${HARPL_ATTENTION_REPORT_BATCHES:-8}"
  --attention_panel_examples "${HARPL_ATTENTION_PANEL_EXAMPLES:-2}"
  --attention_task mixed
  --popout_mode mixed
  --num_distractors 2
  --crowd_size 3
  --cue_frames 5
  --attention_decoder_kind bio
  --attention_apply_stage encoder_layers
  --attention_supervision_target silhouette
  --attention_supervision_scale bio
  --attention_class_feedback_mode target
  --attention_use_task_embedding
  --attention_supervise_all_layers
  --device "$DEVICE"
  --spritevid_device cpu
  --num_workers "${HARPL_ATTENTION_NUM_WORKERS:-0}"
  --logger tensorboard
  --wandb_group mnist_bio_attention
  --checkpoint_dir checkpoints
)

run_condition() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  echo "=== ${name} ===" | tee "$log_path"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" --experiment_name "$name" "$@" 2>&1 | tee -a "$log_path"
}

run_condition mnist_bio_attention_cls_only \
  --attention_class_loss_weight 1 \
  --attention_supervision_weight 0

run_condition mnist_bio_attention_joint_bce_weak \
  --attention_class_loss_weight 1 \
  --attention_supervision_weight 0.25 \
  --attention_supervision_mse_weight 0 \
  --attention_supervision_bce_weight 1 \
  --attention_supervision_positive_weight 20

run_condition mnist_bio_attention_joint_bce \
  --attention_class_loss_weight 1 \
  --attention_supervision_weight 1 \
  --attention_supervision_mse_weight 0 \
  --attention_supervision_bce_weight 1 \
  --attention_supervision_positive_weight 20

run_condition mnist_bio_attention_joint_bce_strong \
  --attention_class_loss_weight 1 \
  --attention_supervision_weight 5 \
  --attention_supervision_mse_weight 0 \
  --attention_supervision_bce_weight 1 \
  --attention_supervision_positive_weight 40

run_condition mnist_bio_attention_mask_only_bce \
  --attention_class_loss_weight 0 \
  --attention_supervision_weight 1 \
  --attention_supervision_mse_weight 0 \
  --attention_supervision_bce_weight 1 \
  --attention_supervision_positive_weight 20
