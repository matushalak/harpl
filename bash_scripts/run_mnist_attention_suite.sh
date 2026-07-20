#!/usr/bin/env bash
set -euo pipefail

read -r -a PYTHON_CMD <<< "${PYTHON:-python}"
DEVICE="${HARPL_DEVICE:-auto}"
SPRITE_DEVICE="${HARPL_SPRITEVID_DEVICE:-cpu}"
EPOCHS="${HARPL_ATTENTION_EPOCHS:-150}"
NUM_SEQUENCES="${HARPL_ATTENTION_NUM_SEQUENCES:-24000}"
BATCH_SIZE="${HARPL_ATTENTION_BATCH_SIZE:-64}"
NUM_WORKERS="${HARPL_ATTENTION_NUM_WORKERS:-4}"
CHECKPOINT_DIR="${HARPL_ATTENTION_CHECKPOINT_DIR:-checkpoints}"
WANDB_GROUP="${WANDB_RUN_GROUP:-mnist-sprites-attention}"
RPL_MODEL_PATH="${HARPL_RPL_MODEL_PATH:-checkpoints/mnist_sprites_cts_noiseOnTop0.1_pred_0/model_final.pt}"
HRPL_MODEL_PATH="${HARPL_HRPL_MODEL_PATH:-checkpoints/mnist_sprites_greedy_cts_noiseOnTop0.1_hier_pred_0/model_final.pt}"
ARCHES="${HARPL_ATTENTION_ARCHES:-rpl hrpl}"
VARIANTS="${HARPL_ATTENTION_VARIANTS:-mask_unweighted no_mask cls_feedback_task}"

LOG_ARGS=()
if [[ "${HARPL_NOLOG:-0}" == "1" ]]; then
    LOG_ARGS+=(--nolog)
else
    LOG_ARGS+=(--logger wandb --wandb_group "$WANDB_GROUP")
fi

COMMON_ARGS=(
    --device "$DEVICE"
    --spritevid_device "$SPRITE_DEVICE"
    --attention_dataset mnist_sprites
    --spritevid_max_sprites 10
    --spritevid_min_scale 0.4
    --spritevid_max_scale 1.2
    --spritevid_output_size 64
    --spritevid_noise_type gaussian
    --spritevid_noise_level 0.1
    --sprite_noise_on_top
    --seq_len 32
    --num_sequences "$NUM_SEQUENCES"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --num_workers "$NUM_WORKERS"
    --lr "${HARPL_ATTENTION_LR:-3e-4}"
    --attention_decoder_kind spatial
    --attention_decoder_layers "${HARPL_ATTENTION_DECODER_LAYERS:-1}"
    --checkpoint_dir "$CHECKPOINT_DIR"
)

RPL_ARGS=(
    --encoder conv2d
    --use_bn
    --enc_n_layers 6
    --enc_kernel_size 5,5 5,5 5,5 5,5 5,5 5,5
    --enc_stride 2,2 2,2 1,1 1,1 1,1 1,1
    --enc_padding 2,2 2,2 2,2 2,2 2,2 2,2
    --enc_output_dim 32
    --flatten_enc_output
    --integrator lstm
    --ctx_dim 512
    --predictor mlp
    --pred_hidden_dim 512
    --pred_steps 1
    --prediction_target enc
)

HRPL_ARGS=(
    --hierarchical
    --n_areas 6
    --area_encoders_kind conv2d
    --area_enc_bn
    --area_enc_n_layers 1
    --area_enc_kernel_sizes 5,5 5,5 5,5 5,5 5,5 5,5
    --area_enc_strides 2,2 2,2 1,1 1,1 1,1 1,1
    --area_enc_paddings 2,2 2,2 2,2 2,2 2,2 2,2
    --area_enc_dims 32 32 32 32 32 32
    --flatten_area_enc_output
    --area_integrators_kind lstm
    --area_ctx_dims 512 512 512 512 512 512
    --area_predictors_kind mlp
    --area_pred_hidden_dims 512 512 512 512 512 512
    --pred_steps 1
    --prediction_target enc
    --attention_readout_area -1
)

variant_args() {
    case "$1" in
        mask_unweighted)
            echo "--attention_mask_loss_weight 1.0 --attention_mask_positive_weight 1.0 --attention_use_task_embedding --attention_use_prompt_embedding"
            ;;
        no_mask)
            echo "--attention_mask_loss_weight 0.0 --no-attention_use_task_embedding --no-attention_use_prompt_embedding"
            ;;
        cls_feedback_task)
            echo "--attention_mask_loss_weight 0.0 --attention_use_task_embedding --attention_use_prompt_embedding"
            ;;
        *)
            echo "Unknown variant: $1" >&2
            return 1
            ;;
    esac
}

for arch in $ARCHES; do
    for variant in $VARIANTS; do
        read -r -a VARIANT_ARGS <<< "$(variant_args "$variant")"
        if [[ "$arch" == "rpl" ]]; then
            model_path="$RPL_MODEL_PATH"
            arch_args=("${RPL_ARGS[@]}")
        elif [[ "$arch" == "hrpl" ]]; then
            model_path="$HRPL_MODEL_PATH"
            arch_args=("${HRPL_ARGS[@]}")
        else
            echo "Unknown arch: $arch" >&2
            exit 1
        fi
        experiment_name="mnist_attention_${arch}_${variant}"
        echo "Running ${experiment_name}"
        "${PYTHON_CMD[@]}" -m harpl.scripts.cli_attention \
            "${LOG_ARGS[@]}" \
            --model_path "$model_path" \
            --experiment_name "$experiment_name" \
            "${COMMON_ARGS[@]}" \
            "${arch_args[@]}" \
            "${VARIANT_ARGS[@]}" \
            "$@"
    done
done
