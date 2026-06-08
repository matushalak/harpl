#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=HarplMovingAnimals
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_moving_animals%A_%a.out
#SBATCH --error=slurm_moving_animals%A_%a.err

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d harpl && ! -f pyproject.toml ]]; then
    cd harpl
fi

source activate harpl-gpu

export HARPL_DEVICE=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
export WANDB_PROJECT="${WANDB_PROJECT:-HARPL}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-full-runs}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

COMMON_ARGS=(
    --device "$HARPL_DEVICE"
    --epochs "${HARPL_EPOCHS:-1000}"
    --seed "${HARPL_SEED:-2026}"
    --loss pred
    --num_workers "${HARPL_NUM_WORKERS:-8}" \
    --prediction_target enc
    --pred_lr_mult 10
)

case "$TASK_ID" in
    0)
        EXTRA_ARGS=(
            --experiment_name animals_cts_noiseOnTop0.1_predFULL_1
        )
        ;;
    1)
        EXTRA_ARGS=(
            --experiment_name animals_cts_noiseOnTop0.1_predFULL_2
            --spritevid_max_sprites 10
        )
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID: $TASK_ID" >&2
        exit 1
        ;;
esac

bash bash_scripts/moving_animals.sh \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
