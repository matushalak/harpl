#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplMovingAnimalsParallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=09:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_moving_animals_parallel_%A_%a.out
#SBATCH --error=slurm_moving_animals_parallel_%A_%a.err

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
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-moving-animals}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Config arrays for 3 parallel jobs
SEEDS=(1 2 3)
LOSSES=(lejepa lejepa lejepa)
TARGETS=(enc enc enc)
LR_MULTS=(10 10 10)
EXP_NAMES=(
    "animals_cts_noiseOnTop0.1_lejepa_1"
    "animals_cts_noiseOnTop0.1_lejepa_2"
    "animals_cts_noiseOnTop0.1_lejepa_3"
)
FREEZES=(0 0 0)
CUSTOM_LRS=(0 0 0)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
LOSS="${LOSSES[$SLURM_ARRAY_TASK_ID]}"
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
LR_MULT="${LR_MULTS[$SLURM_ARRAY_TASK_ID]}"
EXP_NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
FREEZE="${FREEZES[$SLURM_ARRAY_TASK_ID]}"
USE_CUSTOM_LR="${CUSTOM_LRS[$SLURM_ARRAY_TASK_ID]}"

# Build arguments
ARGS="--seed $SEED --experiment_name $EXP_NAME --loss $LOSS"

[[ -n "$TARGET" && "$TARGET" != "" ]] && ARGS="$ARGS --prediction_target $TARGET"
[[ -n "$LR_MULT" && "$LR_MULT" != "" ]] && ARGS="$ARGS --pred_lr_mult $LR_MULT"
[[ "$FREEZE" == "1" ]] && ARGS="$ARGS --freeze"
[[ "$USE_CUSTOM_LR" == "1" ]] && ARGS="$ARGS --lr 1e-3"

srun bash bash_scripts/moving_animals.sh $ARGS
