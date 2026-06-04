#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplGpuLRDiag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_gpu_lrdiag_%A_%a.out
#SBATCH --error=slurm_gpu_lrdiag_%A_%a.err

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d harpl && ! -f pyproject.toml ]]; then
    cd harpl
fi

source activate harpl-gpu

export HARPL_DEVICE=cuda
export HARPL_LR_DIAG_INDEX="$SLURM_ARRAY_TASK_ID"
export HARPL_LR_DIAG_EPOCHS="${HARPL_LR_DIAG_EPOCHS:-10}"
export HARPL_LR_DIAG_NUM_SEQUENCES="${HARPL_LR_DIAG_NUM_SEQUENCES:-16000}"
export HARPL_LR_DIAG_NUM_WORKERS="${HARPL_LR_DIAG_NUM_WORKERS:-8}"
export HARPL_LR_DIAG_VAL_BATCH_SIZE="${HARPL_LR_DIAG_VAL_BATCH_SIZE:-128}"
export HARPL_LR_DIAG_OFFLINE_BATCH_SIZE="${HARPL_LR_DIAG_OFFLINE_BATCH_SIZE:-128}"
export HARPL_LR_DIAG_OFFLINE_TASK="${HARPL_LR_DIAG_OFFLINE_TASK:-none}"
export HARPL_LR_DIAG_OFFLINE_EPOCHS="${HARPL_LR_DIAG_OFFLINE_EPOCHS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
export WANDB_PROJECT="${WANDB_PROJECT:-HARPL}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpu-lr-scaling-diagnostics}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

srun bash bash_scripts/run_gpu_lr_scaling_diagnostics.sh
