#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplGpuDiag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_gpu_diag_%A_%a.out
#SBATCH --error=slurm_gpu_diag_%A_%a.err

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d harpl && ! -f pyproject.toml ]]; then
    cd harpl
fi

source activate harpl-gpu

export HARPL_DEVICE=cuda
export HARPL_DIAG_INDEX="$SLURM_ARRAY_TASK_ID"
export HARPL_DIAG_EPOCHS="${HARPL_DIAG_EPOCHS:-2}"
export HARPL_DIAG_NUM_SEQUENCES="${HARPL_DIAG_NUM_SEQUENCES:-16000}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
export WANDB_PROJECT="${WANDB_PROJECT:-HARPL}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-gpu-diagnostics}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

srun bash bash_scripts/run_gpu_diagnostics.sh
