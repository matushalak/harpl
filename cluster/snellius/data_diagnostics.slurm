#!/usr/bin/env bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplDataDiag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --array=0-0
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

srun bash bash_scripts/benchmark_moving_animals_data_device.sh