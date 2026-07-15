#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplMnistSprites
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_mnist_sprites_%A_%a.out
#SBATCH --error=slurm_mnist_sprites_%A_%a.err

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d harpl && ! -f pyproject.toml ]]; then
    cd harpl
fi

source activate harpl-gpu

export HARPL_DEVICE=cuda
export HARPL_NUM_WORKERS="${HARPL_NUM_WORKERS:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
export WANDB_PROJECT="${WANDB_PROJECT:-HARPL}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mnist-sprites}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK_ID" in
    0)
        srun bash bash_scripts/run_mnist_sprites_rpl.sh "$@"
        ;;
    1)
        srun bash bash_scripts/run_mnist_sprites_hrpl.sh "$@"
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID: $TASK_ID" >&2
        exit 1
        ;;
esac
