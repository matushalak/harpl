#!/usr/bin/env bash
#SBATCH --job-name=mnist-sprites-rpl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
# module load cuda/12.1
# module load mamba

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${HARPL_ENV:-harpl}"

export HARPL_DEVICE="${HARPL_DEVICE:-cuda}"
export HARPL_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export HARPL_BATCH_SIZE="${HARPL_BATCH_SIZE:-128}"
export HARPL_VAL_BATCH_SIZE="${HARPL_VAL_BATCH_SIZE:-128}"
export HARPL_NUM_SEQUENCES="${HARPL_NUM_SEQUENCES:-16000}"
export HARPL_EPOCHS="${HARPL_EPOCHS:-500}"
export HARPL_OFFLINE_EPOCHS="${HARPL_OFFLINE_EPOCHS:-250}"

case "${HARPL_RUN_KIND:-rpl}" in
    rpl)
        bash bash_scripts/run_mnist_sprites_rpl.sh "$@"
        ;;
    hrpl)
        bash bash_scripts/run_mnist_sprites_hrpl.sh "$@"
        ;;
    *)
        echo "HARPL_RUN_KIND must be rpl or hrpl" >&2
        exit 2
        ;;
esac
