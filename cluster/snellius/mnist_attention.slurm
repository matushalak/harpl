#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=HarplMnistAttention
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_mnist_attention_%A_%a.out
#SBATCH --error=slurm_mnist_attention_%A_%a.err

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -d harpl && ! -f pyproject.toml ]]; then
    cd harpl
fi

source activate harpl-gpu

export HARPL_DEVICE=cuda
export HARPL_SPRITEVID_DEVICE=cpu
export HARPL_ATTENTION_EPOCHS="${HARPL_ATTENTION_EPOCHS:-300}"
export HARPL_ATTENTION_NUM_SEQUENCES="${HARPL_ATTENTION_NUM_SEQUENCES:-48000}"
export HARPL_ATTENTION_BATCH_SIZE="${HARPL_ATTENTION_BATCH_SIZE:-128}"
export HARPL_ATTENTION_NUM_WORKERS="${HARPL_ATTENTION_NUM_WORKERS:-8}"
export HARPL_ATTENTION_DECODER_LAYERS="${HARPL_ATTENTION_DECODER_LAYERS:-3}"
export HARPL_ATTENTION_LR="${HARPL_ATTENTION_LR:-3e-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-9}"
export WANDB_PROJECT="${WANDB_PROJECT:-HARPL}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mnist-sprites-attention}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
    0)
        export HARPL_ATTENTION_ARCHES=rpl
        export HARPL_ATTENTION_VARIANTS=mask_unweighted
        ;;
    1)
        export HARPL_ATTENTION_ARCHES=rpl
        export HARPL_ATTENTION_VARIANTS=no_mask
        ;;
    2)
        export HARPL_ATTENTION_ARCHES=rpl
        export HARPL_ATTENTION_VARIANTS=cls_feedback_task
        ;;
    3)
        export HARPL_ATTENTION_ARCHES=hrpl
        export HARPL_ATTENTION_VARIANTS=mask_unweighted
        ;;
    4)
        export HARPL_ATTENTION_ARCHES=hrpl
        export HARPL_ATTENTION_VARIANTS=no_mask
        ;;
    5)
        export HARPL_ATTENTION_ARCHES=hrpl
        export HARPL_ATTENTION_VARIANTS=cls_feedback_task
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID: $TASK_ID" >&2
        exit 1
        ;;
esac

srun bash bash_scripts/run_mnist_attention_suite.sh "$@"
