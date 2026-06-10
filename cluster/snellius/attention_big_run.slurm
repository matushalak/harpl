#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=HarplMovingAnimalsAttn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=24:00:00
#SBATCH --array=0
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
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-attention}"

which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

python -m harpl.scripts.cli_attention \
    --device cuda \
    --model_path checkpoints/animals_cts_noiseOnTop0.1_predFULL_1/model_final.pt \
    --epochs 300 \
    --num_workers 8 \
    --batch_size 128 \
    --num_sequences 24000 \
    --lr 3e-4 \
    --sprite_noise_on_top \
    --attention_decoder_layers 3 \
    --spritevid_max_sprites 8
