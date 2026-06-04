# HARPL

Branch: `gpu-cuda`

HARPL is a focused fork of the original Recurrent Predictive Learning codebase.
It keeps only the code needed to train and evaluate on:

- `animals`: synthetic moving-animal sprite videos
- `mnist`: generated MNIST triplet sequences

The fork removes LibriSpeech, mouse videos, PFC oddballs, notebooks, raw baselines,
and CUDA-only launcher defaults.

This branch is intended for CUDA GPU runs. It keeps the cleaned HARPL dataset
scope while restoring the original CUDA-oriented install and `torchrun` launcher
shape from `fmi-basel/recurrent-predictive-learning`.

## Setup

From this directory:

```bash
mamba env create -f environment.yml
mamba activate harpl-gpu
```

The Conda environment uses Python 3.10 and pip-installs `requirements.txt`,
including CUDA 11 PyTorch, NVIDIA runtime wheels, NCCL, and Triton. Verify CUDA
before launching training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run

TensorBoard is the default logger. Use `--nolog` to disable experiment logging.

```bash
HARPL_NPROC_PER_NODE=1 bash bash_scripts/mnist_triplets.sh --epochs 1 --offline_epochs 1
HARPL_NPROC_PER_NODE=1 bash bash_scripts/moving_animals.sh --epochs 1 --offline_epochs 1
HARPL_NPROC_PER_NODE=1 bash bash_scripts/hRPL.sh --epochs 1 --offline_epochs 1
```

Increase `HARPL_NPROC_PER_NODE` for multi-GPU training.

TensorBoard logs are written under `runs/<experiment_name>/` by default:

```bash
bash bash_scripts/mnist_triplets.sh --epochs 1 --offline_epochs 1
tensorboard --logdir runs
```

Use W&B explicitly with `--logger wandb`.

This branch defaults distributed training to the `nccl` backend and is not meant
for Apple MPS.
