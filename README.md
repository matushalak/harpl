# HARPL

Branch: `gpu-cuda`

HARPL is a focused fork of the original Recurrent Predictive Learning codebase.
It keeps only the code needed to train and evaluate on:

- `animals`: synthetic moving-animal sprite videos
- `mnist`: generated MNIST triplet sequences

The fork removes LibriSpeech, mouse videos, PFC oddballs, notebooks, raw baselines,
and CUDA-only launcher defaults.

This branch is intended for CUDA GPU runs. It keeps the cleaned HARPL dataset
scope while using the CUDA-oriented PyTorch dependency stack from
`fmi-basel/recurrent-predictive-learning`.

## Setup

From this directory:

```bash
mamba env create -f environment.yml
mamba activate harpl-gpu
```

The Conda environment uses Python 3.10 and pip-installs `requirements.txt`,
including CUDA 11 PyTorch, NVIDIA runtime wheels, NCCL, and Triton. HARPL itself
currently runs single-process CUDA; its argument checks still reject distributed
training. Verify CUDA before launching training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run

W&B is the default logger. Log in once per machine before running:

```bash
wandb login
```

Use `--nolog` to disable experiment logging, or `--logger tensorboard` to use
local TensorBoard logs instead.

```bash
bash bash_scripts/mnist_triplets.sh --device cuda --epochs 1 --offline_epochs 1
bash bash_scripts/moving_animals.sh --device cuda --epochs 1 --offline_epochs 1
bash bash_scripts/hRPL.sh --device cuda --epochs 1 --offline_epochs 1
```

TensorBoard logs are written under `runs/<experiment_name>/` when using
`--logger tensorboard`:

```bash
bash bash_scripts/mnist_triplets.sh --epochs 1 --offline_epochs 1
tensorboard --logdir runs
```

This branch is not meant for Apple MPS. Multi-GPU DDP is not enabled by default
because `check_args()` currently rejects `--distributed`.
