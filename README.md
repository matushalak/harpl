# HARPL

HARPL is a focused fork of the original Recurrent Predictive Learning codebase.
It keeps only the code needed to train and evaluate on:

- `animals`: synthetic moving-animal sprite videos
- `mnist`: generated MNIST triplet sequences

The fork removes LibriSpeech, mouse videos, PFC oddballs, notebooks, raw baselines,
and CUDA-only launcher defaults.

## Why The Original Was Not MPS Friendly

- The original bash scripts launched every job with `torchrun` and defaulted to the
  `nccl` distributed backend, which is CUDA-only.
- `repl/scripts/utils.py:init_distributed()` unconditionally called
  `torch.cuda.set_device()` and initialized a process group, even for normal
  single-process runs.
- Entry points selected only `cuda` or `cpu`, so Apple MPS was never chosen.
- Several logging and checkpoint paths called `dist.get_rank()` / `dist.barrier()`
  without checking whether distributed training was initialized.
- `requirements.txt` pinned CUDA-oriented PyTorch packages (`nvidia-*`, `triton`,
  old `torch==2.0.1`) and included dependencies for datasets that are not part of
  this fork.
- The moving-animal dataset generated tensors on CUDA by default when available.
  HARPL keeps dataset generation on CPU by default, then moves training batches to
  the selected compute device.

## Setup

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Apple Silicon, install a current PyTorch build with MPS support. HARPL will use
`mps` automatically when available:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Run

Use `--nolog` to skip Weights & Biases.

```bash
bash bash_scripts/mnist_triplets.sh --nolog --epochs 1 --offline_epochs 1
bash bash_scripts/moving_animals.sh --nolog --epochs 1 --offline_epochs 1
bash bash_scripts/hRPL.sh --nolog --epochs 1 --offline_epochs 1
```

You can force a device with `--device cpu`, `--device mps`, or `--device cuda`.
The default is `--device auto`.
