# HARPL

Branch: `jsc-cpu`

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

## Setup On JSC CPU

From this directory:

```bash
bash cluster/jsc-cpu/setup.sh
source cluster/jsc-cpu/activate.sh
```

This branch follows JSC's official `sc_venv_template` pattern: load modules in
`modules.sh`, create a virtual environment with `--system-site-packages`, then
install extra pip packages from `requirements.txt`.

The JSC CPU profile intentionally does not load JSC's PyTorch module. It installs
CPU-only PyTorch wheels and then installs HARPL editable from this checkout:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run

W&B is the default logger. Log in once on the cluster after activating the
environment:

```bash
wandb login
```

Use `--nolog` to disable experiment logging, or `--logger tensorboard` to use
local TensorBoard logs instead.

```bash
bash bash_scripts/mnist_triplets.sh --nolog --epochs 1 --offline_epochs 1
bash bash_scripts/moving_animals.sh --nolog --epochs 1 --offline_epochs 1
bash bash_scripts/hRPL.sh --nolog --epochs 1 --offline_epochs 1
```

TensorBoard logs are written under `runs/<experiment_name>/` when using
`--logger tensorboard`:

```bash
bash bash_scripts/mnist_triplets.sh --epochs 1 --offline_epochs 1
tensorboard --logdir runs
```

This branch is intended for JSC CPU jobs. Run with `--device cpu`.
