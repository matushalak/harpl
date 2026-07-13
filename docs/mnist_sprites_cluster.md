# MNIST Sprites RPL/hRPL Runs

These runs keep the animal-sprite RPL and hRPL architecture flags and replace only the foreground sprite source with MNIST train digits resized to 32x32. The generator still tracks the moving-animal latent readouts: digit identity, rotation direction, x/y/z direction, discretized x/y/z position, discretized orientation, speed, rotation speed, continuous x/y/z position, continuous x/y/z velocity, sin(theta), and cos(theta). `mnist_sprites` also adds a sequence-level color readout.

MNIST sampling is class balanced: every sequence samples the digit class uniformly from 0-9, then samples the exact MNIST train-set example uniformly within that class. The MNIST test split is not used. The MNIST-sprite scripts use `--spritevid_min_scale 0.4 --spritevid_max_scale 1.2`; the original animal default remains `0.2..1.0`.

## Local smoke tests

```bash
PYTHON="uv run python" bash bash_scripts/run_mnist_sprites_rpl.sh --nolog --epochs 1 --offline_epochs 1 --num_sequences 64 --batch_size 8 --val_batch_size 8 --offline_batch_size 8 --num_workers 0
PYTHON="uv run python" bash bash_scripts/run_mnist_sprites_hrpl.sh --nolog --epochs 1 --offline_epochs 1 --num_sequences 64 --batch_size 8 --val_batch_size 8 --offline_batch_size 8 --num_workers 0
```

## Full runs

```bash
PYTHON="uv run python" bash bash_scripts/run_mnist_sprites_rpl.sh
PYTHON="uv run python" bash bash_scripts/run_mnist_sprites_hrpl.sh
```

Useful overrides:

```bash
HARPL_DEVICE=cuda HARPL_BATCH_SIZE=256 HARPL_NUM_WORKERS=8 bash bash_scripts/run_mnist_sprites_rpl.sh
HARPL_DEVICE=cuda HARPL_BATCH_SIZE=128 HARPL_NUM_WORKERS=8 bash bash_scripts/run_mnist_sprites_hrpl.sh
```

## Slurm

Submit the base RPL run:

```bash
sbatch bash_scripts/slurm_mnist_sprites.sh
```

Submit the hRPL run:

```bash
HARPL_RUN_KIND=hrpl sbatch bash_scripts/slurm_mnist_sprites.sh
```

If your cluster uses different module names, edit only the module/conda section at the top of `bash_scripts/slurm_mnist_sprites.sh`. The actual training commands are delegated to `run_mnist_sprites_rpl.sh` and `run_mnist_sprites_hrpl.sh`.

## Comparison notebook

Open `notebooks/mnist_sprites_vs_animals.ipynb` from the repository root. It shows first-frame examples from the original animal generator and the MNIST-sprite replacement, and prints the sequence/dense label shapes.
