import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from harpl.scripts.args import (
    add_data_args,
    add_model_args,
    add_offline_eval_args,
    add_reproducibility_args,
    add_validation_args,
)
from harpl.scripts.evaluate_saved_readouts import build_base_model, load_data
from harpl.scripts.eval_utils import compute_readout
from harpl.scripts.utils import get_data_specs, seed_everything, select_device


RPL_COLOR = "#994455"
TRUE_COLOR = "black"
TEACHER_COLOR = "#AAAAAA"
OBS_COLOR = "#444444"


def _apply_style():
    sns.set_theme(style="ticks", context="paper", font_scale=0.9)
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7


def _head_indices(args):
    num_classes = get_data_specs(
        args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )[1]
    seq_classes, dense_classes, seq_regression, dense_regression = num_classes
    names = list(seq_classes) + list(dense_classes) + list(seq_regression) + list(dense_regression)
    return {name: idx for idx, name in enumerate(names)}


def _decode_dense_regression(features, readout, args, head_idx, target_length):
    pred = compute_readout(
        data=features,
        target_length=target_length,
        readout=readout[head_idx],
        readout_input=args.offline_input,
        task="seq2seq",
        dense_prediction=args.dense_prediction,
        single_readout=args.offline_single_timestep_readout,
        pred_steps=args.pred_steps,
        full_spatial_readout=args.offline_full_spatial_readout,
    )
    return pred.reshape(features.shape[0], target_length)


def _theta_to_sector(theta):
    return torch.floor(((theta + 0.125) % 1.0) * 4.0).to(torch.long)


def _orientation_bin_to_sector_map(device):
    bins = torch.arange(36, device=device, dtype=torch.float32)
    return torch.floor(((bins / 36.0 + 0.125) % 1.0) * 4.0).to(torch.long)


def _decode_theta4(features, readout, args, head_idx, target_length):
    logits = compute_readout(
        data=features,
        target_length=target_length,
        readout=readout[head_idx],
        readout_input=args.offline_input,
        task="seq2seq",
        dense_prediction=args.dense_prediction,
        single_readout=args.offline_single_timestep_readout,
        pred_steps=args.pred_steps,
        full_spatial_readout=args.offline_full_spatial_readout,
    )
    probs = torch.softmax(logits, dim=1)
    sector_map = _orientation_bin_to_sector_map(logits.device)
    sector_probs = torch.zeros(logits.shape[0], 4, device=logits.device, dtype=probs.dtype)
    sector_probs.index_add_(1, sector_map, probs)
    return sector_probs.argmax(dim=1).reshape(features.shape[0], target_length)


def _encode(model, x):
    data = x
    preprocess_args = {}
    if model.preprocess is not None:
        data, preprocess_args = model.preprocess(data)
    z = model.encoder(data)
    if model.postprocess is not None:
        z = model.postprocess(z, *preprocess_args)
    return z


def _teacher_context(model, z):
    return model.integrator(z)


def _recursive_context(model, z, warmup, rollout_steps):
    if warmup < 1:
        raise ValueError("--warmup must be at least 1.")
    _, hidden = model.integrator.backbone(z[:, :warmup])
    ctx_last = hidden[0][-1:].transpose(0, 1) if isinstance(hidden, tuple) else hidden[-1:].transpose(0, 1)
    z_next = model.predictor(ctx_last)

    ctx_rollout = []
    for _ in range(rollout_steps):
        ctx_t, hidden = model.integrator.backbone(z_next, hidden)
        ctx_rollout.append(ctx_t)
        z_next = model.predictor(ctx_t)
    return torch.cat(ctx_rollout, dim=1)


def _collect(args, model, readout, readout_mean, readout_std, loader, device):
    head_idx = _head_indices(args)
    targets = {key: [] for key in ("x", "y", "z", "sin", "cos", "theta4")}
    teacher = {key: [] for key in targets}
    rollout = {key: [] for key in targets}

    model.eval()
    readout.eval()
    n_seen = 0
    with torch.no_grad():
        for x, labels in tqdm(loader, desc="internal extrapolation"):
            x = x.to(device)
            _, _, _, dense_targets, aux_labels = [item.to(device) for item in labels]
            z = _encode(model, x)
            ctx_teacher = _teacher_context(model, z)
            steps = min(args.rollout_steps, ctx_teacher.shape[1] - args.warmup)
            if steps <= 0:
                continue
            ctx_rollout = _recursive_context(model, z, args.warmup, steps)
            ctx_teacher_future = ctx_teacher[:, args.warmup : args.warmup + steps]
            dense_future = dense_targets[:, args.warmup : args.warmup + steps]
            aux_future = aux_labels[:, args.warmup : args.warmup + steps, 0]

            ctx_teacher_future = (ctx_teacher_future - readout_mean) / (readout_std + 1e-8)
            ctx_rollout = (ctx_rollout - readout_mean) / (readout_std + 1e-8)

            mapping = {
                "x": ("x-position", 0),
                "y": ("y-position", 1),
                "z": ("z-position", 2),
                "sin": ("sin", 6),
                "cos": ("cos", 7),
            }
            for key, (task_name, target_col) in mapping.items():
                targets[key].append(dense_future[:, :, target_col].detach().cpu())
                teacher[key].append(_decode_dense_regression(ctx_teacher_future, readout, args, head_idx[task_name], steps).detach().cpu())
                rollout[key].append(_decode_dense_regression(ctx_rollout, readout, args, head_idx[task_name], steps).detach().cpu())

            targets["theta4"].append(_theta_to_sector(aux_future).detach().cpu())
            teacher["theta4"].append(_decode_theta4(ctx_teacher_future, readout, args, head_idx["orientation"], steps).detach().cpu())
            rollout["theta4"].append(_decode_theta4(ctx_rollout, readout, args, head_idx["orientation"], steps).detach().cpu())

            n_seen += x.shape[0]
            if args.max_sequences and n_seen >= args.max_sequences:
                break

    return (
        {key: torch.cat(value).numpy() for key, value in targets.items()},
        {key: torch.cat(value).numpy() for key, value in teacher.items()},
        {key: torch.cat(value).numpy() for key, value in rollout.items()},
    )


def _horizon_metrics(targets, teacher, rollout):
    rows = []
    for key in ("x", "y", "z", "sin", "cos"):
        for h in range(targets[key].shape[1]):
            y = targets[key][:, h]
            for source, pred in (("teacher_forced", teacher[key][:, h]), ("recursive", rollout[key][:, h])):
                rows.append({
                    "variable": key,
                    "source": source,
                    "horizon": h + 1,
                    "metric": "r2",
                    "value": float(r2_score(y, pred)),
                })
    for h in range(targets["theta4"].shape[1]):
        y = targets["theta4"][:, h]
        for source, pred in (("teacher_forced", teacher["theta4"][:, h]), ("recursive", rollout["theta4"][:, h])):
            rows.append({
                "variable": "theta4",
                "source": source,
                "horizon": h + 1,
                "metric": "accuracy",
                "value": float(np.mean(y == pred)),
            })
    return rows


def _write_metrics(rows, output_prefix):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variable", "source", "horizon", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_horizon(rows, output_prefix):
    variables = ["x", "y", "z", "sin", "cos", "theta4"]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True, constrained_layout=True)
    for ax, var in zip(axes.flat, variables):
        for source, color, ls, label in (
            ("teacher_forced", TEACHER_COLOR, "--", "teacher forced"),
            ("recursive", RPL_COLOR, "-", "recursive"),
        ):
            data = [r for r in rows if r["variable"] == var and r["source"] == source]
            ax.plot([r["horizon"] for r in data], [r["value"] for r in data], color=color, ls=ls, lw=1.7, label=label)
        ax.set_title(var)
        ax.set_xlabel("future step")
        ax.set_ylabel("Acc." if var == "theta4" else r"$R^2$")
        ax.axhline(0, color="black", lw=0.5)
        sns.despine(fig=fig, ax=ax)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.savefig(Path(output_prefix).with_name(Path(output_prefix).name + "_horizon_metrics.png"), dpi=200)
    plt.close(fig)


def _plot_examples(targets, teacher, rollout, warmup, output_prefix, n_examples):
    n = min(n_examples, targets["x"].shape[0])
    fig, axes = plt.subplots(2, n, figsize=(1.65 * n, 3.3), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    future_t = np.arange(warmup, warmup + targets["x"].shape[1])
    for i in range(n):
        ax = axes[0, i]
        ax.plot(targets["x"][i], targets["y"][i], color=TRUE_COLOR, lw=1.5, label="true future")
        ax.plot(teacher["x"][i], teacher["y"][i], color=TEACHER_COLOR, lw=1.2, ls="--", label="teacher")
        ax.plot(rollout["x"][i], rollout["y"][i], color=RPL_COLOR, lw=1.5, label="recursive")
        ax.scatter(rollout["x"][i, 0], rollout["y"][i, 0], color=OBS_COLOR, s=12, zorder=5)
        ax.set_title(f"seq {i}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        sns.despine(fig=fig, ax=ax)

        ax = axes[1, i]
        theta_true = np.degrees(np.arctan2(targets["sin"][i], targets["cos"][i]))
        theta_pred = np.degrees(np.arctan2(rollout["sin"][i], rollout["cos"][i]))
        ax.plot(future_t, theta_true, color=TRUE_COLOR, lw=1.2, label="true")
        ax.plot(future_t, theta_pred, color=RPL_COLOR, lw=1.2, label="recursive")
        ax.set_ylim(-185, 185)
        ax.set_xlabel("frame")
        if i == 0:
            ax.set_ylabel("theta")
        sns.despine(fig=fig, ax=ax)

    axes[0, 0].legend(frameon=False, fontsize=6, loc="best")
    fig.savefig(Path(output_prefix).with_name(Path(output_prefix).name + "_examples.png"), dpi=200)
    plt.close(fig)


def make_parser():
    parser = argparse.ArgumentParser(description="RPL internal recursive extrapolation for MNIST-sprite checkpoints.")
    parser.add_argument("--readout_path", required=True)
    parser.add_argument("--readout_stats_path", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=24)
    parser.add_argument("--max_sequences", type=int, default=512)
    parser.add_argument("--n_examples", type=int, default=6)
    add_reproducibility_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    return parser


def main():
    _apply_style()
    args = make_parser().parse_args()
    if args.offline_task != "multitask":
        raise ValueError("--offline_task must be multitask.")
    seed_everything(args.seed)
    device = select_device(args.device)
    (_, _, _, _, test_loader, _) = load_data(args)
    model, readout = build_base_model(args, device)
    readout.load_state_dict(torch.load(args.readout_path, map_location=device))
    stats = torch.load(args.readout_stats_path, map_location=device)
    targets, teacher, rollout = _collect(args, model, readout, stats["mean"].to(device), stats["std"].to(device), test_loader, device)
    rows = _horizon_metrics(targets, teacher, rollout)
    _write_metrics(rows, args.output_prefix)
    _plot_horizon(rows, args.output_prefix)
    _plot_examples(targets, teacher, rollout, args.warmup, args.output_prefix, args.n_examples)
    print(f"Wrote extrapolation outputs to {Path(args.output_prefix).parent}")


if __name__ == "__main__":
    main()
