import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from harpl.scripts.evaluate_saved_readouts import (
    build_base_model,
    build_hierarchical_model,
    load_data,
)
from harpl.scripts.args import (
    add_data_args,
    add_model_args,
    add_offline_eval_args,
    add_reproducibility_args,
    add_validation_args,
    size_tuple,
)
from harpl.scripts.eval_utils import compute_readout
from harpl.scripts.utils import get_data_specs, seed_everything, select_device


SECTOR_LABELS = ["0 deg", "90 deg", "180 deg", "270 deg"]
RPL_COLOR = "#994455"
HIER_COLOR = "#AAAAAA"
CONFUSION_CMAP = LinearSegmentedColormap.from_list("rpl_confusion", ["#fbf7f8", RPL_COLOR])


def _apply_style():
    sns.set_theme(style="ticks", context="paper", font_scale=0.9)
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7


def _read_csv_metrics(path):
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["value"] = float(row["value"])
            rows.append(row)
    return rows


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < n_classes and 0 <= pred < n_classes:
            matrix[int(true), int(pred)] += 1
    return matrix


def _row_normalized(matrix):
    denom = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom != 0)


def _plot_matrix(matrix, labels, title, path, normalize=True):
    values = _row_normalized(matrix) if normalize else matrix
    fig_w = max(4.8, 0.55 * len(labels) + 2.0)
    fig_h = max(4.2, 0.50 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    image = ax.imshow(values, cmap=CONFUSION_CMAP, vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{values[i, j]:.2f}" if normalize else str(matrix[i, j])
            ax.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > 0.55 else "black", fontsize=6)
    sns.despine(fig=fig, ax=ax, left=False, bottom=False)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric_bars(rows, metric, title, path):
    filtered = [r for r in rows if r["metric"] == metric]
    if not filtered:
        return
    tasks = sorted({f"{r['group']}: {r['task']}" for r in filtered})
    series = sorted({f"{r['model']}" + (f" area {r['area']}" if str(r.get("area", "")) != "" else "") for r in filtered})
    values = {name: {task: np.nan for task in tasks} for name in series}
    for row in filtered:
        name = f"{row['model']}" + (f" area {row['area']}" if str(row.get("area", "")) != "" else "")
        values[name][f"{row['group']}: {row['task']}"] = row["value"]

    x = np.arange(len(tasks))
    width = min(0.8 / max(len(series), 1), 0.18)
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(tasks)), 4.8), constrained_layout=True)
    palette = []
    for name in series:
        if name == "RPL":
            palette.append(RPL_COLOR)
        else:
            try:
                area = int(name.rsplit(" ", 1)[-1])
            except ValueError:
                area = 0
            shade = 0.35 + 0.08 * area
            palette.append(str(min(shade, 0.82)))
    for idx, name in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, [values[name][task] for task in tasks], width=width, label=name, color=palette[idx])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=60, ha="right")
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "R2")
    ax.set_title(title, fontsize=8)
    ax.legend(ncols=min(4, max(1, len(series))), fontsize=8)
    sns.despine(fig=fig, ax=ax)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _theta_to_sector(theta):
    return torch.floor(((theta + 0.125) % 1.0) * 4.0).to(torch.long)


def _orientation_bin_to_sector_map(device):
    bins = torch.arange(36, device=device, dtype=torch.float32)
    return torch.floor(((bins / 36.0 + 0.125) % 1.0) * 4.0).to(torch.long)


def _collapse_orientation_logits(logits36):
    probs = torch.softmax(logits36, dim=1)
    sector_map = _orientation_bin_to_sector_map(logits36.device)
    sector_probs = torch.zeros(logits36.shape[0], 4, device=logits36.device, dtype=probs.dtype)
    sector_probs.index_add_(1, sector_map, probs)
    return sector_probs


def _task_indices(args):
    num_classes = get_data_specs(
        args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )[1]
    seq_classes, dense_classes, _, _ = num_classes
    dense_names = list(dense_classes.keys())
    return {
        "digit": list(seq_classes.keys()).index("digit"),
        "readout_digit": list(seq_classes.keys()).index("digit"),
        "readout_orientation": len(seq_classes) + dense_names.index("orientation"),
    }


def _model_output(args, model, x, area=None):
    if args.offline_input == "enc":
        idx = 0
    elif args.offline_input == "ctx":
        idx = 1
    elif args.offline_input == "pred":
        idx = 2
    else:
        raise ValueError(f"Invalid offline input: {args.offline_input}")
    output = model(x)[idx]
    if area is not None:
        output = output[area]
    return output


def _collect_from_multitask(args, model, readout, loader, device, mean, std, area=None):
    indices = _task_indices(args)
    out = {
        "digit_true": [],
        "digit_pred": [],
        "theta4_true": [],
        "theta4_pred": [],
        "theta4_digit": [],
    }
    model.eval()
    readout.eval()
    with torch.no_grad():
        for x, labels in tqdm(loader, desc=f"collect area {area}" if area is not None else "collect"):
            x = x.to(device)
            seq_labels, _, _, _, aux_labels = [item.to(device) for item in labels]
            features = _model_output(args, model, x, area=area)
            features = (features - mean) / (std + 1e-8)

            digit_logits = compute_readout(
                data=features,
                target_length=None,
                readout=readout[indices["readout_digit"]],
                readout_input=args.offline_input,
                task="seq2label",
                dense_prediction=args.dense_prediction,
                single_readout=args.offline_single_timestep_readout,
                pred_steps=args.pred_steps,
                full_spatial_readout=args.offline_full_spatial_readout,
            )
            orientation_logits = compute_readout(
                data=features,
                target_length=aux_labels.shape[1],
                readout=readout[indices["readout_orientation"]],
                readout_input=args.offline_input,
                task="seq2seq",
                dense_prediction=args.dense_prediction,
                single_readout=args.offline_single_timestep_readout,
                pred_steps=args.pred_steps,
                full_spatial_readout=args.offline_full_spatial_readout,
            )
            theta4_probs = _collapse_orientation_logits(orientation_logits)
            theta4_true = _theta_to_sector(aux_labels[:, :, 0]).reshape(-1)
            digit_true = seq_labels[:, indices["digit"]]

            out["digit_true"].append(digit_true.cpu())
            out["digit_pred"].append(digit_logits.argmax(dim=1).cpu())
            out["theta4_true"].append(theta4_true.cpu())
            out["theta4_pred"].append(theta4_probs.argmax(dim=1).cpu())
            out["theta4_digit"].append(digit_true[:, None].expand(-1, aux_labels.shape[1]).reshape(-1).cpu())

    return {key: torch.cat(value).numpy() for key, value in out.items()}


def _collect_theta4_readout(args, model, loader, bundle_path, device):
    bundle = torch.load(bundle_path, map_location=device)
    linear = nn.Linear(bundle["weight"]["weight"].shape[1], 4).to(device)
    linear.load_state_dict(bundle["weight"])
    mean = bundle["mean"].to(device)
    std = bundle["std"].to(device)

    out = {"theta4_true": [], "theta4_pred": [], "theta4_digit": []}
    model.eval()
    linear.eval()
    with torch.no_grad():
        for x, labels in tqdm(loader, desc="collect theta4 bundle"):
            x = x.to(device)
            seq_labels, _, _, _, aux_labels = [item.to(device) for item in labels]
            features = _model_output(args, model, x)
            features = (features - mean) / (std + 1e-8)
            logits = linear(features.reshape(-1, features.shape[-1]))
            out["theta4_true"].append(_theta_to_sector(aux_labels[:, :, 0]).reshape(-1).cpu())
            out["theta4_pred"].append(logits.argmax(dim=1).cpu())
            out["theta4_digit"].append(seq_labels[:, 0][:, None].expand(-1, aux_labels.shape[1]).reshape(-1).cpu())
    return {key: torch.cat(value).numpy() for key, value in out.items()}


def _write_confusion_outputs(collected, name, plot_dir):
    plot_dir = Path(plot_dir)
    digit_cm = _confusion_matrix(collected["digit_true"], collected["digit_pred"], 10) if "digit_true" in collected else None
    theta4_cm = _confusion_matrix(collected["theta4_true"], collected["theta4_pred"], 4)
    payload = {"theta4": theta4_cm.tolist()}
    if digit_cm is not None:
        payload["digit"] = digit_cm.tolist()
        _plot_matrix(digit_cm, [str(i) for i in range(10)], f"{name} digit confusion", plot_dir / f"{name}_digit_confusion.png")
    _plot_matrix(theta4_cm, SECTOR_LABELS, f"{name} theta4 confusion", plot_dir / f"{name}_theta4_confusion.png")

    per_digit = {}
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), constrained_layout=True)
    for digit, ax in enumerate(axes.flat):
        mask = collected["theta4_digit"] == digit
        cm = _confusion_matrix(collected["theta4_true"][mask], collected["theta4_pred"][mask], 4)
        per_digit[str(digit)] = cm.tolist()
        values = _row_normalized(cm)
        image = ax.imshow(values, cmap=CONFUSION_CMAP, vmin=0.0, vmax=1.0)
        ax.set_title(f"digit {digit}", fontsize=8)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(["0", "90", "180", "270"], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(["0", "90", "180", "270"], fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white" if values[i, j] > 0.55 else "black", fontsize=6)
        sns.despine(fig=fig, ax=ax, left=False, bottom=False)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    fig.savefig(plot_dir / f"{name}_theta4_confusion_by_digit.png", dpi=180)
    plt.close(fig)

    payload["theta4_by_digit"] = per_digit
    _save_json(plot_dir / f"{name}_confusions.json", payload)


def _one_param_per_hier_area(args):
    for attr in ("area_enc_kernel_sizes", "area_enc_strides", "area_enc_paddings", "area_enc_pool_sizes"):
        value = getattr(args, attr, None)
        if value is not None and len(value) > args.hier_n_areas:
            setattr(args, attr, value[: args.hier_n_areas])
    return args


def make_parser():
    parser = argparse.ArgumentParser(
        description="Plot saved MNIST-sprite readout diagnostics and confusion matrices.",
        conflict_handler="resolve",
    )
    add_reproducibility_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    parser.add_argument("--readout_path", required=True)
    parser.add_argument("--readout_stats_path", required=True)
    parser.add_argument("--base_metrics_csv", required=True)
    parser.add_argument("--hier_metrics_csv", required=True)
    parser.add_argument("--hier_model_path", required=True)
    parser.add_argument("--hier_readout_path", required=True)
    parser.add_argument("--hier_readout_stats_dir", required=True)
    parser.add_argument("--theta4_bundle_path", default=None)
    parser.add_argument("--plot_dir", required=True)
    parser.add_argument("--hier_n_areas", type=int, default=6)
    parser.add_argument("--n_areas", type=int, default=6)
    parser.add_argument("--area_encoders_kind", default="conv2d")
    parser.add_argument("--area_enc_dims", type=int, nargs="+", default=[32, 32, 32, 32, 32, 32])
    parser.add_argument("--area_enc_n_layers", type=int, default=1)
    parser.add_argument("--area_enc_kernel_sizes", type=size_tuple, nargs="+", default=None)
    parser.add_argument("--area_enc_strides", type=size_tuple, nargs="+", default=None)
    parser.add_argument("--area_enc_paddings", type=size_tuple, nargs="+", default=None)
    parser.add_argument("--area_enc_pool_sizes", type=size_tuple, nargs="+", default=None)
    parser.add_argument("--flatten_area_enc_output", action="store_true")
    parser.add_argument("--area_enc_bn", action="store_true")
    parser.add_argument("--area_integrators_kind", default="lstm")
    parser.add_argument("--area_ctx_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    parser.add_argument("--area_ctx_n_layers", type=int, default=1)
    parser.add_argument("--area_predictors_kind", default="mlp")
    parser.add_argument("--area_pred_n_hidden_layers", type=int, default=1)
    parser.add_argument("--area_pred_hidden_dims", type=int, nargs="+", default=[512, 512, 512, 512, 512, 512])
    return parser


def main():
    args = make_parser().parse_args()
    _apply_style()
    if args.offline_task != "multitask":
        raise ValueError("--offline_task must be multitask.")
    seed_everything(args.seed)
    device = select_device(args.device)
    (_, _, _, _, test_loader, _) = load_data(args)

    rows = _read_csv_metrics(args.base_metrics_csv) + _read_csv_metrics(args.hier_metrics_csv)
    plot_dir = Path(args.plot_dir)
    _plot_metric_bars(rows, "accuracy", "Saved linear probe classification accuracy", plot_dir / "linear_probe_classification_accuracy.png")
    _plot_metric_bars(rows, "r2", "Saved linear probe regression R2", plot_dir / "linear_probe_regression_r2.png")

    base_model, base_readout = build_base_model(args, device)
    base_readout.load_state_dict(torch.load(args.readout_path, map_location=device))
    base_stats = torch.load(args.readout_stats_path, map_location=device)
    base_collected = _collect_from_multitask(
        args,
        base_model,
        base_readout,
        test_loader,
        device,
        base_stats["mean"].to(device),
        base_stats["std"].to(device),
    )
    _write_confusion_outputs(base_collected, "base_rpl", plot_dir)

    if args.theta4_bundle_path is not None and Path(args.theta4_bundle_path).exists():
        theta4_collected = _collect_theta4_readout(args, base_model, test_loader, args.theta4_bundle_path, device)
        _write_confusion_outputs(theta4_collected, "base_rpl_theta4_pretrained", plot_dir)

    hier_args = argparse.Namespace(**vars(args))
    hier_args.hierarchical = True
    hier_args.model_path = args.hier_model_path
    hier_args.readout_path = args.hier_readout_path
    hier_args.readout_stats_dir = args.hier_readout_stats_dir
    hier_args.n_areas = args.hier_n_areas
    hier_args.area_enc_n_layers = int(hier_args.area_enc_n_layers)
    hier_args = _one_param_per_hier_area(hier_args)
    hier_model, hier_readout = build_hierarchical_model(hier_args, device)
    hier_readout.load_state_dict(torch.load(args.hier_readout_path, map_location=device))
    for area in range(args.hier_n_areas):
        stats = torch.load(Path(args.hier_readout_stats_dir) / f"offline_{args.offline_input}_readout_stats_area{area}.pt", map_location=device)
        collected = _collect_from_multitask(
            hier_args,
            hier_model,
            hier_readout[area],
            test_loader,
            device,
            stats["mean"].to(device),
            stats["std"].to(device),
            area=area,
        )
        _write_confusion_outputs(collected, f"hrpl_area{area}", plot_dir)

    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
