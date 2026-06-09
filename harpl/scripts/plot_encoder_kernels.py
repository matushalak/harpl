import argparse
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch


def _load_state_dict(checkpoint_path, map_location):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state_dict-like checkpoint, got {type(checkpoint).__name__}.")

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            clean_key = key.removeprefix("module.")
            state_dict[clean_key] = value.detach().cpu()
    if not state_dict:
        raise ValueError(f"No tensor parameters found in checkpoint: {checkpoint_path}")
    return state_dict


def _encoder_name(key):
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] == "areas" and parts[2] == "encoder":
        return f"area{parts[1]}_encoder"
    if "encoder" in parts:
        return "encoder"
    return None


def _layer_sort_key(key, fallback_index):
    parts = key.split(".")
    for part in parts:
        if part.startswith("conv") or part.startswith("dense"):
            suffix = "".join(ch for ch in part if ch.isdigit())
            if suffix:
                return int(suffix), fallback_index
    return fallback_index, fallback_index


def _collect_encoder_weights(state_dict):
    grouped = defaultdict(list)
    for index, (key, tensor) in enumerate(state_dict.items()):
        if not key.endswith(".weight") or tensor.ndim not in (3, 4):
            continue
        encoder_name = _encoder_name(key)
        if encoder_name is None:
            continue
        grouped[encoder_name].append((key, tensor, _layer_sort_key(key, index)))

    for encoder_name in grouped:
        grouped[encoder_name].sort(key=lambda item: item[2])
    return dict(sorted(grouped.items()))


def _select_layers(layer_items, start_layer, max_layers):
    if start_layer < 0:
        raise ValueError("--start_layer must be >= 0.")
    if max_layers < 1:
        raise ValueError("--max_layers must be >= 1.")
    return layer_items[start_layer : start_layer + max_layers]


def _normalize_image(image):
    image = image.astype(np.float32)
    image_min = float(np.nanmin(image))
    image_max = float(np.nanmax(image))
    if not np.isfinite(image_min) or not np.isfinite(image_max) or image_max == image_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def _reduce_conv2d_kernel(weight, input_channel, channel_mode):
    if input_channel is not None:
        if input_channel >= weight.shape[1]:
            raise ValueError(
                f"--input_channel {input_channel} is out of range for weight with "
                f"{weight.shape[1]} input channel(s)."
            )
        return weight[:, input_channel], False

    if channel_mode == "auto" and weight.shape[1] == 3:
        return np.moveaxis(weight[:, :3], 1, -1), True
    if channel_mode == "first":
        return weight[:, 0], False
    if channel_mode == "rms":
        return np.sqrt(np.mean(np.square(weight), axis=1)), False
    return np.mean(weight, axis=1), False


def _plot_conv2d_layer(fig, spec, key, tensor, args):
    weight = tensor.numpy()
    filters, is_rgb = _reduce_conv2d_kernel(weight, args.input_channel, args.channel_mode)
    filters = filters[: args.max_filters]

    n_filters = filters.shape[0]
    cols = min(args.cols, n_filters)
    rows = math.ceil(n_filters / cols)
    subgrid = spec.subgridspec(rows, cols, wspace=0.06, hspace=0.22)

    for filter_index in range(rows * cols):
        ax = fig.add_subplot(subgrid[filter_index // cols, filter_index % cols])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        if filter_index >= n_filters:
            ax.axis("off")
            continue

        image = filters[filter_index]
        if is_rgb:
            ax.imshow(_normalize_image(image))
        else:
            scale = float(np.max(np.abs(image)))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            ax.imshow(image, cmap=args.cmap, vmin=-scale, vmax=scale)

        if filter_index == 0:
            ax.set_title(f"{key}\n{tuple(tensor.shape)}", fontsize=7, loc="left")
        else:
            ax.set_title(str(filter_index), fontsize=6)


def _conv1d_matrix(tensor, args):
    weight = tensor.numpy()
    if args.input_channel is not None:
        if args.input_channel >= weight.shape[1]:
            raise ValueError(
                f"--input_channel {args.input_channel} is out of range for weight with "
                f"{weight.shape[1]} input channel(s)."
            )
        return weight[: args.max_filters, args.input_channel, :]

    weight = weight[: args.max_filters, : args.max_input_channels, :]
    if weight.shape[-1] == 1:
        return weight[:, :, 0]
    return weight.reshape(weight.shape[0], -1)


def _plot_heatmap_layer(fig, spec, key, tensor, args):
    matrix = _conv1d_matrix(tensor, args)
    ax = fig.add_subplot(spec)
    scale = float(np.max(np.abs(matrix)))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=-scale, vmax=scale)
    ax.set_title(f"{key} {tuple(tensor.shape)}", fontsize=8, loc="left")
    ax.set_xlabel("input channel" if args.input_channel is None else "kernel position")
    ax.set_ylabel("output filter")


def _layer_height(tensor, args):
    if tensor.ndim == 3:
        return 2.2
    n_filters = min(tensor.shape[0], args.max_filters)
    return max(1.1, math.ceil(n_filters / min(args.cols, n_filters)) * 1.05)


def _output_path(output_dir, checkpoint_path, encoder_name, fmt):
    checkpoint_stem = checkpoint_path.stem
    return output_dir / f"{checkpoint_stem}_{encoder_name}_kernels.{fmt}"


def _plot_encoder(encoder_name, layer_items, checkpoint_path, args):
    selected = _select_layers(layer_items, args.start_layer, args.max_layers)
    if not selected:
        return None

    height_ratios = [_layer_height(tensor, args) for _, tensor, _ in selected]
    width = max(6.0, min(args.cols, args.max_filters) * 1.25)
    height = max(2.5, sum(height_ratios) + 0.8)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    fig.suptitle(f"{encoder_name} kernels from {checkpoint_path}", fontsize=10)
    grid = fig.add_gridspec(len(selected), 1, height_ratios=height_ratios)

    for row, (key, tensor, _) in enumerate(selected):
        if tensor.ndim == 4:
            _plot_conv2d_layer(fig, grid[row], key, tensor, args)
        elif tensor.ndim == 3:
            _plot_heatmap_layer(fig, grid[row], key, tensor, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(args.output_dir, checkpoint_path, encoder_name, args.format)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Conv1d/Conv2d kernels of encoder layers stored in a HARPL "
            "checkpoint. The checkpoint is inspected directly, so model args are "
            "not required."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a model checkpoint.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to <checkpoint parent>/kernel_plots.",
    )
    parser.add_argument("--max_layers", type=int, default=4, help="Number of encoder layers to plot.")
    parser.add_argument("--start_layer", type=int, default=0, help="First encoder layer index to include.")
    parser.add_argument("--max_filters", type=int, default=32, help="Maximum output filters to show per layer.")
    parser.add_argument("--max_input_channels", type=int, default=128, help="Maximum input channels shown for Conv1d heatmaps.")
    parser.add_argument("--cols", type=int, default=8, help="Filter-grid columns for Conv2d layers.")
    parser.add_argument(
        "--input_channel",
        type=int,
        default=None,
        help="Plot kernels for a specific input channel instead of reducing across input channels.",
    )
    parser.add_argument(
        "--channel_mode",
        choices=("auto", "mean", "rms", "first"),
        default="auto",
        help="How to display multi-channel Conv2d kernels when --input_channel is not set.",
    )
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap for signed kernels.")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png", help="Output file format.")
    parser.add_argument("--device", default="cpu", help="Device used only for torch.load map_location.")
    return parser


def _validate_args(parser, args):
    for name in ("max_layers", "max_filters", "max_input_channels", "cols", "dpi"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1.")
    if args.start_layer < 0:
        parser.error("--start_layer must be >= 0.")
    if args.input_channel is not None and args.input_channel < 0:
        parser.error("--input_channel must be >= 0.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / "kernel_plots"

    state_dict = _load_state_dict(args.checkpoint, map_location=args.device)
    grouped_weights = _collect_encoder_weights(state_dict)
    if not grouped_weights:
        raise ValueError(f"No encoder Conv1d/Conv2d weight tensors found in {args.checkpoint}.")

    output_paths = []
    for encoder_name, layer_items in grouped_weights.items():
        output_path = _plot_encoder(encoder_name, layer_items, args.checkpoint, args)
        if output_path is not None:
            output_paths.append(output_path)

    if not output_paths:
        raise ValueError("No encoder layers matched the requested layer range.")

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
