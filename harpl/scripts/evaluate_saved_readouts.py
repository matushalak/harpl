import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from harpl.networks.utils import additional_data_process, prepare_model
from harpl.scripts.args import (
    add_data_args,
    add_greedy_model_args,
    add_model_args,
    add_offline_eval_args,
    add_reproducibility_args,
    add_validation_args,
)
from harpl.scripts.eval_utils import compute_readout_loss, prepare_readout
from harpl.scripts.utils import get_data_specs, prepare_data, seed_everything, select_device


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _merge_metric_dict(results, split, kind, metrics):
    losses, accs_or_r2 = metrics
    for group_idx, group_name in enumerate(("sequence", "frame")):
        for task_name, value in losses[group_idx].items():
            results.append({
                "split": split,
                "group": group_name,
                "task": task_name,
                "metric": f"{kind}_loss",
                "value": _to_float(value),
            })
        metric_name = "accuracy" if kind == "classification" else "r2"
        for task_name, value in accs_or_r2[group_idx].items():
            results.append({
                "split": split,
                "group": group_name,
                "task": task_name,
                "metric": metric_name,
                "value": _to_float(value),
            })


def evaluate_loader(args, model, readout, loader, model_output_idx, readout_mean, readout_std, device, area=None):
    classifier_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    regression_criterion = nn.MSELoss().to(device)
    num_classes = get_data_specs(
        args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )[1]
    seq_classes, dense_classes, seq_regression, dense_regression = num_classes

    clf_loss = ({key: 0.0 for key in seq_classes}, {key: 0.0 for key in dense_classes})
    clf_acc = ({key: 0.0 for key in seq_classes}, {key: 0.0 for key in dense_classes})
    reg_loss = ({key: 0.0 for key in seq_regression}, {key: 0.0 for key in dense_regression})
    reg_r2 = ({key: 0.0 for key in seq_regression}, {key: 0.0 for key in dense_regression})

    model.eval()
    readout.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="evaluate"):
            x = x.to(device)
            y = tuple(yi.to(device) for yi in y)
            model_output = model(x)[model_output_idx]
            if area is not None:
                model_output = model_output[area]
            model_output = (model_output - readout_mean) / (readout_std + 1e-8)
            batch_clf_loss, batch_clf_acc, _, batch_reg_loss, batch_reg_r2 = compute_readout_loss(
                data=model_output,
                labels=y,
                readout=readout,
                classifier_criterion=classifier_criterion,
                regression_criterion=regression_criterion,
                target_length=None,
                downstream_input=args.offline_input,
                dense_prediction=args.dense_prediction,
                single_readout=args.offline_single_timestep_readout,
                pred_steps=args.pred_steps,
                task=args.offline_task,
                full_spatial_readout=args.offline_full_spatial_readout,
            )
            weight = 1.0 / len(loader)
            for task_name in seq_classes:
                clf_loss[0][task_name] += batch_clf_loss[0][task_name] * weight
                clf_acc[0][task_name] += batch_clf_acc[0][task_name] * weight
            for task_name in dense_classes:
                clf_loss[1][task_name] += batch_clf_loss[1][task_name] * weight
                clf_acc[1][task_name] += batch_clf_acc[1][task_name] * weight
            for task_name in seq_regression:
                reg_loss[0][task_name] += batch_reg_loss[0][task_name] * weight
                reg_r2[0][task_name] += batch_reg_r2[0][task_name] * weight
            for task_name in dense_regression:
                reg_loss[1][task_name] += batch_reg_loss[1][task_name] * weight
                reg_r2[1][task_name] += batch_reg_r2[1][task_name] * weight

    rows = []
    _merge_metric_dict(rows, "eval", "classification", (clf_loss, clf_acc))
    _merge_metric_dict(rows, "eval", "regression", (reg_loss, reg_r2))
    return rows


def load_data(args):
    return prepare_data(
        args.dataset,
        args.data_input_dir,
        seq_len=args.seq_len,
        batch_size=args.offline_batch_size,
        val_batch_size=args.offline_batch_size,
        distributed=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        grayscale=args.grayscale,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_max_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        spritevid_exclude_latent_regions=False,
        spritevid_discretize_latents=args.spritevid_discretize_latents,
        spritevid_noise_type=args.spritevid_noise_type,
        spritevid_noise_level=args.spritevid_noise_level,
        spritevid_frozen_noise=args.spritevid_frozen_noise,
        sprite_noise_on_top=args.sprite_noise_on_top,
        spritevid_grid_enabled=args.spritevid_grid_enabled,
        spritevid_frozen_grid=args.spritevid_frozen_grid,
        spritevid_occlude_n_frames=args.spritevid_occlude_n_frames,
        spritevid_min_scale=args.spritevid_min_scale,
        spritevid_max_scale=args.spritevid_max_scale,
        spritevid_normalization_samples=args.spritevid_normalization_samples,
        spritevid_device=args.spritevid_device,
        num_sequences=args.num_sequences,
        inter_trial_interval=args.inter_trial_interval,
    )


def build_base_model(args, device):
    input_size, num_classes = get_data_specs(
        args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )
    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_enc_output)
    state_dict = torch.load(args.model_path, map_location=device)
    model = prepare_model(
        encoder_kind=args.encoder,
        integrator_kind=args.integrator,
        predictor_kind=args.predictor,
        input_size=input_size,
        n_in_channels=1 if args.grayscale else 3,
        enc_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        ctx_n_layers=args.ctx_n_layers,
        pred_n_hidden_layers=args.pred_n_hidden_layers,
        pred_hidden_dim=args.pred_hidden_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        use_bn_enc=args.use_bn,
        enc_n_layers=args.enc_n_layers,
        enc_kernel_size=args.enc_kernel_size,
        enc_stride=args.enc_stride,
        enc_padding=args.enc_padding,
        enc_pool_size=args.enc_pool_size,
        preprocess=preprocess,
        postprocess=postprocess,
        state_dict=state_dict,
        return_full_features=args.offline_full_spatial_readout or args.flatten_enc_output,
        flatten_enc_output=args.flatten_enc_output,
    ).to(device)
    readout = prepare_readout(
        task=args.offline_task,
        downstream_input=args.offline_input,
        input_spatial_size=input_size,
        single_timestep_readout=args.offline_single_timestep_readout,
        full_spatial_readout=args.offline_full_spatial_readout,
        num_classes=num_classes,
        seq_len=args.seq_len,
        evaluate_concat_features=args.evaluate_concat_features,
        enc_output_dim=args.enc_output_dim,
        ctx_dim=args.ctx_dim,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        model=model,
    ).to(device)
    return model, readout


def build_hierarchical_model(args, device):
    input_size, num_classes = get_data_specs(
        args.dataset,
        target_label=args.offline_task,
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )
    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_area_enc_output)
    state_dict = torch.load(args.model_path, map_location=device)
    model = prepare_model(
        encoder_kind=args.area_encoders_kind,
        integrator_kind=args.area_integrators_kind,
        predictor_kind=args.area_predictors_kind,
        input_size=input_size,
        n_in_channels=1 if args.grayscale else 3,
        enc_dim=args.area_enc_dims,
        ctx_dim=args.area_ctx_dims,
        ctx_n_layers=args.area_ctx_n_layers,
        pred_n_hidden_layers=args.area_pred_n_hidden_layers,
        pred_hidden_dim=args.area_pred_hidden_dims,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        use_bn_enc=args.area_enc_bn,
        enc_n_layers=args.area_enc_n_layers,
        enc_kernel_size=args.area_enc_kernel_sizes,
        enc_stride=args.area_enc_strides,
        enc_padding=args.area_enc_paddings,
        enc_pool_size=args.area_enc_pool_sizes,
        preprocess=preprocess,
        postprocess=postprocess,
        state_dict=state_dict,
        return_full_features=args.offline_full_spatial_readout or args.flatten_area_enc_output,
        flatten_enc_output=args.flatten_area_enc_output,
        n_areas=args.n_areas,
        frozen_areas=[True] * args.n_areas,
    ).to(device)
    readout = prepare_readout(
        task=args.offline_task,
        downstream_input=args.offline_input,
        input_spatial_size=input_size,
        single_timestep_readout=args.offline_single_timestep_readout,
        full_spatial_readout=args.offline_full_spatial_readout,
        num_classes=num_classes,
        seq_len=args.seq_len,
        evaluate_concat_features=args.evaluate_concat_features,
        enc_output_dim=args.area_enc_dims,
        ctx_dim=args.area_ctx_dims,
        pred_steps=args.pred_steps,
        dense_prediction=args.dense_prediction,
        prediction_target=args.prediction_target,
        pred_target_dim_override=args.pred_target_dim_override,
        model=model,
        n_areas=args.n_areas,
    ).to(device)
    return model, readout


def write_outputs(rows, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2)
    with output_prefix.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "area", "split", "group", "task", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def make_parser(hierarchical):
    parser = argparse.ArgumentParser(description="Evaluate saved offline readouts without training new probes.")
    parser.add_argument("--hierarchical", action="store_true", default=hierarchical)
    parser.add_argument("--readout_path", required=True)
    parser.add_argument("--readout_stats_path", default=None)
    parser.add_argument("--readout_stats_dir", default=None)
    parser.add_argument("--output_prefix", required=True)
    add_reproducibility_args(parser)
    add_data_args(parser)
    if hierarchical:
        add_greedy_model_args(parser)
    else:
        add_model_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    return parser


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--hierarchical", action="store_true")
    pre_args, _ = pre.parse_known_args()
    parser = make_parser(pre_args.hierarchical)
    args = parser.parse_args()
    if args.offline_task is None or args.offline_task == "none":
        raise ValueError("--offline_task must be set to multitask for MNIST-sprite readout evaluation.")

    seed_everything(args.seed)
    device = select_device(args.device)
    (_, _, _, _, test_loader, _) = load_data(args)

    if args.offline_input == "enc":
        model_output_idx = 0
    elif args.offline_input == "ctx":
        model_output_idx = 1
    elif args.offline_input == "pred":
        model_output_idx = 2
    else:
        raise ValueError(f"Invalid offline input: {args.offline_input}")

    rows = []
    if args.hierarchical:
        model, readout = build_hierarchical_model(args, device)
        readout.load_state_dict(torch.load(args.readout_path, map_location=device))
        if args.readout_stats_dir is None:
            args.readout_stats_dir = str(Path(args.readout_path).parent)
        for area in range(args.n_areas):
            stats = torch.load(Path(args.readout_stats_dir) / f"offline_{args.offline_input}_readout_stats_area{area}.pt", map_location=device)
            area_rows = evaluate_loader(
                args, model, readout[area], test_loader, model_output_idx,
                stats["mean"].to(device), stats["std"].to(device), device, area=area
            )
            for row in area_rows:
                row["model"] = "hRPL"
                row["area"] = area
            rows.extend(area_rows)
    else:
        model, readout = build_base_model(args, device)
        readout.load_state_dict(torch.load(args.readout_path, map_location=device))
        stats = torch.load(args.readout_stats_path, map_location=device)
        rows = evaluate_loader(
            args, model, readout, test_loader, model_output_idx,
            stats["mean"].to(device), stats["std"].to(device), device
        )
        for row in rows:
            row["model"] = "RPL"
            row["area"] = ""

    write_outputs(rows, args.output_prefix)
    for row in rows:
        if row["metric"] in {"accuracy", "r2"}:
            area = f" area={row['area']}" if row["area"] != "" else ""
            print(f"{row['model']}{area} {row['group']} {row['task']} {row['metric']}={row['value']:.4f}")


if __name__ == "__main__":
    main()
