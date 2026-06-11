import argparse
import gc
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from harpl.data.attention_sprites_dataset import (
    MovingAnimalAttentionDataset,
    TASK_TO_ID,
)
from harpl.scripts.eval_utils import (
    evaluate_attention_model,
    evaluate_cross_decode_sprites,
    evaluate_pretrained_attention_tasks,
)
from harpl.networks.harpl import (
    ARPLmodel,
    AttentionDecoder,
    ClassificationHead,
)
from harpl.networks.utils import additional_data_process, prepare_model
from harpl.scripts.args import (
    add_model_args,
    add_optimization_args,
    add_reproducibility_args,
)
from harpl.scripts.utils import (
    close_logger,
    cuda_memory_stats,
    get_data_specs,
    init_logger,
    is_cuda_device,
    log_variable,
    seed_everything,
    select_device,
)


def _normalize_output_size(value):
    if isinstance(value, int):
        return (value, value)
    value = list(value)
    if len(value) == 1:
        return (value[0], value[0])
    if len(value) == 2:
        return (value[0], value[1])
    raise ValueError("--spritevid_output_size expects one value or two values: height width.")


def _parse_task_values(values):
    if values is None:
        return None
    parsed = []
    for value in values:
        if isinstance(value, str) and value.lstrip("-").isdigit():
            parsed.append(int(value))
        else:
            parsed.append(value)
    return parsed


def _parse_optional_scalar_or_channels(value):
    if value is None:
        return None
    value = list(value)
    if len(value) == 1:
        return value[0]
    if len(value) == 3:
        return tuple(value)
    raise ValueError("Expected either one scalar value or three channel values.")


def _dataloader_kwargs(args, device, shuffle=True):
    if torch.device(args.spritevid_device).type == "cuda" and args.num_workers != 0:
        raise ValueError("CUDA attention dataset rendering requires --num_workers 0.")
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        if args.prefetch_factor is not None:
            kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _infer_context_dim(integrator, fallback):
    if hasattr(integrator, "get_output_dim"):
        try:
            return int(integrator.get_output_dim())
        except NotImplementedError:
            pass
    backbone = getattr(integrator, "backbone", None)
    hidden_size = getattr(backbone, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    return int(fallback)


def _infer_first_encoder_shape(encoder, input_size, n_in_channels):
    encoder_first, _, _ = ARPLmodel._split_encoder(encoder)
    encoder_first.eval()
    height, width = _normalize_output_size(input_size)
    dummy = torch.zeros(1, 1, n_in_channels, height, width)
    with torch.no_grad():
        features = ARPLmodel._forward_video_layers(encoder_first, dummy)
    if features.ndim != 5:
        raise RuntimeError(
            "Expected first encoder block to return spatial features shaped "
            f"(B, T, C, H, W), got {tuple(features.shape)}"
        )
    return tuple(int(dim) for dim in features.shape[2:])


def _attention_output_shape(first_encoder_shape, attention_dims):
    channels, height, width = first_encoder_shape
    if attention_dims == "features":
        return (channels, 1, 1)
    if attention_dims == "spatial":
        return (1, height, width)
    if attention_dims == "features+spatial":
        return (channels, height, width)
    raise ValueError("--attention_dims must be one of: features, spatial, features+spatial")


def _infer_predictor_output_dim(predictor, fallback):
    last_linear = None
    for module in predictor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return int(last_linear.out_features)
    return int(fallback)


def _default_readout_path(model_path):
    return Path(model_path).parent / "online_ctx_readout.pt"


def _load_frozen_classification_head(head, readout_path, device):
    readout_path = Path(readout_path)
    if not readout_path.exists():
        raise FileNotFoundError(
            f"Frozen online readout checkpoint not found: {readout_path}. "
            "Pass --attention_readout_path to use a different readout."
        )
    state_dict = torch.load(readout_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected state_dict-like readout checkpoint, got {type(state_dict).__name__}.")

    candidates = [
        ("0.layers.readout.weight", "0.layers.readout.bias"),
        ("layers.readout.weight", "layers.readout.bias"),
        ("readout.weight", "readout.bias"),
    ]
    for weight_key, bias_key in candidates:
        if weight_key in state_dict and bias_key in state_dict:
            readout_state = {
                "weight": state_dict[weight_key],
                "bias": state_dict[bias_key],
            }
            break
    else:
        raise KeyError(
            "Could not find a sprite-classification readout in "
            f"{readout_path}; expected 0.layers.readout.*, layers.readout.*, or readout.* keys."
        )

    expected_weight_shape = tuple(head.readout.weight.shape)
    actual_weight_shape = tuple(readout_state["weight"].shape)
    if actual_weight_shape != expected_weight_shape:
        raise RuntimeError(
            f"Readout weight shape {actual_weight_shape} does not match attention head "
            f"shape {expected_weight_shape}."
        )
    head.readout.load_state_dict(readout_state)
    head.to(device)
    head.eval()
    for param in head.parameters():
        param.requires_grad_(False)
    return readout_path


def _make_attention_targets(task_info, seq_len, cue_frames):
    targets = task_info[:, 1].unsqueeze(1).expand(-1, seq_len)
    mask = torch.ones_like(targets, dtype=torch.bool)
    grouping = task_info[:, 0] == TASK_TO_ID["perceptual_grouping"]
    if grouping.any():
        mask[grouping, :cue_frames] = False
    return targets, mask


def _move_labels(labels, device, non_blocking):
    if isinstance(labels, dict):
        labels = labels["task_info"]
    return labels.to(device, non_blocking=non_blocking)


def _snapshot_named_params(named_params):
    return {
        name: param.detach().cpu().clone()
        for name, param in named_params
    }


def _max_abs_delta(snapshot, named_params):
    max_delta = 0.0
    for name, param in named_params:
        before = snapshot[name].to(device=param.device, dtype=param.dtype)
        delta = (param.detach() - before).abs().max().item()
        max_delta = max(max_delta, delta)
    return max_delta


def _release_memory(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    elif torch.device(device).type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _mps_memory_stats(device):
    if torch.device(device).type != "mps" or not hasattr(torch, "mps"):
        return {}
    stats = {}
    if hasattr(torch.mps, "current_allocated_memory"):
        stats["MPS/memory_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
    if hasattr(torch.mps, "driver_allocated_memory"):
        stats["MPS/driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
    if hasattr(torch.mps, "recommended_max_memory"):
        stats["MPS/recommended_max_memory_gb"] = torch.mps.recommended_max_memory() / 1e9
    return stats


def _prepare_attention_dataset(args, split="train", num_sequences=None):
    output_size = _normalize_output_size(args.spritevid_output_size)
    base_output_size = _normalize_output_size(args.attention_base_output_size)
    return MovingAnimalAttentionDataset(
        data_dir=args.data_input_dir,
        split=split,
        task=args.attention_task,
        tasks=_parse_task_values(args.attention_tasks),
        output_size=output_size,
        base_output_size=base_output_size,
        scale_pixel_parameters=args.attention_scale_pixel_parameters,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences if num_sequences is None else num_sequences,
        sprite_img_dir=args.sprite_img_dir,
        max_sprites=args.spritevid_max_sprites,
        seed=args.seed,
        background=args.attention_background,
        device=args.spritevid_device,
        noise_type=args.spritevid_noise_type,
        noise_level=args.spritevid_noise_level,
        training_noise_level=args.attention_training_noise_level,
        object_recognition_noise_level=args.attention_object_recognition_noise_level,
        freeze_noise=args.spritevid_frozen_noise,
        noise_on_top=args.sprite_noise_on_top,
        popout_mode=args.popout_mode,
        num_distractors=args.num_distractors,
        crowd_size=args.crowd_size,
        cue_frames=args.cue_frames,
        occluder_count=args.attention_occluder_count,
        occluder_min_size=args.attention_occluder_min_size,
        occluder_max_size=args.attention_occluder_max_size,
        fixation_size=args.attention_fixation_size,
        scale_range=tuple(args.attention_scale_range),
        velocity_range=tuple(args.attention_velocity_range),
        scale_velocity_range=tuple(args.attention_scale_velocity_range),
        slow_speed_range=tuple(args.attention_slow_speed_range),
        fast_speed_range=tuple(args.attention_fast_speed_range),
        angular_speed_range=tuple(args.attention_angular_speed_range),
        slow_rotation_speed_range=tuple(args.attention_slow_rotation_speed_range),
        fast_rotation_speed_range=tuple(args.attention_fast_rotation_speed_range),
        velocity_popout_kind=args.attention_velocity_popout_kind,
        rotation_popout_kind=args.attention_rotation_popout_kind,
        normalize=args.attention_normalize,
        mean=_parse_optional_scalar_or_channels(args.attention_mean),
        std=_parse_optional_scalar_or_channels(args.attention_std),
        return_metadata=args.attention_return_metadata,
    )


def _prepare_arpl_model(args, dataset, device):
    if args.model_path is None:
        raise ValueError("Specify a pretrained RePL checkpoint with --model_path.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    output_size = _normalize_output_size(args.spritevid_output_size)
    input_size, _ = get_data_specs(
        dataset="animals",
        target_label="multitask",
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=output_size,
        flatten_images=args.flatten_images,
    )
    preprocess, postprocess = additional_data_process("animals", args.flatten_enc_output)
    return_full_features = args.return_full_features
    if return_full_features is None:
        return_full_features = args.flatten_enc_output

    state_dict = torch.load(args.model_path, map_location="cpu")
    repl = prepare_model(
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
        return_full_features=return_full_features,
        flatten_enc_output=args.flatten_enc_output,
        state_dict=state_dict,
    )

    context_dim = _infer_context_dim(repl.integrator, args.ctx_dim)
    predictor_output_dim = _infer_predictor_output_dim(repl.predictor, context_dim)
    first_encoder_shape = _infer_first_encoder_shape(
        repl.encoder,
        output_size,
        n_in_channels=1 if args.grayscale else 3,
    )
    attention_shape = _attention_output_shape(first_encoder_shape, args.attention_dims)
    head = ClassificationHead(input_dim=context_dim, num_classes=len(dataset.sprites))
    readout_path = args.attention_readout_path or _default_readout_path(args.model_path)
    _load_frozen_classification_head(head, readout_path, device)
    decoder = AttentionDecoder(
        input_dim=predictor_output_dim,
        output_dim=None,
        output_shape=attention_shape,
        hidden_dim=args.attention_hidden_dim or context_dim,
        n_hidden_layers=args.attention_decoder_layers,
    )
    model = ARPLmodel(
        encoder=repl.encoder,
        integrator=repl.integrator,
        predictor=repl.predictor,
        head=head,
        decoder=decoder,
        num_tasks=len(TASK_TO_ID),
        preprocess=repl.preprocess,
        postprocess=repl.postprocess,
        freeze_repl=True,
        eval_frozen=True,
        decoder_input_dim=predictor_output_dim,
        use_task_embedding=getattr(args, "attention_use_task_embedding", True),
        class_prompt_value=getattr(args, "attention_class_prompt_value", 10.0),
    )
    model.head.eval()
    for param in model.head.parameters():
        param.requires_grad_(False)
    return model.to(device)


def _prepare_optimizer(args, model):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable ARPL attention parameters were found.")
    if args.optimizer == "adam":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def _expected_trainable_prefixes(args):
    prefixes = ["decoder.", "class_feedback."]
    if getattr(args, "attention_use_task_embedding", True):
        prefixes.append("task_embedding.")
    return tuple(prefixes)


def _validate_trainable_scope(model, args):
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    frozen_named = [(name, param) for name, param in model.named_parameters() if not param.requires_grad]
    expected_prefixes = _expected_trainable_prefixes(args)
    unexpected = [
        name
        for name, _ in trainable_named
        if not name.startswith(expected_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Unexpected trainable parameters outside attention decoder/feedback modules: "
            + ", ".join(unexpected)
        )
    for prefix in expected_prefixes:
        if not any(name.startswith(prefix) for name, _ in trainable_named):
            raise RuntimeError(f"No trainable parameters found under expected prefix {prefix!r}.")
    return trainable_named, frozen_named


def _validate_optimizer_scope(optimizer, trainable_named):
    trainable_param_ids = {id(param) for _, param in trainable_named}
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    if optimizer_param_ids != trainable_param_ids:
        raise RuntimeError("Optimizer parameter set does not exactly match trainable ARPL parameters.")


def train(args, device):
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    dataset = _prepare_attention_dataset(args)
    val_dataset = _prepare_attention_dataset(args, split="val", num_sequences=args.attention_val_sequences)
    test_dataset = _prepare_attention_dataset(args, split="test", num_sequences=args.attention_test_sequences)
    loader = DataLoader(dataset, **_dataloader_kwargs(args, device, shuffle=True))
    val_loader = DataLoader(val_dataset, **_dataloader_kwargs(args, device, shuffle=False))
    test_loader = DataLoader(test_dataset, **_dataloader_kwargs(args, device, shuffle=False))
    model = _prepare_arpl_model(args, dataset, device)
    trainable_named, frozen_named = _validate_trainable_scope(model, args)
    optimizer = _prepare_optimizer(args, model)
    _validate_optimizer_scope(optimizer, trainable_named)
    criterion = nn.CrossEntropyLoss()

    trainable_snapshot = _snapshot_named_params(trainable_named)
    frozen_snapshot = _snapshot_named_params(frozen_named)

    trainable_count = sum(param.numel() for _, param in trainable_named)
    frozen_count = sum(param.numel() for _, param in frozen_named)
    print(f"trainable_params={trainable_count} frozen_params={frozen_count}")
    print("trainable_modules=" + ", ".join(name for name, _ in trainable_named[:12]))
    if len(trainable_named) > 12:
        print(f"trainable_modules_truncated={len(trainable_named) - 12}")

    non_blocking = args.pin_memory and is_cuda_device(device)
    first_loss = None
    last_loss = None
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        model.encoder_first.eval()
        model.encoder_tail.eval()
        model.integrator.eval()
        model.predictor.eval()
        model.head.eval()

        total_loss = 0.0
        total_correct = 0
        total_items = 0
        total_batches = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        last_batch_end = time.perf_counter()
        for batch_idx, (video, task_info) in enumerate(progress):
            data_time = time.perf_counter() - last_batch_end
            step_start = time.perf_counter()
            video = video.to(device, non_blocking=non_blocking)
            task_info = _move_labels(task_info, device, non_blocking=non_blocking)

            logits = model((video, task_info), return_logits_only=True)
            targets, mask = _make_attention_targets(task_info, logits.size(1), args.cue_frames)
            loss = criterion(logits[mask], targets[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            step_time = time.perf_counter() - step_start

            with torch.no_grad():
                preds = logits[mask].argmax(dim=-1)
                correct = (preds == targets[mask]).sum().item()
                count = int(mask.sum().item())
                loss_value = loss.item()
                if first_loss is None:
                    first_loss = loss_value
                last_loss = loss_value
                batch_acc = correct / max(count, 1)
                samples_per_sec = video.shape[0] / step_time if step_time > 0 else 0.0
                total_loss += loss_value
                total_correct += correct
                total_items += count
                total_batches += 1
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    acc=f"{batch_acc:.3f}",
                )
                if not args.nolog:
                    log_variable(epoch + 1, "Attention/epoch", commit=False)
                    log_variable(loss_value, "Attention/train_loss_step", commit=False)
                    log_variable(batch_acc, "Attention/train_acc_step", commit=False)
                    log_variable(count, "Attention/supervised_items_step", commit=False)
                    log_variable(data_time, "Attention/timing_data_sec", commit=False)
                    log_variable(step_time, "Attention/timing_step_sec", commit=False)
                    log_variable(samples_per_sec, "Attention/timing_samples_per_sec", commit=False)
                    for metric_name, metric_value in cuda_memory_stats(device).items():
                        log_variable(metric_value, f"Attention/{metric_name}", commit=False)
                    for metric_name, metric_value in _mps_memory_stats(device).items():
                        log_variable(metric_value, f"Attention/{metric_name}", commit=False)
                    log_variable(global_step, "Attention/global_step", commit=True)

            stop_after_batch = args.max_batches and batch_idx + 1 >= args.max_batches
            del video, task_info, logits, targets, mask, loss
            global_step += 1
            last_batch_end = time.perf_counter()
            if stop_after_batch:
                break

        avg_loss = total_loss / max(total_batches, 1)
        avg_acc = total_correct / max(total_items, 1)
        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.6f} "
            f"train_acc={avg_acc:.4f} batches={total_batches}"
        )
        if not args.nolog:
            log_variable(epoch + 1, "Attention/epoch_summary", commit=False)
            log_variable(avg_loss, "Attention/train_loss_epoch", commit=False)
            log_variable(avg_acc, "Attention/train_acc_epoch", commit=False)
            log_variable(total_batches, "Attention/batches_epoch", commit=True)
        if args.attention_eval_every and (epoch + 1) % args.attention_eval_every == 0:
            evaluate_attention_model(
                args,
                model,
                val_loader,
                device,
                "val",
                _make_attention_targets,
                _move_labels,
                use_attention=True,
            )
        progress.close()
        _release_memory(device)

    if first_loss is None or last_loss is None:
        raise RuntimeError("No training batches were processed.")

    frozen_delta = _max_abs_delta(frozen_snapshot, frozen_named)
    trainable_delta = _max_abs_delta(trainable_snapshot, trainable_named)
    print(
        f"first_batch_loss={first_loss:.6f} last_batch_loss={last_loss:.6f} "
        f"trainable_max_delta={trainable_delta:.6g} frozen_max_delta={frozen_delta:.6g}"
    )
    if not args.nolog:
        log_variable(first_loss, "Attention/first_batch_loss", commit=False)
        log_variable(last_loss, "Attention/last_batch_loss", commit=False)
        log_variable(trainable_delta, "Attention/trainable_max_delta", commit=False)
        log_variable(frozen_delta, "Attention/frozen_max_delta", commit=True)
    if frozen_delta > args.freeze_tolerance:
        raise RuntimeError(
            f"Frozen RePL parameters changed by {frozen_delta:.6g}, "
            f"above --freeze_tolerance {args.freeze_tolerance}."
        )
    if trainable_delta <= 0.0:
        raise RuntimeError("No trainable attention decoder or feedback parameter changed during training.")

    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "attention_model_final.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "attention_optimizer_final.pt")
        print(f"saved_checkpoint={checkpoint_dir / 'attention_model_final.pt'}")
    evaluate_attention_model(
        args,
        model,
        test_loader,
        device,
        "test",
        _make_attention_targets,
        _move_labels,
        use_attention=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train ARPL attention decoder/head on MovingAnimalAttentionDataset."
    )
    add_reproducibility_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)

    # Paths and sprite rendering.
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained RePL checkpoint.")
    parser.add_argument("--data_input_dir", type=str, default="datasets")
    parser.add_argument("--sprite_img_dir", type=str, default="animals")
    parser.add_argument("--spritevid_max_sprites", type=int, default=8)
    parser.add_argument("--spritevid_output_size", type=int, nargs="+", default=[64])
    parser.add_argument("--spritevid_device", type=str, default="cpu")

    # Dataset size and loader behavior.
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--num_sequences", type=int, default=16000)
    parser.add_argument("--attention_val_sequences", type=int, default=1024)
    parser.add_argument("--attention_test_sequences", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin-memory", "--pin_memory", dest="pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", "--persistent_workers", dest="persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", "--prefetch_factor", dest="prefetch_factor", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    # Attention dataset noise and normalization.
    parser.add_argument("--spritevid_noise_type", choices=["gaussian", "salt_pepper", "none"], default="gaussian")
    parser.add_argument("--spritevid_noise_level", type=float, default=0.1)
    parser.add_argument("--spritevid_frozen_noise", action="store_true")
    parser.add_argument("--sprite_noise_on_top", action="store_true")
    parser.add_argument("--attention_training_noise_level", type=float, default=0.1)
    parser.add_argument("--attention_object_recognition_noise_level", type=float, default=0.35)
    parser.add_argument("--attention_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention_mean", type=float, nargs="+", default=None, help="Scalar or three channel means for attention dataset normalization.")
    parser.add_argument("--attention_std", type=float, nargs="+", default=None, help="Scalar or three channel stds for attention dataset normalization.")

    # Base model input options.
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--flatten_images", action="store_true")
    parser.add_argument("--return_full_features", action=argparse.BooleanOptionalAction, default=None)

    # Attention task composition.
    parser.add_argument("--attention_task", default="mixed")
    parser.add_argument("--attention_tasks", nargs="*", default=None)
    parser.add_argument("--attention_base_output_size", type=int, nargs="+", default=[64])
    parser.add_argument("--attention_scale_pixel_parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_background", type=float, default=0.5)
    parser.add_argument("--popout_mode", choices=["class", "rotation", "velocity", "mixed"], default="class")
    parser.add_argument("--num_distractors", type=int, default=1)
    parser.add_argument("--crowd_size", type=int, default=2)
    parser.add_argument("--cue_frames", type=int, default=5)

    # Occluders and fixation cue.
    parser.add_argument("--attention_occluder_count", type=int, default=4)
    parser.add_argument("--attention_occluder_min_size", type=int, default=8)
    parser.add_argument("--attention_occluder_max_size", type=int, default=18)
    parser.add_argument("--attention_fixation_size", type=float, default=3.0)

    # Motion ranges. Defaults match the continuous SpriteVideoDataset where possible.
    parser.add_argument("--attention_scale_range", type=float, nargs=2, default=[0.2, 1.0])
    parser.add_argument("--attention_velocity_range", type=float, nargs=2, default=[-8.0, 8.0], help="Per-axis x/y velocity component range in pixels per frame.")
    parser.add_argument("--attention_scale_velocity_range", type=float, nargs=2, default=[-0.125, 0.125], help="Z/scale velocity range per frame.")
    parser.add_argument("--attention_angular_speed_range", type=float, nargs=2, default=[-30.0, 30.0])

    # Popout ranges. Slow/fast speed ranges are absolute movement or rotation speeds.
    parser.add_argument("--attention_slow_speed_range", type=float, nargs=2, default=[0.8, 2.0])
    parser.add_argument("--attention_fast_speed_range", type=float, nargs=2, default=[4.8, 7.0])
    parser.add_argument("--attention_velocity_popout_kind", choices=["fast", "slow", "mixed"], default="fast")
    parser.add_argument("--attention_slow_rotation_speed_range", type=float, nargs=2, default=[3.0, 10.0])
    parser.add_argument("--attention_fast_rotation_speed_range", type=float, nargs=2, default=[20.0, 30.0])
    parser.add_argument("--attention_rotation_popout_kind", choices=["fast", "slow", "mixed"], default="fast")
    parser.add_argument("--attention_return_metadata", action=argparse.BooleanOptionalAction, default=False)

    # Attention model and evaluation.
    parser.add_argument("--attention_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_decoder_layers", type=int, default=1)
    parser.add_argument("--attention_dims", choices=["features", "spatial", "features+spatial"], default="features+spatial")
    parser.add_argument("--attention_readout_path", type=str, default=None, help="Frozen online sprite-classification readout checkpoint. Defaults to online_ctx_readout.pt beside --model_path.")
    parser.add_argument("--attention_use_task_embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_class_prompt_value", type=float, default=10.0)
    parser.add_argument("--attention_eval_every", type=int, default=5, help="Run attention validation every N epochs; 0 disables periodic validation.")
    parser.add_argument("--eval_pretrained_attention_tasks", action="store_true", help="Evaluate the frozen pretrained model and online readout on each attention task with attention disabled, then exit.")
    parser.add_argument("--eval_cross_decode_sprites", action="store_true", help="Evaluate frozen pretrained ctx readouts on generated held-out sprite identities, then exit.")
    parser.add_argument("--cross_decode_sprite_indices", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--cross_decode_sequences", type=int, default=512)
    parser.add_argument("--cross_decode_readout_path", type=str, default=None, help="Readout checkpoint for cross-decoding. Defaults to online_ctx_readout.pt beside --model_path.")
    parser.add_argument("--cross_decode_readout_stats_path", type=str, default=None, help="Optional mean/std stats for offline readouts.")
    parser.add_argument("--max_batches", type=int, default=0, help="Stop each epoch after this many batches; 0 means full epoch.")
    parser.add_argument("--freeze_tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="attention")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    parser.add_argument("--nolog", action="store_true", help="disable experiment logging")
    parser.add_argument("--logger", choices=["tensorboard", "wandb", "none"], default="wandb", help="experiment logger backend")
    parser.add_argument("--wandb_project", type=str, default="HARPL", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Weights & Biases run id to resume")
    parser.add_argument("--wandb_resume", choices=["allow", "must", "never", "auto"], default=None, help="Weights & Biases resume mode")
    parser.add_argument("--wandb_group", type=str, default="attention", help="Weights & Biases run group")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log root directory")

    parser.set_defaults(
        encoder="conv2d",
        use_bn=True,
        enc_n_layers=6,
        enc_kernel_size=[(5, 5)] * 6,
        enc_stride=[(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
        enc_padding=[(2, 2)] * 6,
        enc_output_dim=32,
        flatten_enc_output=True,
        integrator="lstm",
        ctx_dim=512,
        predictor="mlp",
        pred_hidden_dim=512,
        pred_steps=1,
        prediction_target="enc",
        dense_prediction=False,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.spritevid_noise_type == "none":
        args.spritevid_noise_type = None
    seed_everything(args.seed, args.deterministic)
    init_logger(args)
    device = select_device(args.device)
    try:
        if args.eval_pretrained_attention_tasks:
            evaluate_pretrained_attention_tasks(
                args,
                device,
                _prepare_attention_dataset,
                _prepare_arpl_model,
                _dataloader_kwargs,
                _make_attention_targets,
                _move_labels,
            )
        elif args.eval_cross_decode_sprites:
            evaluate_cross_decode_sprites(
                args,
                device,
                _prepare_attention_dataset,
                _prepare_arpl_model,
                _default_readout_path,
                _normalize_output_size,
            )
        else:
            train(args, device)
    finally:
        close_logger()


if __name__ == "__main__":
    main()
