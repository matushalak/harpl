import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from harpl.data.attention_sprites_dataset import (
    MovingAnimalAttentionDataset,
    TASK_TO_ID,
)
from harpl.networks.harpl import (
    ARPLmodel,
    ChannelAttentionDecoder,
    ClassificationHead,
)
from harpl.networks.utils import additional_data_process, prepare_model
from harpl.scripts.args import (
    add_model_args,
    add_optimization_args,
    add_reproducibility_args,
)
from harpl.scripts.utils import (
    get_data_specs,
    is_cuda_device,
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


def _dataloader_kwargs(args, device):
    if torch.device(args.spritevid_device).type == "cuda" and args.num_workers != 0:
        raise ValueError("CUDA attention dataset rendering requires --num_workers 0.")
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
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


def _infer_first_encoder_channels(encoder):
    backbone = getattr(encoder, "backbone", None)
    if backbone is None:
        raise TypeError("Cannot infer attention channels from an encoder without a backbone.")
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.out_channels)
    raise TypeError("Cannot infer attention channels because the encoder has no Conv2d layer.")


def _infer_predictor_output_dim(predictor, fallback):
    last_linear = None
    for module in predictor.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return int(last_linear.out_features)
    return int(fallback)


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


def _prepare_attention_dataset(args):
    output_size = _normalize_output_size(args.spritevid_output_size)
    return MovingAnimalAttentionDataset(
        data_dir=args.data_input_dir,
        task=args.attention_task,
        tasks=_parse_task_values(args.attention_tasks),
        output_size=output_size,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        sprite_img_dir=args.sprite_img_dir,
        max_sprites=args.spritevid_max_sprites,
        seed=args.seed,
        device=args.spritevid_device,
        noise_type=args.spritevid_noise_type,
        noise_level=args.spritevid_noise_level,
        freeze_noise=args.spritevid_frozen_noise,
        noise_on_top=args.sprite_noise_on_top,
        popout_mode=args.popout_mode,
        num_distractors=args.num_distractors,
        crowd_size=args.crowd_size,
        cue_frames=args.cue_frames,
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
    attention_channels = _infer_first_encoder_channels(repl.encoder)
    head = ClassificationHead(input_dim=context_dim, num_classes=len(dataset.sprites))
    decoder = ChannelAttentionDecoder(
        input_dim=predictor_output_dim,
        output_dim=attention_channels,
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
    )
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


def train(args, device):
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    dataset = _prepare_attention_dataset(args)
    loader = DataLoader(dataset, **_dataloader_kwargs(args, device))
    model = _prepare_arpl_model(args, dataset, device)
    optimizer = _prepare_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    frozen_named = [(name, param) for name, param in model.named_parameters() if not param.requires_grad]
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

    for epoch in range(args.epochs):
        model.train()
        model.encoder_first.eval()
        model.encoder_tail.eval()
        model.integrator.eval()
        model.predictor.eval()

        total_loss = 0.0
        total_correct = 0
        total_items = 0
        total_batches = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_idx, (video, task_info) in enumerate(progress):
            video = video.to(device, non_blocking=non_blocking)
            task_info = _move_labels(task_info, device, non_blocking=non_blocking)

            *_, logits = model((video, task_info))
            targets, mask = _make_attention_targets(task_info, logits.size(1), args.cue_frames)
            loss = criterion(logits[mask], targets[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits[mask].argmax(dim=-1)
                correct = (preds == targets[mask]).sum().item()
                count = int(mask.sum().item())
                loss_value = loss.item()
                if first_loss is None:
                    first_loss = loss_value
                last_loss = loss_value
                total_loss += loss_value
                total_correct += correct
                total_items += count
                total_batches += 1
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    acc=f"{correct / max(count, 1):.3f}",
                )

            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break

        avg_loss = total_loss / max(total_batches, 1)
        avg_acc = total_correct / max(total_items, 1)
        print(
            f"epoch={epoch + 1} train_loss={avg_loss:.6f} "
            f"train_acc={avg_acc:.4f} batches={total_batches}"
        )

    if first_loss is None or last_loss is None:
        raise RuntimeError("No training batches were processed.")

    frozen_delta = _max_abs_delta(frozen_snapshot, frozen_named)
    trainable_delta = _max_abs_delta(trainable_snapshot, trainable_named)
    print(
        f"first_batch_loss={first_loss:.6f} last_batch_loss={last_loss:.6f} "
        f"trainable_max_delta={trainable_delta:.6g} frozen_max_delta={frozen_delta:.6g}"
    )
    if frozen_delta > args.freeze_tolerance:
        raise RuntimeError(
            f"Frozen RePL parameters changed by {frozen_delta:.6g}, "
            f"above --freeze_tolerance {args.freeze_tolerance}."
        )
    if trainable_delta <= 0.0:
        raise RuntimeError("No trainable attention/head parameter changed during training.")

    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "attention_model_final.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "attention_optimizer_final.pt")
        print(f"saved_checkpoint={checkpoint_dir / 'attention_model_final.pt'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train ARPL attention decoder/head on MovingAnimalAttentionDataset."
    )
    add_reproducibility_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)

    parser.add_argument("--model_path", type=str, required=True, help="Pretrained RePL checkpoint.")
    parser.add_argument("--data_input_dir", type=str, default="datasets")
    parser.add_argument("--sprite_img_dir", type=str, default="animals")
    parser.add_argument("--spritevid_max_sprites", type=int, default=8)
    parser.add_argument("--spritevid_output_size", type=int, nargs="+", default=[64])
    parser.add_argument("--spritevid_noise_type", choices=["gaussian", "salt_pepper", "none"], default="gaussian")
    parser.add_argument("--spritevid_noise_level", type=float, default=0.1)
    parser.add_argument("--spritevid_frozen_noise", action="store_true")
    parser.add_argument("--sprite_noise_on_top", action="store_true")
    parser.add_argument("--spritevid_device", type=str, default="cpu")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin-memory", "--pin_memory", dest="pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--persistent-workers", "--persistent_workers", dest="persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", "--prefetch_factor", dest="prefetch_factor", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--flatten_images", action="store_true")
    parser.add_argument("--return_full_features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention_task", default="mixed")
    parser.add_argument("--attention_tasks", nargs="*", default=None)
    parser.add_argument("--popout_mode", choices=["class", "rotation", "velocity", "mixed"], default="class")
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--crowd_size", type=int, default=5)
    parser.add_argument("--cue_frames", type=int, default=6)
    parser.add_argument("--attention_hidden_dim", type=int, default=None)
    parser.add_argument("--attention_decoder_layers", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=0, help="Stop each epoch after this many batches; 0 means full epoch.")
    parser.add_argument("--freeze_tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="attention")
    parser.add_argument("--torch_num_threads", type=int, default=0)

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
    device = select_device(args.device)
    train(args, device)


if __name__ == "__main__":
    main()
