import argparse
import os
import time

import torch

from harpl.networks.utils import (
    additional_data_process,
    load_repl_weights_into_exposed_model,
    prepare_model,
)
from harpl.scripts.args import (
    add_data_args,
    add_model_args,
    add_reproducibility_args,
)
from harpl.scripts.utils import (
    get_data_specs,
    prepare_data,
    seed_everything,
    select_device,
)


DEFAULT_SUPERVISED_CHECKPOINT = "checkpoints/animals_cts_noiseOnTop0.1_supervised_1/model_final.pt"


def _add_args():
    parser = argparse.ArgumentParser(
        description="Verify RePLModelExposed output parity and timing against RePLModel."
    )
    add_reproducibility_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    parser.add_argument("--checkpoint", default=DEFAULT_SUPERVISED_CHECKPOINT)
    parser.add_argument("--max_batches", type=int, default=0, help="0 verifies the full test loader.")
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--compile_exposed", action="store_true")
    parser.add_argument("--compile_mode", default=None)
    parser.add_argument("--return_activations", action="store_true")
    parser.add_argument("--batch_static_layers", action="store_true")
    parser.add_argument("--torch_num_threads", type=int, default=1, help="Set <=0 to keep PyTorch default.")

    parser.set_defaults(
        dataset="animals",
        spritevid_max_sprites=8,
        spritevid_noise_type="gaussian",
        spritevid_noise_level=0.1,
        spritevid_output_size=[64],
        sprite_noise_on_top=True,
        seq_len=32,
        num_sequences=16000,
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
        dense_prediction=False,
        prediction_target="enc",
        pred_target_dim_override=0,
        val_batch_size=128,
        num_workers=0,
        persistent_workers=False,
        prefetch_factor=None,
        pin_memory=False,
        device="auto",
        spritevid_device="cpu",
    )
    return parser


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _move_batch(batch, device, non_blocking=False):
    data = batch[0] if isinstance(batch, (tuple, list)) else batch
    return data.to(device, non_blocking=non_blocking)


def _forward(model, data, return_activations):
    if return_activations:
        output = model(data, return_activations=True)
        return output[:3]
    return model(data)


def _compare_outputs(reference, exposed):
    names = ("z", "context", "pred")
    stats = []
    strict_equal = True
    allclose = True
    for name, ref, exp in zip(names, reference, exposed):
        equal = torch.equal(ref, exp)
        close = torch.allclose(ref, exp, atol=_compare_outputs.atol, rtol=_compare_outputs.rtol)
        diff = (ref - exp).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        stats.append((name, equal, close, max_abs, mean_abs, tuple(ref.shape)))
        strict_equal = strict_equal and equal
        allclose = allclose and close
    return strict_equal, allclose, stats


def _print_stats(prefix, stats):
    for name, equal, close, max_abs, mean_abs, shape in stats:
        print(
            f"{prefix} {name}: shape={shape} "
            f"equal={equal} allclose={close} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
        )


def _prepare_models(args, input_size, n_in_channels, preprocess, postprocess, device):
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    common_kwargs = dict(
        encoder_kind=args.encoder,
        integrator_kind=args.integrator,
        predictor_kind=args.predictor,
        input_size=input_size,
        n_in_channels=n_in_channels,
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
        return_full_features=args.flatten_enc_output,
        flatten_enc_output=args.flatten_enc_output,
    )

    reference = prepare_model(**common_kwargs, state_dict=state_dict)
    exposed = prepare_model(
        **common_kwargs,
        exposed=True,
        exposed_kwargs={
            "freeze_repl": True,
            "eval_frozen": True,
            "compile_model": args.compile_exposed,
            "compile_mode": args.compile_mode,
            "batch_static_layers": args.batch_static_layers,
        },
    )
    load_repl_weights_into_exposed_model(exposed, state_dict)

    reference.eval()
    exposed.eval()
    return reference.to(device), exposed.to(device)


def main(args):
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    seed_everything(args.seed)
    device = select_device(args.device)
    input_size, _ = get_data_specs(
        args.dataset,
        target_label="multitask",
        mnist_seqtype=args.mnist_seqtype,
        spritevid_num_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        flatten_images=args.flatten_images,
    )
    preprocess, postprocess = additional_data_process(args.dataset, args.flatten_enc_output)
    n_in_channels = 1 if args.grayscale else 3
    reference, exposed = _prepare_models(args, input_size, n_in_channels, preprocess, postprocess, device)

    *_, test_loader, _ = prepare_data(
        args.dataset,
        args.data_input_dir,
        val_size=args.val_size,
        seq_len=args.seq_len,
        batch_size=args.val_batch_size,
        val_batch_size=args.val_batch_size,
        distributed=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        grayscale=args.grayscale,
        target_label="multitask",
        mnist_seqtype=args.mnist_seqtype,
        spritevid_max_sprites=args.spritevid_max_sprites,
        spritevid_output_size=args.spritevid_output_size,
        spritevid_exclude_latent_regions=args.spritevid_exclude_latent_regions,
        spritevid_discretize_latents=args.spritevid_discretize_latents,
        spritevid_noise_type=args.spritevid_noise_type,
        spritevid_noise_level=args.spritevid_noise_level,
        spritevid_frozen_noise=args.spritevid_frozen_noise,
        sprite_noise_on_top=args.sprite_noise_on_top,
        spritevid_grid_enabled=args.spritevid_grid_enabled,
        spritevid_frozen_grid=args.spritevid_frozen_grid,
        spritevid_occlude_n_frames=args.spritevid_occlude_n_frames,
        spritevid_device=args.spritevid_device,
        num_sequences=args.num_sequences,
        inter_trial_interval=args.inter_trial_interval,
    )

    _compare_outputs.atol = args.atol
    _compare_outputs.rtol = args.rtol

    ref_time = 0.0
    exposed_time = 0.0
    compared_batches = 0
    strict_equal = True
    allclose = True
    first_failed_stats = None
    non_blocking = args.pin_memory and device.type == "cuda"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_batches and compared_batches >= args.max_batches:
                break

            data = _move_batch(batch, device, non_blocking=non_blocking)

            for _ in range(args.warmup_batches if batch_idx == 0 else 0):
                _ = _forward(reference, data, False)
                _ = _forward(exposed, data, args.return_activations)

            _sync(device)
            start = time.perf_counter()
            ref_out = _forward(reference, data, False)
            _sync(device)
            ref_time += time.perf_counter() - start

            _sync(device)
            start = time.perf_counter()
            exposed_out = _forward(exposed, data, args.return_activations)
            _sync(device)
            exposed_time += time.perf_counter() - start

            batch_equal, batch_close, stats = _compare_outputs(ref_out, exposed_out)
            strict_equal = strict_equal and batch_equal
            allclose = allclose and batch_close
            if not batch_close and first_failed_stats is None:
                first_failed_stats = (batch_idx, stats)
            compared_batches += 1

    if compared_batches == 0:
        raise RuntimeError("No test batches were evaluated.")

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"batches: {compared_batches}")
    print(f"strict_equal: {strict_equal}")
    print(f"allclose(atol={args.atol}, rtol={args.rtol}): {allclose}")
    print(f"RePLModel total: {ref_time:.6f}s ({ref_time / compared_batches:.6f}s/batch)")
    print(f"RePLModelExposed total: {exposed_time:.6f}s ({exposed_time / compared_batches:.6f}s/batch)")
    print(f"slowdown: {exposed_time / ref_time:.3f}x")

    if first_failed_stats is not None:
        batch_idx, stats = first_failed_stats
        _print_stats(f"first mismatch batch {batch_idx}", stats)
        raise SystemExit(1)


if __name__ == "__main__":
    main(_add_args().parse_args())
