import argparse
from copy import deepcopy
import glob
import os
from pathlib import Path

from harpl.scripts.args import (
    add_data_args,
    add_ddp_args,
    add_logging_args,
    add_model_args,
    add_offline_eval_args,
    add_reproducibility_args,
    add_validation_args,
    check_args,
)
from harpl.scripts.offline_eval import main as offline_eval_main
from harpl.scripts.utils import (
    close_logger,
    init_distributed,
    init_logger,
    seed_everything,
    select_device,
)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_int_list(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return [int(item) for item in value.split()]


def _moving_animals_defaults():
    return {
        "dataset": "animals",
        "spritevid_max_sprites": 8,
        "spritevid_noise_type": "gaussian",
        "spritevid_noise_level": 0.1,
        "spritevid_output_size": _env_int_list("HARPL_SPRITEVID_OUTPUT_SIZE", [64]),
        "sprite_noise_on_top": True,
        "seq_len": 32,
        "num_sequences": _env_int("HARPL_NUM_SEQUENCES", 16000),
        "encoder": "conv2d",
        "use_bn": True,
        "enc_n_layers": 6,
        "enc_kernel_size": [(5, 5)] * 6,
        "enc_stride": [(2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1)],
        "enc_padding": [(2, 2)] * 6,
        "enc_output_dim": 32,
        "flatten_enc_output": True,
        "integrator": "lstm",
        "ctx_dim": 512,
        "predictor": "mlp",
        "pred_hidden_dim": 512,
        "pred_steps": 1,
        "num_workers": _env_int("HARPL_NUM_WORKERS", 8),
        "offline_task": "multitask",
        "offline_input": "ctx",
        "offline_batch_size": _env_int("HARPL_OFFLINE_BATCH_SIZE", 128),
        "offline_epochs": _env_int("HARPL_OFFLINE_EPOCHS", 250),
        "save_offline_readout": True,
        "use_sklearn_regression": True,
        "experiment_name": None,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the moving_animals offline post-training evaluation for one "
            "or more model checkpoints and append the metrics to the matching "
            "Weights & Biases run."
        )
    )
    add_reproducibility_args(parser)
    add_logging_args(parser)
    add_data_args(parser)
    add_model_args(parser)
    add_validation_args(parser)
    add_offline_eval_args(parser)
    add_ddp_args(parser)
    parser.add_argument(
        "--checkpoint",
        "--checkpoints",
        "--checkpoint_path",
        "--checkpoint_paths",
        dest="checkpoint_paths",
        nargs="+",
        default=None,
        help="Model checkpoint path(s), for example checkpoints/<run>/model_final.pt.",
    )
    parser.add_argument(
        "--allow_new_wandb_run",
        action="store_true",
        help="Create a new wandb run if the existing run cannot be resolved by name.",
    )
    parser.add_argument(
        "--no_wandb_config",
        action="store_true",
        help="Do not use the resolved wandb run config as defaults before applying CLI overrides.",
    )
    parser.set_defaults(**_moving_animals_defaults())
    return parser


def _has_option(argv, *option_names):
    for item in argv:
        if item in option_names:
            return True
        if any(item.startswith(f"{option}=") for option in option_names):
            return True
    return False


def _expand_checkpoint_paths(args):
    checkpoint_paths = args.checkpoint_paths
    if checkpoint_paths is None and args.model_path is not None:
        checkpoint_paths = [args.model_path]
    if not checkpoint_paths:
        raise ValueError("Specify at least one checkpoint with --checkpoint.")

    expanded = []
    for path in checkpoint_paths:
        matches = glob.glob(path)
        expanded.extend(matches if matches else [path])

    missing = [path for path in expanded if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): {}".format(", ".join(missing)))
    return sorted(expanded)


def _wandb_project_path(args):
    if args.wandb_entity:
        return f"{args.wandb_entity}/{args.wandb_project}"
    return args.wandb_project


def _resolve_wandb_run(args, run_name):
    import wandb

    api = wandb.Api()
    project_path = _wandb_project_path(args)
    try:
        runs = list(api.runs(project_path, filters={"display_name": run_name}, per_page=100))
    except Exception:
        runs = []
    if not runs:
        runs = [run for run in api.runs(project_path, per_page=1000) if run.name == run_name]
    if not runs:
        return None
    runs.sort(key=lambda run: getattr(run, "created_at", "") or "", reverse=True)
    if len(runs) > 1:
        print(f"Found {len(runs)} wandb runs named {run_name!r}; using latest id {runs[0].id}.")
    return runs[0]


def _parser_destinations(parser):
    return {
        action.dest
        for action in parser._actions
        if action.dest is not argparse.SUPPRESS
    }


def _args_with_run_config(argv, base_args, run):
    if run is None or base_args.no_wandb_config:
        return deepcopy(base_args)

    parser = build_parser()
    protected = {
        "allow_new_wandb_run",
        "checkpoint_paths",
        "model_path",
        "no_wandb_config",
        "wandb_resume",
        "wandb_run_id",
    }
    valid_dests = _parser_destinations(parser)
    defaults = {
        key: value
        for key, value in run.config.items()
        if key in valid_dests and key not in protected
    }
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def _run_name_for_checkpoint(args, checkpoint_path, experiment_name_explicit):
    if experiment_name_explicit and args.experiment_name:
        return args.experiment_name
    return Path(checkpoint_path).parent.name


def _should_use_wandb(args):
    return not args.nolog and args.logger == "wandb"


def _run_one(argv, initial_args, checkpoint_path, experiment_name_explicit):
    run_name = _run_name_for_checkpoint(initial_args, checkpoint_path, experiment_name_explicit)
    run = None
    wandb_run_id = initial_args.wandb_run_id

    if _should_use_wandb(initial_args) and wandb_run_id is None and not initial_args.allow_new_wandb_run:
        run = _resolve_wandb_run(initial_args, run_name)
        if run is None:
            raise RuntimeError(
                f"Could not find a wandb run named {run_name!r} in "
                f"{_wandb_project_path(initial_args)!r}. Pass --wandb_run_id or "
                "--allow_new_wandb_run."
            )
        wandb_run_id = run.id

    args = _args_with_run_config(argv, initial_args, run)
    args.model_path = checkpoint_path
    args.experiment_name = run_name
    if wandb_run_id is not None:
        args.wandb_run_id = wandb_run_id
        args.wandb_resume = args.wandb_resume or "must"

    if args.offline_task == "none":
        args.offline_task = None
    if args.offline_task is None:
        raise ValueError("Offline task is None. For moving_animals this should usually be --offline_task multitask.")

    print(f"Running offline evaluation for {checkpoint_path}")
    print(f"Logging as experiment {args.experiment_name!r}" + (f" to wandb run id {args.wandb_run_id}" if args.wandb_run_id else ""))

    seed_everything(args.seed, args.deterministic)
    device = select_device(args.device)
    if args.distributed:
        init_distributed(args.dist_backend, args.dist_url, args.local_rank, device)

    init_logger(args)
    try:
        check_args(args)
        offline_eval_main(args, device)
    finally:
        close_logger()


def main(argv=None):
    if argv is None:
        import sys

        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    argv = list(argv)
    checkpoint_paths = _expand_checkpoint_paths(args)
    experiment_name_explicit = _has_option(argv, "--experiment_name")

    for checkpoint_path in checkpoint_paths:
        _run_one(argv, args, checkpoint_path, experiment_name_explicit)


if __name__ == "__main__":
    main()
