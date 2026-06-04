import os
import random
import json
import atexit
import numpy as np
import torch
import torch.distributed as dist

from harpl.data.image_dataloader import ImageSequencesDataLoader, SpriteVideoDataLoader
from harpl.data._valid_names_lists import DATASET_NAMES
from harpl.modules.criterion import PredLoss, InvLoss, SupervisedLoss, LeJEPALoss
from harpl.modules.utils import LinearWarmupCosineAnnealingLR


_LOGGER_BACKEND = "none"
_TENSORBOARD_WRITER = None
_PENDING_LOGS = {}
_LOG_STEP = 0


def get_data_specs(dataset, 
                   target_label=None, 
                   mnist_seqtype=None,
                   spritevid_num_sprites=None,
                   flatten_images=False):
    f"""Get the data specifications for the specified dataset.
    
    Args:
        dataset (str): The dataset name. One of {DATASET_NAMES}.
        target_label (str): The target label for the dataset.
        mnist_seqtype (str): The sequence type for MNIST.
        flatten_images (bool): Whether to flatten the images.

    Returns:
        int: The input size.
        int or tuple: The number of classes. If multiple targets, returns a tuple of dicts
                        where each dict corresponds to the number of classes for each target.
                        The size of the tuple is five, corresponding to discrete sequence targets,
                        discrete frame/sequence item targets, continuous sequence targets,
                        continuous frame/sequence item targets, and additional targets that
                        might be needed for further offline analysis.
    """
    if dataset == "mnist":
        if mnist_seqtype == "triplets":
            num_classes = ({}, {"digit": 10, "digit_cluster": 4, "triplet": 18}, {}, {})
        else:
            raise ValueError(f"Invalid sequence type: {mnist_seqtype}")
        input_size = 28 * 28 if flatten_images else 28
    elif dataset == "animals":
        input_size = 64 * 64 if flatten_images else 64
        num_classes = ({"sprite_idx": spritevid_num_sprites, "rotation_direction": 3},
                       {"x-direction": 3, "y-direction": 3, "z-direction": 3, "x-position (discr)": 33, "y-position (discr)": 33, "z-position (discr)": 17, "orientation": 36},
                       {"speed": 1, "rotation_speed": 1},
                       {"x-position": 1, "y-position": 1, "z-position": 1, "x-velocity": 1, "y-velocity": 1, "z-velocity": 1, "sin": 1, "cos": 1})
    else:
        raise ValueError("HARPL supports only the 'mnist' and 'animals' datasets")
    return input_size, num_classes


def prepare_data(
        dataset,
        data_input_dir,
        val_size=0.0,
        seq_len=49,
        distributed=False, 
        batch_size=128, 
        val_batch_size=256, 
        target_label="seq2label",
        num_workers=16,
        grayscale=False,
        mnist_seqtype=None,
        spritevid_max_sprites=16,
        spritevid_exclude_latent_regions=False,
        spritevid_discretize_latents=False,
        spritevid_noise_type=None,
        spritevid_noise_level=0.1,
        spritevid_frozen_noise=False,
        sprite_noise_on_top=False,
        spritevid_grid_enabled=False,
        spritevid_frozen_grid=False,
        spritevid_occlude_n_frames=0,
        num_sequences=10000,
        inter_trial_interval=0,
        ):
    f"""Prepare the data for training and evaluation.

    Args:
        dataset (str): The dataset name. One of {DATASET_NAMES}.
        data_input_dir (str): The directory containing the dataset.
        val_size (float): The validation size.
        seq_len (int): The sequence length.
        distributed (bool): Whether to use distributed training.
        batch_size (int): The batch size.
        val_batch_size (int): The validation batch size.
        target_label (str): The target label for the dataset.
        num_workers (int): The number of workers for the data loader.
        grayscale (bool): Whether to load the data in grayscale.
        mnist_seqtype (str): The sequence type for the dataset (only for MNIST, FashionMNIST).
        spritevid_max_sprites (int): The maximum number of sprites (only for sprite videos).
        spritevid_exclude_latent_regions (bool): Whether to exclude latent regions during training to test generalization (only for sprite videos).
        spritevid_discretize_latents (bool): Whether to discretize latents (only for sprite videos).
        spritevid_noise_type (str): The noise type (only for sprite videos).
        spritevid_noise_level (float): The noise level (only for sprite videos).
        spritevid_frozen_noise (bool): Whether to use frozen noise (only for sprite videos).
        sprtite_noise_on_top (bool): Whether to add noise on top of the sprite video instead of background (only for sprite videos).
        spritevid_grid_enabled (bool): Whether to use grid (only for sprite videos).
        spritevid_frozen_grid (bool): Whether to use frozen grid (only for sprite videos).
        spritevid_occlude_n_frames (int): The number of frames to occlude (only for sprite videos).
        num_sequences (int): The number of sequences (only for MNIST, FashionMNIST).
        inter_trial_interval (int): The inter-trial interval (only for MNIST, FashionMNIST).

    Returns:
        torch.utils.data.DataLoader: The training data loader.
        torch.utils.data.Sampler: The training sampler.
        torch.utils.data.DataLoader: The validation data loader.
        torch.utils.data.Sampler: The validation sampler.
        torch.utils.data.DataLoader: The test data loader.
        torch.utils.data.Sampler: The test sampler.
    """
    if dataset == "animals":
        dataloader = SpriteVideoDataLoader(
            data_dir=data_input_dir,
            num_workers=num_workers,
            val_size=val_size,
            seq_len=seq_len,
            num_sequences=num_sequences,
            max_sprites=spritevid_max_sprites,
            exclude_latent_regions=spritevid_exclude_latent_regions,
            discretize_latents=spritevid_discretize_latents,
            noise_type=spritevid_noise_type,
            noise_level=spritevid_noise_level,
            frozen_noise=spritevid_frozen_noise,
            noise_on_top=sprite_noise_on_top,
            grid_enabled=spritevid_grid_enabled,
            frozen_grid=spritevid_frozen_grid,
            sprite_imgs=dataset,
            grayscale=grayscale,
            occlude_n_frames=spritevid_occlude_n_frames,
        )
        train_loader, train_sampler = dataloader.get_train(batch_size)
        val_loader, val_sampler = dataloader.get_validation(val_batch_size)
        test_loader, test_sampler = dataloader.get_test(val_batch_size)
    elif dataset == "mnist":
        dataloader = ImageSequencesDataLoader(
            data_dir=data_input_dir,
            num_workers=num_workers,
            seq_len=seq_len,
            seq_type=mnist_seqtype,
            num_sequences=num_sequences,
            val_size=val_size,
            inter_trial_interval=inter_trial_interval
        )
        train_loader, train_sampler = dataloader.get_train(batch_size)
        val_loader, val_sampler = dataloader.get_validation(val_batch_size)
        test_loader, test_sampler = dataloader.get_test(val_batch_size)
    else:
        raise ValueError("HARPL supports only the 'mnist' and 'animals' datasets")

    return (
        train_loader,
        train_sampler,
        val_loader,
        val_sampler,
        test_loader,
        test_sampler
    )


def get_criterion_input(z, context, pred, y, loss, downstream_input, prediction_target):
    """Get the input for the loss function (criterion)

    Args:
        z (torch.Tensor): The latent tensor (encoder output)
        context (torch.Tensor): The context tensor (integrator output)
        pred (torch.Tensor): The predictor tensor
        y (torch.Tensor): The target (label) tensor
        loss (str): The loss function (criterion)
        downstream_input (str): The input to the downstream task. For supervised loss, it can be "enc", "ctx", or "pred"; for other losses, it can be "enc" or "ctx".
        prediction_target (str): The target for the prediction. One of ["enc", "ctx"].

    Returns:
        tuple: The input for the criterion
        - If loss is "supervised", returns the relevant embedding tensor and the target tensor.
        - If loss is "inv", returns the embedding tensor.
        - Otherwise, returns the embedding tensor and the predictor tensor.

    Relevant embedding tensor is determined by the downstream_input parameter.
    """
    if loss == "supervised":
        if downstream_input == "enc":
            return z, y
        elif downstream_input == "ctx":
            return context, y
        else:
            return pred, y
    else:
        if prediction_target == "enc":
            pred_target = z
        elif prediction_target == "ctx":
            pred_target = context
        elif prediction_target == "pred":
            assert loss == "inv", "Prediction target 'pred' is only valid for inv loss"
            pred_target = pred
        else:
            raise ValueError(f"Invalid prediction target: {prediction_target}")
        if loss == "inv":
            return (pred_target,)
        else:
            return pred_target, pred


def prepare_criterion(
        loss, 
        pred_steps, 
        discount_factor=0.0, 
        pull_coef=1.0, 
        push_coef=1.0, 
        decorr_coef=1.0, 
        regularize_over="batch",
        readout=None,
        classification_task=None,
        dense_prediction=False,
        pred_loss_type="cosine",
        full_spatial_readout=False,
        no_sg=False,
        sigreg_lambd_ = 1.0,
        sigreg_knots = 17):
    """Prepare the loss function (criterion)

    Args:
        loss (str): The loss function
        pred_steps (int): The number of prediction steps
        discount_factor (float): The discount factor
        pull_coef (float): The pull coefficient
        push_coef (float): The push coefficient
        decorr_coef (float): The decorrelation coefficient
        regularize_over (str): The regularization dimension. One of ["batch", "batch+time", "time"]
        readout (nn.Module): The readout network
        classification_task (str): The classification task
        dense_prediction (bool): Whether to predict dense output, i.e. make predictions for each time step up to pred_steps
        full_spatial_readout (bool): Whether to use full spatial readout (for supervised loss)
        no_sg (bool): Whether to use stop-gradient for the target network. Only used for non-contrastive SSL losses.

    Returns:
        nn.Module: The loss function (criterion)
    """
    if loss == "pred":
        criterion = PredLoss(pred_steps=pred_steps, 
                             discount_factor=discount_factor,
                             dense_prediction=dense_prediction,
                             pred_loss_type=pred_loss_type,
                             no_sg=no_sg)
    elif loss == "inv":
        criterion = InvLoss(pred_steps=pred_steps, 
                            pull_coef=pull_coef,
                            push_coef=push_coef, 
                            decorr_coef=decorr_coef,
                            reg_dim=regularize_over,
                            dense_prediction=dense_prediction,
                            no_sg=no_sg)
    elif loss == "supervised":
        criterion = SupervisedLoss(readout=readout, task=classification_task, full_spatial_readout=full_spatial_readout)
    elif loss == "lejepa":
        criterion = LeJEPALoss(pred_steps=pred_steps, 
                               discount_factor=discount_factor,
                               dense_prediction=dense_prediction,
                               pred_loss_type=pred_loss_type,
                               no_sg=True,
                               sigreg_lambd_=sigreg_lambd_,
                               sigreg_knots=sigreg_knots)

    else:
        raise NotImplementedError(f"Loss {loss} not implemented")
    return criterion


def prepare_model_optimization(
        model,
        criterion,
        readout,
        optimizer,
        use_scheduler,
        epochs,
        lr,
        weight_decay,
        pred_lr_mult,
        online_lr,
        online_weight_decay):
    """Prepare the model optimization parameters

    Args:
        model (RePLModel): The model
        criterion (nn.Module): The loss function (criterion)
        readout (nn.Module): The readout network
        optimizer (str): The optimizer. One of ["sgd", "adam"]
        use_scheduler (bool): Whether to use the scheduler
        epochs (int): The number of epochs
        lr (float): The learning rate
        weight_decay (float): The weight decay
        pred_lr_mult (float): The learning rate multiplier for the predictor
        online_lr (float): The learning rate for online evaluation (readout)
        online_weight_decay (float): The weight decay for online evaluation (readout)

    Returns:
        tuple: The optimizer and the scheduler
    """
    params = []
    params_predictor = []
    params_readout = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif "predictor" in name:
            params_predictor.append(param)
        else:
            params.append(param)
    for name, param in criterion.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
    # readout is trained separately
    for name, param in readout.named_parameters():
        if not param.requires_grad:
            continue
        params_readout.append(param)
    
    model_params =  [{'params': params, 'lr': lr, 'weight_decay': weight_decay},
                     {'params': params_predictor, 'lr': lr * pred_lr_mult, 'weight_decay': weight_decay},
                     {'params': params_readout, 'lr': online_lr, 'weight_decay': online_weight_decay}]
    if optimizer == "adam":
        optimizer = torch.optim.AdamW(model_params)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model_params)
    
    # set up scheduler TODO: disable scheduler for readout
    if use_scheduler:
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=epochs)
    else:
        scheduler = None

    return optimizer, scheduler


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process():
    return get_rank() == 0


def safe_barrier():
    if is_dist_initialized():
        dist.barrier()


def init_distributed(backend, url, local_rank, device):
    """Initialize distributed training

    Args:
        backend (str): The backend.
        url (str): The URL.
        local_rank (int): The local rank.

    Returns:
        list[int]: The device IDs
    """
    if device.type == "mps":
        raise RuntimeError("Distributed training is not supported on MPS. Run single-process HARPL instead.")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    elif backend == "nccl":
        raise RuntimeError("The nccl backend requires CUDA. Use --dist_backend gloo for CPU distributed runs.")
    dist.init_process_group(
        backend=backend,
        init_method=url,
    )
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {get_rank()}, backend={dist.get_backend()}"
    )
    if device.type == "cuda":
        n = torch.cuda.device_count() // dist.get_world_size()
        device_ids = list(range(get_rank() * n, (get_rank() + 1) * n))
    else:
        device_ids = None
    return device_ids


def seed_everything(seed, deterministic=False):
    """Seed everything for reproducibility

    Args:
        seed (int): The seed
        deterministic (bool): Whether to use deterministic training
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def dist_reduce_mean(tensor):
    """Reduce tensor mean across all processes
    
    Args:
        tensor (torch.Tensor): Tensor to reduce
        
    Returns:
        torch.Tensor: Reduced tensor
    """
    if not is_dist_initialized():
        return tensor
    size = dist.get_world_size()
    if size > 1:
        tensor = tensor.detach().clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= size
    return tensor


def init_logger(args):
    """Initialize the configured experiment logger on the main process."""
    global _LOGGER_BACKEND, _TENSORBOARD_WRITER, _PENDING_LOGS, _LOG_STEP

    _PENDING_LOGS = {}
    _LOG_STEP = 0
    if getattr(args, "nolog", False) or getattr(args, "logger", "tensorboard") == "none":
        _LOGGER_BACKEND = "none"
        return
    if not is_main_process():
        _LOGGER_BACKEND = "none"
        return

    _LOGGER_BACKEND = args.logger
    if _LOGGER_BACKEND == "wandb":
        import wandb
        wandb.init(project="HARPL", name=args.experiment_name, config=args)
    elif _LOGGER_BACKEND == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(args.log_dir, args.experiment_name)
        _TENSORBOARD_WRITER = SummaryWriter(log_dir=log_dir)
        _TENSORBOARD_WRITER.add_text("config", _format_args_for_tensorboard(args), 0)
    else:
        raise ValueError(f"Unknown logger backend: {_LOGGER_BACKEND}")


def close_logger():
    """Flush and close the configured experiment logger."""
    global _TENSORBOARD_WRITER
    if _LOGGER_BACKEND == "tensorboard" and _TENSORBOARD_WRITER is not None:
        _TENSORBOARD_WRITER.flush()
        _TENSORBOARD_WRITER.close()
        _TENSORBOARD_WRITER = None


atexit.register(close_logger)


def _format_args_for_tensorboard(args):
    args_dict = vars(args)
    formatted = json.dumps(args_dict, indent=2, sort_keys=True)
    return f"```json\n{formatted}\n```"


def _as_loggable_value(variable):
    if isinstance(variable, torch.Tensor):
        variable = variable.detach().clone()
        variable = dist_reduce_mean(variable)
        if variable.numel() == 1:
            return variable.item()
        return variable.detach().cpu().flatten()
    if isinstance(variable, np.ndarray):
        if variable.size == 1:
            return variable.item()
        return variable.flatten()
    return variable


def _write_tensorboard_scalar(label, value, step):
    if _TENSORBOARD_WRITER is None:
        return
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            _TENSORBOARD_WRITER.add_scalar(label, value.item(), step)
        else:
            for idx, item in enumerate(value.flatten()):
                _TENSORBOARD_WRITER.add_scalar(f"{label}/{idx}", item, step)
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            _TENSORBOARD_WRITER.add_scalar(label, value[0], step)
        else:
            for idx, item in enumerate(value):
                _TENSORBOARD_WRITER.add_scalar(f"{label}/{idx}", item, step)
    else:
        _TENSORBOARD_WRITER.add_scalar(label, value, step)


def log_variable(variable, label, commit=True):
    """Log a variable to the configured experiment logger.

    Args:
        variable (torch.Tensor): Variable to log
        label (str): Label for the variable
        commit (bool, optional): Whether to commit the log batch. Defaults to True.

    Returns:
        torch.Tensor: Logged variable
    """
    global _PENDING_LOGS, _LOG_STEP
    variable = _as_loggable_value(variable)
    if is_main_process():
        if _LOGGER_BACKEND == "wandb":
            import wandb
            wandb.log({label: variable}, commit=commit)
        elif _LOGGER_BACKEND == "tensorboard":
            _PENDING_LOGS[label] = variable
            if commit:
                for pending_label, pending_value in _PENDING_LOGS.items():
                    _write_tensorboard_scalar(pending_label, pending_value, _LOG_STEP)
                _TENSORBOARD_WRITER.flush()
                _PENDING_LOGS = {}
                _LOG_STEP += 1
    return variable
