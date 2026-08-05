from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import shutil
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

import torch
from torch.utils.data import DataLoader

from utils.checkpointing import CheckpointManager
from utils.dataset_helpers import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)
from utils.experiment import (
    WarmupThenCosineScheduler,
    WarmupThenPlateauScheduler,
    _TeeStream,
    build_experiment_directory,
    plot_loss_curves,
    save_loss_history,
)
from utils.metrics import save_epoch_metrics
from weaver.utils.dataset import SimpleIterDataset


def add_common_training_args(
    parser: argparse.ArgumentParser,
    *,
    default_lr: float = 1e-3,
    default_batch_size: int = 96,
    default_epochs: int = 50,
    default_num_workers: int = 4,
    default_keep_best_k: int = 5,
    default_warmup_fraction: float = 0.05,
    default_min_lr: float = 1e-6,
    default_weight_decay: float = 0.01,
    default_grad_clip: float = 1.0,
    default_train_fraction: float = 0.8,
    default_save_every: int = 5,
) -> None:
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=default_epochs)
    parser.add_argument('--batch-size', type=int, default=default_batch_size)
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--weight-decay', type=float, default=default_weight_decay)
    parser.add_argument('--warmup-fraction', type=float, default=default_warmup_fraction)
    parser.add_argument('--min-lr', type=float, default=default_min_lr)
    parser.add_argument('--grad-clip', type=float, default=default_grad_clip)
    parser.add_argument('--train-fraction', type=float, default=default_train_fraction)
    parser.add_argument('--val-data-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=default_num_workers)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=default_save_every)
    parser.add_argument('--keep-best-k', type=int, default=default_keep_best_k)
    parser.add_argument('--resume', type=str, default=None)


def setup_experiment(args) -> tuple[str, str, str]:
    """Returns (experiment_dir, checkpoints_dir, tensorboard_dir).
    Configures root logger + tees stderr to training.log."""
    resume_dir = None
    if args.resume is not None:
        resume_dir = os.path.dirname(os.path.dirname(args.resume))
    experiment_dir, checkpoints_dir, tensorboard_dir = build_experiment_directory(
        args.experiments_dir, args.model_name, resume_dir,
    )

    log_file = os.path.join(experiment_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    log_file_handle = open(log_file, 'a')  # noqa: SIM115
    sys.stderr = _TeeStream(sys.stderr, log_file_handle)
    return experiment_dir, checkpoints_dir, tensorboard_dir


def build_data_loaders(args, logger: logging.Logger) -> tuple[
    DataLoader, DataLoader, Any, str, int, int, int,
]:
    """Returns (train_loader, val_loader, data_config, experiment_dir,
    steps_per_epoch, mask_input_index, label_input_index)."""
    load_in_memory = not args.no_in_memory

    train_parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(
        f'Found {len(train_parquet_files)} train parquet files in {args.data_dir}'
    )
    train_file_dict = {'data': train_parquet_files}
    num_train_files = len(train_parquet_files)

    if args.val_data_dir is not None:
        val_parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
        logger.info(
            f'Found {len(val_parquet_files)} val parquet files in {args.val_data_dir}'
        )
        val_file_dict = {'data': val_parquet_files}
        num_val_files = len(val_parquet_files)
        train_range = ((0.0, 1.0), 1.0)
        val_range = ((0.0, 1.0), 1.0)
    else:
        val_file_dict = train_file_dict
        num_val_files = num_train_files
        train_range = ((0.0, args.train_fraction), 1.0)
        val_range = ((args.train_fraction, 1.0), 1.0)

    train_num_workers = min(args.num_workers, num_train_files)
    # async_load spawns a ThreadPoolExecutor inside the dataset BEFORE torch
    # forks its DataLoader workers; the children inherit the executor's lock
    # in a locked state and hang forever after their first fetch. in_memory
    # mode gains nothing from async prefetch, so it stays off.
    train_dataset = SimpleIterDataset(
        train_file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=train_range,
        fetch_by_files=True,
        fetch_step=num_train_files,
        in_memory=load_in_memory,
        async_load=False,
    )
    data_config = train_dataset.config

    val_dataset = SimpleIterDataset(
        val_file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=val_range,
        fetch_by_files=True,
        fetch_step=num_val_files,
        in_memory=load_in_memory,
        async_load=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=train_num_workers,
        persistent_workers=train_num_workers > 0,
    )
    val_num_workers = (
        min(max(1, train_num_workers // 2), num_val_files)
        if train_num_workers > 0 else 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=val_num_workers,
        persistent_workers=val_num_workers > 0,
    )

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = 100
        logger.warning(
            f'--steps-per-epoch not set, defaulting to {steps_per_epoch}.',
        )
    logger.info(f'Steps per epoch: {steps_per_epoch}')
    logger.info(
        f'DataLoader workers: train={train_num_workers}, val={val_num_workers}'
    )

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    return (
        train_loader, val_loader, data_config,
        steps_per_epoch, mask_input_index, label_input_index,
    )


def copy_auto_yaml(args, experiment_dir: str, logger: logging.Logger) -> None:
    auto_yaml_pattern = args.data_config.replace('.yaml', '.*.auto.yaml')
    for auto_yaml_path in glob.glob(auto_yaml_pattern):
        shutil.copy2(auto_yaml_path, experiment_dir)
        logger.info(
            f'Copied auto.yaml to experiment dir: {os.path.basename(auto_yaml_path)}'
        )


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    args,
    steps_per_epoch: int,
    logger: logging.Logger,
):
    """Build cosine or plateau scheduler with the standard warmup-fraction
    formula. args needs: epochs, warmup_fraction, min_lr; optional scheduler
    ('cosine' default), plateau_factor, plateau_patience."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(int(args.warmup_fraction * total_steps), 2000)
    scheduler_kind = getattr(args, 'scheduler', 'cosine')

    if scheduler_kind == 'cosine':
        warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
        num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
        cosine_power = getattr(args, 'cosine_power', 1.0)
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'CosineAnnealingLR over {num_post_warmup_epochs} epochs '
            f'(cosine_power={cosine_power})',
        )
        return WarmupThenCosineScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_post_warmup_epochs=num_post_warmup_epochs,
            min_lr=args.min_lr,
            cosine_power=cosine_power,
        )

    logger.info(
        f'LR schedule: {warmup_steps} warmup steps, then '
        f'ReduceLROnPlateau (factor={args.plateau_factor})',
    )
    return WarmupThenPlateauScheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        min_lr=args.min_lr,
    )


def setup_torch_compile(
    model: torch.nn.Module,
    args,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    use_compile = (
        not getattr(args, 'no_compile', False)
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        logging.getLogger('torch._inductor').setLevel(logging.WARNING)
        logging.getLogger('torch._dynamo').setLevel(logging.WARNING)
        logger.info('Compiling model with torch.compile...')
        model = torch.compile(model, dynamic=True)
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')
    return model


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_scaler,
    device: torch.device,
    data_config,
    epoch: int,
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    label_input_index: int,
    logger: logging.Logger,
    *,
    grad_clip_max_norm: float = 1.0,
    compute_loss_fn: Callable | None = None,
    pop_keys: tuple[str, ...] = (),
    preprocess_inputs: Callable | None = None,
    on_step_end: Callable | None = None,
) -> tuple[dict[str, float], int]:
    """Generic per-epoch training loop.

    compute_loss_fn(model, points, features, lorentz_vectors, mask, track_labels) -> loss_dict.
    Default: model.compute_loss(...).
    pop_keys: extra keys to pop from loss_dict before accumulating (e.g. '_scores').
    preprocess_inputs(points, features, lorentz, mask, labels) -> 5-tuple (e.g. augmentation).
    on_step_end(model, loss_dict, popped, batch_index) -> None (e.g. EMA, Spearman sample)."""
    if compute_loss_fn is None:
        compute_loss_fn = lambda model_, points_, features_, lorentz_, mask_, labels_: (
            model_.compute_loss(points_, features_, lorentz_, mask_, labels_)
        )
    pop_keys_full = ('_scores',) + tuple(pop_keys)

    model.train()
    loss_accumulators: dict[str, torch.Tensor] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, (X, _, _) in enumerate(train_loader):
        if batch_index >= steps_per_epoch:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        padded_length = inputs[0].shape[2]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)

        if batch_index == 0:
            trimmed_length = inputs[0].shape[2]
            logger.info(
                f'Epoch {epoch} | Trim: {padded_length} → {trimmed_length} '
                f'({100 * (1 - trimmed_length / padded_length):.0f}% '
                f'padding removed)',
            )

        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, features, lorentz_vectors, mask = model_inputs

        if preprocess_inputs is not None:
            points, features, lorentz_vectors, mask = preprocess_inputs(
                points, features, lorentz_vectors, mask, track_labels,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            loss_dict = compute_loss_fn(
                model, points, features, lorentz_vectors, mask, track_labels,
            )
        popped = {key: loss_dict.pop(key, None) for key in pop_keys_full}
        loss = loss_dict['total_loss']

        if not torch.isfinite(loss).item():
            logger.warning(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Skipping batch with non-finite loss',
            )
            optimizer.zero_grad(set_to_none=True)
            global_batch_count += 1
            continue

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()

        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                grad_clip_max_norm,
            )

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        if on_step_end is not None:
            on_step_end(model, loss_dict, popped, batch_index)

        scheduler.step_batch()

        if loss_accumulators is None:
            loss_accumulators = {
                key: torch.zeros(1, device=loss.device) for key in loss_dict
            }
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].detach()

        num_batches += 1
        global_batch_count += 1

        if batch_index % 20 == 0:
            elapsed = time.time() - start_time
            avg_loss = loss_accumulators['total_loss'].item() / num_batches
            components = ' | '.join(
                f'{key.replace("_loss", "")}: '
                f'{value.item() / num_batches:.5f}'
                for key, value in loss_accumulators.items()
                if key != 'total_loss'
            )
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | Avg: {avg_loss:.5f} | '
                f'{components} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                f'Time: {elapsed:.1f}s',
            )

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': torch.zeros(1)}
    loss_averages = {
        key: value.item() / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    components = ' | '.join(
        f'{key.replace("_loss", "")}: {value:.5f}'
        for key, value in loss_averages.items()
        if key != 'total_loss'
    )
    logger.info(
        f'Epoch {epoch} train | total: {loss_averages["total_loss"]:.5f} | '
        f'{components}',
    )
    return loss_averages, global_batch_count


@torch.no_grad()
def validate_loop(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    *,
    compute_loss_fn: Callable | None = None,
    metrics_accumulator,
    pop_keys: tuple[str, ...] = (),
    update_metrics_fn: Callable | None = None,
    bn_train_mode: bool = False,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Generic validation loop.

    compute_loss_fn: same shape as train_one_epoch.
    pop_keys: extra keys to pop from loss_dict (besides _scores).
    update_metrics_fn(accumulator, popped, model_inputs, track_labels) -> None.
        Default: accumulator.update(popped['_scores'], track_labels, mask).
    bn_train_mode: when True, flip model to train() mode for the forward call
        (cascade BN-batch-stats workaround), then back to eval()."""
    if compute_loss_fn is None:
        compute_loss_fn = lambda model_, points_, features_, lorentz_, mask_, labels_: (
            model_.compute_loss(points_, features_, lorentz_, mask_, labels_)
        )
    pop_keys_full = ('_scores',) + tuple(pop_keys)

    model.eval()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0

    for batch_index, (X, _, _) in enumerate(val_loader):
        if max_steps is not None and batch_index >= max_steps:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, features, lorentz_vectors, mask = model_inputs

        if bn_train_mode:
            model.train()
        loss_dict = compute_loss_fn(
            model, points, features, lorentz_vectors, mask, track_labels,
        )
        if bn_train_mode:
            model.eval()

        popped = {key: loss_dict.pop(key, None) for key in pop_keys_full}

        if update_metrics_fn is not None:
            update_metrics_fn(
                metrics_accumulator, popped,
                (points, features, lorentz_vectors, mask), track_labels,
                model,
            )
        else:
            scores = popped['_scores'].detach()
            metrics_accumulator.update(scores, track_labels, mask)

        if loss_accumulators is None:
            loss_accumulators = {key: 0.0 for key in loss_dict}
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].item()

        num_batches += 1

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    metrics = metrics_accumulator.compute()
    return loss_averages, metrics


def save_per_epoch_artifacts(
    *,
    experiment_dir: str,
    tensorboard_writer: 'SummaryWriter',
    loss_history: dict,
    metric_labels: dict | None,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
    val_metrics: dict[str, float],
    train_eval_metrics: dict[str, float] | None,
    current_lr: float,
    epoch: int,
    extra_history: dict | None = None,
    epoch_metrics_extras: dict | None = None,
) -> None:
    val_loss = val_losses['total_loss']
    tensorboard_writer.add_scalar(
        'Loss/train_epoch', train_losses['total_loss'], epoch,
    )
    tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
    for metric_key, metric_value in val_metrics.items():
        if metric_key == 'total_gt_tracks':
            continue
        tensorboard_writer.add_scalar(
            f'Metrics/val_{metric_key}', metric_value, epoch,
        )
    if train_eval_metrics:
        for metric_key, metric_value in train_eval_metrics.items():
            if metric_key == 'total_gt_tracks':
                continue
            tensorboard_writer.add_scalar(
                f'Metrics/train_{metric_key}', metric_value, epoch,
            )
    tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)
    if extra_history:
        for key, value in extra_history.items():
            tensorboard_writer.add_scalar(f'Metrics/{key}', value, epoch)

    loss_history['train'].append(train_losses['total_loss'])
    loss_history['val'].append(val_loss)
    loss_history['lr'].append(current_lr)
    for prefix, metrics in [('val', val_metrics), ('train', train_eval_metrics or {})]:
        for metric_key, metric_value in metrics.items():
            if metric_key == 'total_gt_tracks':
                continue
            history_key = f'{prefix}_{metric_key}'
            if history_key not in loss_history:
                loss_history[history_key] = []
            loss_history[history_key].append(metric_value)
    save_loss_history(
        loss_history, experiment_dir, metric_labels=metric_labels,
    )

    epoch_metrics = {
        'epoch': epoch,
        'train_loss': train_losses['total_loss'],
        'val_loss': val_loss,
        'lr': current_lr,
    }
    if epoch_metrics_extras:
        epoch_metrics.update(epoch_metrics_extras)
    for metric_key, metric_value in val_metrics.items():
        epoch_metrics[f'val_{metric_key}'] = metric_value
    if train_eval_metrics:
        for metric_key, metric_value in train_eval_metrics.items():
            epoch_metrics[f'train_{metric_key}'] = metric_value
    save_epoch_metrics(epoch_metrics, experiment_dir, epoch)


def run_training(
    *,
    args,
    logger: logging.Logger,
    network_module_path: str,
    get_model_kwargs: dict[str, Any],
    selection_metric: str,
    criterion_name_short: str,
    optimizer_factory: Callable,
    scheduler_factory: Callable | None = None,
    metric_labels: dict[str, str] | None = None,
    compute_loss_train_fn: Callable | None = None,
    compute_loss_val_fn: Callable | None = None,
    metrics_accumulator_factory: Callable,
    pop_keys_train: tuple[str, ...] = (),
    pop_keys_val: tuple[str, ...] = (),
    preprocess_train_inputs: Callable | None = None,
    on_epoch_start: Callable | None = None,
    on_step_end: Callable | None = None,
    update_val_metrics_fn: Callable | None = None,
    bn_train_mode_for_val: bool = False,
    also_validate_train: bool = True,
    train_eval_steps_divisor: int = 4,
    log_metrics_summary: Callable | None = None,
    make_checkpoint_dict_fn: Callable,
    on_resume: Callable | None = None,
    load_model_fn: Callable | None = None,
    final_cleanup_fn: Callable | None = None,
    extra_loss_history_keys: tuple[str, ...] = (),
    epoch_metrics_extras_fn: Callable | None = None,
    use_torch_compile: bool = True,
) -> None:
    """Top-level training driver. Sets up experiment, data loaders, model,
    optimizer, scheduler, GradScaler, checkpoint manager, tensorboard, then
    runs the train→validate→save loop. Each callback hook is optional;
    defaults match the simplest common case.

    Returns nothing — checkpoints + history written to experiment_dir.
    """
    device = torch.device(args.device)

    experiment_dir, checkpoints_dir, tensorboard_dir = setup_experiment(args)
    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    (
        train_loader, val_loader, data_config,
        steps_per_epoch, mask_input_index, label_input_index,
    ) = build_data_loaders(args, logger)
    copy_auto_yaml(args, experiment_dir, logger)

    network_module = load_network_module(network_module_path)
    model, _ = network_module.get_model(data_config, **get_model_kwargs)
    original_model = model
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f'Total parameters: {total_params:,} | Trainable: {trainable_params:,}',
    )

    if use_torch_compile:
        model = setup_torch_compile(model, args, device, logger)

    optimizer = optimizer_factory(model, args)
    if scheduler_factory is None:
        scheduler = build_warmup_scheduler(
            optimizer, args, steps_per_epoch, logger,
        )
    else:
        scheduler = scheduler_factory(
            optimizer, args, steps_per_epoch, logger,
        )

    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        logger.info('AMP enabled (fp16 forward + GradScaler backward).')

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name=criterion_name_short,
    )

    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    start_epoch = 1
    best_selection_value = 0.0
    best_val_loss = float('inf')
    best_val_epoch = 0
    global_batch_count = 0
    loss_history: dict[str, list] = {'train': [], 'val': [], 'lr': []}
    for key in extra_loss_history_keys:
        loss_history[key] = []

    resume_state = {
        'best_selection_value': best_selection_value,
        'best_val_loss': best_val_loss,
        'best_val_epoch': best_val_epoch,
        'global_batch_count': global_batch_count,
        'start_epoch': start_epoch,
    }
    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        if load_model_fn is None:
            original_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            load_model_fn(original_model, checkpoint)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, ValueError) as exc:
            logger.warning(
                f'Skipping optimizer.load_state_dict: {exc}. Training '
                f'resumes with fresh optimizer state.',
            )
        resume_state['start_epoch'] = checkpoint.get('epoch', 0) + 1
        resume_state['best_val_loss'] = checkpoint.get(
            'best_val_loss', float('inf'),
        )
        resume_state['best_val_epoch'] = checkpoint.get('best_val_epoch', 0)
        resume_state['global_batch_count'] = checkpoint.get(
            'global_batch_count', 0,
        )
        if on_resume is not None:
            on_resume(checkpoint, original_model, optimizer, args, device, resume_state)
        start_epoch = resume_state['start_epoch']
        best_selection_value = resume_state['best_selection_value']
        best_val_loss = resume_state['best_val_loss']
        best_val_epoch = resume_state['best_val_epoch']
        global_batch_count = resume_state['global_batch_count']
        logger.info(
            f'Resumed from epoch {start_epoch - 1}, '
            f'best {criterion_name_short}={best_selection_value:.5f}',
        )

    logger.info('=== Training ===')

    val_losses: dict[str, float] = {}
    val_metrics: dict[str, float] = {}
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            if on_epoch_start is not None:
                on_epoch_start(original_model, model, epoch, args)

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler, grad_scaler,
                device, data_config, epoch, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index, logger,
                grad_clip_max_norm=args.grad_clip,
                compute_loss_fn=compute_loss_train_fn,
                pop_keys=pop_keys_train,
                preprocess_inputs=preprocess_train_inputs,
                on_step_end=on_step_end,
            )

            eval_steps = max(1, steps_per_epoch // train_eval_steps_divisor)

            val_losses, val_metrics = validate_loop(
                model, val_loader, device, data_config,
                mask_input_index, label_input_index,
                compute_loss_fn=compute_loss_val_fn,
                metrics_accumulator=metrics_accumulator_factory(),
                pop_keys=pop_keys_val,
                update_metrics_fn=update_val_metrics_fn,
                bn_train_mode=bn_train_mode_for_val,
                max_steps=eval_steps,
            )
            train_eval_metrics: dict[str, float] | None = None
            if also_validate_train:
                _, train_eval_metrics = validate_loop(
                    model, train_loader, device, data_config,
                    mask_input_index, label_input_index,
                    compute_loss_fn=compute_loss_val_fn,
                    metrics_accumulator=metrics_accumulator_factory(),
                    pop_keys=pop_keys_val,
                    update_metrics_fn=update_val_metrics_fn,
                    bn_train_mode=bn_train_mode_for_val,
                    max_steps=eval_steps,
                )

            val_loss = val_losses['total_loss']
            selection_value = val_metrics.get(selection_metric, 0.0)

            is_best = selection_value > best_selection_value
            if is_best:
                best_selection_value = selection_value
                best_val_epoch = epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if log_metrics_summary is not None:
                log_metrics_summary(
                    epoch=epoch, val_metrics=val_metrics,
                    train_eval_metrics=train_eval_metrics,
                    val_losses=val_losses, train_losses=train_losses,
                    is_best=is_best,
                    best_value=best_selection_value,
                    best_epoch=best_val_epoch,
                    selection_value=selection_value,
                )

            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            extra_history = {}
            for key in extra_loss_history_keys:
                if key in train_losses:
                    extra_history[key] = train_losses[key]
            epoch_extras = {}
            if epoch_metrics_extras_fn is not None:
                epoch_extras = epoch_metrics_extras_fn(args, epoch)

            save_per_epoch_artifacts(
                experiment_dir=experiment_dir,
                tensorboard_writer=tensorboard_writer,
                loss_history=loss_history,
                metric_labels=metric_labels,
                train_losses=train_losses,
                val_losses=val_losses,
                val_metrics=val_metrics,
                train_eval_metrics=train_eval_metrics,
                current_lr=current_lr,
                epoch=epoch,
                extra_history=extra_history,
                epoch_metrics_extras=epoch_extras,
            )

            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                checkpoint_dict = make_checkpoint_dict_fn(
                    epoch=epoch,
                    original_model=original_model,
                    optimizer=optimizer,
                    best_selection_value=best_selection_value,
                    best_val_loss=best_val_loss,
                    best_val_epoch=best_val_epoch,
                    global_batch_count=global_batch_count,
                    val_losses=val_losses,
                    val_metrics=val_metrics,
                    args=args,
                )
                checkpoint_manager.save_checkpoint(
                    checkpoint_dict, epoch, selection_value, is_best,
                )

    except Exception:
        logger.error(
            f'Training failed with exception:\n{traceback.format_exc()}',
        )
        raise

    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)

    if final_cleanup_fn is not None:
        final_cleanup_fn(
            args=args, model=model, original_model=original_model,
            optimizer=optimizer, train_loader=train_loader,
            val_loader=val_loader, data_config=data_config,
            mask_input_index=mask_input_index,
            label_input_index=label_input_index,
            device=device,
            checkpoints_dir=checkpoints_dir,
            best_selection_value=best_selection_value,
            best_val_loss=best_val_loss,
            best_val_epoch=best_val_epoch,
            global_batch_count=global_batch_count,
            val_losses=val_losses, val_metrics=val_metrics,
            logger=logger,
        )

    logger.info(
        f'Training complete. Best {criterion_name_short}: {best_selection_value:.5f}',
    )
    logger.info(f'Experiment: {experiment_dir}')
