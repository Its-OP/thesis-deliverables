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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

import torch

torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

from utils.experiment import (
    WarmupThenCosineScheduler,
    _TeeStream,
    build_experiment_directory,
    plot_loss_curves,
    save_loss_history,
)
from utils.checkpointing import CheckpointManager
from utils.metrics import (
    CoupleMetricsAccumulator,
    format_couple_metrics_table,
    save_epoch_metrics,
)
from utils.dataset_helpers import (
    extract_label_from_inputs,
    load_network_module,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('train_couple_reranker')


def _build_metric_labels(
    k_values_tracks: list[int],
    k_values_couples: list[int],
) -> dict[str, str]:
    labels: dict[str, str] = {
        'train': 'Train loss (couple ranking, mean per epoch)',
        'val': 'Validation loss (couple ranking)',
        'lr': 'Learning rate',
        'val_eligible_events':
            'Eligible events (val): events with ≥1 GT couple in candidate pool',
        'val_total_events':
            'Total events (val) seen during validation',
        'val_events_with_full_triplet':
            'Events (val) with all 3 GT pions in cascade Stage 1 top-K1',
        'val_mean_first_gt_rank_couples':
            'Mean rank of best GT couple in reranker output (1-indexed; '
            'lower is better; averaged over eligible events)',
    }
    for k in k_values_tracks:
        labels[f'val_d_at_{k}_tracks'] = (
            f'D@{k}_tracks: events with ≥2 GT pions in ParT top-{k} tracks'
        )
    for k in k_values_couples:
        labels[f'val_c_at_{k}_couples'] = (
            f'C@{k}_couples: events with ≥1 GT couple in top-{k} of reranker output'
        )
        labels[f'val_rc_at_{k}_couples'] = (
            f'RC@{k}_couples: C@{k}_couples AND full triplet in cascade Stage 1 top-K1=256'
        )
    return labels


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: 'SummaryWriter | None',
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    label_input_index: int,
    grad_clip_max_norm: float = 1.0,
    grad_scaler: 'torch.amp.GradScaler | None' = None,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_accumulators: dict[str, torch.Tensor] | None = None
    num_batches = 0
    non_finite_batches = 0
    start_time = time.time()
    # Spearman ρ between batch loss and batch mean-first-GT-rank: positive ρ
    # means loss tracks ranking quality; near 0 means loss is rank-insensitive.
    batch_loss_samples: list[float] = []
    batch_mean_gt_rank_samples: list[float] = []

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

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
        with torch.no_grad():
            scores_t = loss_dict.get('_scores')
            labels_t = loss_dict.get('_couple_labels')
            mask_t = loss_dict.get('_couple_mask')
            if scores_t is not None and labels_t is not None and mask_t is not None:
                batch_mean_rank = _compute_batch_mean_first_gt_rank(
                    scores_t, labels_t, mask_t,
                )
                if batch_mean_rank is not None:
                    batch_mean_gt_rank_samples.append(batch_mean_rank)
                    batch_loss_samples.append(float(loss_dict['total_loss'].item()))
        loss_dict.pop('_scores', None)
        loss_dict.pop('_couple_labels', None)
        loss_dict.pop('_couple_mask', None)
        loss_dict.pop('_n_gt_in_top_k1', None)
        loss_dict.pop('_n_gt_in_top_k_tracks', None)
        loss = loss_dict['total_loss']

        if not torch.isfinite(loss).item():
            non_finite_batches += 1
            logger.warning(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Skipping batch with non-finite loss '
                f'(total non-finite: {non_finite_batches})',
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
            logger.info(
                f'Epoch {epoch} | Batch {batch_index} | '
                f'Loss: {loss.item():.5f} | Avg: {avg_loss:.5f} | '
                f'LR: {scheduler.get_last_lr()[0]:.2e} | '
                f'Time: {elapsed:.1f}s',
            )

        del inputs, model_inputs, track_labels, loss_dict

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': torch.zeros(1)}
    loss_averages = {
        key: value.item() / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    if non_finite_batches > 0:
        logger.warning(
            f'Epoch {epoch} train | {non_finite_batches} non-finite batches '
            f'skipped out of {num_batches + non_finite_batches}',
        )
    logger.info(
        f'Epoch {epoch} train | total: {loss_averages["total_loss"]:.5f}',
    )
    loss_averages['non_finite_batches'] = float(non_finite_batches)

    if batch_mean_gt_rank_samples:
        mean_gt_rank_avg = float(
            sum(batch_mean_gt_rank_samples) / len(batch_mean_gt_rank_samples),
        )
        loss_averages['mean_first_gt_rank_train'] = mean_gt_rank_avg
        # Restrict ρ to last 50% of batches: warmup phase has loss collapsing
        # while ranks haven't stabilised, which inflates ρ artificially.
        tail_start = len(batch_loss_samples) // 2
        tail_losses = batch_loss_samples[tail_start:]
        tail_ranks = batch_mean_gt_rank_samples[tail_start:]
        rho = _spearman_rho(tail_losses, tail_ranks)
        loss_averages['loss_rank_spearman_rho'] = rho
        logger.info(
            f'Epoch {epoch} train | mean_first_gt_rank_train: '
            f'{mean_gt_rank_avg:.3f} | ρ(loss, rank): {rho:+.3f} '
            f'(last {len(tail_losses)} batches)',
        )

    return loss_averages, global_batch_count


def _compute_batch_mean_first_gt_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    """Returns batch-average 1-indexed rank of the best GT couple, or None
    if no event in the batch has a GT couple among its valid candidates."""
    batch_size = scores.shape[0]
    ranks: list[float] = []
    for batch_index in range(batch_size):
        valid_mask = mask[batch_index] > 0.5
        if not valid_mask.any():
            continue
        gt_mask = (labels[batch_index] > 0.5) & valid_mask
        if not gt_mask.any():
            continue
        event_scores = scores[batch_index].clone()
        event_scores = event_scores.masked_fill(~valid_mask, float('-inf'))
        sorted_indices = torch.argsort(event_scores, descending=True)
        sorted_gt = gt_mask[sorted_indices]
        first_gt_position = int(sorted_gt.float().argmax().item())
        ranks.append(float(first_gt_position + 1))
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return 0.0
    stat = spearmanr(xs, ys).statistic
    if stat is None:
        return 0.0
    return 0.0 if math.isnan(stat) else float(stat)


def calibrate_reranker_batchnorm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    calibration_steps: int = 200,
) -> None:
    """Reset and recalibrate CoupleReranker BN running stats post-training."""
    for module in model.couple_reranker.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()

    model.couple_reranker.train()
    model.cascade.train()

    with torch.no_grad():
        for batch_index, (X, _, _) in enumerate(train_loader):
            if batch_index >= calibration_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, _ = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            dummy_labels = torch.zeros_like(mask)
            couple_inputs = model._build_couple_inputs(
                points, features, lorentz_vectors, mask, dummy_labels,
            )
            model.couple_reranker(couple_inputs['couple_features'])

            if (batch_index + 1) % 50 == 0:
                logger.info(
                    f'BN calibration: {batch_index + 1}/{calibration_steps}',
                )

    model.eval()


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int | None = None,
    k_values_couples: tuple[int, ...] = (50, 75, 100, 200),
    k_values_tracks: tuple[int, ...] = (30, 50, 75, 100, 200),
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    couple_metrics_accumulator = CoupleMetricsAccumulator(
        k_values_couples=tuple(k_values_couples),
        k_values_tracks=tuple(k_values_tracks),
    )

    for batch_index, (X, _, _) in enumerate(val_loader):
        if max_steps is not None and batch_index >= max_steps:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
        model_inputs, track_labels = extract_label_from_inputs(
            inputs, label_input_index,
        )
        points, features, lorentz_vectors, mask = model_inputs

        # train() mode for BN batch stats — Stage 1+2 BN constructed with
        # track_running_stats=False; the reranker BN matches.
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        model.eval()

        couple_scores = loss_dict.pop('_scores').detach()
        couple_labels = loss_dict.pop('_couple_labels').detach()
        couple_mask = loss_dict.pop('_couple_mask').detach()
        n_gt_in_top_k1 = loss_dict.pop('_n_gt_in_top_k1').detach()
        n_gt_in_top_k_tracks = loss_dict.pop('_n_gt_in_top_k_tracks').detach()

        couple_metrics_accumulator.update(
            couple_scores, couple_labels, couple_mask,
            n_gt_in_top_k1=n_gt_in_top_k1,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )

        if loss_accumulators is None:
            loss_accumulators = {key: 0.0 for key in loss_dict}
        for key in loss_accumulators:
            loss_accumulators[key] += loss_dict[key].item()

        num_batches += 1
        del inputs, model_inputs, track_labels, loss_dict

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    metrics = couple_metrics_accumulator.compute()
    return loss_averages, metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CoupleReranker (Stage 3) on top of a frozen cascade',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--cascade-checkpoint', type=str, required=True)
    parser.add_argument('--top-k2', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='CoupleReranker')
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--cosine-power', type=float, default=2.0)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--val-data-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--keep-best-k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--couple-hidden-dim', type=int, default=256)
    parser.add_argument('--couple-num-residual-blocks', type=int, default=4)
    parser.add_argument('--couple-dropout', type=float, default=0.1)
    parser.add_argument('--couple-ranking-num-samples', type=int, default=50)
    parser.add_argument('--couple-ranking-temperature', type=float, default=1.0)
    parser.add_argument('--couple-label-smoothing', type=float, default=0.10)
    parser.add_argument('--couple-projector-dim', type=int, default=32)

    parser.add_argument('--bn-calibration-steps', type=int, default=200)

    parser.add_argument(
        '--k-values-tracks', type=int, nargs='+',
        default=[30, 50, 75, 100, 200],
    )
    parser.add_argument(
        '--k-values-couples', type=int, nargs='+',
        default=[50, 75, 100, 200],
        help='Must include 100 (selection criterion is C@100_couples).',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    if 100 not in args.k_values_couples:
        parser.error(
            '--k-values-couples must include 100 (the selection criterion '
            f'is C@100_couples). Got: {args.k_values_couples}',
        )

    metric_labels = _build_metric_labels(
        k_values_tracks=args.k_values_tracks,
        k_values_couples=args.k_values_couples,
    )

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

    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    load_in_memory = not args.no_in_memory

    train_parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(f'Found {len(train_parquet_files)} train parquet files in {args.data_dir}')
    train_file_dict = {'data': train_parquet_files}
    num_train_files = len(train_parquet_files)

    if args.val_data_dir is not None:
        val_parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
        logger.info(f'Found {len(val_parquet_files)} val parquet files in {args.val_data_dir}')
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
    train_dataset = SimpleIterDataset(
        train_file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=train_range,
        fetch_by_files=True,
        fetch_step=num_train_files,
        in_memory=load_in_memory,
    )
    data_config = train_dataset.config

    auto_yaml_pattern = args.data_config.replace('.yaml', '.*.auto.yaml')
    for auto_yaml_path in glob.glob(auto_yaml_pattern):
        shutil.copy2(auto_yaml_path, experiment_dir)
        logger.info(f'Copied auto.yaml to experiment dir: {os.path.basename(auto_yaml_path)}')

    val_dataset = SimpleIterDataset(
        val_file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=val_range,
        fetch_by_files=True,
        fetch_step=num_val_files,
        in_memory=load_in_memory,
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
        if train_num_workers > 0
        else 0
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
    logger.info(f'DataLoader workers: train={train_num_workers}, val={val_num_workers}')

    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(
        data_config,
        cascade_checkpoint=args.cascade_checkpoint,
        top_k2=args.top_k2,
        k_values_tracks=tuple(args.k_values_tracks),
        couple_hidden_dim=args.couple_hidden_dim,
        couple_num_residual_blocks=args.couple_num_residual_blocks,
        couple_dropout=args.couple_dropout,
        couple_ranking_num_samples=args.couple_ranking_num_samples,
        couple_ranking_temperature=args.couple_ranking_temperature,
        couple_label_smoothing=args.couple_label_smoothing,
        couple_projector_dim=args.couple_projector_dim,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f'Total parameters: {total_params:,} | Trainable: {trainable_params:,}')

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')

    trainable_parameter_iter = filter(
        lambda parameter: parameter.requires_grad, model.parameters(),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameter_iter, lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(int(args.warmup_fraction * total_steps), max_warmup_steps)
    warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
    num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
    logger.info(
        f'LR schedule: {warmup_steps} warmup steps, then '
        f'CosineAnnealingLR over {num_post_warmup_epochs} epochs '
        f'(cosine_power={args.cosine_power})',
    )
    scheduler = WarmupThenCosineScheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_post_warmup_epochs=num_post_warmup_epochs,
        min_lr=args.min_lr,
        cosine_power=args.cosine_power,
    )

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='C@100',
    )

    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    start_epoch = 1
    best_val_c_at_100 = 0.0
    best_val_epoch = 0
    global_batch_count = 0
    loss_history: dict[str, list] = {
        'train': [], 'val': [], 'lr': [],
    }

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        # Slim ckpt: only the trainable couple_reranker; cascade rebuilt at startup.
        model.couple_reranker.load_state_dict(
            checkpoint['couple_reranker_state_dict'],
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_c_at_100 = checkpoint.get('best_val_c_at_100', 0.0)
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        logger.info(
            f'Resumed from epoch {start_epoch - 1}, '
            f'best C@100={best_val_c_at_100:.5f}',
        )

    logger.info(f'=== Training CoupleReranker (top_k2={args.top_k2}) ===')

    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        logger.info('AMP enabled (fp16 forward + GradScaler backward).')

    val_losses: dict[str, float] = {}
    val_metrics: dict[str, float] = {}
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index,
                grad_clip_max_norm=args.grad_clip,
                grad_scaler=grad_scaler,
            )

            eval_steps = max(1, steps_per_epoch // 2)

            val_losses, val_metrics = validate(
                model, val_loader, device, data_config,
                mask_input_index, label_input_index,
                max_steps=eval_steps,
                k_values_couples=tuple(args.k_values_couples),
                k_values_tracks=tuple(args.k_values_tracks),
            )

            val_loss = val_losses['total_loss']
            val_c_at_100 = val_metrics.get('c_at_100_couples', 0.0)

            is_best = val_c_at_100 > best_val_c_at_100
            if is_best:
                best_val_c_at_100 = val_c_at_100
                best_val_epoch = epoch

            val_table = format_couple_metrics_table(
                val_metrics,
                train_loss=train_losses['total_loss'],
                val_loss=val_loss,
                epoch=epoch,
                is_best=is_best,
                best_val_criterion=best_val_c_at_100,
                best_val_epoch=best_val_epoch,
                criterion_name='C@100c',
                k_values_tracks=tuple(args.k_values_tracks),
                k_values_couples=tuple(args.k_values_couples),
            )
            logger.info('\n' + val_table)

            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            tensorboard_writer.add_scalar('Loss/train_epoch', train_losses['total_loss'], epoch)
            tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            for metric_key, metric_value in val_metrics.items():
                tensorboard_writer.add_scalar(
                    f'Metrics/val_{metric_key}', metric_value, epoch,
                )
            tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)
            if 'mean_first_gt_rank_train' in train_losses:
                tensorboard_writer.add_scalar(
                    'Metrics/mean_first_gt_rank_train',
                    train_losses['mean_first_gt_rank_train'], epoch,
                )
            if 'loss_rank_spearman_rho' in train_losses:
                tensorboard_writer.add_scalar(
                    'Metrics/loss_rank_spearman_rho',
                    train_losses['loss_rank_spearman_rho'], epoch,
                )

            loss_history['train'].append(train_losses['total_loss'])
            loss_history['val'].append(val_loss)
            loss_history['lr'].append(current_lr)
            for metric_key, metric_value in val_metrics.items():
                history_key = f'val_{metric_key}'
                if history_key not in loss_history:
                    loss_history[history_key] = []
                loss_history[history_key].append(metric_value)
                if metric_key in loss_history:
                    loss_history[metric_key].append(metric_value)
            save_loss_history(
                loss_history, experiment_dir, metric_labels=metric_labels,
            )

            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_loss,
                'lr': current_lr,
                'top_k2': args.top_k2,
            }
            for metric_key, metric_value in val_metrics.items():
                epoch_metrics[f'val_{metric_key}'] = metric_value
            save_epoch_metrics(epoch_metrics, experiment_dir, epoch)

            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                checkpoint = {
                    'epoch': epoch,
                    'couple_reranker_state_dict':
                        model.couple_reranker.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_c_at_100': best_val_c_at_100,
                    'best_val_epoch': best_val_epoch,
                    'global_batch_count': global_batch_count,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }
                checkpoint_manager.save_checkpoint(
                    checkpoint, epoch, val_c_at_100, is_best,
                )

    except Exception:
        logger.error(f'Training failed:\n{traceback.format_exc()}')
        raise

    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)

    if args.bn_calibration_steps > 0:
        logger.info(
            f'Calibrating CoupleReranker BN running stats '
            f'({args.bn_calibration_steps} steps)...',
        )
        calibrate_reranker_batchnorm(
            model, train_loader, device, data_config,
            mask_input_index, label_input_index,
            calibration_steps=args.bn_calibration_steps,
        )
        calibrated_checkpoint = {
            'epoch': args.epochs,
            'couple_reranker_state_dict':
                model.couple_reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_c_at_100': best_val_c_at_100,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
        }
        calibrated_path = os.path.join(
            checkpoints_dir, 'best_model_calibrated.pt',
        )
        torch.save(calibrated_checkpoint, calibrated_path)
        logger.info(f'Saved calibrated checkpoint: {calibrated_path}')

    logger.info(f'Training complete. Best C@100: {best_val_c_at_100:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    main()
