from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from utils.checkpointing import CheckpointManager
from utils.couple_dump_data import CoupleDumpDataset
from utils.couple_features import (
    COUPLE_FEATURE_DIM_TOTAL,
    COUPLE_REST_DIM,
    TRACK_EMBED_DIM,
)
from utils.dataset_helpers import (
    extract_label_from_inputs,
    trim_to_max_valid_tracks,
)
from utils.experiment import plot_loss_curves
from utils.metrics import (
    CoupleMetricsAccumulator,
    format_couple_metrics_table,
)
from utils.training import (
    add_common_training_args,
    build_warmup_scheduler,
    run_training,
    save_per_epoch_artifacts,
    setup_experiment,
)
from weaver.nn.model.CoupleDumpModel import CoupleDumpModel
from weaver.nn.model.CoupleReranker import CoupleReranker

logger = logging.getLogger('train_couple_reranker')

torch.set_float32_matmul_precision('high')


def _build_metric_labels(
    k_values_tracks: list[int],
    k_values_couples: list[int],
) -> dict[str, str]:
    labels: dict[str, str] = {
        'train': 'Train loss (couple ranking, mean per epoch)',
        'val': 'Validation loss (couple ranking)',
        'lr': 'Learning rate',
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
            f'RC@{k}_couples: C@{k}_couples AND full triplet in cascade Stage 1 top-K1'
        )
    return labels


def _compute_batch_mean_first_gt_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
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
            model_inputs, _ = extract_label_from_inputs(inputs, label_input_index)
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


def _couple_checkpoint_dict(*, epoch, couple_reranker, optimizer,
                            best_selection_value, best_val_epoch,
                            global_batch_count, val_losses, val_metrics,
                            args) -> dict:
    return {
        'epoch': epoch,
        'couple_reranker_state_dict': couple_reranker.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_c_at_100': best_selection_value,
        'best_val_epoch': best_val_epoch,
        'global_batch_count': global_batch_count,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'args': vars(args),
        'feature_layout': {
            'track_embed_dim': TRACK_EMBED_DIM,
            'rest_dim': COUPLE_REST_DIM,
            'couple_feature_dim_total': COUPLE_FEATURE_DIM_TOTAL,
        },
    }


# ---------------------------------------------------------------------------
# Dump mode: train from precomputed cascade dumps (no Stage 1/2 forward).
# Compact loop instead of utils.training.run_training because run_training's
# batch interface is the 5-tuple cascade input contract; it reuses the shared
# scheduler / CheckpointManager / per-epoch-artifact helpers and reproduces
# the cascade-mode checkpoint layout and logging format.
# ---------------------------------------------------------------------------

def _expand_dump_files(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No dump parquet files match: {pattern}')
    return paths


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _dump_train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_scaler,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    global_batch_count: int,
    grad_clip_max_norm: float,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_accumulators: dict[str, torch.Tensor] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, batch in enumerate(train_loader):
        if batch_index >= steps_per_epoch:
            break
        batch = _batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            loss_dict = model.compute_loss(batch)
        for key in [name for name in loss_dict if name.startswith('_')]:
            loss_dict.pop(key)
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
                model.parameters(), grad_clip_max_norm,
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

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': torch.zeros(1)}
    loss_averages = {
        key: value.item() / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }
    logger.info(
        f'Epoch {epoch} train | total: {loss_averages["total_loss"]:.5f}',
    )
    return loss_averages, global_batch_count


@torch.no_grad()
def _dump_validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    metrics_accumulator: CoupleMetricsAccumulator,
) -> tuple[dict[str, float], dict[str, float]]:
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    for batch in val_loader:
        batch = _batch_to_device(batch, device)
        # train-mode forward for parity with cascade mode's
        # bn_train_mode_for_val=True (keeps C@K comparable across modes).
        model.train()
        loss_dict = model.compute_loss(batch)
        model.eval()

        metrics_accumulator.update(
            loss_dict.pop('_scores').detach(),
            loss_dict.pop('_couple_labels').detach(),
            loss_dict.pop('_couple_mask').detach(),
            n_gt_in_top_k1=loss_dict.pop('_n_gt_in_top_k1').detach(),
            n_gt_in_top_k_tracks=loss_dict.pop(
                '_n_gt_in_top_k_tracks').detach(),
        )
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
    return loss_averages, metrics_accumulator.compute()


def _dump_calibrate_batchnorm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    calibration_steps: int,
) -> None:
    logger.info(
        f'Calibrating CoupleReranker BN running stats '
        f'({calibration_steps} steps)...',
    )
    for module in model.couple_reranker.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()
    model.train()
    with torch.no_grad():
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= calibration_steps:
                break
            batch = _batch_to_device(batch, device)
            couple_inputs = model._build_couple_inputs(batch)
            model.couple_reranker(couple_inputs['couple_features'])
            if (batch_index + 1) % 50 == 0:
                logger.info(
                    f'BN calibration: {batch_index + 1}/{calibration_steps}',
                )
    model.eval()


def run_dump_training(args) -> None:
    device = torch.device(args.device)
    experiment_dir, checkpoints_dir, tensorboard_dir = setup_experiment(args)
    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    train_dataset = CoupleDumpDataset(_expand_dump_files(args.train_dump))
    val_dataset = CoupleDumpDataset(_expand_dump_files(args.val_dump))
    logger.info(
        f'Dump datasets: train={len(train_dataset)} events '
        f'(K1={train_dataset.top_k1}), val={len(val_dataset)} events',
    )
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, pin_memory=pin_memory, num_workers=args.num_workers,
        collate_fn=CoupleDumpDataset.collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, pin_memory=pin_memory, num_workers=args.num_workers,
        collate_fn=CoupleDumpDataset.collate,
    )

    couple_reranker = CoupleReranker(
        hidden_dim=args.couple_hidden_dim,
        num_residual_blocks=args.couple_num_residual_blocks,
        dropout=args.couple_dropout,
        ranking_num_samples=args.couple_ranking_num_samples,
        ranking_temperature=args.couple_ranking_temperature,
        label_smoothing=args.couple_label_smoothing,
        couple_projector_dim=args.couple_projector_dim,
        rest_dim=COUPLE_REST_DIM,
        track_embed_dim=TRACK_EMBED_DIM,
    )
    model = CoupleDumpModel(
        couple_reranker=couple_reranker,
        top_k2=args.top_k2,
        k_values_tracks=tuple(args.k_values_tracks),
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(
        f'CoupleDumpModel: {trainable_parameters:,} trainable parameters '
        f'(input_dim={couple_reranker.input_dim})',
    )

    # One-off companion-cone precompute: the cone block is deterministic per
    # (event, K2); caching it removes the dominant per-batch builder cost.
    cache_start = time.time()
    logger.info('Precomputing companion-cone cache (train)...')
    train_dataset.cone_cache = model.build_cone_cache(
        train_dataset, args.batch_size, device,
        num_workers=args.num_workers,
    )
    logger.info('Precomputing companion-cone cache (val)...')
    val_dataset.cone_cache = model.build_cone_cache(
        val_dataset, args.batch_size, device,
        num_workers=args.num_workers,
    )
    cache_gib = (
        train_dataset.cone_cache.numel()
        + val_dataset.cone_cache.numel()
    ) * 2 / 2 ** 30
    logger.info(
        f'Cone caches ready in {time.time() - cache_start:.1f}s '
        f'({cache_gib:.1f} GiB host RAM)',
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = args.steps_per_epoch or max(1, len(train_loader))
    logger.info(f'Steps per epoch: {steps_per_epoch}')
    scheduler = build_warmup_scheduler(optimizer, args, steps_per_epoch, logger)
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None
    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='C@100',
    )
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)
    metric_labels = _build_metric_labels(
        args.k_values_tracks, args.k_values_couples,
    )
    loss_history: dict[str, list] = {'train': [], 'val': [], 'lr': []}

    best_selection_value = float('-inf')
    best_val_epoch = 0
    global_batch_count = 0
    val_losses: dict[str, float] = {}
    val_metrics: dict[str, float] = {}

    logger.info('=== Training (dump mode) ===')
    for epoch in range(1, args.epochs + 1):
        logger.info(f'=== Epoch {epoch}/{args.epochs} ===')
        train_losses, global_batch_count = _dump_train_one_epoch(
            model, train_loader, optimizer, scheduler, grad_scaler, device,
            epoch, steps_per_epoch, global_batch_count, args.grad_clip,
        )
        val_losses, val_metrics = _dump_validate(
            model, val_loader, device,
            metrics_accumulator=CoupleMetricsAccumulator(
                k_values_couples=tuple(args.k_values_couples),
                k_values_tracks=tuple(args.k_values_tracks),
            ),
        )
        selection_value = val_metrics.get('c_at_100_couples', 0.0)
        is_best = selection_value > best_selection_value
        if is_best:
            best_selection_value = selection_value
            best_val_epoch = epoch

        val_table = format_couple_metrics_table(
            val_metrics,
            train_loss=train_losses['total_loss'],
            val_loss=val_losses['total_loss'],
            epoch=epoch,
            is_best=is_best,
            best_val_criterion=best_selection_value,
            best_val_epoch=best_val_epoch,
            criterion_name='C@100c',
            k_values_tracks=tuple(args.k_values_tracks),
            k_values_couples=tuple(args.k_values_couples),
        )
        logger.info('\n' + val_table)

        scheduler.step_epoch(val_losses['total_loss'])
        save_per_epoch_artifacts(
            experiment_dir=experiment_dir,
            tensorboard_writer=tensorboard_writer,
            loss_history=loss_history,
            metric_labels=metric_labels,
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=val_metrics,
            train_eval_metrics=None,
            current_lr=scheduler.get_last_lr()[0],
            epoch=epoch,
            epoch_metrics_extras={'top_k2': args.top_k2},
        )
        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            checkpoint_manager.save_checkpoint(
                _couple_checkpoint_dict(
                    epoch=epoch,
                    couple_reranker=model.couple_reranker,
                    optimizer=optimizer,
                    best_selection_value=best_selection_value,
                    best_val_epoch=best_val_epoch,
                    global_batch_count=global_batch_count,
                    val_losses=val_losses,
                    val_metrics=val_metrics,
                    args=args,
                ),
                epoch, selection_value, is_best,
            )

    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)

    if args.bn_calibration_steps > 0:
        _dump_calibrate_batchnorm(
            model, train_loader, device, args.bn_calibration_steps,
        )
        calibrated_path = os.path.join(
            checkpoints_dir, 'best_model_calibrated.pt',
        )
        torch.save(
            _couple_checkpoint_dict(
                epoch=args.epochs,
                couple_reranker=model.couple_reranker,
                optimizer=optimizer,
                best_selection_value=best_selection_value,
                best_val_epoch=best_val_epoch,
                global_batch_count=global_batch_count,
                val_losses=val_losses,
                val_metrics=val_metrics,
                args=args,
            ),
            calibrated_path,
        )
        logger.info(f'Saved calibrated checkpoint: {calibrated_path}')

    logger.info(
        f'Training complete. Best C@100: {best_selection_value:.5f}',
    )
    logger.info(f'Experiment: {experiment_dir}')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CoupleReranker (Stage 3) on top of a frozen cascade',
    )
    add_common_training_args(
        parser, default_lr=5e-4, default_batch_size=16, default_epochs=50,
        require_data_args=False,
    )
    parser.add_argument('--stage1-checkpoint', type=str, default=None)
    parser.add_argument('--stage2-checkpoint', type=str, default=None)
    parser.add_argument(
        '--train-dump', type=str, default=None,
        help='glob of Stage-3 input dump parquets '
             '(eval_cascade_pipeline --dump-stage3-inputs); together with '
             '--val-dump switches to dump mode (no cascade forward)',
    )
    parser.add_argument('--val-dump', type=str, default=None)
    parser.add_argument('--top-k2', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='CoupleReranker')
    parser.add_argument('--cosine-power', type=float, default=2.0)

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


def main(argv: list[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if 100 not in args.k_values_couples:
        parser.error(
            '--k-values-couples must include 100 (the selection criterion '
            f'is C@100_couples). Got: {args.k_values_couples}',
        )

    dump_mode = args.train_dump is not None or args.val_dump is not None
    if dump_mode:
        if not (args.train_dump and args.val_dump):
            parser.error('--train-dump and --val-dump are required together.')
        run_dump_training(args)
        return
    if not (args.stage1_checkpoint and args.stage2_checkpoint):
        parser.error(
            'cascade mode requires --stage1-checkpoint and '
            '--stage2-checkpoint (or pass --train-dump/--val-dump '
            'for dump mode).',
        )
    if not (args.data_config and args.data_dir and args.network):
        parser.error(
            'cascade mode requires --data-config, --data-dir and --network.',
        )

    state = {'batch_loss_samples': [], 'batch_mean_gt_rank_samples': []}

    def compute_loss_train(model, points, features, lorentz, mask, labels):
        out = model.compute_loss(points, features, lorentz, mask, labels)
        scores_t = out.get('_scores')
        labels_t = out.get('_couple_labels')
        mask_t = out.get('_couple_mask')
        if scores_t is not None and labels_t is not None and mask_t is not None:
            with torch.no_grad():
                rank = _compute_batch_mean_first_gt_rank(scores_t, labels_t, mask_t)
                if rank is not None:
                    state['batch_mean_gt_rank_samples'].append(rank)
                    state['batch_loss_samples'].append(
                        float(out['total_loss'].item()),
                    )
        return out

    def update_val_metrics(accumulator, popped, model_inputs, labels, model):
        accumulator.update(
            popped['_scores'].detach(),
            popped['_couple_labels'].detach(),
            popped['_couple_mask'].detach(),
            n_gt_in_top_k1=popped['_n_gt_in_top_k1'].detach(),
            n_gt_in_top_k_tracks=popped['_n_gt_in_top_k_tracks'].detach(),
        )

    def make_checkpoint_dict(*, epoch, original_model, optimizer,
                              best_selection_value, best_val_loss,
                              best_val_epoch, global_batch_count,
                              val_losses, val_metrics, args):
        return {
            'epoch': epoch,
            'couple_reranker_state_dict':
                original_model.couple_reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_c_at_100': best_selection_value,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
            'feature_layout': {
                'track_embed_dim': TRACK_EMBED_DIM,
                'rest_dim': COUPLE_REST_DIM,
                'couple_feature_dim_total': COUPLE_FEATURE_DIM_TOTAL,
            },
        }

    def log_summary(*, epoch, val_metrics, train_eval_metrics, val_losses,
                     train_losses, is_best, best_value, best_epoch,
                     selection_value):
        loss_samples = state['batch_loss_samples']
        rank_samples = state['batch_mean_gt_rank_samples']
        if rank_samples:
            mean_rank = sum(rank_samples) / len(rank_samples)
            tail_start = len(loss_samples) // 2
            rho = _spearman_rho(loss_samples[tail_start:], rank_samples[tail_start:])
            train_losses['mean_first_gt_rank_train'] = mean_rank
            train_losses['loss_rank_spearman_rho'] = rho
            logger.info(
                f'Epoch {epoch} train | mean_first_gt_rank_train: '
                f'{mean_rank:.3f} | ρ(loss, rank): {rho:+.3f} '
                f'(last {len(loss_samples) - tail_start} batches)',
            )
        state['batch_loss_samples'] = []
        state['batch_mean_gt_rank_samples'] = []

        val_table = format_couple_metrics_table(
            val_metrics,
            train_loss=train_losses['total_loss'],
            val_loss=val_losses['total_loss'],
            epoch=epoch,
            is_best=is_best,
            best_val_criterion=best_value,
            best_val_epoch=best_epoch,
            criterion_name='C@100c',
            k_values_tracks=tuple(args.k_values_tracks),
            k_values_couples=tuple(args.k_values_couples),
        )
        logger.info('\n' + val_table)

    def final_cleanup(*, args, model, original_model, optimizer, train_loader,
                       val_loader, data_config, mask_input_index,
                       label_input_index, device, checkpoints_dir,
                       best_selection_value, best_val_loss, best_val_epoch,
                       global_batch_count, val_losses, val_metrics, logger):
        if args.bn_calibration_steps <= 0:
            return
        logger.info(
            f'Calibrating CoupleReranker BN running stats '
            f'({args.bn_calibration_steps} steps)...',
        )
        calibrate_reranker_batchnorm(
            model, train_loader, device, data_config,
            mask_input_index, label_input_index,
            calibration_steps=args.bn_calibration_steps,
        )
        calibrated_path = os.path.join(checkpoints_dir, 'best_model_calibrated.pt')
        torch.save({
            'epoch': args.epochs,
            'couple_reranker_state_dict':
                original_model.couple_reranker.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_c_at_100': best_selection_value,
            'best_val_epoch': best_val_epoch,
            'global_batch_count': global_batch_count,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'args': vars(args),
            'feature_layout': {
                'track_embed_dim': TRACK_EMBED_DIM,
                'rest_dim': COUPLE_REST_DIM,
                'couple_feature_dim_total': COUPLE_FEATURE_DIM_TOTAL,
            },
        }, calibrated_path)
        logger.info(f'Saved calibrated checkpoint: {calibrated_path}')

    def epoch_metrics_extras(args, epoch):
        return {'top_k2': args.top_k2}

    def metrics_factory():
        return CoupleMetricsAccumulator(
            k_values_couples=tuple(args.k_values_couples),
            k_values_tracks=tuple(args.k_values_tracks),
        )

    def optimizer_factory(model, args):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def load_model_fn(model, checkpoint):
        # Per-stage save: 'couple_reranker_state_dict' is bare CoupleReranker weights.
        model.couple_reranker.load_state_dict(
            checkpoint['couple_reranker_state_dict'],
        )

    run_training(
        args=args,
        logger=logger,
        network_module_path=args.network,
        get_model_kwargs={
            'stage1_checkpoint': args.stage1_checkpoint,
            'stage2_checkpoint': args.stage2_checkpoint,
            'top_k2': args.top_k2,
            'k_values_tracks': tuple(args.k_values_tracks),
            'couple_hidden_dim': args.couple_hidden_dim,
            'couple_num_residual_blocks': args.couple_num_residual_blocks,
            'couple_dropout': args.couple_dropout,
            'couple_ranking_num_samples': args.couple_ranking_num_samples,
            'couple_ranking_temperature': args.couple_ranking_temperature,
            'couple_label_smoothing': args.couple_label_smoothing,
            'couple_projector_dim': args.couple_projector_dim,
        },
        selection_metric='c_at_100_couples',
        criterion_name_short='C@100',
        optimizer_factory=optimizer_factory,
        metric_labels=_build_metric_labels(
            args.k_values_tracks, args.k_values_couples,
        ),
        compute_loss_train_fn=compute_loss_train,
        metrics_accumulator_factory=metrics_factory,
        pop_keys_train=('_couple_labels', '_couple_mask',
                        '_n_gt_in_top_k1', '_n_gt_in_top_k_tracks'),
        pop_keys_val=('_couple_labels', '_couple_mask',
                      '_n_gt_in_top_k1', '_n_gt_in_top_k_tracks'),
        update_val_metrics_fn=update_val_metrics,
        bn_train_mode_for_val=True,
        also_validate_train=False,
        train_eval_steps_divisor=2,
        log_metrics_summary=log_summary,
        make_checkpoint_dict_fn=make_checkpoint_dict,
        load_model_fn=load_model_fn,
        final_cleanup_fn=final_cleanup,
        epoch_metrics_extras_fn=epoch_metrics_extras,
        use_torch_compile=False,
    )


if __name__ == '__main__':
    main()
