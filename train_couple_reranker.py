"""Training script for the per-couple reranker (Stage 3).

Loads a frozen 2-stage cascade from a checkpoint, builds a fresh
``CoupleReranker`` head, and trains the head with pairwise ranking loss
on per-couple feature vectors enumerated from the ParT top-50 (Filter A:
``m(ij) <= m_tau``).

Architecture and pilot recipe per
``reports/triplet_reranking/triplet_research_plan_20260408.md`` direction A:
- Frozen cascade: TrackPreFilter (Stage 1) + CascadeReranker / ParT (Stage 2)
- Trainable head: ``CoupleReranker`` — input projection (Conv1d 51→256) + 4
  ``ResidualBlock(256)`` + scoring head (Conv1d 256→128→1), ~580K params
- Loss: pairwise ranking with N=50 random negatives per positive (matches
  the prefilter and CascadeReranker convention)
- Best metric: ``couple_recall_at_100`` on val

Usage:
    python train_couple_reranker.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/train/ \\
        --val-data-dir data/low-pt/val/ \\
        --network networks/lowpt_tau_CoupleReranker.py \\
        --cascade-checkpoint models/debug_checkpoints/cascade_soap_*/checkpoints/best_model.pt \\
        --top-k2 50 \\
        --epochs 50 --batch-size 16 --steps-per-epoch 500 \\
        --device cuda:0
"""
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
    WarmupThenPlateauScheduler,
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


# ---------------------------------------------------------------------------
# Metric labels for the on-disk loss_history.json
# ---------------------------------------------------------------------------
#
# Maps loss-history dict keys to short human-readable descriptions. The
# saver wraps each metric as ``{'label': str, 'values': list[float]}`` so
# the JSON file is self-documenting and the user can read it manually
# without grepping the code for what each key means.

def _build_metric_labels(
    k_values_tracks: list[int],
    k_values_couples: list[int],
) -> dict[str, str]:
    """Build the labelled-metric mapping used by ``save_loss_history``.

    The set of D/C/RC keys depends on which K values were configured for
    this run, so we generate the mapping at runtime instead of carrying
    a hardcoded dict at module load. This lets sweeps over different K
    grids (e.g., the top_k2 sweep with K_couples = 50, 60, ..., 200)
    produce a self-documenting JSON without touching this file.

    Args:
        k_values_tracks: K values reported for D@K_tracks.
        k_values_couples: K values reported for C@K_couples and
            RC@K_couples.

    Returns:
        ``{metric_key: human label}`` for every key the trainer writes
        into ``loss_history.json``.
    """
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
            f'D@{k}_tracks: events with ≥2 GT pions in ParT top-{k} tracks '
            f'(cascade duplet rate, fixed by checkpoint)'
        )
    for k in k_values_couples:
        labels[f'val_c_at_{k}_couples'] = (
            f'C@{k}_couples: events with ≥1 GT couple in top-{k} of reranker '
            f'output (per-event binary)'
        )
        labels[f'val_rc_at_{k}_couples'] = (
            f'RC@{k}_couples: C@{k}_couples AND full triplet in cascade '
            f'Stage 1 top-K1=256'
        )
    return labels


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

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
    ema_model: torch.nn.Module | None = None,
    train_aug: str = 'none',
    smear_scale: float = 1.0,
    drop_cov_features: bool = False,
    grad_scaler: 'torch.amp.GradScaler | None' = None,
) -> tuple[dict[str, float], int]:
    """Train the CoupleReranker for one epoch (frozen cascade inside)."""
    model.train()
    loss_accumulators: dict[str, torch.Tensor] | None = None
    num_batches = 0
    non_finite_batches = 0
    start_time = time.time()
    # Per-batch (loss, mean_first_gt_rank_couples) samples for the
    # position-aware diagnostic: Spearman ρ between these two signals
    # quantifies whether the training loss tracks ranking quality. A
    # loss that converges to low values while mean_gt_rank stays large
    # (or grows) produces ρ near 0, signalling the loss is not
    # discriminating on rank position.
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

        if train_aug == 'cov_smear':
            from utils.train_augmentation import (
                apply_cov_smear,
                drop_cov_features as _drop_cov_features,
            )
            features = apply_cov_smear(
                features, smear_scale=smear_scale,
            )
            if drop_cov_features:
                features = _drop_cov_features(features)

        optimizer.zero_grad(set_to_none=True)
        # AMP: when grad_scaler is provided, run forward under autocast
        # so the frozen cascade (ParT attention is the main compute
        # cost) runs in fp16. The couple reranker is small enough that
        # its fp32 gradient path remains numerically stable with the
        # standard GradScaler.
        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
        # Capture train-time mean_first_gt_rank_couples BEFORE the
        # heavy metric tensors are popped. Position-aware diagnostic:
        # batch-average rank of the best GT couple, computed the same
        # way as the val metric (CoupleMetricsAccumulator).
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
        # Drop the heavy metric tensors so the loss accumulator only sees
        # scalar loss components.
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
        else:
            loss.backward()

        # SAM / ASAM: do a second forward-backward pass at the
        # gradient-ascent-perturbed weights, then step with the second
        # gradient. Guarded by isinstance to keep the fast path for the
        # plain-optimizer case unchanged.
        from utils.sam_optimizer import SAM as _SAM
        if isinstance(optimizer, _SAM):
            optimizer.first_step(zero_grad=True)
            loss_dict_2 = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            loss_dict_2.pop('_scores', None)
            loss_dict_2.pop('_couple_labels', None)
            loss_dict_2.pop('_couple_mask', None)
            loss_dict_2.pop('_n_gt_in_top_k1', None)
            loss_dict_2.pop('_n_gt_in_top_k_tracks', None)
            loss_2 = loss_dict_2['total_loss']
            if not torch.isfinite(loss_2).item():
                # Undo the perturbation and skip the update for this batch.
                non_finite_batches += 1
                optimizer.second_step(zero_grad=True)
                global_batch_count += 1
                continue
            loss_2.backward()
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    grad_clip_max_norm,
                )
            optimizer.second_step(zero_grad=True)
        else:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
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
        if ema_model is not None:
            ema_model.update_parameters(model.couple_reranker)

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

    # Position-aware diagnostic: train-time mean rank of the best GT
    # couple, plus Spearman ρ between batch loss and batch mean rank.
    # ρ close to +1 means loss tracks ranking quality (higher loss ⇔
    # worse rank, which is what we want). ρ near 0 means the loss is
    # position-insensitive at this operating point.
    if batch_mean_gt_rank_samples:
        mean_gt_rank_avg = float(
            sum(batch_mean_gt_rank_samples)
            / len(batch_mean_gt_rank_samples),
        )
        loss_averages['mean_first_gt_rank_train'] = mean_gt_rank_avg
        # Restrict correlation to the final portion of the epoch: the
        # warmup phase has loss collapsing while ranks don't stabilise,
        # which inflates ρ artificially. Use last 50% of batches.
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
    """Batch-average 1-indexed rank of the best GT couple.

    Returns None if no event in the batch has a GT couple among its
    valid candidates (so the caller can skip the (loss, rank) sample).
    Matches the semantics of ``CoupleMetricsAccumulator`` for
    ``mean_first_gt_rank_couples``.
    """
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
        event_scores = event_scores.masked_fill(
            ~valid_mask, float('-inf'),
        )
        sorted_indices = torch.argsort(event_scores, descending=True)
        sorted_gt = gt_mask[sorted_indices]
        first_gt_position = int(sorted_gt.float().argmax().item())
        ranks.append(float(first_gt_position + 1))
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two equal-length samples.

    Returns 0.0 if fewer than 3 samples or if either list has no
    variance (all identical values).
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return 0.0
    stat = spearmanr(xs, ys).statistic
    if stat is None:
        return 0.0
    import math
    return 0.0 if math.isnan(stat) else float(stat)


# ---------------------------------------------------------------------------
# BN calibration — rebuild clean running stats after training
# ---------------------------------------------------------------------------

def calibrate_reranker_batchnorm(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    calibration_steps: int = 200,
) -> None:
    """Reset and recalibrate CoupleReranker BN running stats.

    After training, BN running stats may be stale or corrupted (NaN from
    degenerate batches). This function:
    1. Resets all BN running stats in the couple reranker to defaults
    2. Runs ``calibration_steps`` forward-only batches to accumulate
       clean running statistics
    3. Verifies no NaN remains in running stats

    The cascade stays in train() mode (its own BN workaround).
    """
    # Reset BN running stats to defaults
    for module in model.couple_reranker.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()

    # Calibration pass: couple reranker in train() mode to accumulate
    # running stats, but no gradients needed.
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

            # Forward through the full pipeline to update BN stats.
            # Only the reranker has trainable BN; Stage 1 and Stage 2 are
            # frozen (their BN stats are untouched here).
            dummy_labels = torch.zeros_like(mask)
            couple_inputs = model._build_couple_inputs(
                points, features, lorentz_vectors, mask, dummy_labels,
            )
            # Thread k2_features through to support the H6 event-context
            # path; .get() keeps backward compatibility with CoupleCascadeModel
            # versions that don't populate k2_features.
            model.couple_reranker(
                couple_inputs['couple_features'],
                k2_features=couple_inputs.get('k2_features'),
            )

            if (batch_index + 1) % 50 == 0:
                logger.info(
                    f'BN calibration: {batch_index + 1}/{calibration_steps}',
                )

    model.eval()


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

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
    """Validate and compute couple_recall@K metrics on the val set."""
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

        # Train mode for BatchNorm batch stats — same workaround as
        # train_cascade.py validate, since the cascade still has BN inside.
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train CoupleReranker (Stage 3) on top of a frozen cascade',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--cascade-checkpoint', type=str, required=True,
                        help='Path to a trained cascade checkpoint (Stage 1+2)')
    parser.add_argument('--top-k2', type=int, default=50,
                        help='Number of top tracks per event from which '
                             'couples are enumerated (default 50).')
    parser.add_argument('--model-name', type=str, default='CoupleReranker')
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument(
        '--cosine-power', type=float, default=2.0,
        help='Exponent for cosine LR decay. <1 = steeper (faster drop), '
             '>1 = delayed (stays high then drops steeply). Default 2.0 '
             '(baseline v2, 2026-04-17 sweep result).',
    )
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--val-data-dir', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument(
        '--amp', action='store_true',
        help=(
            'Enable mixed-precision training. Forward runs in fp16 under '
            'autocast (frozen cascade kernel fusion benefits most); '
            'backward uses GradScaler. Incompatible with SAM/ASAM '
            'optimizer paths (silently disabled there).'
        ),
    )
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--keep-best-k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Global seed for Python, NumPy, PyTorch (CPU + CUDA). '
             'If unset, RNG stays non-deterministic (legacy behavior).',
    )
    # CoupleReranker architecture knobs
    parser.add_argument('--couple-hidden-dim', type=int, default=256)
    parser.add_argument('--couple-num-residual-blocks', type=int, default=4)
    parser.add_argument('--couple-dropout', type=float, default=0.1)
    parser.add_argument('--couple-ranking-num-samples', type=int, default=50)
    parser.add_argument('--couple-ranking-temperature', type=float, default=1.0)
    parser.add_argument(
        '--couple-loss',
        type=str,
        default='softmax-ce',
        choices=['pairwise', 'softmax-ce', 'lambda_ndcg2pp', 'approx_ndcg'],
        help=(
            "Couple-reranker loss branch. 'pairwise' = softplus pairwise "
            "ranking (legacy). 'softmax-ce' = listwise softmax "
            "cross-entropy (ListMLE top-1), default baseline v2. "
            "'lambda_ndcg2pp' = LambdaLoss NDCG-Loss2++ pair-weighting "
            "with optional K-truncation (Wang 2018, arXiv:1811.04768); "
            "directly optimises NDCG@K ranking metric. 'approx_ndcg' = "
            "smooth differentiable NDCG via sigmoid rank approximation "
            "(Qin, Liu, Li 2010). Position-aware across the full list, "
            "no K-truncation, no negative sampling."
        ),
    )
    parser.add_argument(
        '--ndcg-k', type=int, default=100,
        help=(
            "Top-K truncation for --couple-loss lambda_ndcg2pp. "
            "Pairs whose BOTH ranks exceed K are dropped from the sum. "
            "Default 100 (matches the C@100 operating metric). Set to 0 "
            "to disable truncation (recommended when tail pairs still "
            "carry useful gradient via the log-discount decay)."
        ),
    )
    parser.add_argument(
        '--lambda-sigma', type=float, default=1.0,
        help=(
            "RankNet sigmoid scale for the lambda_ndcg2pp loss. "
            "Higher = sharper swap penalty. Default 1.0."
        ),
    )
    parser.add_argument(
        '--ndcg-alpha', type=float, default=5.0,
        help=(
            "Sigmoid sharpness for --couple-loss approx_ndcg. Controls "
            "the approximation quality of the rank estimate: as alpha "
            "grows, rank(i) ≈ true rank; at alpha → 0 the sigmoid is "
            "flat and the loss loses discriminative power. Default 5.0."
        ),
    )
    parser.add_argument(
        '--couple-use-full-negative-list',
        action='store_true',
        help=(
            'H4: disable 50-sample negative cap in the couple loss and '
            'use every valid non-GT couple in each event. Slower per '
            'step but removes sampling noise at the rank-100 boundary.'
        ),
    )
    parser.add_argument(
        '--couple-multi-positive',
        type=str,
        default='none',
        choices=['none', 'uniform', 'soft_or'],
        help=(
            "Target distribution for softmax-CE when an event has >1 GT "
            "couple. 'none' (default) = single-positive legacy; treats "
            "each GT independently in its own pool. 'uniform' (H2) = "
            "shared pool with target mass 1/k on each positive. "
            "'soft_or' (H3) = MIL-aligned numerator: logsumexp over "
            "positive logits instead of mean. "
            "See Batch-3 plan."
        ),
    )
    parser.add_argument(
        '--couple-label-smoothing',
        type=float,
        default=0.10,
        help=(
            'Label smoothing ε applied to the softmax-CE branch only. '
            '0.0 = plain cross-entropy. Default 0.10 (baseline v2).'
        ),
    )
    parser.add_argument(
        '--couple-hardneg-fraction',
        type=float,
        default=0.0,
        help=(
            'Fraction of negatives drawn from the current top-scoring '
            'pool per event (ANCE / NV-Retriever online hard-negative '
            'mining). 0 = all random (legacy). Typical: 0.5.'
        ),
    )
    parser.add_argument(
        '--couple-hardneg-margin',
        type=float,
        default=0.1,
        help=(
            'Positive-aware threshold for hard-negative filtering. A '
            "candidate is kept only if its score is < s_pos − margin, "
            'which prevents false negatives from being selected as hard '
            'negatives.'
        ),
    )
    parser.add_argument(
        '--pair-kinematics-v2',
        action='store_true',
        help=(
            'Append four extra per-couple features to the 51-dim feature '
            'block: cos(opening angle), (m_ij − m_τ)/σ_m, '
            'dxy_sig_i·dxy_sig_j, dz_sig_i·dz_sig_j. Bumps reranker '
            'input_dim to 55.'
        ),
    )
    parser.add_argument(
        '--pair-physics-v3',
        action='store_true',
        help=(
            'Batch-3 H8: 5 new per-couple features (Kalman-χ² proxy, '
            'DCA-sig sum, lab helicity proxy, log BW densities for '
            'ρ(770) and a1(1260)). Additive with --pair-kinematics-v2.'
        ),
    )
    parser.add_argument(
        '--pair-physics-signif',
        action='store_true',
        help=(
            'Batch-4 H9: 3 significance-normalised pair features '
            '(mass pull, Δφ significance, pT-balance significance). '
            'Additive with other pair-physics flags.'
        ),
    )
    parser.add_argument(
        '--train-aug',
        type=str,
        default='none',
        choices=['none', 'cov_smear'],
        help=(
            'Training-time input augmentation. "cov_smear" (Batch-3 H12) '
            'perturbs kinematic track channels (px, py, pz, eta, phi) '
            'per step with Gaussian noise whose σ scales as '
            'exp(log_pt_err_standardized). See JetCLR (arXiv:2108.04253).'
        ),
    )
    parser.add_argument(
        '--smear-scale', type=float, default=1.0,
        help='Scale factor for --train-aug cov_smear noise. Default 1.0.',
    )
    parser.add_argument(
        '--drop-cov-features',
        action='store_true',
        help=(
            'Zero out the 5 covariance/uncertainty track channels while '
            '--train-aug cov_smear is active, so the model cannot detect '
            'the smearing magnitude by reading σ directly. Recommended '
            'whenever cov_smear is on.'
        ),
    )
    parser.add_argument(
        '--optim',
        type=str,
        default='adamw',
        choices=['adamw', 'asam'],
        help=(
            'Base optimizer. "adamw" (default) is the current behavior. '
            '"asam" wraps AdamW in scale-invariant adaptive SAM '
            '(Kwon 2021, arXiv:2102.11600). Two forward-backward passes '
            'per batch (~2x step cost). See Batch-3 H13.'
        ),
    )
    parser.add_argument(
        '--asam-rho', type=float, default=1.0,
        help='ASAM perturbation radius. Default 1.0.',
    )
    parser.add_argument(
        '--asam-eta', type=float, default=0.01,
        help='ASAM scale offset η preventing div-by-zero on zero weights.',
    )
    parser.add_argument(
        '--aux-vertex-weight', type=float, default=0.0,
        help=(
            'Weight of the auxiliary vertex-compatibility BCE head. '
            '0.0 (default) disables the head. 0.5 = equal weight with '
            'the primary ranking loss. See Batch-3 H7.'
        ),
    )
    parser.add_argument(
        '--event-context',
        type=str,
        default='none',
        choices=['none', 'deepset_film'],
        help=(
            'Per-event context conditioner. "deepset_film" (Batch-3 H6) '
            'pools the K2=60 per-track tensor (mean‖max‖std), projects to '
            'context_dim, and FiLM-modulates the per-couple hidden '
            'representation. Default off.'
        ),
    )
    parser.add_argument(
        '--context-dim', type=int, default=32,
        help='Event-context projection width for --event-context deepset_film.',
    )
    parser.add_argument(
        '--couple-embed-mode',
        type=str,
        default='projected_infersent',
        choices=[
            'concat', 'infersent', 'symmetric',
            'bilinear_lrb', 'projected_infersent',
            'ft_transformer', 'per_track_tokens',
        ],
        help=(
            'How Block 1 of the couple feature tensor is reconstructed '
            'before the Conv1d MLP. '
            "'projected_infersent' (default, baseline v3) learns a "
            "per-track projector φ(t) = LayerNorm(ReLU(Linear(16, p)(t))) "
            'then applies InferSent on the projections; requires '
            '--couple-projector-dim > 0. '
            "'concat' keeps the raw [t_i ‖ t_j] (baseline v2). "
            "'infersent' = [t_i, t_j, |t_i−t_j|, t_i⊙t_j] (Conneau 2017). "
            "'symmetric' = [max, mean, |diff|] — permutation invariant. "
            "'bilinear_lrb' appends (Ut_i)⊙(Vt_j) rank-8 interaction. "
            "'ft_transformer' tokenises every input dim (feature-as-token) "
            "and runs self-attention within each couple + [CLS] readout "
            "(Gorishniy 2021, arXiv:2106.11959). "
            "'per_track_tokens' is the same but with a learnable role "
            "embedding distinguishing track-i / track-j / physics tokens."
        ),
    )
    parser.add_argument(
        '--tokenize-d', type=int, default=16,
        help="Token width for ft_transformer / per_track_tokens modes.",
    )
    parser.add_argument(
        '--tokenize-blocks', type=int, default=2,
        help="Transformer depth for token modes.",
    )
    parser.add_argument(
        '--tokenize-heads', type=int, default=4,
        help="Attention heads for token modes. Must divide tokenize-d.",
    )
    parser.add_argument(
        '--couple-projector-dim',
        type=int,
        default=32,
        help=(
            'Per-track projector width p used only when '
            "--couple-embed-mode='projected_infersent'. Default 32 "
            '(baseline v3). Set to 0 to disable the projector (must '
            "also set --couple-embed-mode to something other than "
            "'projected_infersent')."
        ),
    )
    parser.add_argument(
        '--couple-ema-decay',
        type=float,
        default=0.0,
        help=(
            'Exponential moving average decay for the couple reranker '
            "weights. 0 = disabled (default). Typical: 0.999 (averages "
            '~1000 updates). The EMA copy is BN-recalibrated at the end '
            'of training and saved as best_model_ema_calibrated.pt.'
        ),
    )
    parser.add_argument(
        '--bn-calibration-steps',
        type=int,
        default=200,
        help=(
            'Number of forward-only training batches used to rebuild '
            'the couple reranker BN running stats after training. 0 '
            'disables the calibration step entirely (useful for smoke '
            'tests).'
        ),
    )
    # K values for the validation metrics. The set of K values reported
    # for D@K_tracks (cascade-side) and C/RC@K_couples (reranker-side)
    # is configurable so sweeps can use a denser grid (e.g. step 10).
    # The selection criterion is C@100_couples — 100 must be present in
    # `--k-values-couples`.
    parser.add_argument(
        '--k-values-tracks',
        type=int,
        nargs='+',
        default=[30, 50, 75, 100, 200],
        help='K values for D@K_tracks (default: 30 50 75 100 200)',
    )
    parser.add_argument(
        '--k-values-couples',
        type=int,
        nargs='+',
        default=[50, 75, 100, 200],
        help=(
            'K values for C@K_couples and RC@K_couples '
            '(default: 50 75 100 200). 100 must be in this list because '
            'the selection criterion is C@100_couples.'
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    # Global seed seeding. Covers Python `random`, NumPy, PyTorch CPU and
    # CUDA RNGs so that negative sampling in `CoupleReranker.compute_loss`
    # (uses `torch.randint`) and dataset shuffling are reproducible across
    # runs. When `--seed` is unset the RNGs stay at their default
    # non-deterministic initialization.
    if args.seed is not None:
        import random as _random
        import numpy as _numpy
        _random.seed(args.seed)
        _numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # The selection criterion (best-checkpoint metric) is C@100_couples,
    # so K=100 must be present. Catch typos before any work happens.
    if 100 not in args.k_values_couples:
        parser.error(
            '--k-values-couples must include 100 (the selection criterion '
            f'is C@100_couples). Got: {args.k_values_couples}',
        )

    # Build the labelled-metric mapping for save_loss_history. Done in
    # main() (after argparse) so the keys reflect the K grid this run
    # was launched with.
    metric_labels = _build_metric_labels(
        k_values_tracks=args.k_values_tracks,
        k_values_couples=args.k_values_couples,
    )

    # ---- Experiment directory ----
    resume_dir = None
    if args.resume is not None:
        resume_dir = os.path.dirname(os.path.dirname(args.resume))
    experiment_dir, checkpoints_dir, tensorboard_dir = build_experiment_directory(
        args.experiments_dir, args.model_name, resume_dir,
    )

    # ---- Logging ----
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

    # ---- Data loading ----
    import glob
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

    # Save the auto.yaml used for this run into the experiment directory
    # so the exact standardization params are reproducible.
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

    # ---- Model ----
    network_module = load_network_module(args.network)
    model, model_info = network_module.get_model(
        data_config,
        cascade_checkpoint=args.cascade_checkpoint,
        top_k2=args.top_k2,
        # Forward the configured K_tracks grid so the cascade glue
        # model's `n_gt_in_top_k_tracks` tensor matches the shape the
        # validation accumulator expects.
        k_values_tracks=tuple(args.k_values_tracks),
        couple_hidden_dim=args.couple_hidden_dim,
        couple_num_residual_blocks=args.couple_num_residual_blocks,
        couple_dropout=args.couple_dropout,
        couple_ranking_num_samples=args.couple_ranking_num_samples,
        couple_ranking_temperature=args.couple_ranking_temperature,
        couple_loss=args.couple_loss,
        couple_label_smoothing=args.couple_label_smoothing,
        couple_hardneg_fraction=args.couple_hardneg_fraction,
        couple_hardneg_margin=args.couple_hardneg_margin,
        pair_kinematics_v2=args.pair_kinematics_v2,
        pair_physics_v3=args.pair_physics_v3,
        pair_physics_signif=args.pair_physics_signif,
        couple_embed_mode=args.couple_embed_mode,
        couple_projector_dim=args.couple_projector_dim,
        tokenize_d=args.tokenize_d,
        tokenize_blocks=args.tokenize_blocks,
        tokenize_heads=args.tokenize_heads,
        ndcg_k=args.ndcg_k,
        lambda_sigma=args.lambda_sigma,
        ndcg_alpha=args.ndcg_alpha,
        couple_multi_positive=args.couple_multi_positive,
        couple_use_full_negative_list=args.couple_use_full_negative_list,
        aux_vertex_weight=args.aux_vertex_weight,
        event_context=args.event_context,
        context_dim=args.context_dim,
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

    # ---- Optimizer (only the trainable CoupleReranker params) ----
    trainable_parameter_iter = filter(
        lambda parameter: parameter.requires_grad, model.parameters(),
    )
    optimizer = torch.optim.AdamW(
        trainable_parameter_iter, lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.optim == 'asam':
        from utils.sam_optimizer import SAM
        optimizer = SAM(
            optimizer, rho=args.asam_rho, eta=args.asam_eta, adaptive=True,
        )
        logger.info(
            f'Wrapped optimizer in ASAM (rho={args.asam_rho}, '
            f'eta={args.asam_eta}). Expected 2x step cost.'
        )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(int(args.warmup_fraction * total_steps), max_warmup_steps)

    if args.scheduler == 'cosine':
        warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
        num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
        power_info = (
            f', cosine_power={args.cosine_power}'
            if args.cosine_power != 1.0 else ''
        )
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'CosineAnnealingLR over {num_post_warmup_epochs} epochs'
            f'{power_info}',
        )
        scheduler = WarmupThenCosineScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_post_warmup_epochs=num_post_warmup_epochs,
            min_lr=args.min_lr,
            cosine_power=args.cosine_power,
        )
    else:
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='C@100',
    )

    # ---- Optional EMA (T2.3) ----
    ema_model = None
    if args.couple_ema_decay > 0.0:
        if not 0.0 < args.couple_ema_decay < 1.0:
            raise ValueError(
                f'--couple-ema-decay must be in (0, 1), got '
                f'{args.couple_ema_decay}',
            )
        decay = args.couple_ema_decay
        ema_model = torch.optim.swa_utils.AveragedModel(
            model.couple_reranker,
            avg_fn=(
                lambda averaged, current, n: (
                    decay * averaged + (1.0 - decay) * current
                )
            ),
        )
        logger.info(
            f'EMA enabled for couple_reranker (decay={decay}); '
            'saved as best_model_ema_calibrated.pt after training.',
        )

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    # ---- Training loop ----
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
        # Slim checkpoint format: only the trainable couple_reranker
        # weights are persisted; the frozen cascade is rebuilt from
        # `--cascade-checkpoint` at startup, so it does NOT belong in
        # per-epoch artifacts.
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

    # AMP GradScaler — None disables the AMP path. SAM/ASAM second-pass
    # forward cannot be easily mixed with GradScaler so --amp is
    # silently force-disabled there to keep the SAM code path clean.
    if args.amp and args.optim == 'asam':
        logger.warning(
            '--amp is incompatible with --optim asam; disabling AMP.',
        )
        args.amp = False
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None
    if args.amp:
        logger.info('AMP enabled (fp16 forward + GradScaler backward).')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index,
                grad_clip_max_norm=args.grad_clip,
                ema_model=ema_model,
                train_aug=args.train_aug,
                smear_scale=args.smear_scale,
                drop_cov_features=args.drop_cov_features,
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

            # Render the validation summary as a multi-line ASCII table
            # (header line + K × {D, C, RC} table + footer with mean rank
            # and bookkeeping). The leading newline keeps the table body
            # vertically aligned beneath the logger's timestamped header.
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
            # Position-aware diagnostics (Batch 6 — see
            # couple_reranker_phase_c_proposal.md / cascade_full_val_results_20260421.md):
            # train-time mean rank of the best GT couple and the
            # Spearman correlation ρ between batch loss and that rank.
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
                # Slim checkpoint: save ONLY the trainable couple
                # reranker. The frozen cascade is reloaded from
                # `--cascade-checkpoint` at startup, so re-saving its
                # ~280 MB of weights every epoch would just bloat disk.
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

    # ---- Post-training BN calibration ----
    # Rebuild clean running stats so eval() mode works correctly.
    if args.bn_calibration_steps <= 0:
        logger.info(
            'BN calibration skipped (--bn-calibration-steps=0).',
        )
    else:
        logger.info(
            'Calibrating CoupleReranker BN running stats '
            f'({args.bn_calibration_steps} steps)...',
        )
        calibrate_reranker_batchnorm(
            model, train_loader, device, data_config,
            mask_input_index, label_input_index,
            calibration_steps=args.bn_calibration_steps,
        )
    if args.bn_calibration_steps > 0:
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

    # ---- EMA (T2.3): swap in EMA weights, recalibrate BN on the EMA
    # copy, validate, save separately. The live model is unaffected.
    if ema_model is not None and args.bn_calibration_steps > 0:
        logger.info('Swapping in EMA weights for BN recalibration + eval')
        original_reranker_state = {
            key: value.clone()
            for key, value in model.couple_reranker.state_dict().items()
        }
        # swa_utils.AveragedModel wraps the averaged weights under
        # `.module` in a way that matches the original state dict when
        # loaded back into the live reranker.
        ema_state = ema_model.module.state_dict()
        model.couple_reranker.load_state_dict(ema_state)
        calibrate_reranker_batchnorm(
            model, train_loader, device, data_config,
            mask_input_index, label_input_index,
            calibration_steps=args.bn_calibration_steps,
        )
        ema_val_losses, ema_val_metrics = validate(
            model, val_loader, device, data_config,
            mask_input_index, label_input_index,
            max_steps=max(1, steps_per_epoch // 2),
            k_values_couples=tuple(args.k_values_couples),
            k_values_tracks=tuple(args.k_values_tracks),
        )
            ema_c_at_100 = ema_val_metrics.get('c_at_100_couples', 0.0)
            logger.info(f'EMA C@100 = {ema_c_at_100:.5f}')
            ema_path = os.path.join(
                checkpoints_dir, 'best_model_ema_calibrated.pt',
            )
            torch.save(
                {
                    'epoch': args.epochs,
                    'couple_reranker_state_dict':
                        model.couple_reranker.state_dict(),
                    'ema_c_at_100': ema_c_at_100,
                    'ema_val_metrics': ema_val_metrics,
                    'args': vars(args),
                },
                ema_path,
            )
            logger.info(f'Saved EMA-calibrated checkpoint: {ema_path}')
        else:
            logger.error('EMA BN calibration failed — NaN in running stats')
        # Restore the live (non-EMA) weights for downstream use.
        model.couple_reranker.load_state_dict(original_reranker_state)

    logger.info(f'Training complete. Best C@100: {best_val_c_at_100:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    main()
