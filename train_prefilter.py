"""Training script for TrackPreFilter (Stage 1 of two-stage pipeline).

Same structure as train_trackfinder.py. Uses weaver's dataset infrastructure
for YAML parsing and parquet loading.

Key differences from train_trackfinder.py:
    - Model uses compute_loss() instead of forward() with track_labels kwarg.
    - Model forward() returns per-track scores, not per_track_logits dict.
    - Evaluation uses the same recall@K, d-prime, and median rank metrics.

Usage:
    python train_prefilter.py \\
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \\
        --data-dir data/low-pt/ \\
        --network networks/lowpt_tau_TrackPreFilter.py \\
        --epochs 50 \\
        --batch-size 96 \\
        --lr 1e-3 \\
        --device cuda:0 \\
        --amp
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

import torch

torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

from pretrain_backbone import (
    WarmupThenCosineScheduler,
    WarmupThenPlateauScheduler,
    _TeeStream,
    build_experiment_directory,
    plot_loss_curves,
    save_loss_history,
)
from utils.optimizers import OPTIMIZER_NAMES, build_optimizer
from utils.training_utils import (
    CheckpointManager,
    MetricsAccumulator,
    extract_label_from_inputs,
    load_network_module,
    save_epoch_metrics,
    trim_to_max_valid_tracks,
)

logger = logging.getLogger('train_prefilter')


# ---------------------------------------------------------------------------
# Metric labels for the on-disk loss_history.json
# ---------------------------------------------------------------------------
#
# Maps loss-history dict keys to short human-readable descriptions. The
# saver wraps each metric as ``{'label': str, 'values': list[float]}`` so
# the JSON file is self-documenting.

METRIC_LABELS: dict[str, str] = {
    'train': 'Train loss (per-track ranking, mean per epoch)',
    'val': 'Validation loss (per-track ranking)',
    'lr': 'Learning rate',
    'd_prime': "Cohen's d' between GT and background score distributions (val)",
    'median_gt_rank': 'Median rank of GT pions in the per-event score order (val)',
}
# K values reported by the val MetricsAccumulator at each epoch. 256 is
# the operating-point K of the cascade (Stage 1 → Stage 2 top-K1), so
# per-event ``perfect_at_256`` is a first-class checkpoint-selection
# criterion alongside the legacy ``recall_at_200``.
VAL_METRICS_K_VALUES: tuple[int, ...] = (
    10, 20, 30, 50, 100, 200, 256, 300, 400, 500, 600, 800,
)
# Criteria that ``--checkpoint-criterion`` accepts. Keep synced with
# VAL_METRICS_K_VALUES — any ``perfect_at_N`` or ``recall_at_N`` with
# ``N`` in the tuple is computed and thus selectable.
CHECKPOINT_CRITERIA: tuple[str, ...] = (
    'recall_at_200',
    'recall_at_256',
    'perfect_at_200',
    'perfect_at_256',
)
for _k in VAL_METRICS_K_VALUES:
    METRIC_LABELS[f'recall_at_{_k}'] = (
        f'R@{_k}: per-event recall at top-{_k} tracks '
        f'(fraction of GT pions in the model top-{_k}, val-averaged)'
    )
    METRIC_LABELS[f'perfect_at_{_k}'] = (
        f'P@{_k}: per-event perfect recall at top-{_k} tracks '
        f'(fraction of events with all 3 GT pions in top-{_k}, val-averaged)'
    )
del _k


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    grad_scaler: torch.amp.GradScaler | None,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: SummaryWriter | None,
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    label_input_index: int,
    grad_clip_max_norm: float = 1.0,
    augmentation: torch.nn.Module | None = None,
    ema_teacher: torch.nn.Module | None = None,
    ema_decay: float = 0.999,
    kl_weight: float = 0.1,
) -> tuple[dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0
    start_time = time.time()

    for batch_index, (X, y, _) in enumerate(train_loader):
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

        # Optional set-friendly augmentation (JetCLR-style).
        if augmentation is not None:
            points, features, lorentz_vectors, mask = augmentation(
                points, features, lorentz_vectors, mask, track_labels,
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            # Contrastive denoising re-enabled (2026-04-07) as a regularizer
            # for the dim256+cutoff overfit. The DRW/temperature-annealing
            # ablation ruled out exotic loss enhancements as the *trigger*
            # (DRW's epoch-31 activation was the loss discontinuity), but
            # denoising itself is a GT-invariance regularizer and helps with
            # the residual ~2pp train/val gap. DRW and temperature annealing
            # stay disabled in the wrapper — see reports/prefilter_analysis_20260406.md.
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
            )
            student_scores = loss_dict.get('_scores')

            # Self-distillation EMA teacher (E12). Teacher runs in eval
            # mode and no-grad. KL between student and teacher logits
            # (softened with temperature 1) is added with kl_weight.
            if ema_teacher is not None and student_scores is not None:
                with torch.no_grad():
                    teacher_scores = ema_teacher(
                        points, features, lorentz_vectors, mask,
                    )
                valid_mask = mask.squeeze(1).bool()
                student_logits = student_scores.masked_fill(
                    ~valid_mask, float('-inf'),
                )
                teacher_logits = teacher_scores.masked_fill(
                    ~valid_mask, float('-inf'),
                )
                student_log_prob = torch.nn.functional.log_softmax(
                    student_logits, dim=-1,
                )
                teacher_prob = torch.nn.functional.softmax(
                    teacher_logits, dim=-1,
                )
                # Per-event KL, averaged over events with at least 1 positive.
                kl_per_event = torch.nn.functional.kl_div(
                    student_log_prob, teacher_prob,
                    reduction='none',
                )
                kl_per_event = kl_per_event.sum(dim=-1)
                kl_loss = kl_per_event.mean()
                loss_dict['kl_loss'] = kl_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + (
                    kl_weight * kl_loss
                )

            # Remove cached scores (non-scalar) before loss accumulation
            loss_dict.pop('_scores', None)
            loss = loss_dict['total_loss']

        # Single GPU→CPU sync instead of two (isnan + isinf)
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

        # EMA teacher update (E12). θ_teacher ← decay · θ_teacher + (1-decay) · θ_student.
        if ema_teacher is not None:
            with torch.no_grad():
                student_params = dict(model.named_parameters())
                for name, teacher_param in ema_teacher.named_parameters():
                    student_param = student_params.get(name)
                    if student_param is None:
                        continue
                    teacher_param.data.mul_(ema_decay).add_(
                        student_param.data, alpha=1.0 - ema_decay,
                    )
                # Buffers (BN running stats) — copy straight through.
                student_buffers = dict(model.named_buffers())
                for name, teacher_buffer in ema_teacher.named_buffers():
                    student_buffer = student_buffers.get(name)
                    if student_buffer is None:
                        continue
                    teacher_buffer.data.copy_(student_buffer.data)

        scheduler.step_batch()

        # Accumulate losses on GPU to avoid per-batch GPU→CPU sync.
        # Only transfer to CPU at logging intervals.
        if loss_accumulators is None:
            loss_accumulators = {
                key: torch.zeros(1, device=loss.device)
                for key in loss_dict
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

        del inputs, model_inputs, track_labels, loss_dict

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


def run_profile(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    num_steps: int,
    output_dir: str,
    grad_scaler: torch.amp.GradScaler | None = None,
    warmup_steps: int = 3,
    wait_steps: int = 2,
    record_shapes: bool = False,
    profile_memory: bool = False,
    export_chrome_trace: bool = False,
) -> None:
    """Run N fwd+bwd batches under torch.profiler, emit summary + optional trace.

    Used to diagnose per-op CPU/CUDA time. Schedule:
    ``wait`` skips initial steps (lets data loader warm up), ``warmup``
    runs without recording (primes compilation/caches), ``active`` records
    ``num_steps`` steps into the trace.

    Output size guidance (200 active steps, P~1100 tracks, BS=256):
      - default (shapes+memory off, no trace) → KB summary only
      - record_shapes=True                   → ~200 MB trace if exported
      - profile_memory=True                  → ~2-4x size over record_shapes
      - both + chrome trace                  → multi-GB (fills disk quickly)

    Args:
        num_steps: Active recording steps — batches whose ops appear in trace.
        output_dir: Directory for ``profile_summary.txt`` + optional
            ``profile_trace.json``.
        grad_scaler: Optional AMP grad scaler — mirrors ``train_one_epoch``.
        record_shapes: Record per-op input shapes. Inflates trace size.
        profile_memory: Record per-op allocations. Inflates further.
        export_chrome_trace: Dump the full chrome trace JSON. Off by
            default because the serialized JSON scales with step × op
            count × shape/memory metadata.
    """
    from torch.profiler import ProfilerActivity, profile, schedule

    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, 'profile_trace.json')
    summary_path = os.path.join(output_dir, 'profile_summary.txt')

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    profiler_schedule = schedule(
        wait=wait_steps, warmup=warmup_steps, active=num_steps, repeat=1,
    )
    total_batches_needed = wait_steps + warmup_steps + num_steps

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-8,
    )

    model.train()

    logger.info(
        f'Profiling: wait={wait_steps} warmup={warmup_steps} '
        f'active={num_steps} → total {total_batches_needed} batches | '
        f'shapes={record_shapes} memory={profile_memory} '
        f'trace={export_chrome_trace}',
    )

    with profile(
        activities=activities,
        schedule=profiler_schedule,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,
    ) as profiler:
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            if batch_count >= total_batches_needed:
                break

            X = batch[0]
            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device.type, enabled=grad_scaler is not None,
            ):
                loss_dict = model.compute_loss(
                    points, features, lorentz_vectors, mask, track_labels,
                )
                loss_dict.pop('_scores', None)
                loss = loss_dict['total_loss']

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            profiler.step()
            batch_count += 1

    if export_chrome_trace:
        profiler.export_chrome_trace(trace_path)
        logger.info(f'Chrome trace written: {trace_path}')
    summary = profiler.key_averages().table(
        sort_by=(
            'cuda_time_total' if device.type == 'cuda' else 'cpu_time_total'
        ),
        row_limit=40,
    )
    with open(summary_path, 'w') as summary_file:
        summary_file.write(summary)

    logger.info(f'Summary written:      {summary_path}')
    print(summary)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    label_input_index: int,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate and compute recall@K metrics with global accumulation.

    Uses MetricsAccumulator to collect raw GT ranks and per-event data
    across all batches, then computes percentiles and breakdowns from
    the full distribution (not batch-averaged).
    """
    model.eval()
    loss_accumulators: dict[str, float] | None = None
    num_batches = 0

    metrics_accumulator = MetricsAccumulator(
        k_values=VAL_METRICS_K_VALUES,
    )

    with torch.no_grad():
        for batch_index, (X, y, _) in enumerate(val_loader):
            if max_steps is not None and batch_index >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_input_index,
            )
            points, features, lorentz_vectors, mask = model_inputs

            # Get loss and scores in a single forward pass.
            # compute_loss() calls forward() internally and caches scores
            # in loss_dict['_scores'] — avoids running the model twice.
            # Denoising force-disabled in the validation path so the val
            # loss stays directly comparable to train's ranking component
            # (denoising is a train-only regularizer, not a metric).
            # model.train() is required for BN batch-stats — the running
            # stats stored in the checkpoint are stale (see reports/
            # experiment_log.md "BatchNorm Fix"). Dropout also activates
            # here in train() mode, but the BN workaround is more load-
            # bearing than dropout determinism during a single val pass.
            model.train()
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask, track_labels,
                use_contrastive_denoising=False,
            )
            model.eval()

            per_track_scores = loss_dict.pop('_scores').detach()

            if loss_accumulators is None:
                loss_accumulators = {key: 0.0 for key in loss_dict}
            for key in loss_accumulators:
                loss_accumulators[key] += loss_dict[key].item()

            # Accumulate raw data for global metric computation
            metrics_accumulator.update(per_track_scores, track_labels, mask)

            num_batches += 1

            del inputs, model_inputs, track_labels, loss_dict
            del per_track_scores

    if loss_accumulators is None:
        loss_accumulators = {'total_loss': 0.0}
    loss_averages = {
        key: value / max(1, num_batches)
        for key, value in loss_accumulators.items()
    }

    # Compute global metrics from accumulated raw data
    metrics = metrics_accumulator.compute()

    return loss_averages, metrics


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Split out from ``main`` so tests can inspect flag defaults and parsing
    without executing training side effects.
    """
    parser = argparse.ArgumentParser(
        description='Train TrackPreFilter (Stage 1)',
    )
    parser.add_argument('--data-config', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='PreFilter')
    parser.add_argument('--experiments-dir', type=str, default='experiments')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'])
    parser.add_argument('--warmup-fraction', type=float, default=0.05)
    parser.add_argument('--plateau-factor', type=float, default=0.5)
    parser.add_argument('--plateau-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='Dropout rate in TrackPreFilter MLP hidden layers (default: 0.1). '
             'Applied after each ReLU in the mlp-mode track_mlp, '
             'neighbor_mlps, and scorer. Set to 0 to disable.',
    )
    # --- Architecture knobs (prefilter improvement campaign) ---
    parser.add_argument(
        '--num-neighbors', type=int, default=16,
        help='k for k-NN neighbor aggregation (default: 16).',
    )
    parser.add_argument(
        '--num-message-rounds', type=int, default=3,
        help='Number of k-NN message-passing rounds (default: 3, '
             'E2a campaign winner). Set to 0 for the '
             'aggregation-ablation experiment.',
    )
    parser.add_argument(
        '--aggregation-mode', type=str, default='max',
        choices=['max', 'pna'],
        help='Neighbor aggregation: max-pool (default) or PNA '
             '(cat of mean, max, min, std).',
    )
    parser.add_argument(
        '--use-edge-features', action=argparse.BooleanOptionalAction,
        default=True,
        help='Append pairwise_lv_fts (ln kT, ln z, ln ΔR, ln m²) '
             'max-pooled over the k-NN to the aggregation input '
             '(+4 channels). Default ON (E2a campaign winner); '
             'pass --no-use-edge-features to disable.',
    )
    # --- Loss switch (prefilter improvement campaign) ---
    parser.add_argument(
        '--loss-type', type=str, default='pairwise',
        choices=[
            'pairwise', 'listwise_ce', 'infonce',
            'logit_adjust', 'object_condensation', 'mpm_pretrain',
        ],
        help='Per-event supervision loss. Default pairwise matches '
             'the historical TrackPreFilter ranking objective.',
    )
    parser.add_argument(
        '--logit-adjust-tau', type=float, default=1.0,
        help='τ for Menon 2007.07314 logit adjustment. Only used when '
             '--loss-type=logit_adjust.',
    )
    parser.add_argument(
        '--listwise-temperature', type=float, default=1.0,
        help='Temperature for listwise_ce / infonce loss.',
    )
    # --- Regularisation / augmentation / SSL ---
    parser.add_argument(
        '--use-augmentation', action='store_true',
        help='Apply set-friendly train-time augmentations '
             '(track dropout, feature jitter, η-φ rotation).',
    )
    parser.add_argument(
        '--ssl-pretrain-ckpt', type=str, default=None,
        help='Load backbone weights (track_mlp, neighbor_mlps) from a '
             'masked-particle-modeling SSL pretrain checkpoint before '
             'supervised training starts.',
    )
    parser.add_argument(
        '--mpm-pretrain-epochs', type=int, default=0,
        help='If > 0, run that many epochs of masked-particle-modeling '
             'pretraining before the supervised phase. Uses the model'
             "'s track_mlp + neighbor_mlps backbone and a reconstruction "
             'head to predict the input features of randomly masked tracks.',
    )
    parser.add_argument(
        '--mpm-mask-ratio', type=float, default=0.15,
        help='Fraction of valid tracks randomly masked during MPM pretrain.',
    )
    # --- Self-distillation EMA teacher (E12) ---
    parser.add_argument(
        '--use-self-distillation', action='store_true',
        help='Enable self-distillation: maintain an EMA teacher copy of '
             'the student, add a KL-divergence loss between their logits.',
    )
    parser.add_argument(
        '--ema-decay', type=float, default=0.999,
        help='EMA decay for the teacher. Higher = slower teacher.',
    )
    parser.add_argument(
        '--kl-weight', type=float, default=0.1,
        help='Weight of the KL-divergence auxiliary loss.',
    )
    # --- Object condensation head (E5) ---
    parser.add_argument(
        '--clustering-dim', type=int, default=8,
        help='Embedding dim for object-condensation loss (E5). '
             'Unused unless --loss-type object_condensation.',
    )
    parser.add_argument(
        '--oc-potential-weight', type=float, default=1.0,
        help='Weight of the OC attractive/repulsive potential term.',
    )
    parser.add_argument(
        '--oc-beta-weight', type=float, default=1.0,
        help='Weight of the OC β-regulariser term.',
    )
    parser.add_argument(
        '--oc-q-min', type=float, default=0.1,
        help='Minimum charge floor for object-condensation q values.',
    )
    # --- XGBoost stub feature (E7) ---
    parser.add_argument(
        '--use-xgb-stub-feature', action='store_true',
        help='Prepend a frozen linear per-track score (16→1) as a 17th '
             'feature channel. Stub placeholder for the real XGBoost '
             'score cache — once the cache exists, replace the stub '
             'with the pre-computed scores. The flag exists so the '
             'input-dim path is exercised.',
    )
    # --- Expressiveness plug-in heads (prefilter P@256 sweep) ---
    parser.add_argument(
        '--feature-embed-mode', type=str, default='per_feature',
        choices=('none', 'per_feature'),
        help='P1 (now baseline). "per_feature" routes each raw input '
             'channel through its own grouped 1×1 Conv + LayerNorm + '
             'ReLU before track_mlp, producing (16 * feature_embed_dim) '
             'channels. Pass "none" to reproduce the pre-P1 E2a anchor.',
    )
    parser.add_argument(
        '--feature-embed-dim', type=int, default=32,
        help='P1 per-feature embedding width (only used when '
             '--feature-embed-mode=per_feature).',
    )
    parser.add_argument(
        '--feature-gate', action='store_true',
        help='P2. Apply an SE-style squeeze-excite gate to the track_mlp '
             'output (per-event, per-channel).',
    )
    parser.add_argument(
        '--feature-gate-bottleneck', type=int, default=16,
        help='P2 SE bottleneck width (only used with --feature-gate).',
    )
    parser.add_argument(
        '--film-head', action='store_true',
        help='P3. Modulate the track_mlp output with FiLM (γ, β) derived '
             'from event-level (mean, std) of the standardised features.',
    )
    parser.add_argument(
        '--film-context-dim', type=int, default=32,
        help='P3 FiLM context hidden width.',
    )
    parser.add_argument(
        '--soft-attention-aggregation', action='store_true',
        help='P4. Replace max-pool in each message-passing round with a '
             'learned soft-attention aggregator over the k-NN '
             'neighbours. Edge features flow into the attention score.',
    )
    parser.add_argument(
        '--soft-attention-bottleneck', type=int, default=64,
        help='P4 score-MLP bottleneck width.',
    )
    # --- Two-tier prefilter (P6) — only read by
    #     networks/lowpt_tau_TwoTierPreFilter.py wrapper.
    parser.add_argument(
        '--two-tier-top-n', type=int, default=600,
        help='P6 top-N cut between coarse and refine tiers.',
    )
    parser.add_argument(
        '--two-tier-coarse-hidden-dim', type=int, default=128,
        help='P6 coarse tier hidden dim.',
    )
    parser.add_argument(
        '--two-tier-refine-hidden-dim', type=int, default=384,
        help='P6 refine tier hidden dim.',
    )
    parser.add_argument(
        '--two-tier-coarse-neighbors', type=int, default=16,
        help='P6 coarse tier kNN k.',
    )
    parser.add_argument(
        '--two-tier-refine-neighbors', type=int, default=32,
        help='P6 refine tier kNN k (must be < --two-tier-top-n).',
    )
    parser.add_argument(
        '--two-tier-coarse-rounds', type=int, default=2,
        help='P6 coarse tier message-passing rounds.',
    )
    parser.add_argument(
        '--two-tier-refine-rounds', type=int, default=3,
        help='P6 refine tier message-passing rounds.',
    )
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data-dir for training (ignored if --val-data-dir set)')
    parser.add_argument('--val-data-dir', type=str, default=None,
                        help='Separate directory with validation parquet files. '
                             'When set, data-dir is used entirely for training '
                             'and val-data-dir entirely for validation.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--no-in-memory', action='store_true')
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--keep-best-k', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument(
        '--optimizer', type=str, default='adamw', choices=OPTIMIZER_NAMES,
        help='Optimizer to use. SOAP and Muon require --amp disabled.',
    )
    parser.add_argument(
        '--checkpoint-criterion', type=str,
        default='recall_at_200', choices=CHECKPOINT_CRITERIA,
        help=(
            'Validation metric used to pick the best checkpoint. '
            '"recall_at_200" preserves the 17-experiment campaign convention; '
            '"perfect_at_256" tracks the per-event perfect-recall target '
            'directly (the prefilter-expressiveness sweep primary metric). '
            f'Choices: {", ".join(CHECKPOINT_CRITERIA)}.'
        ),
    )
    parser.add_argument(
        '--profile-steps', type=int, default=0,
        help=(
            'If >0, run N training batches under torch.profiler and exit '
            'without training. Emits chrome trace + summary table.'
        ),
    )
    parser.add_argument(
        '--profile-output', type=str, default=None,
        help=(
            'Directory for profile artifacts. Defaults to experiment dir '
            'when --profile-steps is set.'
        ),
    )
    parser.add_argument(
        '--profile-record-shapes', action='store_true',
        help='Record per-op input shapes. Inflates trace size.',
    )
    parser.add_argument(
        '--profile-memory', action='store_true',
        help='Record per-op allocations. Further inflates trace size.',
    )
    parser.add_argument(
        '--profile-chrome-trace', action='store_true',
        help=(
            'Export full chrome trace JSON. Off by default — summary '
            'table is enough for op-level hotspots and keeps outputs '
            'in the KB range. Enabling with --profile-steps N results '
            'in trace files that scale roughly linearly with N.'
        ),
    )

    return parser


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

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

    # ---- Force-disable AMP for optimizers that don't support it ----
    # SOAP has no lower-precision support yet; Muon is incompatible with
    # torch.cuda.amp.GradScaler. Rather than erroring, we silently disable
    # --amp for these optimizers so the user can launch with the same
    # train_*.sh invocation that passes --amp unconditionally.
    if args.amp and args.optimizer in ('soap', 'muon'):
        logger.warning(
            f'--amp is incompatible with --optimizer {args.optimizer}; '
            f'force-disabling AMP for this run. SOAP/Muon do not support '
            f'mixed precision yet.'
        )
        args.amp = False

    # ---- Data loading ----
    import glob
    load_in_memory = not args.no_in_memory

    train_parquet_files = sorted(glob.glob(f'{args.data_dir}/*.parquet'))
    logger.info(f'Found {len(train_parquet_files)} train parquet files in {args.data_dir}')
    train_file_dict = {'data': train_parquet_files}
    num_train_files = len(train_parquet_files)

    if args.val_data_dir is not None:
        # Separate train/val directories — use each entirely
        val_parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
        logger.info(f'Found {len(val_parquet_files)} val parquet files in {args.val_data_dir}')
        val_file_dict = {'data': val_parquet_files}
        num_val_files = len(val_parquet_files)
        train_range = ((0.0, 1.0), 1.0)
        val_range = ((0.0, 1.0), 1.0)
    else:
        # Single directory — split by train_fraction
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
    val_num_workers = min(
        max(1, train_num_workers // 2), num_val_files,
    ) if train_num_workers > 0 else 0
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
        dropout=args.dropout,
        num_neighbors=args.num_neighbors,
        num_message_rounds=args.num_message_rounds,
        aggregation_mode=args.aggregation_mode,
        use_edge_features=args.use_edge_features,
        loss_type=args.loss_type,
        logit_adjust_tau=args.logit_adjust_tau,
        listwise_temperature=args.listwise_temperature,
        use_xgb_stub_feature=args.use_xgb_stub_feature,
        clustering_dim=args.clustering_dim,
        feature_embed_mode=args.feature_embed_mode,
        feature_embed_dim=args.feature_embed_dim,
        use_feature_gate=args.feature_gate,
        feature_gate_bottleneck=args.feature_gate_bottleneck,
        use_film_head=args.film_head,
        film_context_dim=args.film_context_dim,
        use_soft_attention_aggregation=args.soft_attention_aggregation,
        soft_attention_bottleneck=args.soft_attention_bottleneck,
        # Two-tier-only kwargs — ignored by the single-tier wrapper
        # (see `networks/lowpt_tau_TrackPreFilter.py`) and consumed by
        # `networks/lowpt_tau_TwoTierPreFilter.py`.
        two_tier_top_n=args.two_tier_top_n,
        two_tier_coarse_hidden_dim=args.two_tier_coarse_hidden_dim,
        two_tier_refine_hidden_dim=args.two_tier_refine_hidden_dim,
        two_tier_coarse_neighbors=args.two_tier_coarse_neighbors,
        two_tier_refine_neighbors=args.two_tier_refine_neighbors,
        two_tier_coarse_rounds=args.two_tier_coarse_rounds,
        two_tier_refine_rounds=args.two_tier_refine_rounds,
    )
    # Post-construction scalar attribute tweaks for the OC / MPM flags
    # that don't change module layout.
    if hasattr(model, 'module'):
        model_root = model.module
    else:
        model_root = model
    model_root.oc_q_min = args.oc_q_min
    model_root.oc_potential_weight = args.oc_potential_weight
    model_root.oc_beta_weight = args.oc_beta_weight
    model_root.mpm_mask_ratio = args.mpm_mask_ratio
    # MPM masking is ON whenever loss_type == 'mpm_pretrain' (combined
    # with ``self.training`` gate inside _forward_mlp).
    model_root.apply_mpm_masking = args.loss_type == 'mpm_pretrain'

    # Optional set-friendly training augmentation. Eval/val path is
    # untouched — augmentation only fires in train mode.
    if args.use_augmentation:
        from utils.set_augmentation import SetAugmentation
        augmentation = SetAugmentation().to(device)
    else:
        augmentation = None

    # EMA teacher for self-distillation (E12). Constructed as a second
    # network_module.get_model call using the same flags; state dict is
    # copied from the student, gradients frozen, set to eval mode.
    if args.use_self_distillation:
        ema_teacher, _ = network_module.get_model(
            data_config,
            dropout=args.dropout,
            num_neighbors=args.num_neighbors,
            num_message_rounds=args.num_message_rounds,
            aggregation_mode=args.aggregation_mode,
            use_edge_features=args.use_edge_features,
            loss_type=args.loss_type,
            logit_adjust_tau=args.logit_adjust_tau,
            listwise_temperature=args.listwise_temperature,
        )
        ema_teacher = ema_teacher.to(device)
        ema_teacher.load_state_dict(model_root.state_dict())
        for parameter in ema_teacher.parameters():
            parameter.requires_grad = False
        ema_teacher.eval()
    else:
        ema_teacher = None
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f'Total parameters: {total_params:,} | Trainable: {trainable_params:,}')

    input_names = list(data_config.input_names)
    mask_input_index = input_names.index('pf_mask')
    label_input_index = input_names.index('pf_label')
    logger.info(f'Mask input index: {mask_input_index} | Label input index: {label_input_index}')

    # ---- torch.compile ----
    original_model = model
    use_compile = (
        not args.no_compile
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        import logging as _logging
        _logging.getLogger('torch._inductor').setLevel(_logging.WARNING)
        _logging.getLogger('torch._dynamo').setLevel(_logging.WARNING)
        logger.info('Compiling model with torch.compile...')
        model = torch.compile(model, dynamic=True)
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')

    # ---- Optimizer ----
    logger.info(f'Building optimizer: {args.optimizer}')
    optimizer = build_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp_enabled=args.amp,
    )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(
        int(args.warmup_fraction * total_steps), max_warmup_steps,
    )

    if args.scheduler == 'cosine':
        warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
        num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'CosineAnnealingLR over {num_post_warmup_epochs} epochs',
        )
        scheduler = WarmupThenCosineScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_post_warmup_epochs=num_post_warmup_epochs,
            min_lr=args.min_lr,
        )
    else:
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps, then '
            f'ReduceLROnPlateau (factor={args.plateau_factor})',
        )
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    checkpoint_manager = CheckpointManager(
        checkpoints_directory=checkpoints_dir,
        keep_best_k=args.keep_best_k,
        criterion_mode='max',
        criterion_name='R@200',
    )

    # ---- TensorBoard ----
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)

    # ---- Training loop ----
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_score = 0.0
    checkpoint_criterion = args.checkpoint_criterion
    best_val_epoch = 0
    global_batch_count = 0
    loss_history: dict[str, list] = {
        'train': [], 'val': [], 'lr': [],
        'd_prime': [], 'median_gt_rank': [],
    }
    for _k in VAL_METRICS_K_VALUES:
        loss_history[f'recall_at_{_k}'] = []
    del _k

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False,
        )
        original_model.load_state_dict(checkpoint['model_state_dict'])
        # Skip loading optimizer state if the saved run used a different
        # optimizer — state dicts are not portable across optimizer types.
        saved_optimizer = checkpoint.get('args', {}).get('optimizer', 'adamw')
        if saved_optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning(
                f'Checkpoint was saved with optimizer={saved_optimizer!r} but '
                f'current run uses {args.optimizer!r}. Skipping '
                f'optimizer.load_state_dict — training resumes with fresh '
                f'optimizer state (model weights still loaded).'
            )
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        # Backward-compatible resume: older checkpoints only stored
        # ``best_val_recall_at_200``. If the new ``best_val_score`` key is
        # absent and the criterion matches the legacy one, inherit from it.
        if 'best_val_score' in checkpoint:
            best_val_score = checkpoint['best_val_score']
        elif checkpoint_criterion == 'recall_at_200':
            best_val_score = checkpoint.get('best_val_recall_at_200', 0.0)
        else:
            best_val_score = 0.0
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        logger.info(
            f'Resumed from epoch {start_epoch - 1}, '
            f'best_val_loss={best_val_loss:.5f}, '
            f'best {checkpoint_criterion}={best_val_score:.5f}',
        )

    # Profile-only entry: run N batches under torch.profiler, then exit.
    # Training checkpoints / tensorboard writes are deliberately skipped —
    # the run's sole output is the chrome trace + summary table.
    if args.profile_steps > 0:
        profile_output_dir = args.profile_output or os.path.join(
            experiment_dir, 'profile',
        )
        logger.info(
            f'=== Profiling mode: {args.profile_steps} active steps === '
            f'→ {profile_output_dir}',
        )
        run_profile(
            model=model,
            train_loader=train_loader,
            device=device,
            data_config=data_config,
            mask_input_index=mask_input_index,
            label_input_index=label_input_index,
            num_steps=args.profile_steps,
            output_dir=profile_output_dir,
            grad_scaler=grad_scaler,
            record_shapes=args.profile_record_shapes,
            profile_memory=args.profile_memory,
            export_chrome_trace=args.profile_chrome_trace,
        )
        return

    logger.info('=== Training ===')

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

            # --- Epoch-dependent training schedule updates ---
            temperature_progress = (epoch - 1) / max(1, args.epochs - 1)
            original_model.set_temperature_progress(temperature_progress)

            drw_warmup_epochs = int(
                args.epochs * original_model.drw_warmup_fraction,
            )
            drw_now_active = epoch > drw_warmup_epochs
            original_model.set_drw_active(drw_now_active)

            logger.info(
                f'Schedule | T={original_model.current_ranking_temperature:.3f}'
                f' | σ={original_model.current_denoising_sigma:.3f}'
                f' | DRW={"ON" if drw_now_active else "off"}'
                f' (warmup={drw_warmup_epochs})',
            )

            train_losses, global_batch_count = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                grad_scaler, device, data_config, epoch,
                tensorboard_writer, global_batch_count,
                steps_per_epoch, mask_input_index, label_input_index,
                grad_clip_max_norm=args.grad_clip,
                augmentation=augmentation,
                ema_teacher=ema_teacher,
                ema_decay=args.ema_decay,
                kl_weight=args.kl_weight,
            )

            eval_steps = max(1, steps_per_epoch // 4)

            val_losses, val_metrics = validate(
                model, val_loader, device, data_config,
                mask_input_index, label_input_index,
                max_steps=eval_steps,
            )
            train_eval_losses, train_eval_metrics = validate(
                model, train_loader, device, data_config,
                mask_input_index, label_input_index,
                max_steps=eval_steps,
            )

            val_loss = val_losses['total_loss']
            val_score = val_metrics.get(checkpoint_criterion, 0.0)

            # Best model is selected by the checkpoint-criterion metric
            # (higher is better), not val loss — DRW and other techniques
            # intentionally increase loss while improving task metrics.
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                best_val_epoch = epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            def _format_metrics(metrics):
                perfect_200 = metrics.get('perfect_at_200', 0.0)
                perfect_256 = metrics.get('perfect_at_256', 0.0)
                recall_256 = metrics.get('recall_at_256', 0.0)
                recall_500 = metrics.get('recall_at_500', 0.0)
                recall_600 = metrics.get('recall_at_600', 0.0)
                rank_p90 = metrics.get('gt_rank_p90', 0.0)
                return (
                    f'R@30: {metrics["recall_at_30"]:.4f} | '
                    f'R@100: {metrics["recall_at_100"]:.4f} | '
                    f'R@200: {metrics["recall_at_200"]:.4f} | '
                    f'R@256: {recall_256:.4f} | '
                    f'R@500: {recall_500:.4f} | '
                    f'R@600: {recall_600:.4f} | '
                    f'P@200: {perfect_200:.4f} | '
                    f'P@256: {perfect_256:.4f} | '
                    f'd\': {metrics["d_prime"]:.3f} | '
                    f'rank: {metrics["median_gt_rank"]:.0f} '
                    f'(p90={rank_p90:.0f})'
                )

            train_summary = _format_metrics(train_eval_metrics)
            val_summary = _format_metrics(val_metrics)

            logger.info(
                f'Epoch {epoch} train_eval | '
                f'total: {train_eval_losses["total_loss"]:.5f} | '
                f'{train_summary}',
            )
            if is_best:
                logger.info(
                    f'Epoch {epoch} val | '
                    f'total: {val_loss:.5f} '
                    f'{checkpoint_criterion}: {val_score:.4f} ★ new best | '
                    f'{val_summary}',
                )
            else:
                epochs_since_best = epoch - best_val_epoch
                logger.info(
                    f'Epoch {epoch} val | '
                    f'total: {val_loss:.5f} '
                    f'(best {checkpoint_criterion}: {best_val_score:.4f}, '
                    f'{epochs_since_best} epochs ago) | '
                    f'{val_summary}',
                )

            previous_lr = scheduler.get_last_lr()[0]
            scheduler.step_epoch(val_loss)
            current_lr = scheduler.get_last_lr()[0]

            # TensorBoard
            tensorboard_writer.add_scalar(
                'Loss/train_epoch', train_losses['total_loss'], epoch,
            )
            tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            for key, value in val_losses.items():
                if key != 'total_loss':
                    tensorboard_writer.add_scalar(
                        f'Loss/val_{key}', value, epoch,
                    )
            for metric_key, metric_value in val_metrics.items():
                if metric_key == 'total_gt_tracks':
                    continue
                tensorboard_writer.add_scalar(
                    f'Metrics/val_{metric_key}', metric_value, epoch,
                )
            for metric_key, metric_value in train_eval_metrics.items():
                if metric_key == 'total_gt_tracks':
                    continue
                tensorboard_writer.add_scalar(
                    f'Metrics/train_{metric_key}', metric_value, epoch,
                )
            tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

            # Loss history
            loss_history['train'].append(train_losses['total_loss'])
            loss_history['val'].append(val_loss)
            loss_history['lr'].append(current_lr)
            for key, value in val_losses.items():
                if key == 'total_loss':
                    continue
                short_key = key.replace('_loss', '')
                if short_key not in loss_history:
                    loss_history[short_key] = []
                loss_history[short_key].append(value)
            for prefix, metrics in [('val', val_metrics), ('train', train_eval_metrics)]:
                for metric_key, metric_value in metrics.items():
                    if metric_key == 'total_gt_tracks':
                        continue
                    history_key = f'{prefix}_{metric_key}'
                    if history_key not in loss_history:
                        loss_history[history_key] = []
                    loss_history[history_key].append(metric_value)
            save_loss_history(
                loss_history, experiment_dir, metric_labels=METRIC_LABELS,
            )

            # Per-epoch metrics JSON for cross-experiment comparison
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_loss,
                'lr': current_lr,
            }
            for metric_key, metric_value in val_metrics.items():
                epoch_metrics[f'val_{metric_key}'] = metric_value
            for metric_key, metric_value in train_eval_metrics.items():
                epoch_metrics[f'train_{metric_key}'] = metric_value
            save_epoch_metrics(epoch_metrics, experiment_dir, epoch)

            # Checkpointing
            if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': original_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_score': best_val_score,
                    'checkpoint_criterion': checkpoint_criterion,
                    'best_val_epoch': best_val_epoch,
                    'global_batch_count': global_batch_count,
                    'val_losses': val_losses,
                    'val_metrics': val_metrics,
                    'args': vars(args),
                }
                checkpoint_manager.save_checkpoint(
                    checkpoint, epoch, val_score, is_best,
                )

    except Exception:
        logger.error(f'Training failed with exception:\n{traceback.format_exc()}')
        raise

    # ---- Final outputs ----
    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best val loss: {best_val_loss:.5f}')
    logger.info(f'Experiment: {experiment_dir}')


if __name__ == '__main__':
    main()
