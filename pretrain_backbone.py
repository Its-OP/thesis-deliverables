"""Custom training script for backbone pretraining via masked track reconstruction.

Bypasses weaver's training loop to avoid fighting the regression mode interface.
Uses weaver's dataset infrastructure for YAML parsing and parquet loading.

Features:
    - torch.compile for optimized GPU kernels (enabled by default on CUDA)
    - Mixed precision (AMP) support
    - TensorBoard logging (per-batch and per-epoch scalars)
    - JSON loss history export (for plotting without TensorBoard)
    - File logging alongside stdout
    - Checkpointing with backbone-only weight extraction
    - Resume from checkpoint

Experiment directory layout:
    experiments/
    └── {model_name}_{timestamp}/
        ├── training.log          # full console output
        ├── loss_history.json     # per-epoch loss values
        ├── loss_curves.png       # generated after training
        ├── checkpoints/
        │   ├── checkpoint_epoch_10.pt
        │   ├── best_model.pt
        │   └── backbone_best.pt
        └── tensorboard/
            └── events.out.tfevents.*

Usage:
    python pretrain_backbone.py \\
        --data-config data/low-pt/lowpt_tau_pretrain.yaml \\
        --data-dir data/low-pt/ \\
        --network networks/lowpt_tau_BackbonePretrain.py \\
        --epochs 100 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --device cuda:0 \\
        --amp \\
        --no-compile  # optional: disable torch.compile
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from weaver.utils.dataset import SimpleIterDataset

logger = logging.getLogger('pretrain_backbone')


def trim_to_max_valid_tracks(
    inputs: list[torch.Tensor],
    mask_input_index: int,
) -> list[torch.Tensor]:
    """Trim padded tensors to the maximum number of valid tracks in the batch.

    Weaver pads all events to a fixed sequence length (e.g. 2800) defined in
    the YAML config. Most of this is padding zeros — median track count is
    ~1130. This wastes GPU compute and, critically, corrupts BatchNorm
    statistics (the input embedding's BN1d sees ~60-80% zeros).

    This function finds the maximum number of valid tracks across the batch
    using pf_mask, then slices all input tensors to that length. Since FPS,
    kNN, and EdgeConv operate on variable-length point clouds, no architecture
    changes are needed.

    Args:
        inputs: List of input tensors, each (B, C_i, P) where P is the padded
            sequence length. Order follows data_config.input_names.
        mask_input_index: Index of the pf_mask tensor in the inputs list.

    Returns:
        List of trimmed tensors, each (B, C_i, P_trimmed) where
        P_trimmed = max valid tracks in the batch.
    """
    mask = inputs[mask_input_index]  # (B, 1, P)

    # Sum over the sequence dimension to count valid tracks per event,
    # then take the batch maximum. This is the tightest trim that
    # preserves all real data in the batch.
    max_valid_tracks = int(mask.sum(dim=2).max().item())

    # Safety: ensure at least 1 track (handles empty-event edge case)
    max_valid_tracks = max(1, max_valid_tracks)

    # Round up to the nearest multiple of 128 to reduce the number of
    # distinct tensor shapes. torch.compile with dynamic=True recompiles
    # for each new shape; bucketing avoids this by limiting to ~22 possible
    # sizes (128, 256, ..., 2816) instead of thousands of unique values.
    bucket_size = 128
    max_valid_tracks = min(
        ((max_valid_tracks + bucket_size - 1) // bucket_size) * bucket_size,
        inputs[0].shape[2],  # don't exceed original padded length
    )

    return [tensor[:, :, :max_valid_tracks] for tensor in inputs]


def build_experiment_directory(
    experiments_base: str,
    model_name: str,
    resume_dir: str | None,
) -> tuple[str, str, str]:
    """Create or resolve the experiment directory structure.

    Layout:
        {experiments_base}/{model_name}_{timestamp}/
            ├── checkpoints/
            └── tensorboard/

    When resuming, reuses the existing experiment directory.

    Args:
        experiments_base: Root experiments folder (e.g. 'experiments').
        model_name: Short model identifier (e.g. 'BackbonePretrain').
        resume_dir: If resuming, path to the existing experiment root.

    Returns:
        Tuple of (experiment_dir, checkpoints_dir, tensorboard_dir).
    """
    if resume_dir is not None:
        experiment_dir = resume_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'{model_name}_{timestamp}'
        experiment_dir = os.path.join(experiments_base, experiment_name)

    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    return experiment_dir, checkpoints_dir, tensorboard_dir


class _TeeStream:
    """Write stream that duplicates output to both original stream and a file.

    Used to redirect stderr so that Python warnings (e.g. RuntimeWarning from
    numpy overflow), DataLoader worker tracebacks, and any other stderr output
    are captured in training.log alongside the structured log messages.
    """

    def __init__(self, original_stream, log_file_handle):
        self.original_stream = original_stream
        self.log_file_handle = log_file_handle

    def write(self, message):
        self.original_stream.write(message)
        self.log_file_handle.write(message)
        self.log_file_handle.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file_handle.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()


def setup_logging(experiment_dir: str):
    """Configure logging to both stdout and a log file in the experiment root.

    Sets up two output channels:
    1. Structured logging (logger.*) → both stdout and training.log
    2. stderr tee → training.log also captures Python warnings, tracebacks,
       and any other unstructured error output from libraries / subprocesses.

    Note: We configure the logger directly instead of using basicConfig(),
    because basicConfig() is silently ignored if any other library (e.g.
    numexpr) has already initialised the root logger.

    Args:
        experiment_dir: Experiment root directory for the log file.
    """
    log_file = os.path.join(experiment_dir, 'training.log')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure our logger (not root) so it works regardless of import order
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Tee stderr → training.log so that Python warnings (e.g. RuntimeWarning),
    # DataLoader worker errors, and unhandled tracebacks are also captured.
    log_file_handle = open(log_file, 'a')  # noqa: SIM115 — kept open for process lifetime
    sys.stderr = _TeeStream(sys.stderr, log_file_handle)


def load_network_module(network_path: str):
    """Load get_model() from the network wrapper file.

    Args:
        network_path: Path to the network wrapper Python file.

    Returns:
        Module with get_model() function.
    """
    spec = importlib.util.spec_from_file_location('network', network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WarmupThenPlateauScheduler:
    """Two-phase LR scheduler: linear warmup → ReduceLROnPlateau.

    Phase 1 (warmup): LR scales linearly from 0 to base_lr over
        num_warmup_steps. Called per training step via step_batch().

    Phase 2 (plateau): LR is reduced by `factor` when val_loss plateaus
        for `patience` epochs. Called per epoch via step_epoch(val_loss).

    With masked track reconstruction the effective training set is
    combinatorially infinite (random masking), so the model never overfits.
    Performance plateaus come from parameter saturation, not memorisation.
    ReduceLROnPlateau keeps the LR high while the model is still improving
    and only reduces when genuinely stuck — unlike cosine annealing which
    decays unconditionally on a fixed schedule.

    Args:
        optimizer: Optimizer whose LR will be controlled.
        num_warmup_steps: Number of linear warmup steps.
        plateau_factor: Multiplicative factor for LR reduction (default 0.5).
        plateau_patience: Number of epochs with no improvement before
            reducing LR (default 5).
        min_lr: Lower bound on the learning rate (default 1e-6).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.warmup_finished = False

        # ReduceLROnPlateau for post-warmup phase
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )

    def step_batch(self):
        """Call once per training batch. Only active during warmup phase.

        During warmup: linearly scales LR from 0 → base_lr.
        After warmup: no-op (LR managed by step_epoch).
        """
        if self.warmup_finished:
            self.current_step += 1
            return

        self.current_step += 1

        if self.current_step >= self.num_warmup_steps:
            # Warmup just finished — restore base LR
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group['lr'] = base_lr
            self.warmup_finished = True
        else:
            # Linear warmup: lr = base_lr × (step / num_warmup_steps)
            warmup_fraction = self.current_step / self.num_warmup_steps
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group['lr'] = base_lr * warmup_fraction

    def step_epoch(self, val_loss: float):
        """Call once per epoch after validation. Active only after warmup.

        Feeds val_loss to ReduceLROnPlateau which decides whether to
        reduce LR based on plateau detection.
        """
        if self.warmup_finished:
            self.plateau_scheduler.step(val_loss)

    def get_last_lr(self) -> list[float]:
        """Return current LR for each param group (for logging)."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs,
            'num_warmup_steps': self.num_warmup_steps,
            'plateau_scheduler_state': self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict):
        """Restore scheduler state from checkpoint."""
        self.current_step = state['current_step']
        self.warmup_finished = state['warmup_finished']
        self.base_lrs = state['base_lrs']
        self.num_warmup_steps = state['num_warmup_steps']
        self.plateau_scheduler.load_state_dict(
            state['plateau_scheduler_state']
        )


class WarmupThenCosineScheduler:
    """Two-phase LR scheduler: linear warmup → power-cosine annealing.

    Phase 1 (warmup): LR scales linearly from 0 to base_lr over
        num_warmup_steps. Called per training step via step_batch().

    Phase 2 (cosine): LR decays following a power-warped cosine curve
        from base_lr to min_lr over the remaining epochs. Called per
        epoch via step_epoch(val_loss).

    LR formula (post-warmup):
        progress = (t / T_max) ^ cosine_power
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

    where t = epoch index within the cosine phase, T_max = total cosine
    epochs, and cosine_power controls the decay steepness:
        - cosine_power = 1.0: standard symmetric cosine (default)
        - cosine_power < 1.0: steeper decay — LR drops faster initially
        - cosine_power > 1.0: delayed decay — LR stays near peak longer,
          then drops steeply at the end

    Args:
        optimizer: Optimizer whose LR will be controlled.
        num_warmup_steps: Number of linear warmup steps.
        num_post_warmup_epochs: Number of epochs for the cosine decay phase.
        min_lr: Lower bound on the learning rate (default 1e-6).
        cosine_power: Exponent applied to the cosine phase progress.
            Controls the shape of the LR decay curve.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_post_warmup_epochs: int,
        min_lr: float = 1e-6,
        cosine_power: float = 1.0,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        self.warmup_finished = False
        self.cosine_power = cosine_power
        self.min_lr = min_lr
        self.t_max = max(1, num_post_warmup_epochs)
        self.cosine_epoch = 0

        if cosine_power == 1.0:
            # Standard cosine: use PyTorch built-in for exact compatibility
            self._use_builtin_cosine = True
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.t_max,
                eta_min=min_lr,
            )
        else:
            # Power-warped cosine: manual LR computation
            self._use_builtin_cosine = False
            self.cosine_scheduler = None

    def step_batch(self):
        """Call once per training batch. Only active during warmup phase.

        During warmup: linearly scales LR from 0 → base_lr.
        After warmup: no-op (LR managed by step_epoch).
        """
        if self.warmup_finished:
            self.current_step += 1
            return

        self.current_step += 1

        if self.current_step >= self.num_warmup_steps:
            # Warmup just finished — restore base LR
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group['lr'] = base_lr
            self.warmup_finished = True
        else:
            # Linear warmup: lr = base_lr × (step / num_warmup_steps)
            warmup_fraction = self.current_step / self.num_warmup_steps
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs
            ):
                param_group['lr'] = base_lr * warmup_fraction

    def step_epoch(self, val_loss: float):
        """Call once per epoch after validation. Active only after warmup.

        Steps the cosine scheduler unconditionally (val_loss is accepted
        for interface compatibility with WarmupThenPlateauScheduler but
        is not used).
        """
        if not self.warmup_finished:
            return

        if self._use_builtin_cosine:
            self.cosine_scheduler.step()
        else:
            # Power-warped cosine:
            # progress = (t / T_max) ^ cosine_power
            # lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
            self.cosine_epoch += 1
            progress = min(self.cosine_epoch / self.t_max, 1.0)
            warped_progress = progress ** self.cosine_power
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * warped_progress))
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs,
            ):
                param_group['lr'] = (
                    self.min_lr + (base_lr - self.min_lr) * cosine_factor
                )

    def get_last_lr(self) -> list[float]:
        """Return current LR for each param group (for logging)."""
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        result = {
            'current_step': self.current_step,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs,
            'num_warmup_steps': self.num_warmup_steps,
            'cosine_power': self.cosine_power,
            'cosine_epoch': self.cosine_epoch,
        }
        if self._use_builtin_cosine:
            result['cosine_scheduler_state'] = self.cosine_scheduler.state_dict()
        return result

    def load_state_dict(self, state: dict):
        """Restore scheduler state from checkpoint."""
        self.current_step = state['current_step']
        self.warmup_finished = state['warmup_finished']
        self.base_lrs = state['base_lrs']
        self.num_warmup_steps = state['num_warmup_steps']
        self.cosine_epoch = state.get('cosine_epoch', 0)
        if self._use_builtin_cosine and 'cosine_scheduler_state' in state:
            self.cosine_scheduler.load_state_dict(
                state['cosine_scheduler_state'],
            )


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupThenPlateauScheduler | WarmupThenCosineScheduler,
    grad_scaler: torch.amp.GradScaler | None,
    device: torch.device,
    data_config,
    epoch: int,
    tensorboard_writer: SummaryWriter | None,
    global_batch_count: int,
    steps_per_epoch: int,
    mask_input_index: int,
    grad_clip_max_norm: float = 1.0,
) -> tuple[float, int]:
    """Train for one epoch.

    Args:
        model: MaskedTrackPretrainer.
        train_loader: DataLoader yielding (X, y, Z) tuples.
        optimizer: AdamW optimizer.
        scheduler: WarmupThenPlateauScheduler (step_batch called per step).
        grad_scaler: GradScaler for mixed precision, or None.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.
        epoch: Current epoch number (for logging).
        tensorboard_writer: Optional TensorBoard SummaryWriter.
        global_batch_count: Running batch counter across epochs.
        steps_per_epoch: Maximum number of batches per epoch.
            SimpleIterDataset is infinite, so we must break manually.
        mask_input_index: Index of pf_mask in the inputs list (from
            data_config.input_names) for trim_to_max_valid_tracks().
        grad_clip_max_norm: Maximum gradient norm for clipping (0 to disable).

    Returns:
        Tuple of (average_loss, updated_global_batch_count).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (X, y, _) in enumerate(train_loader):
        if batch_idx >= steps_per_epoch:
            break

        inputs = [X[k].to(device) for k in data_config.input_names]
        padded_length = inputs[0].shape[2]

        # Trim padding: slice all tensors to the max valid track count
        # in this batch. Eliminates ~60-80% zero padding that corrupts
        # BatchNorm and wastes GPU compute.
        inputs = trim_to_max_valid_tracks(inputs, mask_input_index)

        if batch_idx == 0:
            trimmed_length = inputs[0].shape[2]
            logger.info(
                f'Epoch {epoch} | Trim: {padded_length} → {trimmed_length} '
                f'({100 * (1 - trimmed_length / padded_length):.0f}% padding removed)'
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
            # model returns (B,) per-event losses
            per_event_loss = model(*inputs)
            loss = per_event_loss.mean()

        # Skip batches with NaN loss — prevents poisoning model weights
        # and the running average. Log the occurrence for debugging.
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f'Epoch {epoch} | Batch {batch_idx} | '
                f'Skipping batch with {"NaN" if torch.isnan(loss) else "Inf"} loss'
            )
            optimizer.zero_grad(set_to_none=True)
            global_batch_count += 1
            continue

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            # Unscale before clipping so clip threshold is in true gradient scale
            grad_scaler.unscale_(optimizer)
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_max_norm
                )
            optimizer.step()

        scheduler.step_batch()

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        global_batch_count += 1

        # TensorBoard: per-batch logging
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                'Loss/train_batch', batch_loss, global_batch_count
            )
            tensorboard_writer.add_scalar(
                'LR/train', scheduler.get_last_lr()[0], global_batch_count
            )

        if batch_idx % 20 == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f'Epoch {epoch} | Batch {batch_idx} | '
                f'Loss: {batch_loss:.5f} | '
                f'Avg Loss: {total_loss / num_batches:.5f} | '
                f'LR: {current_lr:.2e} | '
                f'Time: {elapsed:.1f}s'
            )

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, global_batch_count


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    data_config,
    mask_input_index: int,
    max_steps: int | None = None,
) -> float:
    """Validate on held-out data.

    Args:
        model: MaskedTrackPretrainer.
        val_loader: Validation DataLoader.
        device: Target device.
        data_config: Weaver DataConfig for input name ordering.
        mask_input_index: Index of pf_mask in the inputs list for
            trim_to_max_valid_tracks().
        max_steps: Maximum number of validation batches.
            SimpleIterDataset is infinite, so we must limit manually.
            If None, runs until the loader is exhausted (unsafe for
            infinite iterables).

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (X, y, _) in enumerate(val_loader):
            if max_steps is not None and batch_idx >= max_steps:
                break

            inputs = [X[k].to(device) for k in data_config.input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_input_index)
            per_event_loss = model(*inputs)
            total_loss += per_event_loss.mean().item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def _extract_metric_values(entry):
    """Extract the value list from a loss-history entry.

    Supports two on-disk formats:
        - **New (preferred):** ``{'label': str, 'values': list[float]}``
          — every metric carries a human-readable label so the JSON
          file is self-documenting.
        - **Legacy:** plain ``list[float]`` — the pre-2026-04 flat
          format used by `train_cascade.py` and `train_prefilter.py`
          before the labels rework. Still readable so older
          experiment dumps don't break.
    """
    if isinstance(entry, dict) and 'values' in entry:
        return entry['values']
    return entry


def save_loss_history(
    loss_history: dict,
    experiment_dir: str,
    metric_labels: dict[str, str] | None = None,
):
    """Save loss history to JSON for later plotting.

    The on-disk format is the labeled-metric layout: each metric key
    maps to ``{'label': str, 'values': list[float]}``. The label is a
    short human-readable description so the JSON file is
    self-documenting (the user reads it manually). If no label is
    provided for a key, the key itself is used as the label fallback.

    Args:
        loss_history: Dict mapping metric key → ``list[float]``
            (per-epoch values). Can also be the new on-disk format
            already (``dict`` entries with ``label`` / ``values``),
            in which case the entries are passed through.
        experiment_dir: Experiment root directory for the JSON file.
        metric_labels: Optional dict mapping metric key → label string.
            Keys missing from this dict default to ``label = key``.
    """
    metric_labels = metric_labels or {}
    output: dict[str, dict] = {}
    for key, entry in loss_history.items():
        if isinstance(entry, dict) and 'values' in entry:
            label = entry.get('label', metric_labels.get(key, key))
            values = entry['values']
        else:
            label = metric_labels.get(key, key)
            values = entry
        output[key] = {'label': label, 'values': values}
    output_path = os.path.join(experiment_dir, 'loss_history.json')
    with open(output_path, 'w') as file:
        json.dump(output, file, indent=2)


def plot_loss_curves(loss_history: dict, experiment_dir: str):
    """Generate and save loss curve plots to the experiment root.

    Reads the in-memory ``loss_history`` dict, which can be either the
    legacy flat layout or the new labeled-metric layout — each metric
    is unpacked via ``_extract_metric_values``.

    Args:
        loss_history: Dict with 'train', 'val', and 'lr' entries.
        experiment_dir: Experiment root directory for the plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        train_values = _extract_metric_values(loss_history['train'])
        val_values = _extract_metric_values(loss_history['val'])
        lr_values = _extract_metric_values(loss_history['lr'])

        fig, (axis_loss, axis_lr) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_values) + 1)

        # Loss curves
        axis_loss.plot(epochs, train_values, 'b-', label='Train')
        axis_loss.plot(epochs, val_values, 'r-', label='Validation')
        axis_loss.set_xlabel('Epoch')
        axis_loss.set_ylabel('Loss')
        axis_loss.set_title('Reconstruction Loss')
        axis_loss.legend()
        axis_loss.grid(True, alpha=0.3)

        # Learning rate
        axis_lr.plot(epochs, lr_values, 'g-')
        axis_lr.set_xlabel('Epoch')
        axis_lr.set_ylabel('Learning Rate')
        axis_lr.set_title('Learning Rate Schedule')
        axis_lr.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path = os.path.join(experiment_dir, 'loss_curves.png')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f'Saved loss curves: {output_path}')
    except ImportError:
        logger.warning('matplotlib not available, skipping loss curve plot')


def main():
    parser = argparse.ArgumentParser(
        description='Backbone pretraining via masked track reconstruction'
    )
    parser.add_argument('--data-config', type=str, required=True,
                        help='Path to YAML data config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing parquet files')
    parser.add_argument('--network', type=str, required=True,
                        help='Path to network wrapper Python file')
    parser.add_argument('--model-name', type=str, default='BackbonePretrain',
                        help='Short model name for experiment folder naming')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                        help='Root directory for all experiments')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler after warmup: '
                             'plateau=ReduceLROnPlateau (adaptive), '
                             'cosine=CosineAnnealingLR (deterministic)')
    parser.add_argument('--warmup-fraction', type=float, default=0.05,
                        help='Fraction of total steps for linear warmup '
                             '(capped at 2000 steps)')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='LR reduction factor when val loss plateaus '
                             '(new_lr = old_lr × factor). Only used with '
                             '--scheduler plateau')
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='Number of epochs with no val loss improvement '
                             'before reducing LR. Only used with '
                             '--scheduler plateau')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Lower bound on the learning rate for both '
                             'plateau and cosine schedulers')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of data for training (rest is validation)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='DataLoader workers (must be <= number of parquet files)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (enabled by default on CUDA)')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Number of training batches per epoch. Required because '
                             'SimpleIterDataset is infinite. Computed as '
                             'floor(num_train_events / batch_size). If not set, '
                             'defaults to 100 (likely wrong — set explicitly).')
    parser.add_argument('--mask-ratio', type=float, default=None,
                        help='Fraction of tracks to mask (overrides network config default)')
    parser.add_argument('--num-enrichment-layers', type=int, default=None,
                        help='Number of enrichment (MultiScaleEdgeConv) layers '
                             '(overrides network config default)')
    parser.add_argument('--num-decoder-layers', type=int, default=None,
                        help='Number of DETR-style decoder layers '
                             '(self-attn + cross-attn + FFN per layer). '
                             'Default: 1')
    parser.add_argument('--train-matcher', type=str, default=None,
                        choices=['hungarian', 'sinkhorn'],
                        help='Matching algorithm for training assignment. '
                             'hungarian=exact optimal (CPU), '
                             'sinkhorn=approximate non-bijective (GPU). '
                             'Validation always uses exact Hungarian.')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # ---- Experiment directory setup ----
    # When resuming, reuse the existing experiment directory (checkpoint's grandparent).
    # Otherwise, create a new one with a timestamp.
    resume_experiment_dir = None
    if args.resume is not None:
        # Checkpoint lives at experiments/{name}/checkpoints/checkpoint_epoch_N.pt
        # So grandparent = experiment root
        resume_experiment_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(args.resume)
        ))

    experiment_dir, checkpoints_dir, tensorboard_dir = build_experiment_directory(
        experiments_base=args.experiments_dir,
        model_name=args.model_name,
        resume_dir=resume_experiment_dir,
    )

    setup_logging(experiment_dir)
    device = torch.device(args.device)

    logger.info(f'Experiment directory: {experiment_dir}')
    logger.info(f'Arguments: {vars(args)}')

    # ---- TensorBoard ----
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f'TensorBoard logs: {tensorboard_dir}')

    # ---- Data loading via weaver's dataset infrastructure ----
    parquet_files = sorted([
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith('.parquet')
    ])
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {args.data_dir}')
    logger.info(f'Found {len(parquet_files)} parquet files')

    file_dict = {'data': parquet_files}
    num_parquet_files = len(parquet_files)

    # Training split: first train_fraction of each file.
    # in_memory=True loads the full dataset once and reshuffles indices each
    # epoch, avoiding the ~30s parquet re-read on every "Restarting DataIter".
    # fetch_step=num_files loads all files in a single initial fetch.
    # Note: fetch_step=0 does NOT work with fetch_by_files=True — weaver slices
    # filelist[0:0] which gives an empty list and raises RuntimeError.
    train_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=True,
        load_range_and_fraction=((0.0, args.train_fraction), 1.0),
        fetch_by_files=True,
        fetch_step=num_parquet_files,
        in_memory=True,
    )
    data_config = train_dataset.config

    # Validation split: remaining fraction (also in-memory)
    val_dataset = SimpleIterDataset(
        file_dict,
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((args.train_fraction, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=num_parquet_files,
        in_memory=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # ---- Steps per epoch ----
    # SimpleIterDataset is an infinite IterableDataset — it never raises
    # StopIteration. We must manually limit batches per epoch.
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = 100
        logger.warning(
            f'--steps-per-epoch not set, defaulting to {steps_per_epoch}. '
            f'Set explicitly as floor(num_train_events / batch_size).'
        )
    logger.info(f'Steps per epoch: {steps_per_epoch}')

    # ---- Model ----
    network_module = load_network_module(args.network)
    model_kwargs = {}
    if args.mask_ratio is not None:
        model_kwargs['mask_ratio'] = args.mask_ratio
    if args.num_enrichment_layers is not None:
        model_kwargs['num_enrichment_layers'] = args.num_enrichment_layers
    if args.num_decoder_layers is not None:
        model_kwargs['num_decoder_layers'] = args.num_decoder_layers
    if args.train_matcher is not None:
        model_kwargs['train_matcher'] = args.train_matcher
    model, model_info = network_module.get_model(data_config, **model_kwargs)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {num_params:,}')
    logger.info(f'Input names: {data_config.input_names}')

    # Find the index of pf_mask in the input list for trim_to_max_valid_tracks()
    mask_input_index = data_config.input_names.index('pf_mask')
    logger.info(f'Mask input index: {mask_input_index} (pf_mask)')

    # Keep a reference to the original (uncompiled) model for state_dict access.
    # torch.compile wraps the module, so we need the original for clean
    # checkpoint saving and backbone weight extraction.
    original_model = model

    # ---- torch.compile ----
    use_compile = (
        not args.no_compile
        and device.type == 'cuda'
        and hasattr(torch, 'compile')
    )
    if use_compile:
        # dynamic=True: sequence length varies per batch after trim_to_max_valid_tracks(),
        # so torch.compile must use symbolic shapes to avoid recompiling every batch.
        logger.info('Compiling model with torch.compile (mode="default", dynamic=True)...')
        model = torch.compile(model, mode='default', dynamic=True)
        logger.info('Model compiled.')
    else:
        logger.info('torch.compile disabled.')

    # ---- Optimizer and scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * steps_per_epoch
    max_warmup_steps = 2000
    warmup_steps = min(int(args.warmup_fraction * total_steps), max_warmup_steps)

    if args.scheduler == 'cosine':
        # Number of full epochs consumed by warmup (rounded up), so the
        # cosine curve spans exactly the remaining post-warmup epochs.
        warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
        num_post_warmup_epochs = max(1, args.epochs - warmup_epochs)
        logger.info(
            f'LR schedule: {warmup_steps} warmup steps (~{warmup_epochs} epochs), then '
            f'CosineAnnealingLR over {num_post_warmup_epochs} epochs'
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
            f'ReduceLROnPlateau (factor={args.plateau_factor}, '
            f'patience={args.plateau_patience})'
        )
        scheduler = WarmupThenPlateauScheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    # ---- Mixed precision ----
    grad_scaler = torch.amp.GradScaler('cuda') if args.amp else None

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_epoch = 0
    global_batch_count = 0
    loss_history = {'train': [], 'val': [], 'lr': []}

    if args.resume is not None:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # Load into the original (uncompiled) model — torch.compile wraps it,
        # but state_dict keys come from the unwrapped module.
        original_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        global_batch_count = checkpoint.get('global_batch_count', 0)
        loss_history = checkpoint.get('loss_history', loss_history)
        logger.info(
            f'Resumed at epoch {start_epoch}, '
            f'best_val_loss={best_val_loss:.5f} (epoch {best_val_epoch})'
        )

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f'=== Epoch {epoch}/{args.epochs} ===')

        train_loss, global_batch_count = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            grad_scaler, device, data_config, epoch,
            tensorboard_writer, global_batch_count,
            steps_per_epoch=steps_per_epoch,
            mask_input_index=mask_input_index,
            grad_clip_max_norm=args.grad_clip,
        )
        logger.info(f'Epoch {epoch} train loss: {train_loss:.5f}')

        # Validation steps: 20% of data / batch_size.
        # With 19K events, 20% val split, batch 128 → ~29 steps.
        val_steps = max(1, steps_per_epoch // 4)
        val_loss = validate(
            model, val_loader, device, data_config,
            mask_input_index=mask_input_index, max_steps=val_steps,
        )
        # Check if this is the best val loss so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_epoch = epoch

        # Log val loss with patience counter (epochs since best).
        # Patience is only meaningful for plateau scheduler; cosine decays
        # unconditionally regardless of val loss trajectory.
        if is_best:
            logger.info(f'Epoch {epoch} val loss: {val_loss:.5f} ★ new best')
        else:
            epochs_since_best = epoch - best_val_epoch
            if args.scheduler == 'plateau':
                logger.info(
                    f'Epoch {epoch} val loss: {val_loss:.5f} '
                    f'(patience: {epochs_since_best}/{args.plateau_patience}, '
                    f'best: {best_val_loss:.5f})'
                )
            else:
                logger.info(
                    f'Epoch {epoch} val loss: {val_loss:.5f} '
                    f'(best: {best_val_loss:.5f})'
                )

        # Step the scheduler with val loss (no-op during warmup).
        # Plateau uses val_loss for plateau detection; cosine ignores it.
        previous_lr = scheduler.get_last_lr()[0]
        scheduler.step_epoch(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr < previous_lr and args.scheduler == 'plateau':
            logger.info(
                f'ReduceLROnPlateau triggered: LR {previous_lr:.2e} → '
                f'{current_lr:.2e}'
            )

        # TensorBoard: per-epoch logging
        tensorboard_writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        tensorboard_writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        tensorboard_writer.add_scalar('LR/epoch', current_lr, epoch)

        # Loss history for JSON export and plotting
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        loss_history['lr'].append(current_lr)
        save_loss_history(loss_history, experiment_dir)

        if epoch % args.save_every == 0 or is_best or epoch == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': original_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
                'global_batch_count': global_batch_count,
                'loss_history': loss_history,
                'args': vars(args),
            }

            checkpoint_path = os.path.join(
                checkpoints_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            if is_best:
                best_path = os.path.join(checkpoints_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                logger.info(f'New best model (val_loss={val_loss:.5f})')

        # Save backbone-only weights (for downstream use)
        if is_best:
            backbone_state = {
                k.replace('backbone.', ''): v
                for k, v in original_model.state_dict().items()
                if k.startswith('backbone.')
            }
            backbone_path = os.path.join(checkpoints_dir, 'backbone_best.pt')
            torch.save(backbone_state, backbone_path)
            logger.info(f'Saved backbone weights: {backbone_path}')

    # ---- Final outputs ----
    tensorboard_writer.close()
    plot_loss_curves(loss_history, experiment_dir)
    logger.info(f'Training complete. Best val loss: {best_val_loss:.5f}')
    logger.info(f'Experiment: {experiment_dir}')
    logger.info(f'  - Log: training.log')
    logger.info(f'  - Loss history: loss_history.json')
    logger.info(f'  - Loss curves: loss_curves.png')
    logger.info(f'  - Checkpoints: checkpoints/')
    logger.info(f'  - TensorBoard: tensorboard/')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Log the full traceback so it appears in training.log even if the
        # process crashes. The stderr tee will also capture it, but logging
        # it explicitly ensures it gets a proper timestamp.
        logger.error(f'Training failed with exception:\n{traceback.format_exc()}')
        raise
