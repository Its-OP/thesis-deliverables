from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


def build_experiment_directory(
    experiments_base: str,
    model_name: str,
    resume_dir: str | None,
) -> tuple[str, str, str]:
    """Returns (experiment_dir, checkpoints_dir, tensorboard_dir)."""
    if resume_dir is not None:
        experiment_dir = resume_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.join(
            experiments_base, f'{model_name}_{timestamp}',
        )
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    return experiment_dir, checkpoints_dir, tensorboard_dir


class _TeeStream:
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


class WarmupThenPlateauScheduler:
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
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )

    def step_batch(self):
        if self.warmup_finished:
            self.current_step += 1
            return

        self.current_step += 1
        if self.current_step >= self.num_warmup_steps:
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs,
            ):
                param_group['lr'] = base_lr
            self.warmup_finished = True
        else:
            warmup_fraction = self.current_step / self.num_warmup_steps
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs,
            ):
                param_group['lr'] = base_lr * warmup_fraction

    def step_epoch(self, val_loss: float):
        if self.warmup_finished:
            self.plateau_scheduler.step(val_loss)

    def get_last_lr(self) -> list[float]:
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            'current_step': self.current_step,
            'warmup_finished': self.warmup_finished,
            'base_lrs': self.base_lrs,
            'num_warmup_steps': self.num_warmup_steps,
            'plateau_scheduler_state': self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.current_step = state['current_step']
        self.warmup_finished = state['warmup_finished']
        self.base_lrs = state['base_lrs']
        self.num_warmup_steps = state['num_warmup_steps']
        self.plateau_scheduler.load_state_dict(
            state['plateau_scheduler_state'],
        )


class WarmupThenCosineScheduler:
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
            self._use_builtin_cosine = True
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.t_max, eta_min=min_lr,
            )
        else:
            self._use_builtin_cosine = False
            self.cosine_scheduler = None

    def step_batch(self):
        if self.warmup_finished:
            self.current_step += 1
            return

        self.current_step += 1
        if self.current_step >= self.num_warmup_steps:
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs,
            ):
                param_group['lr'] = base_lr
            self.warmup_finished = True
        else:
            warmup_fraction = self.current_step / self.num_warmup_steps
            for param_group, base_lr in zip(
                self.optimizer.param_groups, self.base_lrs,
            ):
                param_group['lr'] = base_lr * warmup_fraction

    def step_epoch(self, val_loss: float):
        if not self.warmup_finished:
            return

        if self._use_builtin_cosine:
            self.cosine_scheduler.step()
            return

        # lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * (t/T_max)^power))
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
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
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
        self.current_step = state['current_step']
        self.warmup_finished = state['warmup_finished']
        self.base_lrs = state['base_lrs']
        self.num_warmup_steps = state['num_warmup_steps']
        self.cosine_epoch = state.get('cosine_epoch', 0)
        if self._use_builtin_cosine and 'cosine_scheduler_state' in state:
            self.cosine_scheduler.load_state_dict(
                state['cosine_scheduler_state'],
            )


def _extract_metric_values(entry):
    if isinstance(entry, dict) and 'values' in entry:
        return entry['values']
    return entry


def save_loss_history(
    loss_history: dict,
    experiment_dir: str,
    metric_labels: dict[str, str] | None = None,
):
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
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        train_values = _extract_metric_values(loss_history['train'])
        val_values = _extract_metric_values(loss_history['val'])
        lr_values = _extract_metric_values(loss_history['lr'])

        fig, (axis_loss, axis_lr) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(train_values) + 1)

        axis_loss.plot(epochs, train_values, 'b-', label='Train')
        axis_loss.plot(epochs, val_values, 'r-', label='Validation')
        axis_loss.set_xlabel('Epoch')
        axis_loss.set_ylabel('Loss')
        axis_loss.set_title('Loss')
        axis_loss.legend()
        axis_loss.grid(True, alpha=0.3)

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
