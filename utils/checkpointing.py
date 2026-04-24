from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        checkpoints_directory: str,
        keep_best_k: int = 5,
        criterion_mode: str = 'max',
        criterion_name: str = 'R@200',
    ):
        self.checkpoints_directory = checkpoints_directory
        self.keep_best_k = keep_best_k
        self.criterion_mode = criterion_mode
        self.criterion_name = criterion_name
        self.tracked_checkpoints: list[tuple[float, int, str]] = []

    def save_checkpoint(
        self,
        checkpoint_data: dict,
        epoch: int,
        criterion_value: float,
        is_best: bool,
    ) -> str:
        checkpoint_path = os.path.join(
            self.checkpoints_directory, f'checkpoint_epoch_{epoch}.pt',
        )
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}')

        self.tracked_checkpoints.append(
            (criterion_value, epoch, checkpoint_path),
        )
        reverse = self.criterion_mode == 'max'
        self.tracked_checkpoints.sort(
            key=lambda entry: entry[0], reverse=reverse,
        )

        if is_best:
            best_path = os.path.join(
                self.checkpoints_directory, 'best_model.pt',
            )
            torch.save(checkpoint_data, best_path)
            logger.info(
                f'New best model ({self.criterion_name}={criterion_value:.5f})',
            )

        self._prune_checkpoints()
        return checkpoint_path

    def _prune_checkpoints(self):
        if self.keep_best_k <= 0:
            return

        while len(self.tracked_checkpoints) > self.keep_best_k:
            worst_value, worst_epoch, worst_path = (
                self.tracked_checkpoints.pop()
            )
            if os.path.exists(worst_path):
                os.remove(worst_path)
                logger.info(
                    f'Pruned checkpoint: epoch {worst_epoch} '
                    f'({self.criterion_name}={worst_value:.5f})',
                )
