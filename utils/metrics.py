from __future__ import annotations

import json
import math
import os

import torch


def extract_per_track_scores(
    output_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Returns (B, P) scores from any head's output dict."""
    if 'per_track_logits' in output_dict:
        return output_dict['per_track_logits']
    if 'beta_scores' in output_dict:
        return output_dict['beta_scores']
    if 'mask_logits' in output_dict:
        return output_dict['mask_logits'].max(dim=1).values
    raise KeyError(
        f'Cannot extract per-track scores from output keys: '
        f'{list(output_dict.keys())}.',
    )


@torch.no_grad()
def compute_recall_at_k_metrics(
    per_track_scores: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    k_values: tuple[int, ...] = (10, 20, 30, 100),
) -> dict[str, float]:
    """per_track_scores: (B, P). track_labels: (B, 1, P). mask: (B, 1, P) bool."""
    batch_size = per_track_scores.shape[0]
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
    valid_mask = mask.squeeze(1).bool()

    masked_scores = per_track_scores.clone()
    masked_scores[~valid_mask] = float('-inf')
    sorted_indices = masked_scores.argsort(dim=1, descending=True)

    # rank_lookup[i] = position of track i in sorted order.
    rank_lookup = torch.argsort(
        torch.argsort(masked_scores, dim=1, descending=True), dim=1,
    )

    recall_sums = {k: 0.0 for k in k_values}
    perfect_event_counts = {k: 0 for k in k_values}
    total_events_with_gt = 0
    total_gt_tracks = 0

    all_gt_scores = []
    all_background_scores = []
    all_gt_ranks = []

    breakdown_k = 200
    compute_breakdown = breakdown_k in k_values
    event_breakdown_counts: dict[str, int] = {}

    for batch_index in range(batch_size):
        gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
        num_gt = len(gt_positions)

        event_valid = valid_mask[batch_index]
        event_labels = labels_flat[batch_index]
        event_scores = per_track_scores[batch_index]

        gt_mask = (event_labels == 1.0) & event_valid
        background_mask = (event_labels == 0.0) & event_valid

        if gt_mask.any():
            all_gt_scores.append(event_scores[gt_mask])
        if background_mask.any():
            all_background_scores.append(event_scores[background_mask])

        if num_gt == 0:
            continue

        total_events_with_gt += 1
        total_gt_tracks += num_gt

        found_at_breakdown_k = 0
        for k in k_values:
            top_k_indices = sorted_indices[batch_index, :k]
            found = torch.isin(gt_positions, top_k_indices).sum().item()
            recall_sums[k] += found / num_gt
            if found == num_gt:
                perfect_event_counts[k] += 1
            if k == breakdown_k:
                found_at_breakdown_k = found

        if compute_breakdown:
            breakdown_key = f'found_{found_at_breakdown_k}_of_{num_gt}'
            event_breakdown_counts[breakdown_key] = (
                event_breakdown_counts.get(breakdown_key, 0) + 1
            )

        event_gt_ranks = rank_lookup[batch_index, gt_positions]
        all_gt_ranks.extend(event_gt_ranks.cpu().tolist())

    metrics = {}
    for k in k_values:
        metrics[f'recall_at_{k}'] = recall_sums[k] / max(1, total_events_with_gt)
        metrics[f'perfect_at_{k}'] = perfect_event_counts[k] / max(1, total_events_with_gt)
    metrics['total_gt_tracks'] = total_gt_tracks
    metrics['total_events_with_gt'] = total_events_with_gt

    # d' = (mu_gt - mu_bg) / sqrt(0.5 * (sigma_gt^2 + sigma_bg^2))
    if all_gt_scores and all_background_scores:
        gt_scores_cat = torch.cat(all_gt_scores)
        background_scores_cat = torch.cat(all_background_scores)
        mu_gt = gt_scores_cat.mean().item()
        mu_background = background_scores_cat.mean().item()
        sigma_gt = gt_scores_cat.std().item()
        sigma_background = background_scores_cat.std().item()
        pooled_std = (0.5 * (sigma_gt ** 2 + sigma_background ** 2)) ** 0.5
        metrics['d_prime'] = (
            (mu_gt - mu_background) / pooled_std if pooled_std > 1e-10 else 0.0
        )
    else:
        metrics['d_prime'] = 0.0

    if all_gt_ranks:
        sorted_ranks = sorted(all_gt_ranks)
        num_ranks = len(sorted_ranks)
        midpoint = num_ranks // 2
        if num_ranks % 2 == 0:
            metrics['median_gt_rank'] = (
                sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]
            ) / 2.0
        else:
            metrics['median_gt_rank'] = float(sorted_ranks[midpoint])

        for percentile in (75, 90, 95):
            index = min(
                int(math.ceil(percentile / 100.0 * num_ranks)) - 1,
                num_ranks - 1,
            )
            metrics[f'gt_rank_p{percentile}'] = float(sorted_ranks[index])
    else:
        metrics['median_gt_rank'] = float('inf')
        for percentile in (75, 90, 95):
            metrics[f'gt_rank_p{percentile}'] = float('inf')

    if compute_breakdown:
        for key, count in event_breakdown_counts.items():
            metrics[f'{key}_at_{breakdown_k}'] = (
                count / max(1, total_events_with_gt)
            )

    return metrics


class MetricsAccumulator:
    def __init__(self, k_values: tuple[int, ...] = (10, 20, 30, 100, 200)):
        self.k_values = k_values

        self.all_gt_ranks: list[int] = []
        self.all_gt_scores: list[torch.Tensor] = []
        self.all_background_scores: list[torch.Tensor] = []

        self.recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
        self.perfect_event_counts: dict[int, int] = {k: 0 for k in k_values}

        self.breakdown_k = 200
        self.compute_breakdown = self.breakdown_k in k_values
        self.event_breakdown_counts: dict[str, int] = {}

        self.total_events_with_gt = 0
        self.total_gt_tracks = 0

    @torch.no_grad()
    def update(
        self,
        per_track_scores: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """per_track_scores: (B, P). track_labels: (B, 1, P). mask: (B, 1, P) bool."""
        batch_size = per_track_scores.shape[0]
        labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
        valid_mask = mask.squeeze(1).bool()

        masked_scores = per_track_scores.clone()
        masked_scores[~valid_mask] = float('-inf')
        sorted_indices = masked_scores.argsort(dim=1, descending=True)

        rank_lookup = torch.argsort(
            torch.argsort(masked_scores, dim=1, descending=True), dim=1,
        )

        for batch_index in range(batch_size):
            gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
            num_gt = len(gt_positions)

            event_valid = valid_mask[batch_index]
            event_labels = labels_flat[batch_index]
            event_scores = per_track_scores[batch_index]
            # Cascade models emit -inf for non-selected tracks; exclude from d'.
            finite_scores_mask = torch.isfinite(event_scores)

            gt_mask = (event_labels == 1.0) & event_valid & finite_scores_mask
            background_mask = (event_labels == 0.0) & event_valid & finite_scores_mask

            if gt_mask.any():
                self.all_gt_scores.append(event_scores[gt_mask].cpu())
            if background_mask.any():
                self.all_background_scores.append(
                    event_scores[background_mask].cpu(),
                )

            if num_gt == 0:
                continue

            self.total_events_with_gt += 1
            self.total_gt_tracks += num_gt

            found_at_breakdown_k = 0
            for k in self.k_values:
                top_k_indices = sorted_indices[batch_index, :k]
                found = torch.isin(gt_positions, top_k_indices).sum().item()
                self.recall_sums[k] += found / num_gt
                if found == num_gt:
                    self.perfect_event_counts[k] += 1
                if k == self.breakdown_k:
                    found_at_breakdown_k = found

            if self.compute_breakdown:
                breakdown_key = f'found_{found_at_breakdown_k}_of_{num_gt}'
                self.event_breakdown_counts[breakdown_key] = (
                    self.event_breakdown_counts.get(breakdown_key, 0) + 1
                )

            event_gt_ranks = rank_lookup[batch_index, gt_positions]
            self.all_gt_ranks.extend(event_gt_ranks.cpu().tolist())

    def compute(self) -> dict[str, float]:
        num_events = max(1, self.total_events_with_gt)
        metrics = {}

        for k in self.k_values:
            metrics[f'recall_at_{k}'] = self.recall_sums[k] / num_events
            metrics[f'perfect_at_{k}'] = (
                self.perfect_event_counts[k] / num_events
            )

        metrics['total_gt_tracks'] = self.total_gt_tracks
        metrics['total_events_with_gt'] = self.total_events_with_gt

        if self.all_gt_scores and self.all_background_scores:
            gt_scores_cat = torch.cat(self.all_gt_scores)
            background_scores_cat = torch.cat(self.all_background_scores)
            mu_gt = gt_scores_cat.mean().item()
            mu_background = background_scores_cat.mean().item()
            sigma_gt = gt_scores_cat.std().item()
            sigma_background = background_scores_cat.std().item()
            pooled_std = (0.5 * (sigma_gt ** 2 + sigma_background ** 2)) ** 0.5
            metrics['d_prime'] = (
                (mu_gt - mu_background) / pooled_std
                if pooled_std > 1e-10
                else 0.0
            )
        else:
            metrics['d_prime'] = 0.0

        if self.all_gt_ranks:
            sorted_ranks = sorted(self.all_gt_ranks)
            num_ranks = len(sorted_ranks)
            midpoint = num_ranks // 2
            if num_ranks % 2 == 0:
                metrics['median_gt_rank'] = (
                    sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]
                ) / 2.0
            else:
                metrics['median_gt_rank'] = float(sorted_ranks[midpoint])

            for percentile in (75, 90, 95):
                index = min(
                    int(math.ceil(percentile / 100.0 * num_ranks)) - 1,
                    num_ranks - 1,
                )
                metrics[f'gt_rank_p{percentile}'] = float(sorted_ranks[index])
        else:
            metrics['median_gt_rank'] = float('inf')
            for percentile in (75, 90, 95):
                metrics[f'gt_rank_p{percentile}'] = float('inf')

        if self.compute_breakdown:
            for key, count in self.event_breakdown_counts.items():
                metrics[f'{key}_at_{self.breakdown_k}'] = count / num_events

        return metrics


def save_epoch_metrics(
    metrics: dict[str, float | int],
    experiment_directory: str,
    epoch: int,
) -> str:
    metrics_directory = os.path.join(experiment_directory, 'metrics')
    os.makedirs(metrics_directory, exist_ok=True)
    filepath = os.path.join(metrics_directory, f'epoch_{epoch}.json')
    with open(filepath, 'w') as file_handle:
        json.dump(metrics, file_handle, indent=2, default=float)
    return filepath
