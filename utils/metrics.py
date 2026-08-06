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
        self.duplet_event_counts: dict[int, int] = {k: 0 for k in k_values}

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
                if found >= 2:
                    self.duplet_event_counts[k] += 1
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
            metrics[f'duplet_at_{k}'] = (
                self.duplet_event_counts[k] / num_events
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
class CoupleMetricsAccumulator:
    """Accumulates three complementary per-event metrics across batches.

    The naming convention uses an explicit unit suffix on every K so that
    track-K and couple-K can never be confused:

    **D@K_tracks** (Duplet at K tracks): per event, at least 2 of the 3 GT
    pions are in the top-K tracks of the cascade's Stage 2 (ParT) score
    ordering. K refers to a number of *tracks*. This is a fixed property
    of the frozen cascade — every reranker run sees the same number for
    the same checkpoint and dataset.

        D@K_tracks = mean over events of:
                       1[ n_gt(top-K_tracks) >= 2 ]

    **C@K_couples** (Couple-found at K couples): per event, at least one
    GT couple is in the top-K of the model's couple ranking. K refers to
    a number of *couples*. This is the primary metric the reranker
    optimizes. Multiple GT couples in the same event do NOT inflate the
    metric: each event contributes 0 or 1. Events with no GT couple in
    the candidate pool contribute 0.

        C@K_couples = mean over ALL events of:
                        1[ any GT couple in top-K_couples of reranker ]

    **RC@K_couples** (Reconstructable at K couples): the joint condition
    that we both found a couple AND have the full triplet available
    downstream:

        RC@K_couples = mean over ALL events of:
                         1[ any GT couple in top-K_couples ]
                       × 1[ n_gt_in_top_k1 == 3 ]

    The Stage 1 condition (full triplet in top-K1) is the prerequisite
    for a future triplet-completion stage. The gap
    ``C@K_couples − RC@K_couples`` is the events where the couple was
    found but the third pion was filtered by Stage 1.

    **mean_first_gt_rank_couples** (Mean rank of best GT couple): per
    eligible event, the 1-indexed rank of the highest-scoring GT couple
    in the model's per-event ranking, averaged across eligible events.
    Lower is better. Independent of K (no top-K cutoff). For an event
    with multiple GT couples, only the BEST rank contributes — this is
    the K-free analogue of C@K_couples.

        mean_first_gt_rank_couples =
            mean over eligible events of:
              1 + min over (GT couples in event) of (rank in sorted order)

    All three metrics (D, C, RC) use the **same denominator**: every
    event the accumulator sees. This is the only way the comparison
    ``RC@K_couples ≤ C@K_couples ≤ D@K_tracks`` (whenever
    ``K_tracks ≥ K2``) holds structurally — and it must, because every
    GT couple in the reranker's input requires the cascade to have
    surfaced both pions in its top-K2 ⊆ top-K_tracks selection.

    Mean rank is the only metric that uses the **eligible-events
    denominator**, because rank is undefined for events without a GT
    couple. The bookkeeping fields ``eligible_events`` and
    ``total_events`` are reported so callers can recover the
    conditional version of any rate (e.g., ``c_at_K * total /
    eligible``) if they need it.

    Usage:
        accumulator = CoupleMetricsAccumulator(
            k_values_couples=(50, 75, 100, 200),
            k_values_tracks=(30, 50, 75, 100, 200),
        )
        for batch in val_loader:
            accumulator.update(
                couple_scores, couple_labels, couple_mask,
                n_gt_in_top_k1=...,            # (B,) for RC@K_couples
                n_gt_in_top_k_tracks=...,      # (B, K_tracks) for D@K_tracks
            )
        metrics = accumulator.compute()
        # → {'d_at_30_tracks': ..., 'c_at_50_couples': ..., 'rc_at_50_couples': ..., ...}

    Args:
        k_values_couples: K values for C@K_couples and RC@K_couples.
        k_values_tracks: K values for D@K_tracks.
        full_triplet_threshold: GT-pion count that signals "full triplet"
            (default 3, the τ → 3π case).
        duplet_threshold: GT-pion count that signals "duplet found"
            (default 2).
    """

    def __init__(
        self,
        k_values_couples: tuple[int, ...] = (50, 75, 100, 200),
        k_values_tracks: tuple[int, ...] = (30, 50, 75, 100, 200),
        full_triplet_threshold: int = 3,
        duplet_threshold: int = 2,
    ):
        self.k_values_couples = k_values_couples
        self.k_values_tracks = k_values_tracks
        self.full_triplet_threshold = full_triplet_threshold
        self.duplet_threshold = duplet_threshold
        # C / RC numerator accumulators (denominator = total_events_count,
        # SAME as D — see class docstring for the comparability invariant).
        self.c_sums: dict[int, float] = {k: 0.0 for k in k_values_couples}
        self.rc_sums: dict[int, float] = {k: 0.0 for k in k_values_couples}
        # Bookkeeping (also used as the denominator for mean rank)
        self.eligible_events_count: int = 0
        self.events_with_full_triplet_count: int = 0
        # Mean-rank accumulator (denominator = eligible events — rank is
        # undefined for events without a GT couple).
        # Sums the 1-indexed rank of the highest-scoring GT couple per event.
        self.first_gt_rank_sum: float = 0.0
        # D accumulator (denominator = all events seen)
        self.d_sums: dict[int, float] = {k: 0.0 for k in k_values_tracks}
        self.total_events_count: int = 0

    @torch.no_grad()
    def update(
        self,
        couple_scores: torch.Tensor,
        couple_labels: torch.Tensor,
        couple_mask: torch.Tensor,
        n_gt_in_top_k1: torch.Tensor | None = None,
        n_gt_in_top_k_tracks: torch.Tensor | None = None,
    ) -> None:
        """Accumulate D@K_tracks, C@K_couples, RC@K_couples from one batch.

        Args:
            couple_scores: ``(B, n_couples)`` per-couple scores.
            couple_labels: ``(B, n_couples)`` 0/1 GT-couple labels.
            couple_mask: ``(B, n_couples)`` validity mask (Filter A).
            n_gt_in_top_k1: ``(B,)`` per-event GT-pion count in Stage 1
                top-K1. Required for RC@K_couples.
            n_gt_in_top_k_tracks: ``(B, len(k_values_tracks))`` per-event
                GT-pion counts in the top-K tracks for each K in
                ``k_values_tracks``. Required for D@K_tracks.
        """
        batch_size = couple_scores.shape[0]
        # All per-batch sums are computed on-device as int64 counts and moved
        # to the host in ONE transfer at the end of the call.
        result_chunks: list[torch.Tensor] = []

        # ---- D@K_tracks accumulation (denominator = all events) ----
        if n_gt_in_top_k_tracks is not None:
            duplet_found = n_gt_in_top_k_tracks >= self.duplet_threshold
            result_chunks.append(duplet_found.sum(dim=0).long())
        self.total_events_count += batch_size

        # ---- C@K_couples / RC@K_couples accumulation ----
        valid_mask = couple_mask > 0.5
        gt_mask = (couple_labels > 0.5) & valid_mask
        eligible = valid_mask.any(dim=1) & gt_mask.any(dim=1)

        # Push invalid couples to -inf so they sort to the bottom
        masked_scores = couple_scores.masked_fill(~valid_mask, float('-inf'))
        sorted_indices = torch.argsort(masked_scores, dim=1, descending=True)
        sorted_gt = gt_mask.gather(1, sorted_indices)

        # First (best) GT couple rank, 1-indexed: argmax returns the first
        # True position of each row; only eligible rows contribute.
        first_gt_position = sorted_gt.float().argmax(dim=1)

        if n_gt_in_top_k1 is not None:
            full_triplet = n_gt_in_top_k1 >= self.full_triplet_threshold
        else:
            full_triplet = torch.zeros(
                batch_size, dtype=torch.bool, device=couple_scores.device,
            )
        full_triplet_eligible = eligible & full_triplet

        couple_in_top_k = torch.stack([
            sorted_gt[:, :k].any(dim=1) & eligible
            for k in self.k_values_couples
        ])
        couple_in_top_k_full = (
            couple_in_top_k & full_triplet_eligible.unsqueeze(0)
        )

        result_chunks.append(eligible.sum().reshape(1))
        result_chunks.append(full_triplet_eligible.sum().reshape(1))
        result_chunks.append(
            ((first_gt_position + 1) * eligible).sum().reshape(1),
        )
        result_chunks.append(couple_in_top_k.sum(dim=1).long())
        result_chunks.append(couple_in_top_k_full.sum(dim=1).long())

        results = torch.cat(result_chunks).cpu().tolist()
        cursor = 0
        if n_gt_in_top_k_tracks is not None:
            for k in self.k_values_tracks:
                self.d_sums[k] += float(results[cursor])
                cursor += 1
        self.eligible_events_count += int(results[cursor])
        cursor += 1
        self.events_with_full_triplet_count += int(results[cursor])
        cursor += 1
        self.first_gt_rank_sum += float(results[cursor])
        cursor += 1
        for k in self.k_values_couples:
            self.c_sums[k] += float(results[cursor])
            cursor += 1
        for k in self.k_values_couples:
            self.rc_sums[k] += float(results[cursor])
            cursor += 1

    def compute(self) -> dict[str, float]:
        """Compute final averages.

        Returns a dict with:
            ``d_at_K_tracks`` for each K in ``k_values_tracks``
                (denominator: all events seen)
            ``c_at_K_couples``, ``rc_at_K_couples`` for each K in
                ``k_values_couples``
                (denominator: all events seen — SAME as D so the
                comparison ``RC ≤ C ≤ D@K_tracks`` (when ``K_tracks ≥ K2``)
                holds structurally)
            ``mean_first_gt_rank_couples``
                (denominator: eligible events — rank is undefined for
                events with no GT couple)
            bookkeeping: ``eligible_events``, ``total_events``,
                ``events_with_full_triplet``
        """
        total = max(1, self.total_events_count)
        metrics: dict[str, float] = {}
        for k in self.k_values_tracks:
            metrics[f'd_at_{k}_tracks'] = self.d_sums[k] / total
        for k in self.k_values_couples:
            metrics[f'c_at_{k}_couples'] = self.c_sums[k] / total
            metrics[f'rc_at_{k}_couples'] = self.rc_sums[k] / total
        # When eligible_events == 0 we report 0.0 (sentinel) — there is
        # no GT couple to rank, so the metric is undefined; the
        # bookkeeping ``eligible_events`` field disambiguates.
        if self.eligible_events_count == 0:
            metrics['mean_first_gt_rank_couples'] = 0.0
        else:
            metrics['mean_first_gt_rank_couples'] = (
                self.first_gt_rank_sum / self.eligible_events_count
            )
        metrics['eligible_events'] = self.eligible_events_count
        metrics['total_events'] = self.total_events_count
        metrics['events_with_full_triplet'] = self.events_with_full_triplet_count
        return metrics


def format_couple_metrics_table(
    val_metrics: dict,
    *,
    train_loss: float,
    val_loss: float,
    epoch: int,
    is_best: bool,
    best_val_criterion: float,
    best_val_epoch: int,
    criterion_name: str = 'C@100c',
    k_values_tracks: tuple = (30, 50, 75, 100, 200),
    k_values_couples: tuple = (50, 75, 100, 200),
) -> str:
    """Render one validation epoch as a multi-line ASCII table.

    The output has three sections:

    1. **Header line** — epoch number, train + val losses, best-marker
       (★ on best epoch, "(N epochs ago)" otherwise).
    2. **K × {D, C, RC} table** — rows are the union of ``k_values_tracks``
       and ``k_values_couples``; cells with no value (e.g., C@30 since
       there is no K=30 in ``k_values_couples``) render as ``-``.
    3. **Footer** — mean rank of best GT couple, eligible/total event
       counts, full-triplet bookkeeping count.

    The function is pure: no logging, no I/O. Pass the result to
    ``logger.info`` (the multi-line string is rendered with the standard
    log prefix on the first line and unprefixed continuation lines).

    Args:
        val_metrics: Dict from ``CoupleMetricsAccumulator.compute()``.
        train_loss: Train loss for the same epoch (mean over batches).
        val_loss: Validation loss for the epoch.
        epoch: Current epoch number.
        is_best: Whether ``val_metrics[criterion_name]`` is a new best.
        best_val_criterion: Best criterion value seen so far.
        best_val_epoch: Epoch at which ``best_val_criterion`` was set.
        criterion_name: Display name for the selection metric (default
            ``C@100c``).
        k_values_tracks: K values reported for D@K_tracks.
        k_values_couples: K values reported for C@K_couples and
            RC@K_couples.

    Returns:
        Multi-line string ready for ``logger.info``.
    """
    # ---- Header line ----
    if is_best:
        header = (
            f'Epoch {epoch} | train: {train_loss:.5f} | val: {val_loss:.5f} '
            f'| ★ new best ({criterion_name}={best_val_criterion:.4f})'
        )
    else:
        epochs_since = epoch - best_val_epoch
        header = (
            f'Epoch {epoch} | train: {train_loss:.5f} | val: {val_loss:.5f} '
            f'| best {criterion_name}={best_val_criterion:.4f} '
            f'({epochs_since} epochs ago)'
        )

    # ---- K × {D, C, RC} table ----
    all_k_values = sorted(set(k_values_tracks) | set(k_values_couples))
    column_headers = ('K', 'D@K_tracks', 'C@K_couples', 'RC@K_couples')
    # Inner widths chosen to be wider than the longest header text in each
    # column, with at least one space of horizontal padding on each side.
    column_widths = (5, 12, 13, 14)

    def _format_row(values: tuple) -> str:
        cells = [
            str(value).center(width)
            for value, width in zip(values, column_widths, strict=True)
        ]
        return '|' + '|'.join(cells) + '|'

    def _separator() -> str:
        return '+' + '+'.join('-' * width for width in column_widths) + '+'

    table_lines = [_separator(), _format_row(column_headers), _separator()]
    for k in all_k_values:
        d_value = val_metrics.get(f'd_at_{k}_tracks')
        c_value = val_metrics.get(f'c_at_{k}_couples')
        rc_value = val_metrics.get(f'rc_at_{k}_couples')
        d_text = f'{d_value:.4f}' if d_value is not None else '-'
        c_text = f'{c_value:.4f}' if c_value is not None else '-'
        rc_text = f'{rc_value:.4f}' if rc_value is not None else '-'
        table_lines.append(_format_row((str(k), d_text, c_text, rc_text)))
    table_lines.append(_separator())

    # ---- Footer ----
    mean_rank = val_metrics.get('mean_first_gt_rank_couples', 0.0)
    eligible_events = int(val_metrics.get('eligible_events', 0))
    total_events = int(val_metrics.get('total_events', 0))
    full_triplet_events = int(val_metrics.get('events_with_full_triplet', 0))
    footer = (
        f'mean_rank: {mean_rank:.1f} | '
        f'eligible: {eligible_events} / {total_events} | '
        f'full_triplet: {full_triplet_events}'
    )

    return '\n'.join([header, *table_lines, footer])


@torch.no_grad()
def compute_conditional_recall(
    per_track_scores: torch.Tensor,
    track_labels: torch.Tensor,
    mask: torch.Tensor,
    raw_features: torch.Tensor,
    feature_index_pt: int = 0,
    feature_index_dxy_significance: int = 6,
    top_k: int = 200,
) -> dict[str, float]:
    """Compute recall@K conditioned on pT and |dxy_significance| bins.

    For each GT pion track, checks whether it lands in the top-K.
    Reports found rate per pT bin, per |dxy_sig| bin, and a 2D grid.

    Args:
        per_track_scores: (B, P) per-track ranking scores.
        track_labels: (B, 1, P) binary labels (1.0 = tau pion).
        mask: (B, 1, P) boolean mask (True = valid track).
        raw_features: (B, C, P) raw features (before standardization).
        feature_index_pt: Index of pT in feature channels (default: 0).
        feature_index_dxy_significance: Index of dxy_significance (default: 6).
        top_k: K value for recall computation (default: 200).

    Returns:
        Dict with recall_pt_{bin}, recall_dxy_{bin}, recall_2d_pt{i}_dxy{j},
        and corresponding count_* entries (70 metrics total).
    """
    batch_size = per_track_scores.shape[0]
    labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
    valid_mask = mask.squeeze(1).bool()

    masked_scores = per_track_scores.clone()
    masked_scores[~valid_mask] = float('-inf')
    sorted_indices = masked_scores.argsort(dim=1, descending=True)

    # Bin edges for pT (GeV) and |dxy_significance|
    pt_bin_edges = [0.0, 0.3, 0.5, 1.0, 2.0, float('inf')]
    dxy_bin_edges = [0.0, 0.5, 1.0, 2.0, 5.0, float('inf')]
    num_pt_bins = len(pt_bin_edges) - 1
    num_dxy_bins = len(dxy_bin_edges) - 1

    # Accumulators: [found, total] per bin
    pt_counts = [[0, 0] for _ in range(num_pt_bins)]
    dxy_counts = [[0, 0] for _ in range(num_dxy_bins)]
    grid_counts = [
        [[0, 0] for _ in range(num_dxy_bins)]
        for _ in range(num_pt_bins)
    ]

    for batch_index in range(batch_size):
        gt_positions = labels_flat[batch_index].nonzero(as_tuple=True)[0]
        if len(gt_positions) == 0:
            continue

        top_k_indices = sorted_indices[batch_index, :top_k]
        top_k_set = set(top_k_indices.cpu().tolist())
        features_event = raw_features[batch_index]  # (C, P)

        for gt_pos in gt_positions.cpu().tolist():
            pt_value = features_event[feature_index_pt, gt_pos].item()
            dxy_value = abs(
                features_event[feature_index_dxy_significance, gt_pos].item(),
            )
            found = 1 if gt_pos in top_k_set else 0

            # Find pT bin
            pt_bin = num_pt_bins - 1
            for bin_index in range(num_pt_bins):
                if pt_value < pt_bin_edges[bin_index + 1]:
                    pt_bin = bin_index
                    break

            # Find |dxy_sig| bin
            dxy_bin = num_dxy_bins - 1
            for bin_index in range(num_dxy_bins):
                if dxy_value < dxy_bin_edges[bin_index + 1]:
                    dxy_bin = bin_index
                    break

            pt_counts[pt_bin][0] += found
            pt_counts[pt_bin][1] += 1
            dxy_counts[dxy_bin][0] += found
            dxy_counts[dxy_bin][1] += 1
            grid_counts[pt_bin][dxy_bin][0] += found
            grid_counts[pt_bin][dxy_bin][1] += 1

    metrics = {}
    pt_labels = ['0_0.3', '0.3_0.5', '0.5_1', '1_2', '2+']
    for bin_index, label in enumerate(pt_labels):
        found, total = pt_counts[bin_index]
        metrics[f'recall_pt_{label}'] = found / max(1, total)
        metrics[f'count_pt_{label}'] = total

    dxy_labels = ['0_0.5', '0.5_1', '1_2', '2_5', '5+']
    for bin_index, label in enumerate(dxy_labels):
        found, total = dxy_counts[bin_index]
        metrics[f'recall_dxy_{label}'] = found / max(1, total)
        metrics[f'count_dxy_{label}'] = total

    for pt_index, pt_label in enumerate(pt_labels):
        for dxy_index, dxy_label in enumerate(dxy_labels):
            found, total = grid_counts[pt_index][dxy_index]
            metrics[f'recall_2d_pt{pt_label}_dxy{dxy_label}'] = (
                found / max(1, total)
            )
            metrics[f'count_2d_pt{pt_label}_dxy{dxy_label}'] = total

    return metrics
