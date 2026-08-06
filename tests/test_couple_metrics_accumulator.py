"""Unit tests for ``CoupleMetricsAccumulator`` (D@K_tracks, C@K_couples, RC@K_couples)."""
from __future__ import annotations

import pytest
import torch

from utils.metrics import (
    CoupleMetricsAccumulator,
    format_couple_metrics_table,
)


def _accumulator(
    k_values_couples=(50, 100, 200),
    k_values_tracks=(30, 50, 75, 100, 200),
):
    return CoupleMetricsAccumulator(
        k_values_couples=k_values_couples,
        k_values_tracks=k_values_tracks,
    )


# ---------------------------------------------------------------------------
# C@K_couples basics
# ---------------------------------------------------------------------------

class TestCCouples:
    def test_perfect_ranking_recall_at_50(self):
        accumulator = _accumulator(k_values_couples=(50, 100, 200))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['c_at_100_couples'] == 1.0
        assert metrics['c_at_200_couples'] == 1.0
        assert metrics['eligible_events'] == 1

    def test_gt_at_rank_75_only_in_top_100_and_200(self):
        accumulator = _accumulator(k_values_couples=(50, 75, 100, 200))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 75] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 0.0
        assert metrics['c_at_75_couples'] == 0.0
        assert metrics['c_at_100_couples'] == 1.0
        assert metrics['c_at_200_couples'] == 1.0

    def test_event_with_no_gt_excluded_from_c(self):
        accumulator = _accumulator(k_values_couples=(50, 100))
        scores = torch.randn(1, 200)
        labels = torch.zeros(1, 200)
        mask = torch.ones(1, 200)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 0
        assert metrics['c_at_50_couples'] == 0.0

    def test_multiple_gt_couples_do_not_inflate_metric(self):
        """If an event has 3 GT couples and 2 are in top-50, C@50 should
        still contribute exactly 1.0 (binary per event)."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 5] = 1.0
        labels[0, 10] = 1.0
        labels[0, 80] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0


# ---------------------------------------------------------------------------
# RC@K_couples
# ---------------------------------------------------------------------------

class TestRcCouples:
    def test_rc_requires_full_triplet_in_top_k1(self):
        accumulator = _accumulator(k_values_couples=(50, 100))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([2])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.0
        assert metrics['rc_at_100_couples'] == 0.0

    def test_rc_equals_c_when_full_triplet_present(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 1.0

    def test_rc_zero_when_couple_not_in_top_k(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 100] = 1.0
        mask = torch.ones(1, n_couples)
        n_gt_in_top_k1 = torch.tensor([3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 0.0
        assert metrics['rc_at_50_couples'] == 0.0

    def test_rc_omitted_when_n_gt_in_top_k1_is_none(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.0
        assert metrics['events_with_full_triplet'] == 0


# ---------------------------------------------------------------------------
# D@K_tracks
# ---------------------------------------------------------------------------

class TestDTracks:
    def test_d_zero_with_no_n_gt_in_top_k_tracks(self):
        accumulator = _accumulator(k_values_tracks=(30, 50))
        scores = torch.zeros(2, 100)
        labels = torch.zeros(2, 100)
        mask = torch.ones(2, 100)
        # No n_gt_in_top_k_tracks → D@K stays at 0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['d_at_30_tracks'] == 0.0
        assert metrics['d_at_50_tracks'] == 0.0
        # total_events should still be incremented (denominator works)
        assert metrics['total_events'] == 2

    def test_d_per_event_threshold_is_geq_2(self):
        """D@K = events with at least 2 GT in top-K tracks. 2 → 1, 1 → 0, 3 → 1."""
        accumulator = _accumulator(k_values_tracks=(50,))
        n_couples = 100
        scores = torch.zeros(3, n_couples)
        labels = torch.zeros(3, n_couples)
        mask = torch.ones(3, n_couples)
        # n_gt_in_top_50_tracks per event:
        # event 0: 1 (below threshold)
        # event 1: 2 (at threshold → counts)
        # event 2: 3 (above threshold → counts)
        n_gt_in_top_k_tracks = torch.tensor([[1], [2], [3]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        # 2 of 3 events have ≥2 GT pions in top-50 tracks
        assert abs(metrics['d_at_50_tracks'] - 2.0 / 3.0) < 1e-6
        assert metrics['total_events'] == 3

    def test_d_at_multiple_k_values(self):
        accumulator = _accumulator(k_values_tracks=(30, 50, 100))
        n_couples = 100
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)
        mask = torch.ones(2, n_couples)
        # event 0: 1 GT in top-30, 2 in top-50, 3 in top-100
        # event 1: 0 GT in top-30, 1 in top-50, 2 in top-100
        n_gt_in_top_k_tracks = torch.tensor([
            [1, 2, 3],
            [0, 1, 2],
        ])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        # D@30: 0/2 events have ≥2 GT in top-30
        assert metrics['d_at_30_tracks'] == 0.0
        # D@50: 1/2 events (event 0) have ≥2 GT in top-50
        assert metrics['d_at_50_tracks'] == 0.5
        # D@100: 2/2 events have ≥2 GT in top-100
        assert metrics['d_at_100_tracks'] == 1.0

    def test_d_denominator_is_total_events_not_eligible(self):
        """D's denominator counts ALL events (it's a property of the
        cascade, independent of whether the event has a GT couple in the
        candidate pool)."""
        accumulator = _accumulator(k_values_tracks=(50,), k_values_couples=(50,))
        n_couples = 100
        # Two events: neither has any GT couple (so eligible_events = 0)
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)  # no GT couples
        mask = torch.ones(2, n_couples)
        # But both events have ≥2 GT pions in top-50 tracks
        n_gt_in_top_k_tracks = torch.tensor([[3], [3]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 0  # no eligible events for C/RC
        assert metrics['total_events'] == 2     # but D's denominator is full
        assert metrics['d_at_50_tracks'] == 1.0
        assert metrics['c_at_50_couples'] == 0.0   # no eligible → 0 (max(1, 0))
        assert metrics['rc_at_50_couples'] == 0.0


# ---------------------------------------------------------------------------
# Padding mask handling
# ---------------------------------------------------------------------------

class TestPaddingMask:
    def test_padded_couples_pushed_to_bottom_of_ranking(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 100:] = 1000.0
        scores[0, 0] = 1.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.zeros(1, n_couples)
        mask[0, :100] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0


# ---------------------------------------------------------------------------
# Multi-event averaging
# ---------------------------------------------------------------------------

class TestMultiEvent:
    def test_average_uses_total_events_not_eligible(self):
        """Denominator for C/RC is total_events (4), not eligible (3).

        Setup: 4 events. Events 0, 1 have GT in top-50; event 2 has GT
        at rank 101 (outside top-50); event 3 has no GT couple at all.
        Numerator for C@50 is 2 (events 0, 1). With denom = 4,
        C@50 = 2/4 = 0.5. Same for RC@50 since all eligible events
        have a full triplet in K1.
        """
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(4, n_couples)
        labels = torch.zeros(4, n_couples)
        mask = torch.ones(4, n_couples)
        scores[0, 0] = 100.0
        labels[0, 0] = 1.0
        scores[1, 0] = 100.0
        labels[1, 0] = 1.0
        scores[2] = torch.linspace(1.0, 0.0, n_couples)
        labels[2, 100] = 1.0
        # Event 3: no GT couple
        n_gt_in_top_k1 = torch.tensor([3, 3, 3, 3])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['total_events'] == 4
        assert metrics['eligible_events'] == 3
        assert metrics['events_with_full_triplet'] == 3
        # 2 of 4 total events have a GT couple in top-50 of the reranker
        assert metrics['c_at_50_couples'] == 0.5
        assert metrics['rc_at_50_couples'] == 0.5

    def test_rc_smaller_than_c_when_some_have_partial_triplet(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(4, n_couples)
        scores[:, 0] = 100.0
        labels = torch.zeros(4, n_couples)
        labels[:, 0] = 1.0
        mask = torch.ones(4, n_couples)
        n_gt_in_top_k1 = torch.tensor([3, 3, 3, 2])
        accumulator.update(scores, labels, mask, n_gt_in_top_k1=n_gt_in_top_k1)
        metrics = accumulator.compute()
        assert metrics['c_at_50_couples'] == 1.0
        assert metrics['rc_at_50_couples'] == 0.75
        assert metrics['events_with_full_triplet'] == 3


# ---------------------------------------------------------------------------
# Cross-batch accumulation
# ---------------------------------------------------------------------------

class TestCrossBatch:
    def test_accumulates_across_multiple_update_calls(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores_1 = torch.zeros(1, n_couples)
        scores_1[0, 0] = 100.0
        labels_1 = torch.zeros(1, n_couples)
        labels_1[0, 0] = 1.0
        accumulator.update(
            scores_1, labels_1, torch.ones(1, n_couples),
            n_gt_in_top_k1=torch.tensor([3]),
        )
        scores_2 = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels_2 = torch.zeros(1, n_couples)
        labels_2[0, 100] = 1.0
        accumulator.update(
            scores_2, labels_2, torch.ones(1, n_couples),
            n_gt_in_top_k1=torch.tensor([3]),
        )
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 2
        assert metrics['c_at_50_couples'] == 0.5
        assert metrics['rc_at_50_couples'] == 0.5


# ---------------------------------------------------------------------------
# Mean rank of first GT couple
# ---------------------------------------------------------------------------

class TestMeanFirstGtRankCouples:
    """Mean rank of the highest-scoring GT couple per event.

    Convention: 1-indexed (rank 1 = best), averaged over eligible events
    only (events with ≥1 GT couple in the candidate pool). Lower is better.
    """

    def test_perfect_ranking_yields_rank_one(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        scores[0, 0] = 100.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['mean_first_gt_rank_couples'] == 1.0

    def test_gt_at_descending_position_75_yields_rank_76(self):
        accumulator = _accumulator(k_values_couples=(50, 100))
        n_couples = 200
        # scores strictly descending: index 0 is rank 1, index 75 is rank 76
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 75] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['mean_first_gt_rank_couples'] == 76.0

    def test_multi_gt_uses_best_rank(self):
        """If an event has 3 GT couples at sorted positions 5, 10, 80, the
        mean should reflect rank 6 (the best of the three), not the worst
        and not the mean within the event."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels = torch.zeros(1, n_couples)
        labels[0, 5] = 1.0
        labels[0, 10] = 1.0
        labels[0, 80] = 1.0
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['mean_first_gt_rank_couples'] == 6.0

    def test_multi_event_averaging(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        # Three events with first GT at sorted ranks 1, 11, 51 → mean = 21
        scores = torch.zeros(3, n_couples)
        labels = torch.zeros(3, n_couples)
        mask = torch.ones(3, n_couples)
        # Event 0: GT at descending rank 1 (highest score)
        scores[0] = torch.linspace(1.0, 0.0, n_couples)
        labels[0, 0] = 1.0
        # Event 1: GT at descending rank 11 (index 10)
        scores[1] = torch.linspace(1.0, 0.0, n_couples)
        labels[1, 10] = 1.0
        # Event 2: GT at descending rank 51 (index 50)
        scores[2] = torch.linspace(1.0, 0.0, n_couples)
        labels[2, 50] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        # (1 + 11 + 51) / 3 = 21
        assert abs(metrics['mean_first_gt_rank_couples'] - 21.0) < 1e-6
        assert metrics['eligible_events'] == 3

    def test_event_with_no_gt_excluded_from_mean(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)
        mask = torch.ones(2, n_couples)
        # Event 0 has GT at rank 1, event 1 has none
        scores[0, 0] = 100.0
        labels[0, 0] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        # Only event 0 is eligible → mean = 1.0 (event 1 ignored)
        assert metrics['eligible_events'] == 1
        assert metrics['mean_first_gt_rank_couples'] == 1.0

    def test_zero_eligible_events_yields_zero(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        labels = torch.zeros(1, n_couples)
        mask = torch.ones(1, n_couples)
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 0
        assert metrics['mean_first_gt_rank_couples'] == 0.0

    def test_padded_couples_do_not_inflate_rank(self):
        """Padded positions sit at the bottom of the ranking and must not
        push the GT couple to a worse rank."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(1, n_couples)
        # Padded couples have huge scores but must be masked out
        scores[0, 100:] = 1000.0
        scores[0, 0] = 1.0
        labels = torch.zeros(1, n_couples)
        labels[0, 0] = 1.0
        mask = torch.zeros(1, n_couples)
        mask[0, :100] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        # Ranking restricted to valid: GT is the only positive score → rank 1
        assert metrics['mean_first_gt_rank_couples'] == 1.0

    def test_accumulates_across_batches(self):
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        # Batch 1: GT at rank 1
        scores_1 = torch.zeros(1, n_couples)
        scores_1[0, 0] = 100.0
        labels_1 = torch.zeros(1, n_couples)
        labels_1[0, 0] = 1.0
        accumulator.update(scores_1, labels_1, torch.ones(1, n_couples))
        # Batch 2: GT at rank 11
        scores_2 = torch.linspace(1.0, 0.0, n_couples).unsqueeze(0)
        labels_2 = torch.zeros(1, n_couples)
        labels_2[0, 10] = 1.0
        accumulator.update(scores_2, labels_2, torch.ones(1, n_couples))
        metrics = accumulator.compute()
        # (1 + 11) / 2 = 6
        assert metrics['eligible_events'] == 2
        assert metrics['mean_first_gt_rank_couples'] == 6.0


# ---------------------------------------------------------------------------
# Denominator invariant: RC ≤ C ≤ D@K_tracks when K_tracks ≥ K2
# ---------------------------------------------------------------------------

class TestDenominatorInvariant:
    """Regression tests for the metric-comparability invariant.

    For any event with a GT couple in the candidate pool (eligible),
    BOTH GT pions of that couple must already be in the cascade's
    top-K2 selection. So when ``K_tracks ≥ K2``, the eligible-events
    set is a subset of the D@K_tracks-positive set, and:

        RC@K_couples ≤ C@K_couples ≤ D@K_tracks

    must hold structurally. This requires C, RC, and D to share the
    same denominator (= total_events). The original implementation
    used ``eligible_events`` for C/RC and ``total_events`` for D,
    which produced a paradox where RC > D was reported.
    """

    def test_rc_does_not_exceed_d_at_same_k(self):
        """4 events. 3 are eligible AND have ≥2 GT pions in top-100t
        AND have full triplet in K1 AND the reranker puts the GT couple
        at the top. The 4th event has nothing.

        Old behavior:  c=rc=3/3=1.000, d=3/4=0.750  →  rc > d (paradox)
        New behavior:  c=rc=3/4=0.750, d=3/4=0.750  →  rc ≤ c ≤ d ✓
        """
        accumulator = _accumulator(
            k_values_couples=(100,),
            k_values_tracks=(100,),
        )
        n_couples = 200
        scores = torch.zeros(4, n_couples)
        labels = torch.zeros(4, n_couples)
        mask = torch.ones(4, n_couples)
        # Events 0-2: GT couple at the top of the score list
        scores[:3, 0] = 100.0
        labels[:3, 0] = 1.0
        # Event 3: no GT couple
        n_gt_in_top_k1 = torch.tensor([3, 3, 3, 0])
        n_gt_in_top_k_tracks = torch.tensor([[3], [3], [3], [0]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k1=n_gt_in_top_k1,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        # All four events count toward the denominator
        assert metrics['total_events'] == 4
        assert metrics['eligible_events'] == 3
        # RC ≤ C ≤ D (all = 0.75)
        assert metrics['d_at_100_tracks'] == 0.75
        assert metrics['c_at_100_couples'] == 0.75
        assert metrics['rc_at_100_couples'] == 0.75
        assert (
            metrics['rc_at_100_couples']
            <= metrics['c_at_100_couples']
            <= metrics['d_at_100_tracks']
        )

    def test_c_uses_total_events_denominator(self):
        """Two events: one eligible with GT at rank 1, one ineligible.
        C@50 should be 1/2, not 1/1."""
        accumulator = _accumulator(
            k_values_couples=(50,),
            k_values_tracks=(50,),
        )
        n_couples = 200
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)
        mask = torch.ones(2, n_couples)
        scores[0, 0] = 100.0
        labels[0, 0] = 1.0
        n_gt_in_top_k_tracks = torch.tensor([[3], [0]])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
        )
        metrics = accumulator.compute()
        assert metrics['total_events'] == 2
        assert metrics['eligible_events'] == 1
        assert metrics['c_at_50_couples'] == 0.5  # 1 / 2, not 1 / 1
        assert metrics['d_at_50_tracks'] == 0.5

    def test_rc_uses_total_events_denominator(self):
        """3 events: 2 eligible with full triplet + GT couple at top,
        1 ineligible. RC@50 = 2/3, not 2/2."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(3, n_couples)
        labels = torch.zeros(3, n_couples)
        mask = torch.ones(3, n_couples)
        scores[:2, 0] = 100.0
        labels[:2, 0] = 1.0
        n_gt_in_top_k1 = torch.tensor([3, 3, 0])
        accumulator.update(
            scores, labels, mask,
            n_gt_in_top_k1=n_gt_in_top_k1,
        )
        metrics = accumulator.compute()
        assert metrics['total_events'] == 3
        assert metrics['eligible_events'] == 2
        assert abs(metrics['c_at_50_couples'] - 2.0 / 3.0) < 1e-6
        assert abs(metrics['rc_at_50_couples'] - 2.0 / 3.0) < 1e-6

    def test_mean_rank_still_uses_eligible_denominator(self):
        """The mean rank metric is undefined for events with no GT
        couple, so it must continue to use eligible_events as its
        denominator (events without GT contribute nothing)."""
        accumulator = _accumulator(k_values_couples=(50,))
        n_couples = 200
        scores = torch.zeros(2, n_couples)
        labels = torch.zeros(2, n_couples)
        mask = torch.ones(2, n_couples)
        # Event 0: GT at rank 1; Event 1: no GT couple
        scores[0, 0] = 100.0
        labels[0, 0] = 1.0
        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()
        assert metrics['eligible_events'] == 1
        # Mean rank averages over the 1 eligible event, not 2
        assert metrics['mean_first_gt_rank_couples'] == 1.0


# ---------------------------------------------------------------------------
# Vectorized update vs the per-event reference loop
# ---------------------------------------------------------------------------

def oracle_update(
    accumulator: CoupleMetricsAccumulator,
    couple_scores: torch.Tensor,
    couple_labels: torch.Tensor,
    couple_mask: torch.Tensor,
    n_gt_in_top_k1: torch.Tensor | None = None,
    n_gt_in_top_k_tracks: torch.Tensor | None = None,
) -> None:
    """couple_scores, couple_labels, couple_mask: (B, n_couples).
    Verbatim copy of the pre-vectorization ``update`` loop, mutating
    ``accumulator`` state exactly as it did."""
    batch_size = couple_scores.shape[0]

    if n_gt_in_top_k_tracks is not None:
        for batch_index in range(batch_size):
            accumulator.total_events_count += 1
            for k_index, k in enumerate(accumulator.k_values_tracks):
                n_gt_at_k = n_gt_in_top_k_tracks[batch_index, k_index].item()
                if n_gt_at_k >= accumulator.duplet_threshold:
                    accumulator.d_sums[k] += 1.0
    else:
        accumulator.total_events_count += batch_size

    for batch_index in range(batch_size):
        valid_mask = couple_mask[batch_index] > 0.5
        if not valid_mask.any():
            continue
        gt_mask = (couple_labels[batch_index] > 0.5) & valid_mask
        if not gt_mask.any():
            continue

        event_scores = couple_scores[batch_index].clone()
        event_scores = event_scores.masked_fill(~valid_mask, float('-inf'))
        sorted_indices = torch.argsort(event_scores, descending=True)
        sorted_gt = gt_mask[sorted_indices]

        accumulator.eligible_events_count += 1
        first_gt_position = int(sorted_gt.float().argmax().item())
        accumulator.first_gt_rank_sum += float(first_gt_position + 1)

        full_triplet_present = False
        if n_gt_in_top_k1 is not None:
            full_triplet_present = bool(
                n_gt_in_top_k1[batch_index].item()
                >= accumulator.full_triplet_threshold
            )
            if full_triplet_present:
                accumulator.events_with_full_triplet_count += 1

        for k in accumulator.k_values_couples:
            if bool(sorted_gt[:k].any().item()):
                accumulator.c_sums[k] += 1.0
                if full_triplet_present:
                    accumulator.rc_sums[k] += 1.0


class TestVectorizedUpdateMatchesLoop:
    def test_randomized_batches_match_reference(self):
        torch.manual_seed(1234)
        k_couples = (5, 20, 50)
        k_tracks = (10, 30, 50)
        vectorized = CoupleMetricsAccumulator(
            k_values_couples=k_couples, k_values_tracks=k_tracks,
        )
        reference = CoupleMetricsAccumulator(
            k_values_couples=k_couples, k_values_tracks=k_tracks,
        )
        # Four batches spanning the argument grid: (n_gt_in_top_k1 present,
        # n_gt_in_top_k_tracks present) in all combinations, plus events with
        # no valid couples, no GT couples, and a guaranteed eligible event.
        argument_grid = [(True, True), (True, False), (False, False),
                         (False, True)]
        for with_top_k1, with_top_k_tracks in argument_grid:
            batch_size, n_couples = 9, 60
            scores = torch.randn(batch_size, n_couples)
            labels = (torch.rand(batch_size, n_couples) < 0.06).float()
            mask = (torch.rand(batch_size, n_couples) < 0.8).float()
            mask[0] = 0.0                        # no valid couples
            labels[1] = 0.0                      # no GT couples
            labels[2, 5] = 1.0                   # guaranteed eligible event
            mask[2, 5] = 1.0
            n_gt_in_top_k1 = (
                torch.randint(0, 4, (batch_size,)).float()
                if with_top_k1 else None
            )
            n_gt_in_top_k_tracks = (
                torch.randint(0, 4, (batch_size, len(k_tracks)))
                if with_top_k_tracks else None
            )
            vectorized.update(
                scores, labels, mask,
                n_gt_in_top_k1=n_gt_in_top_k1,
                n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
            )
            oracle_update(
                reference, scores, labels, mask,
                n_gt_in_top_k1=n_gt_in_top_k1,
                n_gt_in_top_k_tracks=n_gt_in_top_k_tracks,
            )

        vectorized_metrics = vectorized.compute()
        reference_metrics = reference.compute()
        assert set(vectorized_metrics) == set(reference_metrics)
        for key, reference_value in reference_metrics.items():
            assert vectorized_metrics[key] == pytest.approx(
                reference_value, abs=1e-9,
            ), f'metric {key} diverged from the reference loop'


# ---------------------------------------------------------------------------
# Validation log table formatter
# ---------------------------------------------------------------------------

def _example_metrics() -> dict:
    """Realistic val_metrics dict matching one epoch of the actual log."""
    return {
        'd_at_30_tracks': 0.8145,
        'd_at_50_tracks': 0.8500,
        'd_at_75_tracks': 0.8845,
        'd_at_100_tracks': 0.9050,
        'd_at_200_tracks': 0.9555,
        'c_at_50_couples': 0.9341,
        'c_at_75_couples': 0.9606,
        'c_at_100_couples': 0.9753,
        'c_at_200_couples': 0.9935,
        'rc_at_50_couples': 0.8982,
        'rc_at_75_couples': 0.9217,
        'rc_at_100_couples': 0.9353,
        'rc_at_200_couples': 0.9511,
        'mean_first_gt_rank_couples': 12.3,
        'eligible_events': 1699,
        'total_events': 1700,
        'events_with_full_triplet': 1625,
    }


class TestFormatCoupleMetricsTable:
    def test_returns_multiline_string(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        assert isinstance(table, str)
        assert table.count('\n') >= 5  # header + table body + footer

    def test_header_contains_train_and_val_loss(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        assert '0.08874' in table  # train loss
        assert '0.08194' in table  # val loss
        assert 'Epoch 22' in table

    def test_best_marker_when_is_best(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        assert 'best' in table.lower()
        assert '0.9753' in table  # best criterion value

    def test_non_best_shows_epochs_since_best(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=25,
            is_best=False,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        # 25 - 22 = 3 epochs ago
        assert '3' in table
        assert 'best' in table.lower()

    def test_table_includes_union_of_k_values(self):
        """K=30 only has D; K=50/75/100/200 have all three. The K
        column must include 30 even though C and RC are missing."""
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        for k in (30, 50, 75, 100, 200):
            assert str(k) in table, f'K={k} missing from table'

    def test_missing_metric_renders_as_dash(self):
        """K=30 has no C/RC value — those cells must render as a
        dash, not as 0.0000 (which would be a wrong claim)."""
        metrics = _example_metrics()
        # Make sure K=30 has only D
        assert 'c_at_30_couples' not in metrics
        assert 'rc_at_30_couples' not in metrics
        table = format_couple_metrics_table(
            metrics,
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        # Find the row for K=30
        line_for_30 = next(
            (line for line in table.split('\n') if line.lstrip().startswith('|') and ' 30 ' in line),
            None,
        )
        assert line_for_30 is not None
        assert '-' in line_for_30  # placeholder for missing C/RC

    def test_all_d_c_rc_values_appear_in_table(self):
        metrics = _example_metrics()
        table = format_couple_metrics_table(
            metrics,
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        for value_text in (
            '0.8145', '0.8500', '0.8845', '0.9050', '0.9555',  # D
            '0.9341', '0.9606', '0.9753', '0.9935',            # C
            '0.8982', '0.9217', '0.9353', '0.9511',            # RC
        ):
            assert value_text in table, f'value {value_text} missing'

    def test_footer_contains_mean_rank_eligible_full_triplet(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        assert '12.3' in table              # mean rank
        assert '1699' in table              # eligible
        assert '1700' in table              # total
        assert '1625' in table              # full triplet

    def test_table_has_column_headers(self):
        table = format_couple_metrics_table(
            _example_metrics(),
            train_loss=0.08874,
            val_loss=0.08194,
            epoch=22,
            is_best=True,
            best_val_criterion=0.9753,
            best_val_epoch=22,
        )
        assert 'D@K_tracks' in table
        assert 'C@K_couples' in table
        assert 'RC@K_couples' in table

    def test_zero_total_events_does_not_crash(self):
        empty = {
            'd_at_50_tracks': 0.0,
            'c_at_50_couples': 0.0,
            'rc_at_50_couples': 0.0,
            'mean_first_gt_rank_couples': 0.0,
            'eligible_events': 0,
            'total_events': 0,
            'events_with_full_triplet': 0,
        }
        # Must not raise
        table = format_couple_metrics_table(
            empty,
            train_loss=0.0,
            val_loss=0.0,
            epoch=1,
            is_best=False,
            best_val_criterion=0.0,
            best_val_epoch=0,
            k_values_tracks=(50,),
            k_values_couples=(50,),
        )
        assert isinstance(table, str)
