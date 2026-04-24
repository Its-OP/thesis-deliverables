"""TDD tests for MetricsAccumulator — global metric computation across batches.

The problem: percentiles and per-event breakdowns cannot be averaged across
batches. A MetricsAccumulator collects raw data (GT ranks, per-event found
counts, scores) across all validation batches and computes global metrics
at the end.

Tests written BEFORE implementation.
"""
import pytest
import torch


class TestMetricsAccumulatorExists:
    """MetricsAccumulator should be importable from training_utils."""

    def test_importable(self):
        from utils.training_utils import MetricsAccumulator
        assert callable(MetricsAccumulator)


class TestMetricsAccumulatorBasicUsage:
    """Basic API: update() per batch, compute() at the end."""

    def test_update_and_compute(self):
        """update() accepts batch data, compute() returns final metrics dict."""
        from utils.training_utils import MetricsAccumulator

        accumulator = MetricsAccumulator(
            k_values=(10, 200),
        )

        # Single batch: 2 events, 20 tracks each, 3 GT per event
        scores = torch.zeros(2, 20)
        labels = torch.zeros(2, 1, 20)
        mask = torch.ones(2, 1, 20)
        labels[0, 0, 0] = 1; labels[0, 0, 1] = 1; labels[0, 0, 2] = 1
        labels[1, 0, 3] = 1; labels[1, 0, 4] = 1; labels[1, 0, 5] = 1
        scores[0, 0] = 10; scores[0, 1] = 9; scores[0, 2] = 8
        scores[1, 3] = 10; scores[1, 4] = 9; scores[1, 5] = 8

        accumulator.update(scores, labels, mask)
        metrics = accumulator.compute()

        assert 'recall_at_200' in metrics
        assert 'gt_rank_p90' in metrics
        assert 'total_events_with_gt' in metrics

    def test_multiple_updates(self):
        """Multiple update() calls should accumulate data."""
        from utils.training_utils import MetricsAccumulator

        accumulator = MetricsAccumulator(k_values=(10,))

        # Batch 1: 1 event
        scores1 = torch.tensor([[10.0, 5.0, 1.0]])
        labels1 = torch.tensor([[[1.0, 0.0, 0.0]]])
        mask1 = torch.ones(1, 1, 3)
        accumulator.update(scores1, labels1, mask1)

        # Batch 2: 1 event
        scores2 = torch.tensor([[1.0, 5.0, 10.0]])
        labels2 = torch.tensor([[[0.0, 0.0, 1.0]]])
        mask2 = torch.ones(1, 1, 3)
        accumulator.update(scores2, labels2, mask2)

        metrics = accumulator.compute()
        assert metrics['total_events_with_gt'] == 2


class TestGlobalPercentiles:
    """Percentiles must be computed from the GLOBAL rank distribution,
    not averaged across per-batch percentiles."""

    def test_global_percentiles_not_batch_averaged(self):
        """Two batches with different rank distributions.

        Batch 1: GT ranks = [0, 0, 0] (all top-ranked)
        Batch 2: GT ranks = [50, 50, 50] (all at rank 50)

        Global p90 of [0, 0, 0, 50, 50, 50] should be 50.
        Batch-averaged p90 = (0 + 50) / 2 = 25 — WRONG.
        """
        from utils.training_utils import MetricsAccumulator

        accumulator = MetricsAccumulator(k_values=(200,))

        # Batch 1: 1 event, 100 tracks, 3 GT at top
        scores1 = torch.zeros(1, 100)
        labels1 = torch.zeros(1, 1, 100)
        mask1 = torch.ones(1, 1, 100)
        labels1[0, 0, 0] = 1; labels1[0, 0, 1] = 1; labels1[0, 0, 2] = 1
        scores1[0, 0] = 100; scores1[0, 1] = 99; scores1[0, 2] = 98
        # Background scores: 0..96 (all below GT)
        scores1[0, 3:] = torch.linspace(0, 96, 97)
        accumulator.update(scores1, labels1, mask1)

        # Batch 2: 1 event, 100 tracks, 3 GT at rank ~50
        scores2 = torch.zeros(1, 100)
        labels2 = torch.zeros(1, 1, 100)
        mask2 = torch.ones(1, 1, 100)
        labels2[0, 0, 50] = 1; labels2[0, 0, 51] = 1; labels2[0, 0, 52] = 1
        # GT scores are moderate — ~50 background tracks score higher
        scores2[0, 50] = 40; scores2[0, 51] = 39; scores2[0, 52] = 38
        scores2[0, :50] = torch.linspace(100, 41, 50)  # 50 tracks above GT
        scores2[0, 53:] = torch.linspace(37, 0, 47)    # 47 tracks below GT
        accumulator.update(scores2, labels2, mask2)

        metrics = accumulator.compute()

        # Global ranks: [0, 1, 2, 50, 51, 52] (6 values)
        # p90 = value at ceil(0.9 * 6) - 1 = index 4 = 51
        assert metrics['gt_rank_p90'] >= 50, (
            f'p90={metrics["gt_rank_p90"]}, expected >= 50 (global, not batch-averaged)'
        )

    def test_p95_with_single_outlier(self):
        """p95 should reflect the worst-ranked GT track globally."""
        from utils.training_utils import MetricsAccumulator

        accumulator = MetricsAccumulator(k_values=(200,))

        # Batch 1: 1 event, GT at ranks 0, 1, 2 (top)
        scores1 = torch.zeros(1, 500)
        labels1 = torch.zeros(1, 1, 500)
        mask1 = torch.ones(1, 1, 500)
        labels1[0, 0, 0] = 1; labels1[0, 0, 1] = 1; labels1[0, 0, 2] = 1
        scores1[0, 0] = 100; scores1[0, 1] = 99; scores1[0, 2] = 98
        scores1[0, 3:] = torch.linspace(50, 0, 497)
        accumulator.update(scores1, labels1, mask1)

        # Batch 2: 1 event, one GT at rank ~400 (very bad)
        scores2 = torch.zeros(1, 500)
        labels2 = torch.zeros(1, 1, 500)
        mask2 = torch.ones(1, 1, 500)
        labels2[0, 0, 0] = 1; labels2[0, 0, 1] = 1; labels2[0, 0, 499] = 1
        scores2[0, 0] = 100; scores2[0, 1] = 99
        scores2[0, 499] = -100  # This GT track is ranked very low
        scores2[0, 2:499] = torch.linspace(50, -50, 497)
        accumulator.update(scores2, labels2, mask2)

        metrics = accumulator.compute()

        # One GT track should be at rank ~499. p95 must capture this.
        assert metrics['gt_rank_p95'] > 100, (
            f'p95={metrics["gt_rank_p95"]}, should be > 100 to catch the outlier'
        )


class TestGlobalBreakdown:
    """Per-event breakdown must count over ALL events, not average per batch."""

    def test_breakdown_counts_globally(self):
        """Batch 1: 3/3 found. Batch 2: 1/3 found.
        Global: found_3_of_3 = 0.5, found_1_of_3 = 0.5."""
        from utils.training_utils import MetricsAccumulator

        accumulator = MetricsAccumulator(k_values=(200,))

        # Batch 1: perfect event (3/3 in top-200)
        scores1 = torch.zeros(1, 20)
        labels1 = torch.zeros(1, 1, 20)
        mask1 = torch.ones(1, 1, 20)
        labels1[0, 0, 0] = 1; labels1[0, 0, 1] = 1; labels1[0, 0, 2] = 1
        scores1[0, 0] = 10; scores1[0, 1] = 9; scores1[0, 2] = 8
        accumulator.update(scores1, labels1, mask1)

        # Batch 2: poor event (only 1/3 in top-5 of 1000 tracks)
        scores2 = torch.zeros(1, 1000)
        labels2 = torch.zeros(1, 1, 1000)
        mask2 = torch.ones(1, 1, 1000)
        labels2[0, 0, 0] = 1; labels2[0, 0, 500] = 1; labels2[0, 0, 999] = 1
        scores2[0, 0] = 100
        scores2[0, 500] = -50; scores2[0, 999] = -100
        scores2[0, 1:500] = torch.linspace(50, -49, 499)
        scores2[0, 501:999] = torch.linspace(-51, -99, 498)
        accumulator.update(scores2, labels2, mask2)

        metrics = accumulator.compute()

        assert metrics.get('found_3_of_3_at_200') == pytest.approx(0.5)
        assert metrics.get('found_1_of_3_at_200') == pytest.approx(0.5)


class TestGlobalRecallConsistency:
    """R@K and d-prime should be consistent with single-batch computation."""

    def test_single_batch_matches_direct_call(self):
        """MetricsAccumulator with one batch should match compute_recall_at_k_metrics."""
        from utils.training_utils import MetricsAccumulator, compute_recall_at_k_metrics

        scores = torch.randn(4, 50)
        labels = torch.zeros(4, 1, 50)
        mask = torch.ones(4, 1, 50)
        for event_index in range(4):
            labels[event_index, 0, event_index] = 1
            labels[event_index, 0, event_index + 10] = 1
            scores[event_index, event_index] = 10
            scores[event_index, event_index + 10] = 5

        k_values = (5, 10, 200)

        # Direct call
        direct = compute_recall_at_k_metrics(scores, labels, mask, k_values)

        # Via accumulator (single batch)
        accumulator = MetricsAccumulator(k_values=k_values)
        accumulator.update(scores, labels, mask)
        accumulated = accumulator.compute()

        for key in ('recall_at_5', 'recall_at_10', 'recall_at_200',
                    'perfect_at_5', 'd_prime', 'median_gt_rank'):
            assert accumulated[key] == pytest.approx(direct[key], abs=1e-6), (
                f'{key}: accumulated={accumulated[key]}, direct={direct[key]}'
            )
