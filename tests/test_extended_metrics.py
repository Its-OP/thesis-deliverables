"""TDD tests for Phase 0: extended validation metrics.

Tests written BEFORE implementation. Each test specifies the expected
behavior for new metrics functions:
    0.1  compute_recall_at_k_metrics — extended K values, percentiles, breakdown
    0.2  compute_conditional_recall — per-bin recall by pT and dxy_significance
    0.3  save_epoch_metrics — JSON export per epoch
"""
import json
import os
import tempfile

import pytest
import torch

from utils.training_utils import compute_recall_at_k_metrics


# ---------------------------------------------------------------------------
# Helpers: build controlled test events
# ---------------------------------------------------------------------------

def make_event(
    num_tracks: int,
    gt_indices: list[int],
    gt_scores: list[float],
    background_score_range: tuple[float, float] = (-1.0, 0.0),
):
    """Create a single-event batch with controlled scores.

    Args:
        num_tracks: Total tracks including GT.
        gt_indices: Positions of GT tracks (0-indexed).
        gt_scores: Scores to assign to GT tracks (descending = better).
        background_score_range: (low, high) uniform range for background.

    Returns:
        (scores, labels, mask) tensors, each with batch dim = 1.
    """
    scores = torch.zeros(1, num_tracks)
    labels = torch.zeros(1, 1, num_tracks)
    mask = torch.ones(1, 1, num_tracks)

    # Background scores: linearly spaced in range
    low, high = background_score_range
    scores[0] = torch.linspace(low, high, num_tracks)

    for gt_idx, gt_score in zip(gt_indices, gt_scores):
        scores[0, gt_idx] = gt_score
        labels[0, 0, gt_idx] = 1.0

    return scores, labels, mask


# ===========================================================================
# 0.1 Tests: compute_recall_at_k_metrics — extended features
# ===========================================================================


class TestExtendedKValues:
    """Verify that extended K values (300, 400, 500, 600, 800) are supported."""

    def test_extended_k_values_present_in_output(self):
        """Passing K=500 should produce recall_at_500 and perfect_at_500."""
        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0, 1, 2], gt_scores=[10, 9, 8],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask,
            k_values=(100, 200, 300, 400, 500, 600, 800),
        )
        for k in (100, 200, 300, 400, 500, 600, 800):
            assert f'recall_at_{k}' in metrics, f'Missing recall_at_{k}'
            assert f'perfect_at_{k}' in metrics, f'Missing perfect_at_{k}'

    def test_recall_at_large_k_is_one_when_all_gt_found(self):
        """With 20 tracks and K=500, all GT should be found (R@500 = 1.0)."""
        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0, 5, 10], gt_scores=[10, 9, 8],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(500,),
        )
        assert metrics['recall_at_500'] == pytest.approx(1.0)


class TestGTRankPercentiles:
    """Verify GT rank percentiles (p75, p90, p95) are computed."""

    def test_percentile_keys_present(self):
        """Output should contain gt_rank_p75, gt_rank_p90, gt_rank_p95."""
        scores, labels, mask = make_event(
            num_tracks=100, gt_indices=[0, 1, 2], gt_scores=[10, 9, 8],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(10,),
        )
        assert 'gt_rank_p75' in metrics
        assert 'gt_rank_p90' in metrics
        assert 'gt_rank_p95' in metrics

    def test_percentiles_are_ordered(self):
        """p75 <= p90 <= p95 (higher percentile = worse rank)."""
        scores, labels, mask = make_event(
            num_tracks=100, gt_indices=[0, 50, 90], gt_scores=[10, 0.1, -0.9],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(10,),
        )
        assert metrics['gt_rank_p75'] <= metrics['gt_rank_p90']
        assert metrics['gt_rank_p90'] <= metrics['gt_rank_p95']

    def test_perfect_ranking_percentiles(self):
        """When all 3 GT are top-3, all percentiles should be <= 2."""
        scores, labels, mask = make_event(
            num_tracks=100, gt_indices=[0, 1, 2], gt_scores=[10, 9, 8],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(10,),
        )
        assert metrics['gt_rank_p95'] <= 2.0

    def test_percentiles_with_no_gt_are_inf(self):
        """Events with no GT should return inf percentiles."""
        scores = torch.randn(1, 20)
        labels = torch.zeros(1, 1, 20)
        mask = torch.ones(1, 1, 20)
        metrics = compute_recall_at_k_metrics(scores, labels, mask, k_values=(10,))
        assert metrics['gt_rank_p75'] == float('inf')
        assert metrics['gt_rank_p90'] == float('inf')
        assert metrics['gt_rank_p95'] == float('inf')


class TestTotalEventsWithGT:
    """Verify total_events_with_gt is reported."""

    def test_total_events_with_gt_present(self):
        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0], gt_scores=[10],
        )
        metrics = compute_recall_at_k_metrics(scores, labels, mask, k_values=(10,))
        assert 'total_events_with_gt' in metrics
        assert metrics['total_events_with_gt'] == 1

    def test_total_events_with_gt_excludes_empty(self):
        """Batch of 2: one event with GT, one without."""
        scores = torch.randn(2, 20)
        labels = torch.zeros(2, 1, 20)
        mask = torch.ones(2, 1, 20)
        labels[0, 0, 5] = 1.0  # Only first event has GT
        scores[0, 5] = 100.0
        metrics = compute_recall_at_k_metrics(scores, labels, mask, k_values=(10,))
        assert metrics['total_events_with_gt'] == 1


class TestPerEventBreakdown:
    """Verify per-event breakdown (found_N_of_M_at_200) is computed."""

    def test_perfect_event_breakdown(self):
        """All 3 GT in top-200 -> found_3_of_3_at_200 = 1.0."""
        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0, 1, 2], gt_scores=[10, 9, 8],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(200,),
        )
        assert metrics.get('found_3_of_3_at_200', 0) == pytest.approx(1.0)

    def test_partial_event_breakdown(self):
        """2 of 3 GT in top-3 of 1000 tracks -> found_2_of_3_at_200."""
        scores = torch.zeros(1, 1000)
        labels = torch.zeros(1, 1, 1000)
        mask = torch.ones(1, 1, 1000)
        # GT at positions 0, 500, 999
        labels[0, 0, 0] = 1; labels[0, 0, 500] = 1; labels[0, 0, 999] = 1
        # Only first two GT score high; third is at rank > 200
        scores[0, 0] = 10; scores[0, 500] = 9; scores[0, 999] = -100
        # Fill some background with moderate scores to push rank of GT#3 > 200
        scores[0, 1:300] = torch.linspace(5, 0, 299)

        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(200,),
        )
        assert 'found_2_of_3_at_200' in metrics
        assert metrics['found_2_of_3_at_200'] == pytest.approx(1.0)

    def test_breakdown_not_computed_without_k200(self):
        """Breakdown should only appear when 200 is in k_values."""
        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0], gt_scores=[10],
        )
        metrics = compute_recall_at_k_metrics(
            scores, labels, mask, k_values=(10, 50),
        )
        breakdown_keys = [k for k in metrics if 'found_' in k]
        assert len(breakdown_keys) == 0


# ===========================================================================
# 0.2 Tests: compute_conditional_recall
# ===========================================================================


class TestConditionalRecall:
    """Tests for compute_conditional_recall() — per-bin recall by pT and dxy."""

    def test_function_exists_and_importable(self):
        """compute_conditional_recall should be importable."""
        from utils.training_utils import compute_conditional_recall
        assert callable(compute_conditional_recall)

    def test_returns_pt_bin_metrics(self):
        """Should return recall_pt_* and count_pt_* for each pT bin."""
        from utils.training_utils import compute_conditional_recall

        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0, 1, 2], gt_scores=[10, 9, 8],
        )
        features = torch.randn(1, 16, 20)
        features[0, 0, 0] = 0.4   # pT in [0.3, 0.5) bin
        features[0, 0, 1] = 1.5   # pT in [1.0, 2.0) bin
        features[0, 0, 2] = 3.0   # pT in [2.0, inf) bin

        metrics = compute_conditional_recall(
            scores, labels, mask, features,
            feature_index_pt=0, feature_index_dxy_significance=6,
        )
        assert 'recall_pt_0.3_0.5' in metrics
        assert 'recall_pt_1_2' in metrics
        assert 'recall_pt_2+' in metrics
        assert 'count_pt_0.3_0.5' in metrics

    def test_returns_dxy_bin_metrics(self):
        """Should return recall_dxy_* and count_dxy_* for each |dxy| bin."""
        from utils.training_utils import compute_conditional_recall

        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0], gt_scores=[10],
        )
        features = torch.randn(1, 16, 20)
        features[0, 6, 0] = 0.3   # |dxy_sig| in [0, 0.5) bin

        metrics = compute_conditional_recall(
            scores, labels, mask, features,
            feature_index_pt=0, feature_index_dxy_significance=6,
        )
        assert 'recall_dxy_0_0.5' in metrics
        assert 'count_dxy_0_0.5' in metrics

    def test_returns_2d_grid_metrics(self):
        """Should return recall_2d_pt*_dxy* for the full 5x5 grid."""
        from utils.training_utils import compute_conditional_recall

        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0], gt_scores=[10],
        )
        features = torch.randn(1, 16, 20)
        metrics = compute_conditional_recall(
            scores, labels, mask, features,
        )
        # 5 pT bins * 5 dxy bins * 2 (recall + count) = 50
        # Plus 5 pt recall + 5 pt count + 5 dxy recall + 5 dxy count = 20
        # Total = 70
        assert len(metrics) == 70

    def test_found_track_has_recall_one(self):
        """A GT track scored highest should have recall 1.0 in its bin."""
        from utils.training_utils import compute_conditional_recall

        scores, labels, mask = make_event(
            num_tracks=20, gt_indices=[0], gt_scores=[100],
        )
        features = torch.zeros(1, 16, 20)
        features[0, 0, 0] = 1.5   # pT bin [1, 2)
        features[0, 6, 0] = 3.0   # |dxy| bin [2, 5)

        metrics = compute_conditional_recall(
            scores, labels, mask, features,
            feature_index_pt=0, feature_index_dxy_significance=6,
            top_k=200,
        )
        assert metrics['recall_pt_1_2'] == pytest.approx(1.0)
        assert metrics['recall_dxy_2_5'] == pytest.approx(1.0)
        assert metrics['count_pt_1_2'] == 1


# ===========================================================================
# 0.3 Tests: save_epoch_metrics
# ===========================================================================


class TestSaveEpochMetrics:
    """Tests for save_epoch_metrics() — JSON export."""

    def test_function_exists_and_importable(self):
        """save_epoch_metrics should be importable."""
        from utils.training_utils import save_epoch_metrics
        assert callable(save_epoch_metrics)

    def test_creates_json_file(self):
        """Should create metrics/epoch_N.json in experiment directory."""
        from utils.training_utils import save_epoch_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_epoch_metrics({'recall_at_200': 0.629}, tmpdir, 5)
            assert os.path.exists(path)
            assert path.endswith('epoch_5.json')

    def test_json_content_matches_input(self):
        """Written JSON should contain the exact metrics passed."""
        from utils.training_utils import save_epoch_metrics

        input_metrics = {
            'recall_at_200': 0.629,
            'perfect_at_200': 0.368,
            'd_prime': 1.29,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_epoch_metrics(input_metrics, tmpdir, 1)
            with open(path) as file_handle:
                loaded = json.load(file_handle)
            assert loaded['recall_at_200'] == pytest.approx(0.629)
            assert loaded['d_prime'] == pytest.approx(1.29)

    def test_creates_metrics_subdirectory(self):
        """Should create the metrics/ subdirectory if it doesn't exist."""
        from utils.training_utils import save_epoch_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            save_epoch_metrics({}, tmpdir, 1)
            assert os.path.isdir(os.path.join(tmpdir, 'metrics'))
