from __future__ import annotations

import numpy as np
import pytest

from scripts.python.eval_triplet_filter_ranking import (
    K_VALUES,
    deduped_gt_rank,
    feature_columns_for_width,
    use_cpu_inference,
)
from scripts.python.probe_triplet_features import FEATURE_SETS
from utils.triplet_join import FEATURE_NAMES_EXTENDED


def test_each_trained_width_maps_back_to_its_feature_set():
    for name, names in FEATURE_SETS.items():
        resolved, columns, h6_width = feature_columns_for_width(len(names))
        assert resolved == name
        assert [FEATURE_NAMES_EXTENDED[index] for index in columns] == names
        assert h6_width == max(len(names) - 89, 0)


def test_only_the_cone_dependent_sets_request_the_cone_block():
    from utils.triplet_join import H6_NAMES
    cone_start = H6_NAMES.index('n_low_ip_in_cone')
    widths = {name: feature_columns_for_width(len(names))[2]
              for name, names in FEATURE_SETS.items()}
    assert widths['full89'] == 0
    assert widths['vertex'] <= cone_start      # vertex block alone, no cone
    assert widths['vertex_physics'] > cone_start
    assert widths['all22'] > cone_start


def test_an_unknown_width_is_rejected():
    with pytest.raises(ValueError, match='no feature set'):
        feature_columns_for_width(77)


def test_selected_columns_are_positions_in_the_extended_layout():
    _, columns, _ = feature_columns_for_width(len(FEATURE_SETS['vertex']))
    assert len(columns) == 95
    assert columns.max() < len(FEATURE_NAMES_EXTENDED)


class _Booster:
    def __init__(self):
        self.params = {}

    def set_param(self, params):
        self.params.update(params)


class _GpuModel:
    def __init__(self):
        self._booster = _Booster()
        self.device = 'cuda'

    def get_booster(self):
        return self._booster

    def set_params(self, **kwargs):
        self.device = kwargs.get('device', self.device)


def test_gpu_models_are_switched_to_cpu_for_forked_inference():
    model = _GpuModel()
    use_cpu_inference(model)
    assert model.device == 'cpu'
    assert model.get_booster().params['device'] == 'cpu'


def test_estimators_without_a_device_are_left_alone():
    class _Plain:
        pass

    plain = _Plain()
    assert use_cpu_inference(plain) is plain


def test_rank_is_zero_when_the_truth_scores_highest():
    scores = np.array([0.9, 0.5, 0.1])
    is_gt = np.array([True, False, False])
    triplets = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert deduped_gt_rank(scores, is_gt, triplets) == 0


def test_rank_counts_only_higher_scoring_candidates():
    scores = np.array([0.1, 0.9, 0.5])
    is_gt = np.array([True, False, False])
    triplets = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    assert deduped_gt_rank(scores, is_gt, triplets) == 2


def test_duplicate_three_sets_are_collapsed_before_ranking():
    # The same 3-set reached through three couples must not push the truth down
    # by three places.
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    is_gt = np.array([False, False, False, True])
    triplets = np.array([[5, 6, 7], [6, 7, 5], [7, 5, 6], [0, 1, 2]])
    assert deduped_gt_rank(scores, is_gt, triplets) == 1


def test_a_duplicated_truth_takes_its_best_position():
    scores = np.array([0.9, 0.4, 0.8])
    is_gt = np.array([True, True, False])
    triplets = np.array([[0, 1, 2], [2, 1, 0], [3, 4, 5]])
    assert deduped_gt_rank(scores, is_gt, triplets) == 0


def test_missing_truth_returns_none():
    scores = np.array([0.9, 0.5])
    is_gt = np.array([False, False])
    triplets = np.array([[0, 1, 2], [0, 1, 3]])
    assert deduped_gt_rank(scores, is_gt, triplets) is None


def test_ties_keep_a_stable_order():
    scores = np.full(4, 0.5)
    is_gt = np.array([False, False, True, False])
    triplets = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]])
    assert deduped_gt_rank(scores, is_gt, triplets) == 2


def test_k_values_cover_the_published_reference_points():
    assert 10 in K_VALUES and 100 in K_VALUES
    assert list(K_VALUES) == sorted(K_VALUES)


@pytest.mark.parametrize('position', range(4))
def test_rank_matches_a_brute_force_count(position):
    generator = np.random.default_rng(position)
    scores = generator.random(40)
    is_gt = np.zeros(40, dtype=bool)
    is_gt[position] = True
    triplets = np.arange(120).reshape(40, 3)
    expected = int((scores > scores[position]).sum())
    assert deduped_gt_rank(scores, is_gt, triplets) == expected
