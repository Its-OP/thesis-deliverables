from __future__ import annotations

import numpy as np
import pytest

from scripts.python.probe_triplet_features import (
    FEATURE_SETS,
    grouped_folds,
    hit_at_k,
    roc_auc,
)
from utils.triplet_join import FEATURE_NAMES, FEATURE_NAMES_EXTENDED


def test_feature_sets_are_cumulative_and_end_at_the_extended_layout():
    assert FEATURE_SETS['full89'] == FEATURE_NAMES
    assert sorted(FEATURE_SETS['all22']) == sorted(FEATURE_NAMES_EXTENDED)
    for smaller, larger in (('full89', 'vertex'),
                            ('vertex', 'vertex_physics'),
                            ('vertex_physics', 'all22')):
        assert FEATURE_SETS[smaller] == \
            FEATURE_SETS[larger][:len(FEATURE_SETS[smaller])]


def test_roc_auc_is_one_for_a_perfect_ranking():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_is_zero_for_a_perfectly_inverted_ranking():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(scores, labels) == pytest.approx(0.0)


def test_roc_auc_averages_ties_to_one_half():
    scores = np.full(6, 0.5)
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_matches_the_rank_sum_definition_on_random_data():
    generator = np.random.default_rng(0)
    scores = generator.normal(size=400)
    labels = (generator.random(400) < 0.2).astype(float)
    positive, negative = scores[labels > 0.5], scores[labels < 0.5]
    expected = np.mean([(p > negative).mean() + 0.5 * (p == negative).mean()
                        for p in positive])
    assert roc_auc(scores, labels) == pytest.approx(expected, abs=1e-9)


def test_hit_at_k_counts_events_whose_truth_lands_in_the_window():
    event_index = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # Event 0: truth is top ranked. Event 1: truth is last.
    scores = np.array([0.9, 0.1, 0.2, 0.1, 0.8, 0.7])
    assert hit_at_k(scores, labels, event_index, k=1) == pytest.approx(0.5)
    assert hit_at_k(scores, labels, event_index, k=3) == pytest.approx(1.0)


def test_hit_at_k_ignores_events_without_a_ground_truth_row():
    event_index = np.array([0, 0, 1, 1])
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    scores = np.array([0.9, 0.1, 0.5, 0.4])
    assert hit_at_k(scores, labels, event_index, k=1) == pytest.approx(1.0)


def test_hit_at_k_uses_the_best_ground_truth_row_when_several_exist():
    event_index = np.array([0, 0, 0])
    labels = np.array([1.0, 1.0, 0.0])
    scores = np.array([0.1, 0.9, 0.5])
    assert hit_at_k(scores, labels, event_index, k=1) == pytest.approx(1.0)


def test_grouped_folds_keep_every_event_whole():
    event_index = np.repeat(np.arange(9), 4)
    folds = grouped_folds(event_index, 3)
    assert sum(len(fold) for fold in folds) == len(event_index)
    for fold in folds:
        events_in_fold = set(event_index[fold].tolist())
        other_rows = np.setdiff1d(np.arange(len(event_index)), fold)
        assert not events_in_fold & set(event_index[other_rows].tolist())


def test_grouped_folds_are_balanced_when_events_divide_evenly():
    event_index = np.repeat(np.arange(9), 4)
    assert [len(fold) for fold in grouped_folds(event_index, 3)] == [12, 12, 12]
