from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from scripts.python.sweep_triplet_filter import (
    ATTRIBUTION_BLOCKS,
    BASE_ESTIMATORS,
    REFINEMENT_GRID,
    arm_name,
    block_permutation_importance,
    enumerate_arms,
)
from utils.triplet_join import (
    H6_ISOLATION_NAMES,
    H6_PHYSICS_NAMES,
    H6_SV_NAMES,
    H6_VERTEX_NAMES,
)


def test_arm_count_is_the_full_cross_product():
    arms = enumerate_arms(['full89', 'all22'], ['uniform', 'hard'],
                          ['balanced', None])
    assert len(arms) == 2 * 2 * 2 * len(BASE_ESTIMATORS)


def test_refinement_arms_extend_the_grid_on_the_richest_feature_set():
    base = enumerate_arms(['full89', 'all22'], ['uniform'], ['balanced'])
    refined = enumerate_arms(['full89', 'all22'], ['uniform'], ['balanced'],
                             refine=True)
    expected = 1
    for values in REFINEMENT_GRID.values():
        expected *= len(values)
    assert len(refined) == len(base) + expected
    assert all(arm['feature_set'] == 'all22'
               for arm in refined[len(base):])


def test_arm_names_are_unique_and_filesystem_safe():
    arms = enumerate_arms(['full89', 'all22'], ['uniform', 'hard'],
                          ['balanced', None], refine=True)
    names = [arm_name(arm) for arm in arms]
    assert len(set(names)) == len(names)
    assert all(set(name) <= set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.')
        for name in names)


def test_none_class_weight_is_rendered_explicitly():
    arm = enumerate_arms(['full89'], ['uniform'], [None])[0]
    assert '__none__' in arm_name(arm)


def test_attribution_blocks_cover_the_h6_groups_without_overlap():
    assert set(ATTRIBUTION_BLOCKS) == {
        'vertex', 'physics', 'isolation', 'secondary_vertex'}
    covered = sum((ATTRIBUTION_BLOCKS[name] for name in ATTRIBUTION_BLOCKS), [])
    assert sorted(covered) == sorted(
        H6_VERTEX_NAMES + H6_PHYSICS_NAMES + H6_ISOLATION_NAMES + H6_SV_NAMES)
    assert len(set(covered)) == len(covered)


class _SignalModel:
    """Scores on one column only, so exactly one block should matter."""

    def __init__(self, column):
        self.column = column

    def predict_proba(self, features):
        scores = features[:, self.column]
        return np.stack([1.0 - scores, scores], axis=1)


def _attribution_table(names, generator):
    columns = {name: generator.random(200).astype(np.float32) for name in names}
    return pa.table(columns)


def test_block_permutation_credits_only_the_block_the_model_uses():
    generator = np.random.default_rng(0)
    names = H6_VERTEX_NAMES + H6_SV_NAMES
    table = _attribution_table(names, generator)
    signal_column = names.index(H6_VERTEX_NAMES[0])
    values = np.asarray(table.column(names[signal_column]))
    labels = (values > np.median(values)).astype(float)

    importances = block_permutation_importance(
        _SignalModel(signal_column), table, names, labels,
        {'vertex': H6_VERTEX_NAMES, 'secondary_vertex': H6_SV_NAMES})
    assert importances['vertex'] > 0.1
    assert importances['secondary_vertex'] == pytest.approx(0.0, abs=1e-9)


def test_block_permutation_skips_blocks_absent_from_the_feature_set():
    generator = np.random.default_rng(0)
    names = list(H6_VERTEX_NAMES)
    table = _attribution_table(names, generator)
    labels = (np.asarray(table.column(names[0])) > 0.5).astype(float)
    importances = block_permutation_importance(
        _SignalModel(0), table, names, labels, ATTRIBUTION_BLOCKS)
    assert set(importances) == {'vertex'}
