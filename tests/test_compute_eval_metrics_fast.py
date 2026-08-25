from __future__ import annotations

import numpy as np

from scripts.python.compute_eval_metrics import gt_ranks_in_ordering, metrics_from_ranks


def test_gt_ranks_finds_positions_in_ordering():
    ordering = np.array([7, 3, 9, 1, 4])
    ranks = gt_ranks_in_ordering(ordering, frozenset({9, 4, 7}))
    assert ranks.tolist() == [0, 2, 4]


def test_gt_ranks_ignores_gt_missing_from_ordering():
    ranks = gt_ranks_in_ordering(np.array([5, 6]), frozenset({6, 99}))
    assert ranks.tolist() == [1]


def test_metrics_from_ranks_recall_perfect_double():
    # GT ranks 0, 2, 4 with 3 GT tracks total.
    ranks = np.array([0, 2, 4])
    out = metrics_from_ranks(ranks, n_gt=3, k_values=(1, 3, 5),
                             with_double=True)
    assert out['recall_at_K'] == {1: 1 / 3, 3: 2 / 3, 5: 1.0}
    assert out['perfect_at_K'] == {1: 0.0, 3: 0.0, 5: 1.0}
    assert out['double_at_K'] == {1: 0.0, 3: 1.0, 5: 1.0}


def test_metrics_from_ranks_gt_absent_everywhere():
    out = metrics_from_ranks(np.array([], dtype=np.int64), n_gt=3,
                             k_values=(10,), with_double=True)
    assert out['recall_at_K'][10] == 0.0
    assert out['perfect_at_K'][10] == 0.0
    assert out['double_at_K'][10] == 0.0
