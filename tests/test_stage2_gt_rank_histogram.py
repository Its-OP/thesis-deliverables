from __future__ import annotations

import numpy as np

from scripts.python.stage2_gt_rank_histogram import (
    binding_gt_rank,
    classify_event,
    summarize_events,
)


def test_binding_gt_rank_is_second_gt_position_one_based():
    ordering = np.array([7, 3, 9, 1, 5], dtype=np.int64)
    assert binding_gt_rank(ordering, {3, 5}) == 5
    assert binding_gt_rank(ordering, {7, 9}) == 3
    assert binding_gt_rank(ordering, {7, 3, 9}) == 2


def test_binding_gt_rank_none_when_fewer_than_two_gt_in_ordering():
    ordering = np.array([7, 3, 9], dtype=np.int64)
    assert binding_gt_rank(ordering, {3, 42}) is None
    assert binding_gt_rank(ordering, {42, 43}) is None


def test_classify_event_hit_miss_and_bound_lost():
    ordering = np.arange(400, dtype=np.int64)
    hit = classify_event(ordering, {0, 1, 2}, k=200)
    assert hit['status'] == 'hit'
    assert hit['binding_rank'] == 2

    miss = classify_event(ordering, {0, 250, 300}, k=200)
    assert miss['status'] == 'achievable_miss'
    assert miss['binding_rank'] == 251

    lost = classify_event(ordering[:200], {0, 999, 998}, k=200)
    assert lost['status'] == 'bound_lost'
    assert lost['binding_rank'] is None


def test_summarize_events_counts_and_boundary_share():
    orderings = [np.arange(400, dtype=np.int64) for _ in range(4)]
    gt_sets = [
        {0, 1, 2},        # hit
        {0, 219, 300},    # achievable miss, binding 220 (<=260)
        {0, 350, 399},    # achievable miss, binding 351 (>260)
        {0, 998, 999},    # bound lost
    ]
    summary = summarize_events(orderings, gt_sets, k=200, boundary=260)
    assert summary['n_events'] == 4
    assert summary['n_hits'] == 1
    assert summary['n_achievable_misses'] == 2
    assert summary['n_bound_lost'] == 1
    assert summary['share_misses_within_boundary'] == 0.5
    assert summary['miss_binding_ranks'] == [220, 351]
