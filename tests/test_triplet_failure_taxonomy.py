from __future__ import annotations

import numpy as np

from scripts.python.triplet_failure_taxonomy import (
    best_two_overlap_rank, classify_event, deduped_order,
    diversity_capped_gt_rank, distinct_couples_in_top, gt_couple_stage3_rank)


def test_deduped_order_keeps_first_occurrence_in_score_order():
    keys = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9], [4, 5, 6]])
    order = deduped_order(keys)
    assert order.tolist() == [0, 1, 3]


def test_gt_couple_stage3_rank_takes_best_producing_couple():
    is_gt = np.array([False, True, False, True])
    couple_rank = np.array([0, 17, 3, 5])
    assert gt_couple_stage3_rank(is_gt, couple_rank) == 5


def test_gt_couple_stage3_rank_none_when_gt_absent():
    assert gt_couple_stage3_rank(np.zeros(3, dtype=bool),
                                 np.array([0, 1, 2])) is None


def test_best_two_overlap_rank_ignores_gt_itself_and_one_overlaps():
    gt_triple = np.array([10, 20, 30])
    keys = np.array([
        [10, 20, 30],   # the GT triplet itself (3 shared) — not a confusion
        [10, 40, 50],   # 1 shared — irrelevant
        [10, 20, 55],   # 2 shared — first genuine near-clone, rank 3
        [20, 30, 60],   # 2 shared — later
    ])
    assert best_two_overlap_rank(keys, gt_triple) == 3


def test_best_two_overlap_rank_none_without_two_share():
    keys = np.array([[1, 2, 3], [4, 5, 6]])
    assert best_two_overlap_rank(keys, np.array([7, 8, 9])) is None


def test_distinct_couples_in_top_counts_unique_ids():
    couple_ids = np.array([4, 4, 7, 4, 9, 11])
    assert distinct_couples_in_top(couple_ids, 4) == 2
    assert distinct_couples_in_top(couple_ids, 6) == 4


def test_diversity_cap_promotes_gt_buried_under_one_couple():
    # 11 candidates of couple 0 ahead of the GT (couple 1): plain rank 12,
    # capped at 2 per couple the GT surfaces at rank 3.
    couple_ids = np.array([0] * 11 + [1])
    is_gt = np.array([False] * 11 + [True])
    assert diversity_capped_gt_rank(couple_ids, is_gt, cap=None) == 12
    assert diversity_capped_gt_rank(couple_ids, is_gt, cap=2) == 3


def test_diversity_cap_can_drop_the_gt_itself():
    couple_ids = np.array([0, 0, 0])
    is_gt = np.array([False, False, True])
    assert diversity_capped_gt_rank(couple_ids, is_gt, cap=2) is None


def test_classify_event_priorities():
    assert classify_event(gt_rank=None, gt_couple_rank=None,
                          two_overlap_rank=None, n_couples_top10=5) == 'absent'
    assert classify_event(gt_rank=7, gt_couple_rank=0,
                          two_overlap_rank=1, n_couples_top10=1) == 'hit'
    assert classify_event(gt_rank=40, gt_couple_rank=0,
                          two_overlap_rank=4, n_couples_top10=2) \
        == 'third_pion_confusion'
    assert classify_event(gt_rank=15, gt_couple_rank=0,
                          two_overlap_rank=20, n_couples_top10=2) == 'near_miss'
    assert classify_event(gt_rank=200, gt_couple_rank=80,
                          two_overlap_rank=None, n_couples_top10=8) \
        == 'couple_sunk'
    assert classify_event(gt_rank=150, gt_couple_rank=2,
                          two_overlap_rank=None, n_couples_top10=8) == 'far_tail'
    assert classify_event(gt_rank=45, gt_couple_rank=2,
                          two_overlap_rank=None, n_couples_top10=3) \
        == 'crowded_out'
    assert classify_event(gt_rank=45, gt_couple_rank=2,
                          two_overlap_rank=None, n_couples_top10=8) == 'mid_other'
