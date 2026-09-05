from __future__ import annotations

from scripts.python.stage3_miss_anatomy import (
    binding_pool_rank,
    classify_event,
    first_gt_couple_rank,
    summarize_events,
    top_k_composition,
)


def test_binding_pool_rank_is_second_gt_rank():
    assert binding_pool_rank([9, 1, 8, 3, 2], {1, 2, 3}) == 4
    assert binding_pool_rank([9, 1, 8], {1, 2, 3}) is None


def test_summarize_events_reports_pool_rank_quantiles():
    hit = [(1, 2), (4, 5)]
    miss = [(4, 5), (6, 7), (1, 2)]
    summary = summarize_events(
        [hit, miss], [{1, 2, 3}, {1, 2, 3}], [3, 3], k=2,
        pool_orderings=[[1, 2, 3, 9], [9, 8, 1, 2]],
    )
    assert summary['hits']['binding_pool_rank_quantiles']['p50'] == 2.0
    assert summary['misses']['binding_pool_rank_quantiles']['p50'] == 4.0

GT = {1, 2, 3}


def test_first_gt_couple_rank_finds_first_full_gt_pair():
    couples = [(1, 9), (4, 5), (2, 3), (1, 2)]
    assert first_gt_couple_rank(couples, GT) == 3


def test_first_gt_couple_rank_none_when_absent():
    assert first_gt_couple_rank([(1, 9), (4, 5)], GT) is None


def test_top_k_composition_counts_siblings_and_gt_presence():
    couples = [(1, 9), (4, 5), (2, 8), (1, 2), (6, 7)]
    composition = top_k_composition(couples, GT, k=4)
    assert composition['n_gt'] == 1
    assert composition['n_sibling'] == 2
    assert composition['n_unrelated'] == 1
    assert composition['first_sibling_rank'] == 1
    assert composition['n_gt_tracks_present'] == 2
    # track 1 appears twice in the top-4, more than any other track
    assert composition['most_frequent_is_gt'] is True


def test_classify_event_statuses():
    couples = [(1, 9), (4, 5), (2, 3)]
    assert classify_event(couples, GT, n_gt_in_pool=3, k=3)['status'] == 'hit'
    miss = classify_event(couples, GT, n_gt_in_pool=3, k=2)
    assert miss['status'] == 'achievable_miss'
    assert miss['gt_couple_rank'] == 3
    lost = classify_event(couples, GT, n_gt_in_pool=1, k=3)
    assert lost['status'] == 'bound_lost'
    assert lost['gt_couple_rank'] is None


def test_summarize_events_buckets_and_shares():
    hit = [(1, 2), (4, 5)]
    near_miss = [(1, 9), (2, 8), (4, 5), (1, 3)]      # GT couple at rank 4
    deep_miss = [(4, 5), (6, 7), (8, 9), (10, 11)]    # no GT couple listed
    lost = [(4, 5), (6, 7)]
    summary = summarize_events(
        [hit, near_miss, deep_miss, lost],
        [GT, GT, GT, GT],
        [3, 3, 3, 1],
        k=2,
    )
    assert summary['n_events'] == 4
    assert summary['n_hits'] == 1
    assert summary['n_achievable_misses'] == 2
    assert summary['n_bound_lost'] == 1
    assert summary['c_at_k'] == 0.25
    assert summary['miss_rank_buckets'] == {'(2,4]': 1, '>4': 1}
    misses = summary['misses']
    assert misses['n'] == 2
    # near_miss top-2 = two siblings; deep_miss top-2 = none
    assert misses['mean_sibling_share_top_k'] == 0.5
    assert misses['share_events_sibling_majority'] == 0.5
    assert misses['gt_tracks_present_in_top_k'] == {
        '0': 0.5, '1': 0.0, '2': 0.5, '3': 0.0}
    assert summary['miss_rank_quantiles']['p50'] == 4.0
