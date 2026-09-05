from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.python.compute_eval_metrics import (  # noqa: E402
    IDENTITY_COLUMNS,
    build_gt_lookup,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('stage3_miss_anatomy')


def first_gt_couple_rank(sorted_couples, gt) -> int | None:
    """sorted_couples: (N, 2) track indices in rank order; gt: set of GT
    track indices. Returns the 1-based rank of the first couple whose both
    members are GT, or None when no GT couple appears in the list."""
    for rank, (first, second) in enumerate(sorted_couples, start=1):
        if first in gt and second in gt:
            return rank
    return None


def top_k_composition(sorted_couples, gt, *, k: int) -> dict:
    """sorted_couples: (N, 2); gt: set of GT track indices. Describes the
    top-k couples: how many are GT / sibling (exactly one GT member) /
    unrelated, the rank of the first sibling, how many distinct GT tracks
    appear anywhere in the top-k, and whether the most frequent track in
    the top-k is a GT track."""
    n_gt = 0
    n_sibling = 0
    n_unrelated = 0
    first_sibling_rank: int | None = None
    gt_tracks_present: set[int] = set()
    track_counter: Counter = Counter()
    for rank, (first, second) in enumerate(sorted_couples[:k], start=1):
        members_in_gt = (first in gt) + (second in gt)
        if members_in_gt == 2:
            n_gt += 1
        elif members_in_gt == 1:
            n_sibling += 1
            if first_sibling_rank is None:
                first_sibling_rank = rank
        else:
            n_unrelated += 1
        for member in (first, second):
            track_counter[member] += 1
            if member in gt:
                gt_tracks_present.add(member)
    most_frequent_is_gt = False
    if track_counter:
        most_frequent_track, _ = track_counter.most_common(1)[0]
        most_frequent_is_gt = most_frequent_track in gt
    return {
        'n_gt': n_gt,
        'n_sibling': n_sibling,
        'n_unrelated': n_unrelated,
        'first_sibling_rank': first_sibling_rank,
        'n_gt_tracks_present': len(gt_tracks_present),
        'most_frequent_is_gt': most_frequent_is_gt,
    }


def binding_pool_rank(pool_ordering, gt) -> int | None:
    """pool_ordering: (K2,) track indices in stage-2 rank order; gt: set of
    GT track indices. Returns the 1-based pool rank of the SECOND GT track
    (the rank at which a GT couple becomes constructible), or None when
    fewer than two GT tracks are in the pool."""
    ranks = [rank for rank, track in enumerate(pool_ordering, start=1)
             if track in gt]
    if len(ranks) < 2:
        return None
    return ranks[1]


def classify_event(sorted_couples, gt, *, n_gt_in_pool: int, k: int) -> dict:
    """n_gt_in_pool: GT tracks inside the K2 pool the couples were built
    from. Fewer than two means no GT couple was constructible."""
    if n_gt_in_pool < 2:
        return {'status': 'bound_lost', 'gt_couple_rank': None}
    rank = first_gt_couple_rank(sorted_couples, gt)
    if rank is not None and rank <= k:
        return {'status': 'hit', 'gt_couple_rank': rank}
    return {'status': 'achievable_miss', 'gt_couple_rank': rank}


def _rank_bucket(rank: int | None, k: int, list_length: int) -> str:
    if rank is None:
        return f'>{list_length}'
    if rank <= 2 * k:
        return f'({k},{2 * k}]'
    return f'({2 * k},{list_length}]'


def summarize_events(sorted_couple_lists, gt_sets, n_gt_in_pool_values,
                     *, k: int, pool_orderings=None) -> dict:
    """pool_orderings: optional per-event (K2,) stage-2 pool orderings; when
    given, hits and misses also report where the second GT track sits in
    the pool (stage-2 marginality)."""
    counts = Counter()
    miss_rank_buckets: Counter = Counter()
    miss_ranks: list[int] = []
    miss_compositions: list[dict] = []
    hit_compositions: list[dict] = []
    list_length = 0
    if pool_orderings is None:
        pool_orderings = [None] * len(gt_sets)
    for sorted_couples, gt, n_gt_in_pool, pool_ordering in zip(
            sorted_couple_lists, gt_sets, n_gt_in_pool_values,
            pool_orderings):
        list_length = max(list_length, len(sorted_couples))
        record = classify_event(sorted_couples, gt,
                                n_gt_in_pool=n_gt_in_pool, k=k)
        counts[record['status']] += 1
        if record['status'] == 'bound_lost':
            continue
        composition = top_k_composition(sorted_couples, gt, k=k)
        composition['binding_pool_rank'] = (
            binding_pool_rank(pool_ordering, gt)
            if pool_ordering is not None else None)
        if record['status'] == 'hit':
            hit_compositions.append(composition)
        else:
            miss_compositions.append(composition)
            miss_rank_buckets[_rank_bucket(
                record['gt_couple_rank'], k, len(sorted_couples))] += 1
            if record['gt_couple_rank'] is not None:
                miss_ranks.append(record['gt_couple_rank'])

    def aggregate(compositions: list[dict]) -> dict:
        n = len(compositions)
        if n == 0:
            return {'n': 0}
        sibling_shares = [c['n_sibling'] / k for c in compositions]
        gt_present = Counter(c['n_gt_tracks_present'] for c in compositions)
        first_sibling = [c['first_sibling_rank'] for c in compositions
                         if c['first_sibling_rank'] is not None]
        pool_ranks = [c['binding_pool_rank'] for c in compositions
                      if c['binding_pool_rank'] is not None]
        aggregated = {
            'n': n,
            'mean_sibling_share_top_k': float(np.mean(sibling_shares)),
            'share_events_sibling_majority': float(np.mean(
                [share >= 0.5 for share in sibling_shares])),
            'share_events_any_sibling': float(np.mean(
                [c['n_sibling'] > 0 for c in compositions])),
            'gt_tracks_present_in_top_k': {
                str(count): gt_present.get(count, 0) / n
                for count in range(4)},
            'share_most_frequent_track_is_gt': float(np.mean(
                [c['most_frequent_is_gt'] for c in compositions])),
            'first_sibling_rank_p50': (
                float(np.percentile(first_sibling, 50))
                if first_sibling else None),
        }
        if pool_ranks:
            aggregated['binding_pool_rank_quantiles'] = {
                'p25': float(np.percentile(pool_ranks, 25)),
                'p50': float(np.percentile(pool_ranks, 50)),
                'p75': float(np.percentile(pool_ranks, 75)),
            }
        return aggregated

    n_events = sum(counts.values())
    summary = {
        'k': k,
        'n_events': n_events,
        'n_hits': counts['hit'],
        'n_achievable_misses': counts['achievable_miss'],
        'n_bound_lost': counts['bound_lost'],
        'c_at_k': counts['hit'] / n_events if n_events else 0.0,
        'miss_rank_buckets': dict(miss_rank_buckets),
        'misses': aggregate(miss_compositions),
        'hits': aggregate(hit_compositions),
    }
    if miss_ranks:
        summary['miss_rank_quantiles'] = {
            'p50': float(np.percentile(miss_ranks, 50)),
            'p75': float(np.percentile(miss_ranks, 75)),
            'p90': float(np.percentile(miss_ranks, 90)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--couples-parquet', type=str, required=True,
                        help='score_couples_from_dump / --stage couples '
                             'output (stage3_sorted_couples)')
    parser.add_argument('--val-data-dir', type=str, required=True,
                        help='raw source shards for GT labels')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--k2', type=int, default=125,
                        help='stage-3 track pool = top-k2 of '
                             'stage2_sorted_indices')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    shards = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
    gt_lookup = build_gt_lookup(shards)

    table = pq.read_table(
        args.couples_parquet,
        columns=list(IDENTITY_COLUMNS)
        + ['stage2_sorted_indices', 'stage3_sorted_couples'])
    identity_arrays = [table[name].to_numpy(zero_copy_only=False)
                       for name in IDENTITY_COLUMNS]
    pool_column = table['stage2_sorted_indices']
    couples_column = table['stage3_sorted_couples']

    sorted_couple_lists = []
    gt_sets = []
    n_gt_in_pool_values = []
    pool_orderings = []
    n_missing_gt = 0
    for row in range(table.num_rows):
        key = tuple(int(array[row]) for array in identity_arrays)
        gt = gt_lookup.get(key)
        if gt is None:
            n_missing_gt += 1
            continue
        pool = np.asarray(pool_column[row].values, dtype=np.int64)[:args.k2]
        n_gt_in_pool_values.append(int(np.isin(pool, list(gt)).sum()))
        pool_orderings.append(pool.tolist())
        couples = couples_column[row].as_py()
        sorted_couple_lists.append([(int(a), int(b)) for a, b in couples])
        gt_sets.append(gt)
        if (row + 1) % 10000 == 0:
            logger.info(f'{row + 1} / {table.num_rows} rows read')

    summary = summarize_events(sorted_couple_lists, gt_sets,
                               n_gt_in_pool_values, k=args.k,
                               pool_orderings=pool_orderings)
    summary['n_skipped_missing_gt'] = n_missing_gt
    summary['input'] = args.couples_parquet
    with open(args.output, 'w') as handle:
        json.dump(summary, handle, indent=2)
    logger.info(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
