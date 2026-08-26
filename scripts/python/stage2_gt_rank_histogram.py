from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.python.compute_eval_metrics import (  # noqa: E402
    IDENTITY_COLUMNS,
    build_gt_lookup,
    gt_ranks_in_ordering,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger('stage2_gt_rank_histogram')

HISTOGRAM_BIN_WIDTH = 20


def binding_gt_rank(ordering: np.ndarray, gt) -> int | None:
    """ordering: (N,) track indices in rank order; gt: set of GT track
    indices. Returns the 1-based rank of the SECOND GT track in the
    ordering (the rank that decides D@K), or None when fewer than two GT
    tracks appear in the ordering."""
    ranks = gt_ranks_in_ordering(ordering, frozenset(gt))
    if ranks.size < 2:
        return None
    return int(ranks[1]) + 1


def classify_event(ordering: np.ndarray, gt, *, k: int) -> dict:
    rank = binding_gt_rank(ordering, gt)
    if rank is None:
        return {'status': 'bound_lost', 'binding_rank': None}
    if rank <= k:
        return {'status': 'hit', 'binding_rank': rank}
    return {'status': 'achievable_miss', 'binding_rank': rank}


def summarize_events(orderings, gt_sets, *, k: int, boundary: int) -> dict:
    n_hits = 0
    n_bound_lost = 0
    miss_binding_ranks: list[int] = []
    for ordering, gt in zip(orderings, gt_sets):
        record = classify_event(ordering, gt, k=k)
        if record['status'] == 'hit':
            n_hits += 1
        elif record['status'] == 'bound_lost':
            n_bound_lost += 1
        else:
            miss_binding_ranks.append(record['binding_rank'])

    n_misses = len(miss_binding_ranks)
    within_boundary = sum(1 for rank in miss_binding_ranks if rank <= boundary)
    return {
        'n_events': n_hits + n_bound_lost + n_misses,
        'n_hits': n_hits,
        'n_achievable_misses': n_misses,
        'n_bound_lost': n_bound_lost,
        'share_misses_within_boundary': (
            within_boundary / n_misses if n_misses else 0.0
        ),
        'miss_binding_ranks': sorted(miss_binding_ranks),
    }


def _histogram(ranks: list[int], k: int, max_rank: int) -> dict[str, int]:
    edges = list(range(k, max_rank + HISTOGRAM_BIN_WIDTH, HISTOGRAM_BIN_WIDTH))
    counts: dict[str, int] = {}
    for low, high in zip(edges[:-1], edges[1:]):
        counts[f'({low},{high}]'] = sum(1 for r in ranks if low < r <= high)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-parquet', type=str, required=True,
                        help='eval_cascade_pipeline dump parquet')
    parser.add_argument('--val-data-dir', type=str, required=True,
                        help='raw source shards for GT labels')
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--boundary', type=int, default=260)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    import glob
    shards = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
    gt_lookup = build_gt_lookup(shards)

    table = pq.read_table(
        args.eval_parquet,
        columns=list(IDENTITY_COLUMNS) + ['stage2_sorted_indices'])
    identity_arrays = [table[name].to_numpy(zero_copy_only=False)
                       for name in IDENTITY_COLUMNS]
    orderings_column = table['stage2_sorted_indices']

    orderings = []
    gt_sets = []
    n_missing_gt = 0
    for row in range(table.num_rows):
        key = tuple(int(array[row]) for array in identity_arrays)
        gt = gt_lookup.get(key)
        if gt is None or len(gt) < 2:
            n_missing_gt += 1
            continue
        orderings.append(np.asarray(orderings_column[row].values,
                                    dtype=np.int64))
        gt_sets.append(gt)

    summary = summarize_events(orderings, gt_sets,
                               k=args.k, boundary=args.boundary)
    ranks = summary.pop('miss_binding_ranks')
    max_rank = max(ranks) if ranks else args.k
    summary['n_skipped_missing_gt'] = n_missing_gt
    summary['histogram'] = _histogram(ranks, args.k, max_rank)
    if ranks:
        summary['miss_rank_quantiles'] = {
            'p50': float(np.percentile(ranks, 50)),
            'p75': float(np.percentile(ranks, 75)),
            'p90': float(np.percentile(ranks, 90)),
        }
    summary['d_at_k'] = summary['n_hits'] / summary['n_events']

    with open(args.output, 'w') as handle:
        json.dump(summary, handle, indent=2)
    logger.info(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
