from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

COLUMNS = ['filter_score', 'row_kind', 'is_gt']


def serving_stats(scores: np.ndarray, kinds: np.ndarray, is_gt: np.ndarray,
                  tau: float, cap: int) -> tuple[int, bool, bool]:
    """scores/kinds/is_gt: (N,) stored in filter-rank order.
    Returns (survivor count, GT survives the gate, GT within the serving cap).
    """
    serving = kinds == 0
    survivors = serving & (scores >= tau)
    n_survivors = int(survivors.sum())
    gt_positions = np.nonzero(is_gt[survivors])[0]
    gt_survives = gt_positions.size > 0
    gt_within_cap = bool(gt_survives and gt_positions[0] < cap)
    return n_survivors, gt_survives, gt_within_cap


def collect(candidates_path: str, tau: float, cap: int) -> dict:
    table = pq.read_table(candidates_path, columns=COLUMNS)
    counts, survives, within = [], 0, 0
    n_gt_events = 0
    for r in range(table.num_rows):
        scores = np.asarray(table['filter_score'][r].values)
        kinds = np.asarray(table['row_kind'][r].values)
        is_gt = np.asarray(table['is_gt'][r].values, dtype=bool)
        n_survivors, gt_survives, gt_within_cap = serving_stats(
            scores, kinds, is_gt, tau, cap)
        counts.append(n_survivors)
        if is_gt.any():
            n_gt_events += 1
            survives += int(gt_survives)
            within += int(gt_within_cap)
    count_array = np.asarray(counts)
    return {
        'n_events': int(table.num_rows),
        'n_events_with_gt_rows': n_gt_events,
        'survivors': {
            'mean': float(count_array.mean()),
            'p50': float(np.percentile(count_array, 50)),
            'p90': float(np.percentile(count_array, 90)),
            'p99': float(np.percentile(count_array, 99)),
            'max': int(count_array.max()),
            'over_cap_fraction': float((count_array > cap).mean()),
        },
        'gt_survives_gate': survives,
        'gt_within_cap': within,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--cap', type=int, default=3072)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    result = collect(args.candidates, args.tau, args.cap)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
