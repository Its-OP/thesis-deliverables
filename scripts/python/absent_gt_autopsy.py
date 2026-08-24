from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from scripts.python.third_track_concentration import percentile_of


def classify_event(gt_rows_scores: np.ndarray, tau: float) -> str:
    """gt_rows_scores: filter scores of the event's GT rows (may be empty).
    Returns 'present' | 'gate_killed' | 'not_stored'."""
    if gt_rows_scores.size == 0:
        return 'not_stored'
    return 'present' if (gt_rows_scores >= tau).any() else 'gate_killed'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    table = pq.read_table(
        args.candidates,
        columns=['is_gt', 'filter_score', 'row_kind', 'gt_k', 'recon',
                 'track_s1'])
    percentiles: dict[str, list[float]] = {'present': [], 'gate_killed': [],
                                           'not_stored': []}
    counts = {'present': 0, 'gate_killed': 0, 'not_stored': 0,
              'not_reconstructable': 0}
    for r in range(table.num_rows):
        if not bool(table['recon'][r].as_py()):
            counts['not_reconstructable'] += 1
            continue
        gt_k = int(table['gt_k'][r].as_py())
        if gt_k < 0:
            counts['not_reconstructable'] += 1
            continue
        is_gt = np.asarray(table['is_gt'][r].values)
        kinds = np.asarray(table['row_kind'][r].values)
        scores = np.asarray(table['filter_score'][r].values)
        gt_scores = scores[is_gt & (kinds == 0)]
        label = classify_event(gt_scores, args.tau)
        counts[label] += 1
        track_s1 = np.asarray(table['track_s1'][r].values)
        percentiles[label].append(percentile_of(track_s1[gt_k], track_s1))

    summary = {'counts': counts}
    for label, values in percentiles.items():
        if not values:
            continue
        array = np.asarray(values)
        summary[label] = {
            'n': len(array), 'mean': float(array.mean()),
            'p10': float(np.percentile(array, 10)),
            'p25': float(np.percentile(array, 25)),
            'p50': float(np.percentile(array, 50)),
            'p75': float(np.percentile(array, 75)),
            'below_half': float((array < 0.5).mean()),
        }
    with open(args.output, 'w') as handle:
        json.dump(summary, handle, indent=2)
    for label in ('present', 'gate_killed', 'not_stored'):
        if label in summary:
            entry = summary[label]
            print(f"{label}: n={entry['n']} gt-third s1-pctile "
                  f"p10/p50/p75 {entry['p10']:.2f}/{entry['p50']:.2f}/"
                  f"{entry['p75']:.2f} below-median-share "
                  f"{entry['below_half']:.3f}")
    print(f"counts: {counts}")


if __name__ == '__main__':
    main()
