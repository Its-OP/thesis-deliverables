from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

DEFAULT_TOP_N = [250, 500, 1000, 1800, 3000, 5000]


def recall_at_top_n(scores: list[np.ndarray], is_gt: list[np.ndarray],
                    n: int) -> tuple[int, int]:
    """scores, is_gt: per-event arrays. Returns (events with a GT row in the
    per-event top-n by score, total events)."""
    hits = 0
    for event_scores, event_gt in zip(scores, is_gt):
        if not event_gt.any():
            continue
        keep = np.argsort(-event_scores, kind='stable')[:n]
        if event_gt[keep].any():
            hits += 1
    return hits, len(scores)


def threshold_curve(scores: list[np.ndarray], is_gt: list[np.ndarray],
                    thresholds: list[float]) -> list[dict]:
    """Returns one point per threshold: mean survivors/event + event-level
    recall of GT survival."""
    points = []
    for tau in thresholds:
        survivors = 0
        hits = 0
        for event_scores, event_gt in zip(scores, is_gt):
            mask = event_scores >= tau
            survivors += int(mask.sum())
            if event_gt.any() and event_gt[mask].any():
                hits += 1
        points.append({'threshold': float(tau),
                       'mean_survivors': survivors / len(scores),
                       'recall': hits / len(scores)})
    return points


def load_event_arrays(candidates_path: str, score_column: str,
                      max_events: int = 0) -> tuple[list, list]:
    table = pq.read_table(candidates_path,
                          columns=[score_column, 'is_gt', 'row_kind'])
    scores, is_gt = [], []
    n_rows = table.num_rows if not max_events \
        else min(max_events, table.num_rows)
    score_col, gt_col, kind_col = (table[score_column], table['is_gt'],
                                   table['row_kind'])
    for r in range(n_rows):
        kinds = np.asarray(kind_col[r].values)
        serving = kinds == 0
        scores.append(np.asarray(score_col[r].values, dtype=np.float64)[serving])
        is_gt.append(np.asarray(gt_col[r].values)[serving])
    return scores, is_gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--score-column', default='filter_score')
    parser.add_argument('--top-n', default=','.join(map(str, DEFAULT_TOP_N)))
    parser.add_argument('--quantile-thresholds', type=int, default=25)
    parser.add_argument('--max-events', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    scores, is_gt = load_event_arrays(args.candidates, args.score_column,
                                      args.max_events)
    pooled = np.concatenate(scores)
    thresholds = np.quantile(
        pooled, np.linspace(0.05, 0.999, args.quantile_thresholds)).tolist()

    result = {
        'candidates': args.candidates,
        'score_column': args.score_column,
        'n_events': len(scores),
        'ceiling': sum(1 for gt in is_gt if gt.any()) / len(scores),
        'top_n': {},
        'threshold_curve': threshold_curve(scores, is_gt, thresholds),
    }
    for n in (int(x) for x in args.top_n.split(',')):
        hits, total = recall_at_top_n(scores, is_gt, n)
        result['top_n'][n] = hits / total
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    print(f"{args.score_column}: ceiling {result['ceiling']:.4f} " +
          ' '.join(f"R@{n}={v:.4f}" for n, v in result['top_n'].items()))


if __name__ == '__main__':
    main()
