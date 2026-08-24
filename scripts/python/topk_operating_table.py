from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from scripts.python.eval_triplet_rank_baselines import deduped_gt_rank

FINE_K = list(range(1, 21)) + [25, 30, 40, 50, 75, 100]


def rank_distribution(candidates_path: str, scores_path: str,
                      tau: float, max_list: int) -> tuple[list[int], int]:
    """Returns (deduped GT ranks over ranked events, total events)."""
    table = pq.read_table(candidates_path,
                          columns=['cand_i', 'cand_j', 'cand_k', 'is_gt',
                                   'row_kind', 'filter_score'])
    external = pq.read_table(scores_path, columns=['scores'])
    ranks: list[int] = []
    for r in range(table.num_rows):
        kinds = np.asarray(table['row_kind'][r].values)
        stored_scores = np.asarray(table['filter_score'][r].values)
        serving = np.nonzero((kinds == 0) & (stored_scores >= tau))[0]
        if max_list:
            serving = serving[:max_list]
        if serving.size == 0:
            continue
        is_gt = np.asarray(table['is_gt'][r].values)[serving]
        if not is_gt.any():
            continue
        model_scores = np.asarray(external['scores'][r].values)
        assert model_scores.size == serving.size, \
            f'event {r}: {model_scores.size} scores vs {serving.size} rows'
        keys = np.stack([np.asarray(table[name][r].values)[serving]
                         for name in ('cand_i', 'cand_j', 'cand_k')], axis=1)
        keys = np.sort(keys, axis=1)
        order = np.argsort(-model_scores, kind='stable')
        rank = deduped_gt_rank(keys[order], is_gt[order])
        if rank is not None:
            ranks.append(rank)
    return ranks, table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--scores-parquet', required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--max-list', type=int, default=3072)
    parser.add_argument('--reference-k', type=int, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    ranks, n_events = rank_distribution(args.candidates, args.scores_parquet,
                                        args.tau, args.max_list)
    rank_array = np.asarray(ranks)
    curve = {k: float((rank_array <= k).sum() / n_events) for k in FINE_K}
    reference = curve[args.reference_k]
    with open(args.output, 'w') as handle:
        json.dump({'n_events': n_events, 'n_ranked': len(ranks),
                   'T@K': curve, 'reference_k': args.reference_k}, handle,
                  indent=2)
    print(f'{len(ranks)} ranked of {n_events} events; '
          f'T@{args.reference_k} = {reference:.4f}')
    for k in FINE_K:
        delta = curve[k] - reference
        print(f'K={k:3d}  T@K {curve[k]:.4f}  vs K={args.reference_k} '
              f'{delta:+.4f}')


if __name__ == '__main__':
    main()
