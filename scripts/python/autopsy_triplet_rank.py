from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

RANK_BUCKETS = [(1, 10), (11, 50), (51, 100), (101, 512), (513, 1024), (1025, 10 ** 9)]


def _bucket_label(low, high):
    return f'{low}-{high}' if high < 10 ** 9 else f'{low}+'


def autopsy_event(row: dict, window_positions: np.ndarray, tau: float,
                  ks=(1, 5, 10, 20, 50, 100)) -> dict:
    """row: per-event scalars + candidate arrays (recon, gt_i/j/k, cand_i/j/k,
    couple_rank, gbdt6_score, is_gt). window_positions: (W,) candidate positions in
    stage-A score-descending order. Returns bucket + per-event stats."""
    if not row['recon']:
        return {'bucket': 'not_reconstructable'}
    is_gt = np.asarray(row['is_gt'], dtype=bool)
    if not is_gt.any():
        return {'bucket': 'gt_not_enumerated'}
    scores = np.asarray(row['gbdt6_score'], dtype=np.float64)
    surviving = scores >= tau
    if not (is_gt & surviving).any():
        return {'bucket': 'gt_killed_by_tau'}

    window = np.asarray(window_positions, dtype=np.int64)
    window_is_gt = is_gt[window]
    if not window_is_gt.any():
        return {'bucket': 'gt_outside_window'}

    gt_rank = int(np.argmax(window_is_gt)) + 1
    gt_position = window[gt_rank - 1]
    impostor_slots = np.where(~window_is_gt)[0]
    gt_set = {int(row['gt_i']), int(row['gt_j']), int(row['gt_k'])}
    result = {
        'bucket': 'gt_in_window',
        'gt_window_rank': gt_rank,
        'hits': {k: gt_rank <= k for k in ks},
        'n_surviving': int(surviving.sum()),
        'gt_gbdt6': float(scores[gt_position]),
    }
    if impostor_slots.size:
        impostor_position = window[impostor_slots[0]]
        impostor_tracks = {int(row['cand_i'][impostor_position]),
                           int(row['cand_j'][impostor_position]),
                           int(row['cand_k'][impostor_position])}
        result['impostor_shared_tracks'] = len(impostor_tracks & gt_set)
        result['impostor_gbdt6'] = float(scores[impostor_position])
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Miss autopsy: decompose where GT '
                                                 'triplets die along the stage-4 funnel.')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--window', required=True,
                        help='stage-A window dump aligned with --candidates')
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--ks', default='1,5,10,20,50,100')
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--out-json', required=True)
    args = parser.parse_args(argv)
    ks = [int(k) for k in args.ks.split(',')]

    columns = ['recon', 'gt_i', 'gt_j', 'gt_k', 'cand_i', 'cand_j', 'cand_k',
               'couple_rank', 'gbdt6_score', 'is_gt']
    candidates = pq.read_table(args.candidates, columns=columns)
    window_table = pq.read_table(args.window, columns=['window_positions'])
    assert candidates.num_rows == window_table.num_rows, 'candidates/window row mismatch'
    n_events = candidates.num_rows if args.max_events is None else \
        min(args.max_events, candidates.num_rows)

    funnel = {'not_reconstructable': 0, 'gt_not_enumerated': 0,
              'gt_killed_by_tau': 0, 'gt_outside_window': 0, 'gt_in_window': 0}
    hits_by_k = {k: 0 for k in ks}
    rank_histogram = {_bucket_label(*bucket): 0 for bucket in RANK_BUCKETS}
    shared_all = {0: 0, 1: 0, 2: 0}
    shared_miss10 = {0: 0, 1: 0, 2: 0}
    gbdt6_margin_hit10 = []
    gbdt6_margin_miss10 = []
    surviving_and_rank = []

    for r in tqdm(range(n_events), desc='autopsy', mininterval=10):
        row = {name: candidates[name][r].as_py() if name in ('recon', 'gt_i', 'gt_j', 'gt_k')
               else np.asarray(candidates[name][r].values)
               for name in columns}
        window_positions = np.asarray(window_table['window_positions'][r].values)
        result = autopsy_event(row, window_positions, args.tau, ks=tuple(ks))
        funnel[result['bucket']] += 1
        if result['bucket'] != 'gt_in_window':
            continue
        rank = result['gt_window_rank']
        for k in ks:
            hits_by_k[k] += int(result['hits'][k])
        for low, high in RANK_BUCKETS:
            if low <= rank <= high:
                rank_histogram[_bucket_label(low, high)] += 1
                break
        surviving_and_rank.append((result['n_surviving'], rank))
        if 'impostor_shared_tracks' in result:
            shared_all[result['impostor_shared_tracks']] += 1
            margin = result['gt_gbdt6'] - result['impostor_gbdt6']
            if rank > 10:
                shared_miss10[result['impostor_shared_tracks']] += 1
                gbdt6_margin_miss10.append(margin)
            else:
                gbdt6_margin_hit10.append(margin)

    # T@10 by event-busyness quartile: does a global tau starve quiet events (H4)?
    quartile_t10 = {}
    if surviving_and_rank:
        surviving_counts = np.array([s for s, _ in surviving_and_rank])
        ranks = np.array([rank for _, rank in surviving_and_rank])
        edges = np.quantile(surviving_counts, [0.25, 0.5, 0.75])
        quartile_of = np.searchsorted(edges, surviving_counts)
        for q in range(4):
            mask = quartile_of == q
            quartile_t10[f'q{q + 1}'] = {
                'n_surviving_median': float(np.median(surviving_counts[mask])),
                'hit10_fraction_of_in_window': float((ranks[mask] <= 10).mean()),
                'events': int(mask.sum()),
            }

    report = {
        'n_events': n_events,
        'tau': args.tau,
        'funnel': funnel,
        'funnel_fractions': {k: v / n_events for k, v in funnel.items()},
        'hits_by_k': {str(k): hits_by_k[k] for k in ks},
        't_at_k': {str(k): hits_by_k[k] / n_events for k in ks},
        'gt_window_rank_histogram': rank_histogram,
        'impostor_shared_tracks_histogram': shared_all,
        'impostor_shared_tracks_histogram_miss10': shared_miss10,
        'gbdt6_margin_gt_minus_impostor': {
            'hit10_median': float(np.median(gbdt6_margin_hit10)) if gbdt6_margin_hit10 else None,
            'miss10_median': float(np.median(gbdt6_margin_miss10)) if gbdt6_margin_miss10 else None,
        },
        't10_by_n_surviving_quartile': quartile_t10,
    }
    with open(args.out_json, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
