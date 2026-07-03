from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from utils.triplet_split import load_split

TRIPLET_RANK_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                                'data', 'low-pt', 'eval', 'triplet_rank')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
K_VALUES = [1, 5, 10, 20, 50, 100]

# tau values frozen from the GBDT soft-filter pass (recall floors on held-out VAL-20);
# tierH applies no learned filter.
OPERATING_POINTS = {
    'tierH': ('gbdt6_score', 0.0),
    'd6@0.99': ('gbdt6_score', 0.003824),
    'd8@0.95': ('gbdt8_score', 0.090546),
}
ORDERINGS = ['gbdt', 'couple_rank_lex', 'random']


def deduped_gt_rank(keys: np.ndarray, is_gt: np.ndarray) -> int | None:
    """keys: (n, 3) sorted 3-sets, ALREADY in ranking order. is_gt: (n,) bool.

    Returns the 1-based rank of the GT 3-set after collapsing duplicate 3-sets to
    their first occurrence, or None if no GT candidate is present.
    """
    if not is_gt.any():
        return None
    encoded = (keys[:, 0].astype(np.int64) * 4_194_304
               + keys[:, 1].astype(np.int64) * 2048 + keys[:, 2].astype(np.int64))
    _, first_positions = np.unique(encoded, return_index=True)
    gt_first = int(np.flatnonzero(is_gt)[0])
    return int(np.searchsorted(np.sort(first_positions), gt_first)) + 1


def _event_orderings(scores, couple_rank, generator):
    yield 'gbdt', np.argsort(-scores, kind='stable')
    yield 'couple_rank_lex', np.lexsort((-scores, couple_rank))
    yield 'random', generator.permutation(len(scores))


def evaluate_baselines(candidates_path: str, *, operating_point: str,
                       event_indices=None, seed: int = 0) -> dict:
    score_column, tau = OPERATING_POINTS[operating_point]
    table = pq.read_table(candidates_path,
                          columns=['cand_i', 'cand_j', 'cand_k', 'couple_rank',
                                   score_column, 'is_gt'])
    n_rows = table.num_rows
    rows = range(n_rows) if event_indices is None else [int(x) for x in event_indices]
    generator = np.random.default_rng(seed)

    hits = {ordering: {k: 0 for k in K_VALUES} for ordering in ORDERINGS}
    n_ceiling = 0
    survivor_counts = []
    column = {name: table[name].combine_chunks() for name in table.schema.names}

    for r in tqdm(rows, desc=f'baselines[{operating_point}]', mininterval=30):
        arrays = {name: np.asarray(column[name][r].values) for name in column}
        survive = arrays[score_column] >= tau
        survivor_counts.append(int(survive.sum()))
        is_gt = arrays['is_gt'][survive].astype(bool)
        if not is_gt.any():
            continue
        n_ceiling += 1
        keys = np.sort(np.stack([arrays['cand_i'][survive], arrays['cand_j'][survive],
                                 arrays['cand_k'][survive]], axis=1), axis=1)
        scores = arrays[score_column][survive]
        couple_rank = arrays['couple_rank'][survive]
        for ordering_name, order in _event_orderings(scores, couple_rank, generator):
            rank = deduped_gt_rank(keys[order], is_gt[order])
            for k in K_VALUES:
                if rank is not None and rank <= k:
                    hits[ordering_name][k] += 1

    n_events = len(survivor_counts)
    counts = np.asarray(survivor_counts)
    return {
        'operating_point': operating_point,
        'score_column': score_column,
        'tau': tau,
        'n_events': n_events,
        'ceiling': n_ceiling / n_events if n_events else 0.0,
        'k_values': K_VALUES,
        't_at_k': {ordering: {str(k): hits[ordering][k] / n_events for k in K_VALUES}
                   for ordering in ORDERINGS},
        'survivors': {'mean': float(counts.mean()), 'median': float(np.median(counts)),
                      'p99': float(np.percentile(counts, 99)), 'max': int(counts.max())},
    }


def _print_table(results: dict) -> list[str]:
    lines = []
    for slice_name, per_op in results.items():
        lines.append(f'## {slice_name}')
        header = '| op point | ceiling | ordering | ' + ' | '.join(f'T@{k}' for k in K_VALUES) + ' |'
        lines += [header, '|' + '---|' * (3 + len(K_VALUES))]
        for op_name, result in per_op.items():
            for ordering in ORDERINGS:
                cells = ' | '.join(f"{result['t_at_k'][ordering][str(k)]:.4f}" for k in K_VALUES)
                lines.append(f"| {op_name} | {result['ceiling']:.4f} | {ordering} | {cells} |")
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default=os.path.join(TRIPLET_RANK_DIR, 'candidates_val.parquet'))
    ap.add_argument('--split-json', default=None,
                    help='also report the held-out slice of this split file')
    ap.add_argument('--out-json', default=os.path.join(REPORTS_DIR, 'triplet_rank_baselines.json'))
    ap.add_argument('--ceilings-json', default=os.path.join(TRIPLET_RANK_DIR, 'ceilings.json'))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    slices = {'full': None}
    if args.split_json:
        slices['held_out_20'] = load_split(args.split_json, 'test')

    results = {}
    for slice_name, indices in slices.items():
        results[slice_name] = {
            op: evaluate_baselines(args.candidates, operating_point=op,
                                   event_indices=indices, seed=args.seed)
            for op in OPERATING_POINTS
        }
    _print_table(results)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'wrote {args.out_json}')

    ceilings = {
        slice_name: {op: {'tau': r['tau'], 'score_column': r['score_column'],
                          'ceiling': r['ceiling'], 'survivors': r['survivors']}
                     for op, r in per_op.items()}
        for slice_name, per_op in results.items()
    }
    os.makedirs(os.path.dirname(args.ceilings_json), exist_ok=True)
    with open(args.ceilings_json, 'w') as fh:
        json.dump(ceilings, fh, indent=2)
    print(f'wrote {args.ceilings_json}')


if __name__ == '__main__':
    main()
