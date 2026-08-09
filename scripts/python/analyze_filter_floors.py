import argparse
import glob
import json
import os

import joblib
import numpy as np
import pyarrow.parquet as pq

from scripts.python.eval_triplet_filter_ranking import (
    feature_columns_for_width,
    use_cpu_inference,
)
from scripts.python.probe_triplet_features import FEATURE_SETS
from scripts.python.train_triplet_filter import _curve, _factor_at_floor
from utils.triplet_join import (
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    H6_ISOLATION_NAMES,
    H6_PHYSICS_NAMES,
    H6_SV_NAMES,
    H6_VERTEX_NAMES,
)

FLOOR_GRID = (0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.92, 0.90)
PERCENTILES = (0.1, 0.5, 1, 2, 5, 10, 25, 50)
# Features worth contrasting between the pinned tail and the bulk.
PROFILE_NAMES = ("pt_k", "n_pixel_k", "dr_min", "dz_dist", "couple_rank",
                 "m_ijk", "poca_max", "raw_dz_gap_max",
                 "lifetime_positive_count", "dxy_sig_spread")


def compression_grid(gt_scores, sub_scores, sub_weight, recon, n_full, floors):
    """Compression and ground-truth loss at each recall floor."""
    points = _curve(gt_scores, sub_scores, sub_weight, recon, n_full)
    grid = {}
    for floor in floors:
        factor = _factor_at_floor(points, floor)
        grid[floor] = dict(compression=factor,
                           gt_lost=float(recon) * (1.0 - floor))
    return grid


def percentile_table(scores, percentiles=PERCENTILES):
    return {p: float(np.percentile(scores, p)) for p in percentiles}


def bottom_fraction_indices(scores, fraction):
    """Row indices of the lowest-scoring `fraction` of entries, at least one."""
    count = max(1, int(round(len(scores) * fraction)))
    return np.argsort(scores, kind="stable")[:count]


def jaccard(left, right):
    left_set, right_set = set(left.tolist()), set(right.tolist())
    union = left_set | right_set
    if not union:
        return float("nan")
    return len(left_set & right_set) / len(union)


def _rank(matrix):
    order = np.argsort(matrix, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(matrix.shape[0])[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(
        rows, order.shape).astype(np.float64), axis=0)
    return ranks


def max_abs_spearman(new_matrix, legacy_matrix):
    """For each new column, the largest |Spearman| against any legacy column.
    High values mean the column is already spanned by what the model had."""
    new_ranks = _rank(np.asarray(new_matrix, dtype=np.float64))
    legacy_ranks = _rank(np.asarray(legacy_matrix, dtype=np.float64))
    new_centered = new_ranks - new_ranks.mean(axis=0)
    legacy_centered = legacy_ranks - legacy_ranks.mean(axis=0)
    new_norm = np.linalg.norm(new_centered, axis=0)
    legacy_norm = np.linalg.norm(legacy_centered, axis=0)
    # Guard constant columns, whose correlation is undefined.
    new_norm[new_norm == 0] = np.inf
    legacy_norm[legacy_norm == 0] = np.inf
    correlation = np.abs(
        (new_centered / new_norm).T @ (legacy_centered / legacy_norm))
    return correlation.max(axis=1), correlation.argmax(axis=1)


def _columns(table, names):
    return np.column_stack([table[name].to_numpy() for name in names])


def _score(model, table, names, columns):
    return model.predict_proba(_columns(table, names)[:, :])[:, 1] \
        if columns is None else model.predict_proba(
            _columns(table, FEATURE_NAMES_EXTENDED)[:, columns])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Floor sensitivity, tail composition and feature '
                    'redundancy for the third-pion filters.')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--eval-prefix', required=True)
    parser.add_argument('--arms', default='full89__uniform__balanced__d8,'
                                          'vertex__uniform__balanced__d8,'
                                          'vertex_physics__uniform__balanced__d8,'
                                          'all22__uniform__balanced__d8')
    parser.add_argument('--tier-h-compression', type=float, default=None,
                        help='defaults to n_full / total sub weight')
    parser.add_argument('--redundancy-rows', type=int, default=200000)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    gt_table = pq.read_table(args.eval_prefix + '_gt.parquet')
    sub_table = pq.read_table(args.eval_prefix + '_sub.parquet')
    with open(args.eval_prefix + '_meta.json') as handle:
        meta = json.load(handle)['P2']
    weights = sub_table['weight'].to_numpy()
    tier_h = args.tier_h_compression or meta['n_full'] / float(weights.sum())
    print(f'reconstructable {meta["recon"]} of {meta["n_events"]} events; '
          f'Tier-H gate alone {tier_h:.2f}x')

    results = {'tier_h_compression': tier_h, 'meta': meta, 'arms': {}}
    bottom_sets, scores_by_arm = {}, {}
    for arm in args.arms.split(','):
        matches = glob.glob(os.path.join(args.models_dir, f'sweep_{arm}__*.joblib'))
        if not matches:
            print(f'{arm}: no model found, skipping')
            continue
        model = use_cpu_inference(joblib.load(matches[0]))
        width = int(model.n_features_in_)
        name, columns, _ = feature_columns_for_width(width)
        gt_scores = model.predict_proba(
            _columns(gt_table, FEATURE_NAMES_EXTENDED)[:, columns])[:, 1]
        sub_scores = model.predict_proba(
            _columns(sub_table, FEATURE_NAMES_EXTENDED)[:, columns])[:, 1]
        scores_by_arm[arm] = gt_scores
        bottom_sets[arm] = bottom_fraction_indices(gt_scores, 0.01)

        grid = compression_grid(gt_scores, sub_scores, weights, meta['recon'],
                                meta['n_full'], FLOOR_GRID)
        results['arms'][arm] = dict(
            feature_set=name, n_features=width,
            floors={str(floor): dict(entry, beyond_tier_h=(
                entry['compression'] / tier_h if entry['compression'] else None))
                for floor, entry in grid.items()},
            gt_percentiles=percentile_table(gt_scores))
        print(f'\n{arm} ({name}, {width} columns)')
        print(f'  {"floor":>6s} {"compression":>12s} {"x Tier-H":>9s} {"GT lost":>8s}')
        for floor, entry in grid.items():
            factor = entry['compression']
            print(f'  {floor:6.3f} {factor if factor else float("nan"):12.1f} '
                  f'{(factor / tier_h) if factor else float("nan"):9.2f} '
                  f'{entry["gt_lost"]:8.0f}')

    # Tail overlap: is the pinned 1% the same events for every model?
    arms = list(bottom_sets)
    overlap = {f'{a}|{b}': jaccard(bottom_sets[a], bottom_sets[b])
               for index, a in enumerate(arms) for b in arms[index + 1:]}
    results['bottom_one_percent_overlap'] = overlap
    if overlap:
        print('\nbottom-1% overlap (Jaccard)')
        for pair, value in overlap.items():
            left, right = pair.split('|')
            print(f'  {left.split("__")[0]:16s} vs {right.split("__")[0]:16s} {value:.3f}')

    # What distinguishes the pinned tail from the bulk.
    if arms:
        reference = arms[0]
        tail = bottom_sets[reference]
        mask = np.zeros(gt_table.num_rows, dtype=bool)
        mask[tail] = True
        profile = {}
        print(f'\ntail profile ({reference.split("__")[0]}, {mask.sum()} events)')
        print(f'  {"feature":26s} {"tail":>12s} {"bulk":>12s} {"ratio":>8s}')
        for feature in PROFILE_NAMES:
            if feature not in gt_table.schema.names:
                continue
            values = gt_table[feature].to_numpy()
            tail_median = float(np.nanmedian(values[mask]))
            bulk_median = float(np.nanmedian(values[~mask]))
            ratio = tail_median / bulk_median if bulk_median else float('nan')
            profile[feature] = dict(tail=tail_median, bulk=bulk_median, ratio=ratio)
            print(f'  {feature:26s} {tail_median:12.4f} {bulk_median:12.4f} {ratio:8.2f}')
        results['tail_profile'] = profile

    # Redundancy of every new column against the legacy 89.
    rows = min(args.redundancy_rows, sub_table.num_rows)
    sample = np.random.default_rng(0).choice(sub_table.num_rows, rows, replace=False)
    new_names = H6_VERTEX_NAMES + H6_PHYSICS_NAMES + H6_ISOLATION_NAMES + H6_SV_NAMES
    new_matrix = _columns(sub_table, new_names)[sample]
    legacy_matrix = _columns(sub_table, FEATURE_NAMES)[sample]
    finite = np.isfinite(new_matrix).all(axis=1) & np.isfinite(legacy_matrix).all(axis=1)
    best, partner = max_abs_spearman(new_matrix[finite], legacy_matrix[finite])
    redundancy = {name: dict(max_abs_spearman=float(value),
                             closest_legacy=FEATURE_NAMES[int(index)])
                  for name, value, index in zip(new_names, best, partner)}
    results['redundancy'] = redundancy
    print(f'\nredundancy against the legacy 89 ({int(finite.sum())} rows)')
    print(f'  {"feature":26s} {"max |rho|":>9s}  closest legacy column')
    for name, entry in sorted(redundancy.items(),
                              key=lambda kv: -kv[1]['max_abs_spearman']):
        print(f'  {name:26s} {entry["max_abs_spearman"]:9.3f}  '
              f'{entry["closest_legacy"]}')

    if args.output:
        with open(args.output, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f'\n-> {args.output}')


if __name__ == '__main__':
    main()
