import argparse
import itertools
import json
import os

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.python.probe_triplet_features import FEATURE_SETS
from scripts.python.train_triplet_filter import (
    FLOORS,
    _cols,
    _curve,
    _factor_at_floor,
)
from utils.triplet_join import (
    H6_ISOLATION_NAMES,
    H6_PHYSICS_NAMES,
    H6_SV_NAMES,
    H6_VERTEX_NAMES,
)

MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models")
REPORTS = os.path.join(os.path.dirname(__file__), "..", "..", "reports")

BASE_ESTIMATORS = {
    "d6": dict(max_leaf_nodes=31, max_depth=6, learning_rate=0.1, max_iter=200),
    "d8": dict(max_leaf_nodes=63, max_depth=8, learning_rate=0.1, max_iter=300),
}
# Leaves and regularization rather than depth alone: histogram boosters grow
# leaf-wise, so max_depth is the weaker knob.
REFINEMENT_GRID = dict(
    max_leaf_nodes=[31, 63, 127],
    min_samples_leaf=[20, 100],
    learning_rate=[0.05, 0.1],
    max_iter=[200, 400],
)
ATTRIBUTION_BLOCKS = {
    "vertex": H6_VERTEX_NAMES,
    "physics": H6_PHYSICS_NAMES,
    "isolation": H6_ISOLATION_NAMES,
    "secondary_vertex": H6_SV_NAMES,
}


def xgboost_parameters(params, class_weight, positive_count, negative_count):
    """Translates the sklearn HistGradientBoosting arguments this project uses
    into their XGBoost equivalents. `class_weight='balanced'` becomes
    scale_pos_weight, which is the same reweighting expressed per class."""
    translated = dict(
        tree_method="hist", grow_policy="lossguide", device="cuda",
        max_leaves=params.get("max_leaf_nodes", 31),
        max_depth=params.get("max_depth", 0),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("max_iter", 200),
        min_child_weight=params.get("min_samples_leaf", 20),
        reg_lambda=params.get("l2_regularization", 0.0),
        random_state=0,
    )
    if class_weight == "balanced":
        translated["scale_pos_weight"] = (
            negative_count / max(positive_count, 1))
    return translated


def build_estimator(backend, params, class_weight, labels):
    if backend == "sklearn":
        return HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=0, **params)
    if backend != "xgboost":
        raise ValueError(f"unknown backend {backend!r}")
    from xgboost import XGBClassifier
    positive_count = int((labels > 0.5).sum())
    negative_count = int((labels <= 0.5).sum())
    return XGBClassifier(**xgboost_parameters(
        params, class_weight, positive_count, negative_count))


def enumerate_arms(feature_sets, neg_modes, class_weights, refine=False):
    """Returns the list of arm dicts the sweep will fit, in a stable order."""
    arms = []
    for feature_set, neg_mode, class_weight in itertools.product(
            feature_sets, neg_modes, class_weights):
        for estimator_name, estimator in BASE_ESTIMATORS.items():
            arms.append(dict(feature_set=feature_set, neg_mode=neg_mode,
                             class_weight=class_weight,
                             estimator=estimator_name, params=dict(estimator)))
    if refine:
        keys = sorted(REFINEMENT_GRID)
        for values in itertools.product(*(REFINEMENT_GRID[key] for key in keys)):
            params = dict(zip(keys, values))
            arms.append(dict(feature_set=feature_sets[-1], neg_mode=neg_modes[0],
                             class_weight=class_weights[0],
                             estimator="refine", params=params))
    return arms


def arm_name(arm):
    return (f'{arm["feature_set"]}__{arm["neg_mode"]}__'
            f'{arm["class_weight"] or "none"}__{arm["estimator"]}__'
            + "_".join(f'{key}{value}' for key, value in sorted(arm["params"].items())))


def fit_arm(arm, train_table, gt_table, sub_table, meta, pool="P2",
            backend="sklearn"):
    """Fits one arm and returns (model, metrics dict of compression at floors)."""
    names = FEATURE_SETS[arm["feature_set"]]
    train_rows = train_table["pool"].to_numpy(zero_copy_only=False) == pool
    gt_rows = gt_table["pool"].to_numpy(zero_copy_only=False) == pool
    sub_rows = sub_table["pool"].to_numpy(zero_copy_only=False) == pool

    labels = train_table["is_gt"].to_numpy()[train_rows]
    model = build_estimator(backend, arm["params"], arm["class_weight"], labels)
    model.fit(_cols(train_table, names)[train_rows], labels)
    gt_scores = model.predict_proba(_cols(gt_table, names)[gt_rows])[:, 1]
    sub_scores = model.predict_proba(_cols(sub_table, names)[sub_rows])[:, 1]
    points = _curve(gt_scores, sub_scores,
                    sub_table["weight"].to_numpy()[sub_rows],
                    meta[pool]["recon"], meta[pool]["n_full"])
    metrics = {f"compression@{floor}": _factor_at_floor(points, floor)
               for floor in FLOORS}
    # tau at the 0.99 and 0.95 GT quantiles: the operating points the
    # downstream stage freezes.
    metrics["tau@0.99"] = float(np.quantile(gt_scores, 0.01))
    metrics["tau@0.95"] = float(np.quantile(gt_scores, 0.05))
    metrics["points"] = points
    return model, metrics


def block_permutation_importance(model, table, names, labels, blocks, seed=0):
    """Permutes whole correlated blocks together — permuting one column at a
    time splits credit arbitrarily between correlated physics features and
    evaluates the model off the data manifold."""
    generator = np.random.default_rng(seed)
    features = _cols(table, names)
    baseline = model.predict_proba(features)[:, 1]
    reference = float(np.mean(baseline[labels > 0.5])
                      - np.mean(baseline[labels < 0.5]))
    importances = {}
    for block_name, block_names in blocks.items():
        columns = [names.index(name) for name in block_names if name in names]
        if not columns:
            continue
        shuffled = features.copy()
        order = generator.permutation(len(shuffled))
        shuffled[:, columns] = shuffled[order][:, columns]
        scores = model.predict_proba(shuffled)[:, 1]
        separation = float(np.mean(scores[labels > 0.5])
                           - np.mean(scores[labels < 0.5]))
        importances[block_name] = reference - separation
    return importances


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Arm grid over feature sets, negative modes, class '
                    'weighting and estimator hyperparameters.')
    parser.add_argument('--train-table', action='append', required=True,
                        metavar='NEGMODE=PATH')
    parser.add_argument('--eval-prefix', required=True)
    parser.add_argument('--feature-sets', default='full89,vertex,all22')
    parser.add_argument('--class-weights', default='balanced,none')
    parser.add_argument('--backend', choices=('sklearn', 'xgboost'),
                        default='sklearn')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--attribution', action='store_true')
    parser.add_argument('--output', default=os.path.join(
        REPORTS, 'triplet_filter_sweep.json'))
    args = parser.parse_args()

    train_tables = {}
    for entry in args.train_table:
        neg_mode, path = entry.split('=', 1)
        train_tables[neg_mode] = pq.read_table(path)
    gt_table = pq.read_table(args.eval_prefix + '_gt.parquet')
    sub_table = pq.read_table(args.eval_prefix + '_sub.parquet')
    with open(args.eval_prefix + '_meta.json') as handle:
        meta = json.load(handle)

    class_weights = [None if value == 'none' else value
                     for value in args.class_weights.split(',')]
    arms = enumerate_arms(args.feature_sets.split(','), sorted(train_tables),
                          class_weights, refine=args.refine)
    print(f'{len(arms)} arms over {len(train_tables)} training tables')

    results = {}
    for index, arm in enumerate(arms, start=1):
        name = arm_name(arm)
        model, metrics = fit_arm(arm, train_tables[arm['neg_mode']],
                                 gt_table, sub_table, meta,
                                 backend=args.backend)
        results[name] = {key: value for key, value in metrics.items()
                         if key != 'points'}
        results[name]['arm'] = {key: arm[key] for key in
                                ('feature_set', 'neg_mode', 'class_weight',
                                 'estimator', 'params')}
        print(f'[{index}/{len(arms)}] {name}: '
              + '  '.join(f'{key} {value:.1f}' for key, value
                          in results[name].items()
                          if key.startswith('compression') and value))
        if args.save_models:
            os.makedirs(MODELS, exist_ok=True)
            joblib.dump(model, os.path.join(MODELS, f'sweep_{name}.joblib'))
        if args.attribution and arm['feature_set'] == 'all22':
            names = FEATURE_SETS[arm['feature_set']]
            rows = gt_table['pool'].to_numpy(zero_copy_only=False) == 'P2'
            labels = np.ones(int(rows.sum()))
            sub_rows = sub_table['pool'].to_numpy(zero_copy_only=False) == 'P2'
            combined_labels = np.concatenate(
                [labels, np.zeros(int(sub_rows.sum()))])
            import pyarrow as pa
            combined = pa.concat_tables(
                [gt_table.select(names), sub_table.select(names)])
            results[name]['block_importance'] = block_permutation_importance(
                model, combined, names, combined_labels, ATTRIBUTION_BLOCKS)

    with open(args.output, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'-> {args.output}')


if __name__ == '__main__':
    main()
