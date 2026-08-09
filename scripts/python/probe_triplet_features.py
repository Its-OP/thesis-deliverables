import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from utils.triplet_join import (
    FEATURE_NAMES,
    H6_ISOLATION_NAMES,
    H6_PHYSICS_NAMES,
    H6_SV_NAMES,
    H6_VERTEX_NAMES,
)

# Feature sets are cumulative: each adds one measured H6 group on top of the
# legacy 89, so a delta is attributable to the group that was added.
FEATURE_SETS = {
    "full89": FEATURE_NAMES,
    "vertex": FEATURE_NAMES + H6_VERTEX_NAMES,
    "vertex_physics": FEATURE_NAMES + H6_VERTEX_NAMES + H6_PHYSICS_NAMES,
    "all22": (FEATURE_NAMES + H6_VERTEX_NAMES + H6_PHYSICS_NAMES
              + H6_ISOLATION_NAMES + H6_SV_NAMES),
}

PROBE_ESTIMATOR = dict(max_leaf_nodes=31, max_depth=6, learning_rate=0.1,
                       max_iter=200, class_weight="balanced", random_state=0)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """scores, labels: (N,). Rank-based ROC AUC, ties averaged."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores so ties contribute 0.5 each.
    sorted_scores = scores[order]
    start = 0
    for stop in range(1, len(scores) + 1):
        if stop == len(scores) or sorted_scores[stop] != sorted_scores[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    positive = labels > 0.5
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
                 / (n_positive * n_negative))


def hit_at_k(scores: np.ndarray, labels: np.ndarray,
             event_index: np.ndarray, k: int = 10) -> float:
    """Fraction of events whose ground-truth candidate ranks inside the top k
    of that event's own candidate list."""
    hits = total = 0
    for event in np.unique(event_index):
        rows = event_index == event
        event_labels = labels[rows] > 0.5
        if not event_labels.any():
            continue
        total += 1
        event_scores = scores[rows]
        best_truth = event_scores[event_labels].max()
        # Rank of the best GT row: how many candidates strictly outrank it.
        if int((event_scores > best_truth).sum()) < k:
            hits += 1
    return hits / total if total else float("nan")


def grouped_folds(event_index: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Event-grouped fold assignment: every row of an event lands in one fold,
    so no event is split across the train/test boundary."""
    events = np.unique(event_index)
    fold_of_event = {event: index % n_folds for index, event in enumerate(events)}
    assignment = np.array([fold_of_event[event] for event in event_index])
    return [np.flatnonzero(assignment == fold) for fold in range(n_folds)]


def run_probe(table_path: str, n_folds: int = 3, top_k: int = 10,
              max_events: int | None = None) -> dict:
    table = pq.read_table(table_path)
    labels = np.asarray(table.column("is_gt"), dtype=np.float64)
    event_index = np.asarray(table.column("event_index"), dtype=np.int64)
    if max_events is not None:
        keep = event_index < np.unique(event_index)[:max_events].max() + 1
        labels, event_index = labels[keep], event_index[keep]
    else:
        keep = np.ones(len(labels), dtype=bool)

    columns = {name: np.asarray(table.column(name), dtype=np.float32)[keep]
               for name in table.schema.names
               if name not in ("is_gt", "event_index", "pool", "weight")}
    folds = grouped_folds(event_index, n_folds)

    results = {}
    for set_name, feature_names in FEATURE_SETS.items():
        features = np.stack([columns[name] for name in feature_names], axis=1)
        scores = np.empty(len(labels), dtype=np.float64)
        for fold_rows in folds:
            train_rows = np.setdiff1d(np.arange(len(labels)), fold_rows)
            model = HistGradientBoostingClassifier(**PROBE_ESTIMATOR)
            model.fit(features[train_rows], labels[train_rows])
            scores[fold_rows] = model.predict_proba(features[fold_rows])[:, 1]
        results[set_name] = dict(
            n_features=len(feature_names),
            auc=roc_auc(scores, labels),
            hit_at_k=hit_at_k(scores, labels, event_index, top_k),
        )
        print(f'{set_name:16s} features {len(feature_names):3d}  '
              f'AUC {results[set_name]["auc"]:.4f}  '
              f'hit@{top_k} {results[set_name]["hit_at_k"]:.4f}')

    baseline = results["full89"]
    for set_name, entry in results.items():
        entry["delta_auc"] = entry["auc"] - baseline["auc"]
        entry["delta_hit_at_k"] = entry["hit_at_k"] - baseline["hit_at_k"]
    results["_meta"] = dict(table=table_path, n_rows=int(len(labels)),
                            n_events=int(len(np.unique(event_index))),
                            n_positives=int((labels > 0.5).sum()),
                            n_folds=n_folds, top_k=top_k)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Event-grouped probe over the H6 feature groups.')
    parser.add_argument('--table', required=True)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    results = run_probe(args.table, n_folds=args.folds, top_k=args.top_k,
                        max_events=args.max_events)
    for set_name, entry in results.items():
        if set_name.startswith('_'):
            continue
        print(f'{set_name:16s} dAUC {entry["delta_auc"]:+.4f}  '
              f'dHit {entry["delta_hit_at_k"]:+.4f}')
    if args.output:
        with open(args.output, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f'-> {args.output}')


if __name__ == '__main__':
    main()
