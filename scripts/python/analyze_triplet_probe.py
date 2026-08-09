import argparse
import json

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.python.probe_triplet_features import (
    FEATURE_SETS,
    PROBE_ESTIMATOR,
    grouped_folds,
    hit_at_k,
    roc_auc,
)
from utils.triplet_join import H6_ISOLATION_NAMES, H6_SV_NAMES


def load_probe_table(path: str) -> dict:
    table = pq.read_table(path)
    return dict(
        table=table,
        labels=np.asarray(table.column("is_gt"), dtype=np.float64),
        event_index=np.asarray(table.column("event_index"), dtype=np.int64),
        columns={name: np.asarray(table.column(name), dtype=np.float32)
                 for name in table.schema.names
                 if name not in ("is_gt", "event_index", "pool", "weight")},
    )


def cross_fold_scores(train, evaluate, feature_names, n_folds=3, drop=()):
    """Fits on `train` rows and scores `evaluate` rows fold by fold. Folds are
    assigned by event over the union of both tables, so an event never appears
    on both sides of the split even when the two tables sample different
    negatives for the same events."""
    names = [name for name in feature_names if name not in set(drop)]
    train_features = np.stack([train["columns"][name] for name in names], axis=1)
    evaluate_features = np.stack(
        [evaluate["columns"][name] for name in names], axis=1)

    all_events = np.union1d(train["event_index"], evaluate["event_index"])
    fold_of_event = {event: index % n_folds
                     for index, event in enumerate(all_events)}
    train_fold = np.array([fold_of_event[event] for event in train["event_index"]])
    evaluate_fold = np.array(
        [fold_of_event[event] for event in evaluate["event_index"]])

    scores = np.empty(len(evaluate["labels"]), dtype=np.float64)
    for fold in range(n_folds):
        model = HistGradientBoostingClassifier(**PROBE_ESTIMATOR)
        held_in = train_fold != fold
        model.fit(train_features[held_in], train["labels"][held_in])
        held_out = evaluate_fold == fold
        scores[held_out] = model.predict_proba(
            evaluate_features[held_out])[:, 1]
    return scores


def per_event_hit(scores, labels, event_index, k=10):
    """Returns {event: bool} for events that carry a ground-truth row."""
    result = {}
    for event in np.unique(event_index):
        rows = event_index == event
        event_labels = labels[rows] > 0.5
        if not event_labels.any():
            continue
        event_scores = scores[rows]
        best_truth = event_scores[event_labels].max()
        result[int(event)] = int((event_scores > best_truth).sum()) < k
    return result


def event_level_flag(columns, event_index, name):
    """Returns {event: value} for a column that is constant within an event."""
    flags = {}
    for event in np.unique(event_index):
        values = columns[name][event_index == event]
        flags[int(event)] = float(np.nanmax(values)) if len(values) else 0.0
    return flags


def restricted_metrics(scores, labels, event_index, events, k=10):
    keep = np.isin(event_index, np.asarray(sorted(events), dtype=np.int64))
    if not keep.any():
        return dict(n_events=0, auc=float("nan"), hit_at_k=float("nan"))
    return dict(
        n_events=len(events),
        auc=roc_auc(scores[keep], labels[keep]),
        hit_at_k=hit_at_k(scores[keep], labels[keep], event_index[keep], k),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Phase-1.5 follow-ups: the features-versus-negatives '
                    'factorial and the conditioned ablations.')
    parser.add_argument('--uniform-table', required=True)
    parser.add_argument('--hard-table', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    uniform = load_probe_table(args.uniform_table)
    hard = load_probe_table(args.hard_table)
    results = {}

    # 1.5a — every cell is evaluated on the hard distribution, which is the one
    # that resembles inference.
    print('== 1.5a factorial (evaluated on hard negatives) ==')
    factorial = {}
    control_scores = None
    for feature_set in ('full89', 'all22'):
        for train_name, train_table in (('uniform', uniform), ('hard', hard)):
            scores = cross_fold_scores(train_table, hard,
                                       FEATURE_SETS[feature_set], args.folds)
            cell = dict(
                auc=roc_auc(scores, hard['labels']),
                hit_at_k=hit_at_k(scores, hard['labels'], hard['event_index'],
                                  args.top_k))
            factorial[f'{feature_set}__trained_{train_name}'] = cell
            if feature_set == 'full89' and train_name == 'uniform':
                control_scores = scores
            print(f'  {feature_set:8s} trained {train_name:8s} '
                  f'AUC {cell["auc"]:.4f}  hit@{args.top_k} {cell["hit_at_k"]:.4f}')
    results['factorial'] = factorial

    all22_scores = cross_fold_scores(hard, hard, FEATURE_SETS['all22'],
                                     args.folds)
    vertex_physics_scores = cross_fold_scores(
        hard, hard, FEATURE_SETS['vertex_physics'], args.folds)

    # 1.5b — does the gain concentrate where the control already fails?
    print('== 1.5b miss-conditioned ==')
    control_hits = per_event_hit(control_scores, hard['labels'],
                                 hard['event_index'], args.top_k)
    missed = {event for event, hit in control_hits.items() if not hit}
    hit_events = {event for event, hit in control_hits.items() if hit}
    conditioned = {}
    for name, scores in (('all22', all22_scores),
                         ('vertex_physics', vertex_physics_scores)):
        conditioned[name] = dict(
            on_control_misses=restricted_metrics(
                scores, hard['labels'], hard['event_index'], missed, args.top_k),
            on_control_hits=restricted_metrics(
                scores, hard['labels'], hard['event_index'], hit_events,
                args.top_k))
        rescued = conditioned[name]['on_control_misses']['hit_at_k']
        print(f'  {name:16s} recovers {rescued:.4f} of the '
              f'{len(missed)} control misses')
    results['miss_conditioned'] = conditioned

    # 1.5c — the SV block can only pay where a secondary vertex exists.
    print('== 1.5c secondary-vertex category ==')
    has_sv = event_level_flag(hard['columns'], hard['event_index'], 'has_sv')
    with_sv = {event for event, flag in has_sv.items() if flag > 0.5}
    without_sv = {event for event, flag in has_sv.items() if flag <= 0.5}
    category = {}
    for name, scores in (('full89', control_scores),
                         ('vertex_physics', vertex_physics_scores),
                         ('all22', all22_scores)):
        category[name] = dict(
            with_sv=restricted_metrics(scores, hard['labels'],
                                       hard['event_index'], with_sv, args.top_k),
            without_sv=restricted_metrics(scores, hard['labels'],
                                          hard['event_index'], without_sv,
                                          args.top_k))
        print(f'  {name:16s} with_sv hit@{args.top_k} '
              f'{category[name]["with_sv"]["hit_at_k"]:.4f}  '
              f'without_sv {category[name]["without_sv"]["hit_at_k"]:.4f}')
    results['sv_category'] = category
    results['sv_category']['_coverage'] = len(with_sv) / max(len(has_sv), 1)

    # 1.5d — couple_rank predates the Stage-3 replacement.
    print('== 1.5d drop-column ==')
    drops = {'couple_rank': ('couple_rank',),
             'isolation_sv': tuple(H6_ISOLATION_NAMES + H6_SV_NAMES)}
    dropped = {}
    for name, columns in drops.items():
        scores = cross_fold_scores(hard, hard, FEATURE_SETS['all22'],
                                   args.folds, drop=columns)
        dropped[name] = dict(
            auc=roc_auc(scores, hard['labels']),
            hit_at_k=hit_at_k(scores, hard['labels'], hard['event_index'],
                              args.top_k))
        print(f'  without {name:14s} AUC {dropped[name]["auc"]:.4f}  '
              f'hit@{args.top_k} {dropped[name]["hit_at_k"]:.4f}')
    results['drop_column'] = dropped
    results['drop_column']['_reference_all22'] = dict(
        auc=roc_auc(all22_scores, hard['labels']),
        hit_at_k=hit_at_k(all22_scores, hard['labels'], hard['event_index'],
                          args.top_k))

    if args.output:
        with open(args.output, 'w') as handle:
            json.dump(results, handle, indent=2)
        print(f'-> {args.output}')


if __name__ == '__main__':
    main()
