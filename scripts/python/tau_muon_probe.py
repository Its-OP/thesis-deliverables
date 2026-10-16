import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from scripts.python.probe_triplet_features import grouped_folds, roc_auc
from utils import tau_muon_proxy as proxy

POLICIES = ('oracle', 'min_dr', 'min_dz')
# 'none' = the probe model on the baseline score alone; it shares the binning/tie behaviour of the
# augmented models, so muon deltas are read against it rather than against the raw ordering.
PROBE_POLICIES = ('none',) + POLICIES
PROBE_ESTIMATOR = dict(max_leaf_nodes=31, max_depth=6, learning_rate=0.1, max_iter=200,
                       random_state=0)
DUMP_COLUMNS = proxy.IDENTITY_COLUMNS + ['stage2_sorted_indices', 'stage3_sorted_couples',
                                         'stage3_couple_scores']
CANDIDATE_COLUMNS = ['n_tracks', 'cand_i', 'cand_j', 'cand_k', 'gbdt8_score', 'is_gt',
                     'gt_i', 'gt_j', 'gt_k', 'recon']
CANDIDATE_ROW_GROUP = 500
MISS_K = {'couples': 100, 'triplets': 10}


def load_couple_dump(path, identities):
    """Returns {identity: {'pool': (K,), 'couples': (C, 2), 'scores': (C,)}} for the identities
    present in the dump; the dump is joined on the 5-column identity key, never positionally."""
    parquet_file = pq.ParquetFile(path)
    rows = {}
    for group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(group, columns=DUMP_COLUMNS)
        identity = np.stack([table.column(name).to_numpy() for name in proxy.IDENTITY_COLUMNS], axis=1)
        keys = [tuple(int(value) for value in row) for row in identity]
        wanted = [index for index, key in enumerate(keys) if key in identities]
        if not wanted:
            continue
        pool_values, pool_offsets = proxy.list_parts(table.column('stage2_sorted_indices'))
        score_values, score_offsets = proxy.list_parts(table.column('stage3_couple_scores'))
        outer = table.column('stage3_sorted_couples').combine_chunks()
        outer_offsets = np.asarray(outer.offsets)
        inner_offsets = np.asarray(outer.values.offsets)
        inner_values = np.asarray(outer.values.values)
        for index in wanted:
            start, stop = outer_offsets[index], outer_offsets[index + 1]
            flat = inner_values[inner_offsets[start]:inner_offsets[stop]]
            assert len(flat) == 2 * (stop - start), 'stage3_sorted_couples entries must be pairs'
            rows[keys[index]] = dict(
                pool=pool_values[pool_offsets[index]:pool_offsets[index + 1]],
                couples=flat.reshape(-1, 2),
                scores=score_values[score_offsets[index]:score_offsets[index + 1]].astype(np.float64))
    return rows


def load_candidate_rows(path, start, count):
    """Reads events [start, start + count) of a thesis-era candidates parquet (positionally aligned
    with the source shards). Returns a list of dicts: n_tracks, cand (N, 3), score (N,), is_gt (N,), gt, recon."""
    parquet_file = pq.ParquetFile(path)
    first_group = start // CANDIDATE_ROW_GROUP
    last_group = (start + count - 1) // CANDIDATE_ROW_GROUP
    table = parquet_file.read_row_groups(list(range(first_group, last_group + 1)), columns=CANDIDATE_COLUMNS)
    table = table.slice(start - first_group * CANDIDATE_ROW_GROUP, count)
    parts = {name: proxy.list_parts(table.column(name)) for name in ('cand_i', 'cand_j', 'cand_k', 'gbdt8_score', 'is_gt')}
    scalars = {name: table.column(name).to_numpy() for name in ('n_tracks', 'gt_i', 'gt_j', 'gt_k', 'recon')}
    rows = []
    for row in range(table.num_rows):
        span = slice(parts['cand_i'][1][row], parts['cand_i'][1][row + 1])
        rows.append(dict(
            n_tracks=int(scalars['n_tracks'][row]),
            cand=np.stack([parts[name][0][span].astype(np.int64) for name in ('cand_i', 'cand_j', 'cand_k')], axis=1),
            score=parts['gbdt8_score'][0][span].astype(np.float64),
            is_gt=parts['is_gt'][0][span].astype(bool),
            gt=(int(scalars['gt_i'][row]), int(scalars['gt_j'][row]), int(scalars['gt_k'][row])),
            recon=bool(scalars['recon'][row])))
    return rows


def _policy_features(event, members, oracle_index):
    system = proxy.system_from_members(event, members)
    relations = proxy.muon_relations(event, system)
    features = {}
    for policy in POLICIES:
        indices = proxy.policy_indices(relations, event['muon_soft'], policy, oracle_index=oracle_index)
        for name, values in proxy.gather(relations, indices).items():
            features[f'{policy}__{name}'] = values
    return features


def gt_rank(scores, labels):
    """scores, labels: (N,) for one event. Tie-aware rank of the best GT row: impostors scoring
    strictly above it plus half of those tied with it; -1 when the event has no GT row."""
    labels = np.asarray(labels) > 0
    if not labels.any():
        return -1.0
    best = scores[labels].max()
    impostors = scores[~labels]
    return float((impostors > best).sum() + 0.5 * (impostors == best).sum())


def build_couple_rows(event, dump_row, top_k2, oracle_index):
    """One row per stored couple: label, baseline (stage-3) score, muon features per policy;
    gt_rank is the tie-aware rank of the first GT couple in the stored list, -1 if absent."""
    gt = proxy.gt_members(event['track_label'])
    couples = dump_row['couples']
    label = np.isin(couples, gt).all(axis=1)
    rows = dict(label=label.astype(np.int8), baseline_score=dump_row['scores'],
                gt_rank=gt_rank(dump_row['scores'], label),
                n_gt_in_pool=int(np.isin(gt, dump_row['pool'][:top_k2]).sum()))
    rows.update(_policy_features(event, couples, oracle_index))
    return rows


def build_triplet_rows(event, candidate, impostor_cap, oracle_index):
    """Rows = top impostor_cap impostors by the stored gate score plus every GT row, sorted by score;
    gt_rank is the tie-aware rank of the GT triplet within the full candidate list, -1 if absent."""
    assert candidate['n_tracks'] == len(event['track_pt']), 'candidates are not aligned with the shard'
    score, is_gt = candidate['score'], candidate['is_gt']
    order = np.argsort(-score, kind='stable')
    impostors = order[~is_gt[order]][:impostor_cap]
    keep = order[np.isin(order, np.concatenate([impostors, np.flatnonzero(is_gt)]))]
    rows = dict(label=is_gt[keep].astype(np.int8), baseline_score=score[keep],
                gt_rank=gt_rank(score, is_gt))
    rows.update(_policy_features(event, candidate['cand'][keep], oracle_index))
    return rows


def _iterate_shard_events(shard_glob, max_events):
    """Yields (global_offset, events) per row group across the sorted shards."""
    offset = 0
    for shard in sorted(glob.glob(shard_glob)):
        parquet_file = pq.ParquetFile(shard)
        for group in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(group, columns=proxy.SHARD_COLUMNS)
            if max_events is not None:
                table = table.slice(0, max(0, min(table.num_rows, max_events - offset)))
            if table.num_rows == 0:
                return
            yield offset, proxy.events_from_table(table)
            offset += table.num_rows


def _gt_rank_from_scores(scores, labels, event_index):
    """Per event: tie-aware rank of the best GT row under the given scores. Returns {event: rank}."""
    ranks = {}
    order = np.argsort(event_index, kind='stable')
    boundaries = np.flatnonzero(np.diff(event_index[order])) + 1
    for block in np.split(order, boundaries):
        rank = gt_rank(scores[block], labels[block])
        if rank >= 0:
            ranks[int(event_index[block[0]])] = rank
    return ranks


def _rate_at_k(ranks, k_values):
    ranks = np.asarray(ranks)
    return {str(k): float((ranks < k).mean()) if len(ranks) else float('nan') for k in k_values}


def _probe(columns, labels, event_index, feature_names, n_folds):
    features = np.stack([columns[name] for name in feature_names], axis=1)
    scores = np.empty(len(labels), dtype=np.float64)
    for fold_rows in grouped_folds(event_index, n_folds):
        train_rows = np.setdiff1d(np.arange(len(labels)), fold_rows)
        model = HistGradientBoostingClassifier(**PROBE_ESTIMATOR)
        model.fit(features[train_rows], labels[train_rows])
        scores[fold_rows] = model.predict_proba(features[fold_rows])[:, 1]
    return scores


def _group_metrics(columns, labels, event_index, group_mask, baseline_ranks, model_scores, k_values, miss_k):
    events = np.unique(event_index[group_mask])
    block = dict(n_events=int(len(events)),
                 baseline=dict(c_at_k=_rate_at_k([baseline_ranks[event] for event in events], k_values)),
                 feature_auc={}, probe={}, miss_conditioned={})
    misses = [event for event in events if baseline_ranks[event] >= miss_k]
    block['n_baseline_misses_at_k'] = len(misses)
    for policy in POLICIES:
        block['feature_auc'][policy] = {
            name: roc_auc(columns[f'{policy}__{name}'][group_mask & np.isfinite(columns[f'{policy}__{name}'])],
                          labels[group_mask & np.isfinite(columns[f'{policy}__{name}'])])
            for name in proxy.FEATURE_NAMES}
        block['feature_auc'][policy]['coverage'] = float(
            np.isfinite(columns[f'{policy}__delta_r'][group_mask]).mean())
    for policy in PROBE_POLICIES:
        ranks = _gt_rank_from_scores(model_scores[policy][group_mask], labels[group_mask], event_index[group_mask])
        block['probe'][policy] = dict(
            auc=roc_auc(model_scores[policy][group_mask], labels[group_mask]),
            c_at_k=_rate_at_k([ranks[event] for event in events], k_values))
        block['miss_conditioned'][policy] = dict(
            recovered_at_k=float(np.mean([ranks[event] < miss_k for event in misses])) if misses else float('nan'))
    return block


def run(level, shard_glob, source, variant, max_events=None, top_k2=125, impostor_cap=200,
        n_folds=3, k_values=(10, 50, 100, 200), complement_events=None):
    """Builds candidate rows with muon features and measures, per subset / complement, the
    baseline ranking, per-feature AUC, and an event-grouped GBDT probe over baseline score + muon features.
    complement_events caps the muon-less events kept (first come, in shard order)."""
    chunks = []
    n_events = n_gt_in_list = n_subset = n_complement = 0
    dump_rows = None
    if level == 'couples':
        identities = set()
        for _, events in _iterate_shard_events(shard_glob, max_events):
            identities.update(event['identity'] for event in events)
        dump_rows = load_couple_dump(source, identities)
    for offset, events in _iterate_shard_events(shard_glob, max_events):
        candidates = load_candidate_rows(source, offset, len(events)) if level == 'triplets' else None
        for position, event in enumerate(events):
            n_events += 1
            oracle_index = proxy.select_proxy_muon(proxy.gt_relations(event), event['muon_soft'], variant)
            in_subset = oracle_index >= 0
            if level == 'couples':
                dump_row = dump_rows.get(event['identity'])
                if dump_row is None:
                    continue
                rows = build_couple_rows(event, dump_row, top_k2, oracle_index)
            else:
                rows = build_triplet_rows(event, candidates[position], impostor_cap, oracle_index)
            if rows['gt_rank'] < 0:
                continue
            n_gt_in_list += 1
            if in_subset:
                n_subset += 1
            else:
                if complement_events is not None and n_complement >= complement_events:
                    continue
                n_complement += 1
            rows['event_index'] = np.full(len(rows['label']), offset + position, dtype=np.int64)
            rows['in_subset'] = np.full(len(rows['label']), in_subset, dtype=bool)
            rows['baseline_rank'] = np.full(len(rows['label']), rows['gt_rank'], dtype=np.float64)
            chunks.append({key: value for key, value in rows.items() if key not in ('gt_rank', 'n_gt_in_pool')})
    if not chunks:
        raise RuntimeError('no events with the GT candidate inside the stored list')
    columns = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    labels = columns['label'].astype(np.float64)
    event_index = columns['event_index']
    baseline_ranks = {int(event): float(rank) for event, rank in
                      zip(event_index, columns['baseline_rank'])}
    model_scores = {policy: _probe(columns, labels, event_index,
                                   ['baseline_score'] + ([] if policy == 'none' else
                                                         [f'{policy}__{name}' for name in proxy.FEATURE_NAMES]),
                                   n_folds) for policy in PROBE_POLICIES}
    result = dict(level=level, variant=variant, source=source, n_events=n_events,
                  n_events_with_gt_in_list=n_gt_in_list, n_subset=n_subset, n_complement=n_complement,
                  n_rows=int(len(labels)), k_values=list(k_values), miss_k=MISS_K[level], n_folds=n_folds)
    for group, mask in (('subset', columns['in_subset']), ('complement', ~columns['in_subset'])):
        result[group] = _group_metrics(columns, labels, event_index, mask, baseline_ranks, model_scores,
                                       k_values, MISS_K[level]) if mask.any() else None
    return result


def main():
    parser = argparse.ArgumentParser(description='tau-muon feature probe at couple or triplet level')
    parser.add_argument('--level', choices=('couples', 'triplets'), required=True)
    parser.add_argument('--shards', required=True, help='glob of source shards (train or eval)')
    parser.add_argument('--source', required=True, help='couples dump or thesis-era candidates parquet')
    parser.add_argument('--variant', choices=sorted(proxy.PROXY_VARIANTS), default='full')
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--complement-events', type=int, default=None)
    parser.add_argument('--top-k2', type=int, default=125)
    parser.add_argument('--impostor-cap', type=int, default=200)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--k-values', default='10,50,100,200')
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    result = run(level=arguments.level, shard_glob=arguments.shards, source=arguments.source,
                 variant=arguments.variant, max_events=arguments.max_events, top_k2=arguments.top_k2,
                 impostor_cap=arguments.impostor_cap, n_folds=arguments.folds,
                 k_values=tuple(int(value) for value in arguments.k_values.split(',')),
                 complement_events=arguments.complement_events)
    os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
    with open(arguments.output, 'w') as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({key: result[key] for key in ('n_events', 'n_events_with_gt_in_list', 'n_subset', 'n_complement')}))
    for group in ('subset', 'complement'):
        block = result[group]
        if block is None:
            continue
        print(f'[{group}] events {block["n_events"]}  baseline C@K {block["baseline"]["c_at_k"]}')
        for policy in PROBE_POLICIES:
            auc = block['feature_auc'].get(policy, {})
            print(f'  {policy:7s} auc {auc}  probe C@K {block["probe"][policy]["c_at_k"]}  '
                  f'miss-recovered {block["miss_conditioned"][policy]["recovered_at_k"]:.3f}')


if __name__ == '__main__':
    main()
