from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

NEAR_MISS_MAX = 30
FAR_TAIL_MIN = 100
COUPLE_SUNK_MIN = 30
CROWD_MAX_COUPLES = 3
CURVE_K = [1, 5, 10, 15, 20, 30, 50, 100]
DIVERSITY_CAPS = [1, 2, 3, 5, None]
_RADIX = 4096


def deduped_order(keys: np.ndarray) -> np.ndarray:
    """keys: (n, 3) sorted triples in ranking order. Returns first-occurrence
    indices, order preserved."""
    encoded = keys[:, 0] * (_RADIX * _RADIX) + keys[:, 1] * _RADIX + keys[:, 2]
    _, first = np.unique(encoded, return_index=True)
    return np.sort(first)


def gt_couple_stage3_rank(is_gt: np.ndarray,
                          couple_rank: np.ndarray) -> int | None:
    """is_gt, couple_rank: (n,). Best stage-3 rank among couples that
    produced the GT triplet."""
    if not is_gt.any():
        return None
    return int(couple_rank[is_gt].min())


def best_two_overlap_rank(keys: np.ndarray,
                          gt_triple: np.ndarray) -> int | None:
    """keys: (m, 3) deduped triples in ranking order; gt_triple: (3,).
    1-based rank of the first candidate sharing exactly two tracks with GT."""
    shared = np.isin(keys, gt_triple).sum(axis=1)
    hits = np.nonzero(shared == 2)[0]
    return int(hits[0]) + 1 if hits.size else None


def distinct_couples_in_top(couple_ids: np.ndarray, k: int) -> int:
    """couple_ids: (m,) deduped, ranking order."""
    return int(np.unique(couple_ids[:k]).size)


def top_couple_coverage(couple_ids: np.ndarray, k: int, m: int) -> float:
    """couple_ids: (n,) deduped ranking order. Fraction of the first k
    candidates whose couple is among the first m distinct couples to appear."""
    head = couple_ids[:k]
    if head.size == 0:
        return 0.0
    _, first_positions = np.unique(couple_ids, return_index=True)
    leading = couple_ids[np.sort(first_positions)[:m]]
    return float(np.isin(head, leading).mean())


def diversity_capped_gt_rank(couple_ids: np.ndarray, is_gt: np.ndarray,
                             cap: int | None) -> int | None:
    """couple_ids, is_gt: (m,) deduped, ranking order. GT rank after keeping
    at most `cap` candidates per couple; None if the GT gets dropped."""
    if cap is None:
        hits = np.nonzero(is_gt)[0]
        return int(hits[0]) + 1 if hits.size else None
    taken = 0
    per_couple: dict[int, int] = {}
    for couple_id, gt in zip(couple_ids.tolist(), is_gt.tolist()):
        used = per_couple.get(couple_id, 0)
        if used >= cap:
            continue
        per_couple[couple_id] = used + 1
        taken += 1
        if gt:
            return taken
    return None


def classify_event(*, gt_rank: int | None, gt_couple_rank: int | None,
                   two_overlap_rank: int | None,
                   n_couples_top10: int) -> str:
    if gt_rank is None:
        return 'absent'
    if gt_rank <= 10:
        return 'hit'
    if two_overlap_rank is not None and two_overlap_rank <= 10:
        return 'third_pion_confusion'
    if gt_rank <= NEAR_MISS_MAX:
        return 'near_miss'
    if gt_couple_rank is not None and gt_couple_rank >= COUPLE_SUNK_MIN:
        return 'couple_sunk'
    if gt_rank > FAR_TAIL_MIN:
        return 'far_tail'
    if n_couples_top10 <= CROWD_MAX_COUPLES:
        return 'crowded_out'
    return 'mid_other'


def _build_model(checkpoint: dict, device: torch.device):
    from weaver.nn.model.TripletReranker import TripletReranker
    from weaver.nn.model.VertexFit import FIT_NAMES
    args = checkpoint['args']
    fit_stats = None
    if args['fit_mode'] == 'layer':
        fit_stats = {name: checkpoint['norm_stats'][name] for name in FIT_NAMES}
    model = TripletReranker(
        input_mode=args['input_mode'], hidden_dim=args['hidden_dim'],
        num_residual_blocks=args['num_residual_blocks'], dropout=args['dropout'],
        ranking_num_samples=args['num_negatives'],
        ranking_temperature=args['temperature'],
        label_smoothing=args['label_smoothing'],
        projector_dim=args['projector_dim'],
        feature_names=checkpoint['feature_names'], loss_mode=args['loss_mode'],
        num_attention_layers=args['attention_layers'],
        attention_heads=args['attention_heads'],
        track_embed_dim=args['track_embed_dim'], trunk_norm=args['trunk_norm'],
        fusion=args['fusion'], aux_from_b_weight=args['aux_fromb_weight'],
        vertex_fit_layer=args['fit_mode'] == 'layer', fit_norm_stats=fit_stats)
    model.load_state_dict(checkpoint['triplet_reranker_state_dict'])
    return model.to(device).eval()


def _gt_visible_pt(table, r: int, gt_triple: np.ndarray) -> float | None:
    if (gt_triple < 0).any():
        return None
    pt = np.asarray(table.tracks['track_pt'][r].values)[gt_triple]
    phi = np.asarray(table.tracks['track_phi'][r].values)[gt_triple]
    return float(math.hypot((pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum()))


def _summarize(records: list[dict], gate: str, n_events: int) -> dict:
    rows = [record for record in records if record['gate'] == gate]
    ranks = np.asarray([row['gt_rank'] for row in rows
                        if row['gt_rank'] is not None])
    survivors = np.asarray([row['n_survivors'] for row in rows])
    classes = {}
    for row in rows:
        classes[row['class']] = classes.get(row['class'], 0) + 1
    summary = {
        'n_events': n_events,
        'ceiling': float(len(ranks) / n_events),
        'class_counts': dict(sorted(classes.items())),
        'T@K': {k: float((ranks <= k).sum() / n_events) for k in CURVE_K},
        'gt_rank_median': float(np.median(ranks)) if ranks.size else None,
        'gt_rank_p90': float(np.percentile(ranks, 90)) if ranks.size else None,
        'survivors_mean': float(survivors.mean()) if survivors.size else None,
        'survivors_median': float(np.median(survivors)) if survivors.size else None,
    }
    for cap in DIVERSITY_CAPS:
        capped = np.asarray([row['capped_ranks'][str(cap)] for row in rows
                             if row['capped_ranks'][str(cap)] is not None])
        summary[f'T@10_cap{cap}'] = float((capped <= 10).sum() / n_events)
    return summary


def run(args: argparse.Namespace) -> dict:
    from scripts.python.eval_triplet_rank_baselines import (
        OPERATING_POINTS, deduped_gt_rank, load_operating_points)
    from train_triplet_reranker import _batch_kwargs
    from utils.triplet_rank_data import (TripletRankDataset,
                                         collate_triplet_rank_eval)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    model = _build_model(checkpoint, device)
    model_args = checkpoint['args']

    points = dict(OPERATING_POINTS)
    if os.path.exists(args.operating_points):
        points = load_operating_points(args.operating_points)
    gates = {name: points[name][1] for name in args.gates.split(',')}
    serve_tau = min(gates.values())
    logit_taus = {gate: (math.log(tau / (1.0 - tau))
                         if 0.0 < tau < 1.0 else -float('inf'))
                  for gate, tau in gates.items()}

    dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=serve_tau, mode='eval',
        norm_stats=checkpoint['norm_stats'], seed=args.seed,
        extra_features=model_args['extra_features'],
        context_features=model_args['context_features'],
        vertex_fit=model_args['fit_mode'], from_b_targets=False,
        track32=(model_args['input_mode'] == 'hierarchical'
                 and model_args['track_embed_dim'] == 32))
    if dataset.feature_names != checkpoint['feature_names']:
        raise SystemExit('dataset resolves different feature names than the '
                         'checkpoint — wrong artifact or source shards')

    scalars = pq.read_table(args.candidates,
                            columns=['gt_i', 'gt_j', 'gt_k', 'recon',
                                     'n_tierh', 'n_tracks'])
    gt_columns = np.stack([np.asarray(scalars[name])
                           for name in ('gt_i', 'gt_j', 'gt_k')], axis=1)
    n_tierh = np.asarray(scalars['n_tierh'])

    n_rows = dataset.table.num_rows
    if args.events and args.events < n_rows:
        event_indices = np.sort(np.random.default_rng(args.seed).choice(
            n_rows, args.events, replace=False))
    else:
        event_indices = np.arange(n_rows)

    loader = DataLoader(Subset(dataset, [int(r) for r in event_indices]),
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        collate_fn=collate_triplet_rank_eval)
    records: list[dict] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            valid_mask = batch['valid_mask'].to(device)
            scores = model(batch['features'].to(device), valid_mask=valid_mask,
                           **_batch_kwargs(batch, device, for_loss=False))
            scores = scores.masked_fill(~valid_mask, float('-inf')).cpu().numpy()
            counts = batch['counts'].numpy()
            for b in range(scores.shape[0]):
                r = int(event_indices[cursor])
                cursor += 1
                n = int(counts[b])
                if n == 0:
                    continue
                arrays = dataset.table.candidate_arrays(r)
                serving = np.nonzero((arrays['filter_score'] >= serve_tau)
                                     & (arrays['row_kind'] == 0))[0]
                assert serving.size == n, \
                    f'event {r}: {serving.size} serving rows vs {n} in the item'
                keys = batch['keys'][b].numpy()
                pos = batch['pos_mask'][b, :n].numpy()
                logits = batch['filter_logit'][b, :n].numpy()
                event_scores = scores[b, :n]
                couple_ids = (arrays['cand_i'][serving].astype(np.int64) * _RADIX
                              + arrays['cand_j'][serving].astype(np.int64))
                stage3_ranks = arrays['couple_rank'][serving]
                gt_triple = np.sort(gt_columns[r])
                pt_visible = _gt_visible_pt(dataset.table, r, gt_triple)

                for gate, logit_tau in logit_taus.items():
                    mask = logits >= logit_tau
                    record = {
                        'event': r, 'gate': gate,
                        'n_survivors': int(mask.sum()),
                        'n_tierh': int(n_tierh[r]),
                        'gt_pt_visible': pt_visible,
                        'gt_rank': None, 'gt_couple_rank': None,
                        'two_overlap_rank': None, 'n_couples_top10': None,
                        'capped_ranks': {str(cap): None
                                         for cap in DIVERSITY_CAPS},
                    }
                    if not mask.any():
                        record['class'] = 'absent'
                        records.append(record)
                        continue
                    order = np.argsort(-event_scores[mask], kind='stable')
                    keys_ordered = keys[mask][order]
                    dedup = deduped_order(keys_ordered)
                    couples_deduped = couple_ids[mask][order][dedup]
                    is_gt_deduped = pos[mask][order][dedup]
                    record['n_couples_top10'] = distinct_couples_in_top(
                        couples_deduped, 10)
                    for top_k in (50, 100, 500):
                        record[f'n_couples_top{top_k}'] = \
                            distinct_couples_in_top(couples_deduped, top_k)
                    for top_k in (100, 500):
                        for leading in (1, 5, 12):
                            record[f'cov_top{leading}_at{top_k}'] = \
                                top_couple_coverage(couples_deduped, top_k,
                                                    leading)
                    if not pos[mask].any():
                        record['class'] = 'absent'
                        records.append(record)
                        continue
                    record['gt_rank'] = deduped_gt_rank(keys_ordered,
                                                        pos[mask][order])
                    record['gt_couple_rank'] = gt_couple_stage3_rank(
                        pos[mask], stage3_ranks[mask])
                    if (gt_triple >= 0).all():
                        record['two_overlap_rank'] = best_two_overlap_rank(
                            keys_ordered[dedup], gt_triple)
                    record['capped_ranks'] = {
                        str(cap): diversity_capped_gt_rank(
                            couples_deduped, is_gt_deduped, cap)
                        for cap in DIVERSITY_CAPS}
                    record['class'] = classify_event(
                        gt_rank=record['gt_rank'],
                        gt_couple_rank=record['gt_couple_rank'],
                        two_overlap_rank=record['two_overlap_rank'],
                        n_couples_top10=record['n_couples_top10'])
                    records.append(record)

    n_events = len(event_indices)
    result = {
        'checkpoint': args.checkpoint,
        'candidates': args.candidates,
        'n_events': n_events,
        'gates': {gate: _summarize(records, gate, n_events)
                  for gate in gates},
    }
    if args.records_out:
        import pandas as pd
        flat = [{**{key: value for key, value in record.items()
                    if key != 'capped_ranks'},
                 **{f'cap_{cap}': record['capped_ranks'][str(cap)]
                    for cap in DIVERSITY_CAPS}}
                for record in records]
        pd.DataFrame(flat).to_parquet(args.records_out)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2)
    for gate, summary in result['gates'].items():
        print(f"[{gate}] ceiling {summary['ceiling']:.4f} "
              f"T@10 {summary['T@K'][10]:.4f} "
              f"classes {summary['class_counts']}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--operating-points',
                        default='data/triplet_rank_v2/operating_points.json')
    parser.add_argument('--gates', default='tierH,p99,p95')
    parser.add_argument('--output', required=True)
    parser.add_argument('--records-out', default=None)
    parser.add_argument('--events', type=int, default=0)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
