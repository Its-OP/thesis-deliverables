from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger('compute_eval_metrics')

K_TRACKS: tuple[int, ...] = (50, 60, 70, 75, 80, 100, 125, 150, 200, 256,
                             280, 300, 320, 350, 400, 600)
K_COUPLES: tuple[int, ...] = (30, 40, 50, 60, 75, 100, 125, 150, 200)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Aggregate R@K, P@K, D@K, C@K, RC@K from an '
                    'eval_cascade_pipeline parquet + val data → JSON.',
    )
    parser.add_argument('--eval-parquet', required=True)
    parser.add_argument('--val-data-dir', required=True)
    parser.add_argument('--output', required=True)
    return parser


IDENTITY_COLUMNS = ('event_run', 'event_id', 'event_luminosity_block',
                    'source_batch_id', 'source_microbatch_id')


def build_gt_lookup(parquet_files: list[str]) -> dict[tuple, frozenset]:
    """parquet_files: raw source shards. Reads only the identity columns and
    the per-track label column; track order in the shards is the loader
    order (the data config applies no track sorting), so nonzero label
    positions are directly comparable to dumped track indices."""
    gt_lookup: dict[tuple, frozenset] = {}
    for shard in parquet_files:
        table = pq.read_table(
            shard, columns=list(IDENTITY_COLUMNS) + ['track_label_from_tau'])
        identity_arrays = [table[name].to_numpy(zero_copy_only=False)
                           for name in IDENTITY_COLUMNS]
        labels_column = table['track_label_from_tau']
        for row in range(table.num_rows):
            key = tuple(int(array[row]) for array in identity_arrays)
            labels = np.asarray(labels_column[row].values, dtype=np.float32)
            gt_lookup[key] = frozenset(np.nonzero(labels > 0.5)[0].tolist())
        logger.info(f'GT lookup: {len(gt_lookup)} events after {shard}')
    return gt_lookup


def _build_gt_lookup(args) -> dict[tuple, frozenset]:
    parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files in {args.val_data_dir}')
    return build_gt_lookup(parquet_files)


def gt_ranks_in_ordering(ordering: np.ndarray,
                         gt: frozenset) -> np.ndarray:
    """ordering: (N,) track indices in rank order. Returns the sorted rank
    positions of the GT tracks that appear in the ordering."""
    return np.flatnonzero(np.isin(ordering, list(gt)))


def metrics_from_ranks(ranks: np.ndarray, n_gt: int,
                       k_values: tuple[int, ...],
                       *, with_double: bool = False,
                       ) -> dict[str, dict[int, float]]:
    """ranks: sorted rank positions of found GT tracks; n_gt: total GT."""
    out: dict[str, dict[int, float]] = {'recall_at_K': {}, 'perfect_at_K': {}}
    if with_double:
        out['double_at_K'] = {}
    for k in k_values:
        match_count = int(np.searchsorted(ranks, k, side='left'))
        out['recall_at_K'][k] = match_count / n_gt if n_gt > 0 else 0.0
        out['perfect_at_K'][k] = 1.0 if (n_gt > 0 and match_count == n_gt) else 0.0
        if with_double:
            out['double_at_K'][k] = 1.0 if match_count >= 2 else 0.0
    return out


def _track_metrics(
    sorted_indices, gt: frozenset, k_values: tuple[int, ...],
    *, with_double: bool = False,
) -> dict[str, dict[int, float]]:
    ordering = np.asarray(sorted_indices)
    ranks = gt_ranks_in_ordering(ordering, gt)
    return metrics_from_ranks(ranks, len(gt), k_values,
                              with_double=with_double)


def _couple_metrics(
    sorted_couples, gt: frozenset, k_values: tuple[int, ...],
    *, full_triplet_in_stage2: bool,
) -> dict[str, dict[int, float]]:
    sorted_gt = sorted(gt)
    gt_couples = frozenset(
        frozenset({sorted_gt[i], sorted_gt[j]})
        for i in range(len(sorted_gt))
        for j in range(i + 1, len(sorted_gt))
    )
    pairs = np.asarray(sorted_couples).reshape(-1, 2)
    if pairs.size:
        pair_keys = pairs.min(axis=1) * 4096 + pairs.max(axis=1)
        gt_keys = [min(couple) * 4096 + max(couple)
                   for couple in map(sorted, gt_couples)]
        hit_positions = np.flatnonzero(np.isin(pair_keys, gt_keys))
        first_hit = int(hit_positions[0]) if hit_positions.size else None
    else:
        first_hit = None
    out: dict[str, dict[int, float]] = {'c_at_K': {}, 'rc_at_K': {}}
    for k in k_values:
        any_gt_couple = first_hit is not None and first_hit < k
        out['c_at_K'][k] = 1.0 if any_gt_couple else 0.0
        out['rc_at_K'][k] = (
            1.0 if (any_gt_couple and full_triplet_in_stage2) else 0.0
        )
    return out


def _aggregate(
    per_event_metrics: list[dict], n_events: int,
) -> dict[str, dict[str, dict[str, float]]]:
    if n_events == 0:
        return {}
    sums: dict = {}
    for event_metrics in per_event_metrics:
        for stage, stage_metrics in event_metrics.items():
            sums.setdefault(stage, {})
            for metric_name, k_dict in stage_metrics.items():
                sums[stage].setdefault(metric_name, defaultdict(float))
                for k, value in k_dict.items():
                    sums[stage][metric_name][k] += value
    averages: dict = {}
    for stage, stage_sums in sums.items():
        averages[stage] = {}
        for metric_name, k_sums in stage_sums.items():
            averages[stage][metric_name] = {
                str(k): k_sums[k] / n_events for k in sorted(k_sums)
            }
    return averages


def _check_monotone(metric_dict: dict[str, float], label: str) -> None:
    keys_sorted = sorted(int(k) for k in metric_dict)
    previous_value = -1.0
    for k in keys_sorted:
        current = metric_dict[str(k)]
        if current < previous_value - 1e-6:
            logger.warning(
                f'{label} non-monotone at K={k}: '
                f'{current:.4f} < previous {previous_value:.4f}',
            )
        previous_value = current


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args(argv)

    logger.info(f'Building GT lookup from {args.val_data_dir} ...')
    gt_lookup = _build_gt_lookup(args)

    logger.info(f'Reading eval parquet: {args.eval_parquet}')
    needed_columns = [
        'event_run', 'event_id', 'event_luminosity_block',
        'source_batch_id', 'source_microbatch_id',
        'stage1_sorted_indices', 'stage2_sorted_indices',
        'stage3_sorted_couples',
    ]
    table = pq.read_table(args.eval_parquet, columns=needed_columns)
    logger.info(f'Eval parquet: {table.num_rows} rows.')

    identity_arrays = [table[name].to_numpy(zero_copy_only=False)
                       for name in IDENTITY_COLUMNS]
    stage1_column = table['stage1_sorted_indices']
    stage2_column = table['stage2_sorted_indices']
    couples_column = table['stage3_sorted_couples']
    has_stage1 = len(stage1_column[0].values) > 0
    has_stage2 = len(stage2_column[0].values) > 0
    has_couples = len(couples_column[0].values) > 0
    logger.info(
        f'Stages present: stage1={has_stage1}, stage2={has_stage2}, '
        f'couples={has_couples}',
    )

    per_event: list[dict] = []
    n_skipped = 0
    for row in range(table.num_rows):
        key = tuple(int(array[row]) for array in identity_arrays)
        gt = gt_lookup.get(key)
        if gt is None:
            n_skipped += 1
            continue
        event_metrics: dict = {}
        if has_stage1:
            event_metrics['stage1'] = _track_metrics(
                np.asarray(stage1_column[row].values), gt, K_TRACKS,
                with_double=True,
            )
        if has_stage2:
            stage2_indices = np.asarray(stage2_column[row].values)
            event_metrics['stage2'] = _track_metrics(
                stage2_indices, gt, K_TRACKS, with_double=True,
            )
        if has_couples:
            full_triplet = gt.issubset(
                frozenset(stage2_indices.tolist()))
            pairs = np.asarray(
                couples_column[row].values.flatten()).reshape(-1, 2)
            event_metrics['couples'] = _couple_metrics(
                pairs, gt, K_COUPLES,
                full_triplet_in_stage2=full_triplet,
            )
        per_event.append(event_metrics)

    n_events = len(per_event)
    averages = _aggregate(per_event, n_events)

    for stage, stage_metrics in averages.items():
        for metric_name, k_dict in stage_metrics.items():
            _check_monotone(k_dict, f'{stage}.{metric_name}')

    output_dict: dict = {
        'input': args.eval_parquet,
        'n_events': n_events,
        'n_events_skipped_missing_gt': n_skipped,
        'k_tracks': list(K_TRACKS),
        'k_couples': list(K_COUPLES),
    }
    output_dict.update(averages)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
    logger.info(f'Wrote: {args.output}')
    logger.info(
        f'Events: {n_events} processed, {n_skipped} skipped (missing GT).',
    )
    if has_stage1:
        logger.info(
            f"Stage 1 R@200 = {averages['stage1']['recall_at_K']['200']:.4f}",
        )
    if has_stage2:
        logger.info(
            f"Stage 2 R@200 = {averages['stage2']['recall_at_K']['200']:.4f}",
        )
    if has_couples:
        logger.info(
            f"C@100 = {averages['couples']['c_at_K']['100']:.4f} | "
            f"RC@100 = {averages['couples']['rc_at_K']['100']:.4f}",
        )


if __name__ == '__main__':
    main()
