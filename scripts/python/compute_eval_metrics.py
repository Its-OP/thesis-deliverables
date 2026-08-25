from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict

import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from weaver.utils.dataset import SimpleIterDataset

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
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser


def _composite_key_tuple(observers: dict, b: int) -> tuple[int, int, int, int, int]:
    return (
        int(observers['event_run'][b]),
        int(observers['event_id'][b]),
        int(observers['event_luminosity_block'][b]),
        int(observers['source_batch_id'][b]),
        int(observers['source_microbatch_id'][b]),
    )


def _build_gt_lookup(args) -> dict[tuple, frozenset]:
    parquet_files = sorted(glob.glob(f'{args.val_data_dir}/*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files in {args.val_data_dir}')
    dataset = SimpleIterDataset(
        {'data': parquet_files},
        data_config_file=args.data_config,
        for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0),
        fetch_by_files=True,
        fetch_step=len(parquet_files),
        in_memory=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=False, num_workers=args.num_workers,
    )
    gt_lookup: dict[tuple, frozenset] = {}
    for batch_index, (X, _, observers) in enumerate(loader):
        labels = (X['pf_label'].squeeze(1) > 0.5)  # (B, P)
        for b in range(labels.shape[0]):
            key = _composite_key_tuple(observers, b)
            gt_indices = labels[b].nonzero(as_tuple=True)[0].tolist()
            gt_lookup[key] = frozenset(int(i) for i in gt_indices)
        if batch_index % 20 == 0:
            logger.info(
                f'GT lookup batch {batch_index} | events: {len(gt_lookup)}',
            )
    logger.info(f'GT lookup complete: {len(gt_lookup)} events.')
    return gt_lookup


def _track_metrics(
    sorted_indices, gt: frozenset, k_values: tuple[int, ...],
    *, with_double: bool = False,
) -> dict[str, dict[int, float]]:
    n_gt = len(gt)
    out: dict[str, dict[int, float]] = {'recall_at_K': {}, 'perfect_at_K': {}}
    if with_double:
        out['double_at_K'] = {}
    for k in k_values:
        match_count = sum(1 for i in sorted_indices[:k] if int(i) in gt)
        out['recall_at_K'][k] = match_count / n_gt if n_gt > 0 else 0.0
        out['perfect_at_K'][k] = 1.0 if (n_gt > 0 and match_count == n_gt) else 0.0
        if with_double:
            out['double_at_K'][k] = 1.0 if match_count >= 2 else 0.0
    return out


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
    out: dict[str, dict[int, float]] = {'c_at_K': {}, 'rc_at_K': {}}
    for k in k_values:
        any_gt_couple = any(
            frozenset({int(pair[0]), int(pair[1])}) in gt_couples
            for pair in sorted_couples[:k]
        )
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
    dataframe = table.to_pandas()
    logger.info(f'Eval parquet: {len(dataframe)} rows.')

    has_stage1 = (dataframe['stage1_sorted_indices'].apply(len) > 0).any()
    has_stage2 = (dataframe['stage2_sorted_indices'].apply(len) > 0).any()
    has_couples = (dataframe['stage3_sorted_couples'].apply(len) > 0).any()
    logger.info(
        f'Stages present: stage1={has_stage1}, stage2={has_stage2}, '
        f'couples={has_couples}',
    )

    per_event: list[dict] = []
    n_skipped = 0
    for row in dataframe.itertuples(index=False):
        key = (
            int(row.event_run),
            int(row.event_id),
            int(row.event_luminosity_block),
            int(row.source_batch_id),
            int(row.source_microbatch_id),
        )
        gt = gt_lookup.get(key)
        if gt is None:
            n_skipped += 1
            continue
        event_metrics: dict = {}
        if has_stage1:
            event_metrics['stage1'] = _track_metrics(
                row.stage1_sorted_indices, gt, K_TRACKS, with_double=False,
            )
        if has_stage2:
            event_metrics['stage2'] = _track_metrics(
                row.stage2_sorted_indices, gt, K_TRACKS, with_double=True,
            )
        if has_couples:
            stage2_set = frozenset(int(i) for i in row.stage2_sorted_indices)
            full_triplet = gt.issubset(stage2_set)
            event_metrics['couples'] = _couple_metrics(
                row.stage3_sorted_couples, gt, K_COUPLES,
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
