from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import resource
import sys
import time
import traceback

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from networks.lowpt_tau_CascadeReranker import infer_stage1_kwargs
from utils.couple_features import (
    COUPLE_REST_DIM,
    TRACK_EMBED_DIM,
    build_couple_features_batched,
)
from utils.dataset_helpers import (
    extract_label_from_inputs,
    trim_to_max_valid_tracks,
)
from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.CoupleReranker import CoupleReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.dataset import SimpleIterDataset

logger = logging.getLogger('eval_cascade_pipeline')

# Score columns are parallel to their index columns: stageN_scores[m] is the score of
# stageN_sorted_indices[m] (stage3_couple_scores[m] of stage3_sorted_couples[m]), descending.
OUTPUT_SCHEMA = pa.schema([
    pa.field('event_run', pa.int32()),
    pa.field('event_id', pa.int64()),
    pa.field('event_luminosity_block', pa.int32()),
    pa.field('source_batch_id', pa.int32()),
    pa.field('source_microbatch_id', pa.int32()),
    pa.field('stage', pa.string()),
    pa.field('stage1_sorted_indices', pa.list_(pa.int32())),
    pa.field('stage2_sorted_indices', pa.list_(pa.int32())),
    pa.field('stage3_sorted_couples', pa.list_(pa.list_(pa.int32()))),
    pa.field('stage1_scores', pa.list_(pa.float32())),
    pa.field('stage2_scores', pa.list_(pa.float32())),
    pa.field('stage3_couple_scores', pa.list_(pa.float32())),
])

# --dump-stage3-inputs: everything the dump-based Stage-3 trainer needs. The
# k1_* lists are the gathered top-K1 block (fixed length; flat row-major for
# the 2-D tensors); padded pool slots carry -inf in k1_stage2_scores. The
# cone_* lists hold the FULL event's valid tracks (companion-cone
# candidates), variable length.
STAGE3_DUMP_SCHEMA = pa.schema(
    list(OUTPUT_SCHEMA)
    + [
        pa.field('k1_features', pa.list_(pa.float32())),
        pa.field('k1_points', pa.list_(pa.float32())),
        pa.field('k1_lorentz', pa.list_(pa.float32())),
        pa.field('k1_stage1_scores', pa.list_(pa.float32())),
        pa.field('k1_stage2_scores', pa.list_(pa.float32())),
        pa.field('k1_labels', pa.list_(pa.int32())),
        pa.field('k1_original_indices', pa.list_(pa.int32())),
        pa.field('cone_eta', pa.list_(pa.float32())),
        pa.field('cone_phi', pa.list_(pa.float32())),
        pa.field('cone_dz', pa.list_(pa.float32())),
        pa.field('cone_pt', pa.list_(pa.float32())),
    ]
)


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    full_prefix = prefix if prefix.endswith('.') else f'{prefix}.'
    stripped = {
        key[len(full_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(full_prefix)
    }
    return stripped or state_dict


def _load_stage1(
    path: str, data_config, num_neighbors: int, device: str,
) -> TrackPreFilter:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = _strip_prefix(checkpoint['model_state_dict'], 'stage1')
    kwargs = infer_stage1_kwargs(state_dict, stage1_num_neighbors=num_neighbors)
    kwargs['input_dim'] = len(data_config.input_dicts['pf_features'])
    model = TrackPreFilter(**kwargs)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _load_stage2(
    path: str, input_dim: int, device: str,
) -> tuple[CascadeReranker, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {}) or {}
    state_dict = _strip_prefix(checkpoint['model_state_dict'], 'stage2')

    pair_embed_dims = args.get('stage2_pair_embed_dims', '64,64,64')
    if isinstance(pair_embed_dims, str):
        pair_embed_dims = [int(x) for x in pair_embed_dims.split(',')]

    model = CascadeReranker(
        input_dim=input_dim,
        embed_dim=args.get('stage2_embed_dim', 512),
        num_heads=args.get('stage2_num_heads', 8),
        num_layers=args.get('stage2_num_layers', 2),
        pair_input_dim=4,
        pair_extra_dim=args.get('stage2_pair_extra_dim', 6),
        pair_embed_dims=pair_embed_dims,
        pair_embed_mode=args.get('stage2_pair_embed_mode', 'concat'),
        ffn_ratio=args.get('stage2_ffn_ratio', 4),
        dropout=args.get('stage2_dropout', 0.1),
        loss_mode=args.get('stage2_loss_mode', 'pairwise'),
        rs_at_k_target=args.get('stage2_rs_at_k_target', 200),
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval(), int(args.get('top_k1', 256))


def _load_stage3(path: str, device: str) -> tuple[CoupleReranker, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {}) or {}
    state_dict = checkpoint['couple_reranker_state_dict']
    # Feature-layout versioning: checkpoints written after the H6 widening
    # carry their dims; older ones fall back to the live module constants.
    feature_layout = checkpoint.get('feature_layout', {}) or {}
    track_embed_dim = int(feature_layout.get('track_embed_dim',
                                             TRACK_EMBED_DIM))
    rest_dim = int(feature_layout.get('rest_dim', COUPLE_REST_DIM))
    projector_in = state_dict['couple_projector.0.weight'].shape[1]
    if projector_in != track_embed_dim:
        raise ValueError(
            f'Stage-3 checkpoint projector expects {projector_in}-wide '
            f'track blocks but the resolved layout says {track_embed_dim} — '
            'checkpoint and feature layout are from different generations.'
        )
    model = CoupleReranker(
        hidden_dim=args.get('couple_hidden_dim', 256),
        num_residual_blocks=args.get('couple_num_residual_blocks', 4),
        dropout=args.get('couple_dropout', 0.1),
        ranking_num_samples=args.get('couple_ranking_num_samples', 50),
        ranking_temperature=args.get('couple_ranking_temperature', 1.0),
        label_smoothing=args.get('couple_label_smoothing', 0.10),
        couple_projector_dim=args.get('couple_projector_dim', 32),
        rest_dim=rest_dim,
        track_embed_dim=track_embed_dim,
    )
    model.load_state_dict(state_dict)
    return model.to(device).eval(), int(args.get('top_k2', 50))


def _gather_along_tracks(
    tensor: torch.Tensor, indices: torch.Tensor,
) -> torch.Tensor:
    expanded = indices.unsqueeze(1).expand(-1, tensor.shape[1], -1)
    return tensor.gather(2, expanded)


@torch.no_grad()
def _evaluate_batch(
    *,
    stage: str,
    stage1, stage2, stage3,
    points, features, lorentz, mask,
    top_k1, top_k2, num_couples,
    dump_stage3_inputs: bool = False,
    track_labels: torch.Tensor | None = None,
) -> list[dict]:
    s1_scores = stage1(points, features, lorentz, mask)
    valid_mask = mask.squeeze(1).bool()
    s1_masked = torch.where(
        valid_mask, s1_scores, torch.full_like(s1_scores, float('-inf')),
    )
    s1_sorted = torch.argsort(s1_masked, dim=1, descending=True)
    batch_size = s1_scores.size(0)

    def _stage1_row(b):
        n_valid = int(valid_mask[b].sum())
        sorted_indices = s1_sorted[b, :n_valid]
        return {
            'stage1_sorted_indices': sorted_indices.tolist(),
            'stage1_scores': s1_masked[b, sorted_indices].tolist(),
        }

    if stage == 'stage1':
        return [
            {
                **_stage1_row(b),
                'stage2_sorted_indices': [],
                'stage3_sorted_couples': [],
                'stage2_scores': [],
                'stage3_couple_scores': [],
            }
            for b in range(batch_size)
        ]

    selected_k1 = stage1.select_top_k(s1_scores, mask, top_k1)
    f_points = _gather_along_tracks(points, selected_k1)
    f_features = _gather_along_tracks(features, selected_k1)
    f_lorentz = _gather_along_tracks(lorentz, selected_k1)
    f_mask = _gather_along_tracks(mask, selected_k1)
    f_s1 = s1_scores.gather(1, selected_k1)

    s2_scores = stage2(f_points, f_features, f_lorentz, f_mask, f_s1)
    s2_sorted_k1 = torch.argsort(s2_scores, dim=1, descending=True)
    s2_sorted_orig = selected_k1.gather(1, s2_sorted_k1)

    def _stage2_row(b):
        n_valid_k1 = int(torch.isfinite(s2_scores[b]).sum())
        return {
            'stage2_sorted_indices': s2_sorted_orig[b, :n_valid_k1].tolist(),
            'stage2_scores': s2_scores[b, s2_sorted_k1[b, :n_valid_k1]].tolist(),
        }

    if stage == 'part':
        rows = [
            {
                **_stage1_row(b),
                **_stage2_row(b),
                'stage3_sorted_couples': [],
                'stage3_couple_scores': [],
            }
            for b in range(batch_size)
        ]
        if dump_stage3_inputs:
            pool_valid = torch.isfinite(s2_scores)
            labels_flat = track_labels.squeeze(1)
            k1_labels = torch.where(
                pool_valid,
                labels_flat.gather(1, selected_k1),
                torch.zeros_like(s2_scores),
            )
            full_valid = mask.squeeze(1) > 0.5
            full_pt = torch.hypot(lorentz[:, 0, :], lorentz[:, 1, :])
            for b, row in enumerate(rows):
                valid_b = full_valid[b]
                row['k1_features'] = (
                    f_features[b].reshape(-1).tolist())
                row['k1_points'] = f_points[b].reshape(-1).tolist()
                row['k1_lorentz'] = f_lorentz[b].reshape(-1).tolist()
                row['k1_stage1_scores'] = f_s1[b].tolist()
                row['k1_stage2_scores'] = s2_scores[b].tolist()
                row['k1_labels'] = k1_labels[b].int().tolist()
                row['k1_original_indices'] = selected_k1[b].int().tolist()
                row['cone_eta'] = points[b, 0, valid_b].tolist()
                row['cone_phi'] = points[b, 1, valid_b].tolist()
                row['cone_dz'] = points[b, 2, valid_b].tolist()
                row['cone_pt'] = full_pt[b, valid_b].tolist()
        return rows

    top_k2_in_k1 = s2_scores.topk(top_k2, dim=1).indices
    k2_orig = selected_k1.gather(1, top_k2_in_k1)
    k2_features = _gather_along_tracks(f_features, top_k2_in_k1)
    k2_points = _gather_along_tracks(f_points, top_k2_in_k1)
    k2_lorentz = _gather_along_tracks(f_lorentz, top_k2_in_k1)
    k2_s1 = f_s1.gather(1, top_k2_in_k1)
    k2_s2 = s2_scores.gather(1, top_k2_in_k1)
    k2_valid = torch.isfinite(k2_s2)

    couple_inputs = build_couple_features_batched(
        top_k2_features=k2_features,
        top_k2_points=k2_points,
        top_k2_lorentz=k2_lorentz,
        top_k2_stage1_scores=k2_s1,
        top_k2_stage2_scores=k2_s2,
        full_points=points,
        full_lorentz=lorentz,
        full_valid_mask=mask.squeeze(1) > 0.5,
        member_full_indices=k2_orig,
        track_valid_mask=k2_valid,
    )
    s3_scores = stage3(couple_inputs['couple_features'])

    upper_i, upper_j = torch.triu_indices(
        top_k2, top_k2, offset=1, device=s3_scores.device,
    ).unbind(0)

    rows = []
    for b in range(batch_size):
        scores_b = s3_scores[b].clone()
        scores_b[~couple_inputs['filter_a_mask'][b]] = float('-inf')
        order = torch.argsort(scores_b, descending=True)[:num_couples]
        keep = couple_inputs['filter_a_mask'][b][order]
        order = order[keep]
        i_orig = k2_orig[b, upper_i[order]].tolist()
        j_orig = k2_orig[b, upper_j[order]].tolist()
        rows.append({
            **_stage1_row(b),
            **_stage2_row(b),
            'stage3_sorted_couples': [[i, j] for i, j in zip(i_orig, j_orig)],
            'stage3_couple_scores': scores_b[order].tolist(),
        })
    return rows


def _write_parquet(rows: list[dict], output_path: str,
                   schema: pa.Schema = OUTPUT_SCHEMA) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    columns = {field.name: [] for field in schema}
    for row in rows:
        for field in schema:
            columns[field.name].append(row[field.name])
    table = pa.table(columns, schema=schema)
    pq.write_table(table, output_path)


def _composite_key(observers: dict, b: int) -> dict:
    return {
        'event_run': int(observers['event_run'][b]),
        'event_id': int(observers['event_id'][b]),
        'event_luminosity_block': int(observers['event_luminosity_block'][b]),
        'source_batch_id': int(observers['source_batch_id'][b]),
        'source_microbatch_id': int(observers['source_microbatch_id'][b]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Cascade evaluator: prefilter → ParT → couples; '
                    'dumps per-event sorted indices to parquet.',
    )
    parser.add_argument(
        '--stage', choices=('stage1', 'part', 'couples'), required=True,
    )
    parser.add_argument('--stage1-weights', required=True)
    parser.add_argument('--stage2-weights')
    parser.add_argument('--stage3-weights')
    parser.add_argument('--num-couples', type=int, default=200)
    parser.add_argument(
        '--dump-stage3-inputs', action='store_true',
        help='with --stage part: additionally dump the top-K1 block '
             '(features/points/lorentz/scores/labels/indices) and the '
             'full-event cone candidates for dump-based Stage-3 training.')
    parser.add_argument('--val-data-dir', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top-k1', type=int, default=None)
    parser.add_argument('--top-k2', type=int, default=None)
    parser.add_argument('--stage1-num-neighbors', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--start-event', type=int, default=0,
                        help='skip the first N source events (resume a crashed dump '
                             'into a fresh --output part file)')
    parser.add_argument('--log-every', type=int, default=20,
                        help='progress log cadence in batches')
    return parser


def _rss_gb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return max_rss / 2**30 if sys.platform == 'darwin' else max_rss / 2**20


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args(argv)
    if args.device.startswith('mps'):
        raise SystemExit('MPS not supported; use cpu or cuda.')
    if args.stage in {'part', 'couples'} and not args.stage2_weights:
        raise SystemExit('--stage2-weights required for stage=part/couples.')
    if args.stage == 'couples' and not args.stage3_weights:
        raise SystemExit('--stage3-weights required for stage=couples.')
    if args.dump_stage3_inputs and args.stage != 'part':
        raise SystemExit('--dump-stage3-inputs requires --stage part.')

    device = torch.device(args.device)

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
    data_config = dataset.config
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=False, num_workers=args.num_workers,
    )
    input_names = list(data_config.input_names)
    mask_idx = input_names.index('pf_mask')
    label_idx = input_names.index('pf_label')

    stage1 = _load_stage1(
        args.stage1_weights, data_config,
        num_neighbors=args.stage1_num_neighbors, device=device,
    )
    stage2, top_k1 = (None, None)
    stage3, top_k2 = (None, None)
    if args.stage in {'part', 'couples'}:
        stage2, top_k1 = _load_stage2(
            args.stage2_weights,
            input_dim=len(data_config.input_dicts['pf_features']),
            device=device,
        )
        if args.top_k1 is not None:
            top_k1 = args.top_k1
        logger.info(f'top_k1 = {top_k1}')
    if args.stage == 'couples':
        stage3, top_k2 = _load_stage3(args.stage3_weights, device=device)
        if args.top_k2 is not None:
            top_k2 = args.top_k2
        logger.info(f'top_k2 = {top_k2}')

    total_source_events = sum(pq.read_metadata(path).num_rows for path in parquet_files)
    total_to_write = max(total_source_events - args.start_event, 0)
    if args.max_events is not None:
        total_to_write = min(total_to_write, args.max_events)
    logger.info(f'{total_source_events} source events; writing {total_to_write} '
                f'starting at event {args.start_event}')

    rows: list[dict] = []
    events_done = 0   # rows written (after --start-event)
    events_seen = 0   # source rows consumed (including skipped)
    crash: BaseException | None = None
    loop_start = time.time()
    try:
        for batch_index, (X, _, observers) in enumerate(loader):
            batch_events = len(observers['event_run'])
            if events_seen + batch_events <= args.start_event:
                events_seen += batch_events   # whole batch below the resume point
                continue
            inputs = [X[k].to(device) for k in input_names]
            inputs = trim_to_max_valid_tracks(inputs, mask_idx)
            model_inputs, track_labels = extract_label_from_inputs(
                inputs, label_idx)
            points, features, lorentz, mask = model_inputs

            batch_rows = _evaluate_batch(
                stage=args.stage,
                stage1=stage1, stage2=stage2, stage3=stage3,
                points=points, features=features, lorentz=lorentz, mask=mask,
                top_k1=top_k1, top_k2=top_k2, num_couples=args.num_couples,
                dump_stage3_inputs=args.dump_stage3_inputs,
                track_labels=track_labels,
            )
            for b, row in enumerate(batch_rows):
                events_seen += 1
                if events_seen <= args.start_event:
                    continue
                row.update(_composite_key(observers, b))
                row['stage'] = args.stage
                rows.append(row)
                events_done += 1
                if args.max_events is not None and events_done >= args.max_events:
                    break
            if args.max_events is not None and events_done >= args.max_events:
                break
            if batch_index % args.log_every == 0:
                elapsed = max(time.time() - loop_start, 1e-9)
                rate = events_done / elapsed
                eta_min = ((total_to_write - events_done) / rate / 60) if rate > 0 else float('inf')
                logger.info(f'Batch {batch_index} | {events_done}/{total_to_write} events '
                            f'| {rate:.1f} ev/s | ETA {eta_min:.1f} min | RSS {_rss_gb():.1f} GB')
    except BaseException as error:   # salvage partial work on ANY failure, incl. Ctrl-C
        logger.error(f'event loop crashed after {events_done} events:\n{traceback.format_exc()}')
        crash = error
    finally:
        if rows:
            logger.info(f'Total events: {len(rows)} → {args.output}')
            _write_parquet(
                rows, args.output,
                schema=(STAGE3_DUMP_SCHEMA if args.dump_stage3_inputs
                        else OUTPUT_SCHEMA))
        marker = args.output + '.INCOMPLETE'
        if crash is not None:
            with open(marker, 'w') as fh:
                json.dump({'start_event': args.start_event, 'events_written': events_done,
                           'next_start_event': args.start_event + events_done}, fh, indent=2)
            logger.error(f'partial dump: wrote {marker}; resume with '
                         f'--start-event {args.start_event + events_done}')
        elif os.path.exists(marker):
            os.remove(marker)
    if crash is not None:
        raise crash


if __name__ == '__main__':
    main()
