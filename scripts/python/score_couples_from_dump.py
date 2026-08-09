import argparse
import logging
import os

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from scripts.python.eval_cascade_pipeline import (
    OUTPUT_SCHEMA,
    _load_stage3,
    _write_parquet,
)
from utils.couple_dump_data import CoupleDumpDataset
from weaver.nn.model.CoupleDumpModel import CoupleDumpModel

logger = logging.getLogger('score_couples_from_dump')

IDENTITY_COLUMNS = ('event_run', 'event_id', 'event_luminosity_block',
                    'source_batch_id', 'source_microbatch_id')
COPIED_COLUMNS = ('stage1_sorted_indices', 'stage2_sorted_indices',
                  'stage1_scores', 'stage2_scores')


def _read_passthrough_columns(dump_path: str, max_events: int | None) -> dict:
    columns = list(IDENTITY_COLUMNS) + list(COPIED_COLUMNS)
    collected: dict[str, list] = {name: [] for name in columns}
    parquet_file = pq.ParquetFile(dump_path)
    loaded = 0
    for record_batch in parquet_file.iter_batches(
            batch_size=8192, columns=columns):
        if max_events is not None:
            record_batch = record_batch.slice(0, max_events - loaded)
        for name in columns:
            collected[name].extend(record_batch.column(name).to_pylist())
        loaded += record_batch.num_rows
        if max_events is not None and loaded >= max_events:
            break
    return collected


@torch.no_grad()
def score_dump(
    *,
    dump_path: str,
    checkpoint_path: str,
    output_path: str,
    top_k2: int | None = None,
    num_couples: int = 200,
    batch_size: int = 64,
    device: str = 'cpu',
    max_events: int | None = None,
    num_workers: int = 0,
) -> str:
    stage3, checkpoint_top_k2 = _load_stage3(checkpoint_path, device=device)
    if top_k2 is None:
        top_k2 = checkpoint_top_k2
    logger.info(f'top_k2 = {top_k2}, num_couples = {num_couples}')

    dataset = CoupleDumpDataset([dump_path], max_events=max_events)
    passthrough = _read_passthrough_columns(dump_path, max_events)
    if len(dataset) != len(passthrough['event_id']):
        raise ValueError(
            f'dump rows disagree: {len(dataset)} tensor rows vs '
            f'{len(passthrough["event_id"])} identity rows',
        )
    logger.info(f'{len(dataset)} events from {dump_path}')

    model = CoupleDumpModel(couple_reranker=stage3, top_k2=top_k2)
    model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=dataset.collate,
    )
    upper_i, upper_j = torch.triu_indices(
        top_k2, top_k2, offset=1, device=device).unbind(0)

    rows: list[dict] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        couple_inputs = model._build_couple_inputs(batch, with_metrics=False)
        scores = stage3(couple_inputs['couple_features'])
        member_indices = couple_inputs['member_full_indices']
        for row in range(scores.shape[0]):
            event_index = len(rows)
            row_scores = scores[row].clone()
            row_scores[~couple_inputs['filter_a_mask'][row]] = float('-inf')
            order = torch.argsort(row_scores, descending=True)[:num_couples]
            order = order[couple_inputs['filter_a_mask'][row][order]]
            i_original = member_indices[row, upper_i[order]].tolist()
            j_original = member_indices[row, upper_j[order]].tolist()
            rows.append({
                **{name: passthrough[name][event_index]
                   for name in IDENTITY_COLUMNS},
                'stage': 'couples',
                **{name: passthrough[name][event_index]
                   for name in COPIED_COLUMNS},
                'stage3_sorted_couples': [[i, j] for i, j
                                          in zip(i_original, j_original)],
                'stage3_couple_scores': row_scores[order].tolist(),
            })
        if len(rows) % (batch_size * 50) == 0:
            logger.info(f'{len(rows)} / {len(dataset)} events scored')

    _write_parquet(rows, output_path, schema=OUTPUT_SCHEMA)
    logger.info(f'{len(rows)} events -> {output_path}')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Replay Stage 3 over a Stage-3 input dump and write the '
                    'per-stage couples parquet the triplet chain consumes.')
    parser.add_argument('--dump', required=True)
    parser.add_argument('--stage3-weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--top-k2', type=int, default=None)
    parser.add_argument('--num-couples', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')
    device = args.device if torch.cuda.is_available() or \
        not args.device.startswith('cuda') else 'cpu'
    if device != args.device:
        logger.warning(f'{args.device} unavailable, falling back to {device}')
    score_dump(
        dump_path=args.dump,
        checkpoint_path=args.stage3_weights,
        output_path=args.output,
        top_k2=args.top_k2,
        num_couples=args.num_couples,
        batch_size=args.batch_size,
        device=device,
        max_events=args.max_events,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
