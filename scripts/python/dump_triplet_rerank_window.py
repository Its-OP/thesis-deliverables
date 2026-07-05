from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from scripts.python.eval_triplet_reranker import TRIPLET_RANK_DIR, load_model
except ImportError:  # direct-file invocation: scripts/python is sys.path[0]
    from eval_triplet_reranker import TRIPLET_RANK_DIR, load_model

from utils.triplet_rank_data import TripletRankDataset, collate_triplet_rank_eval

WINDOW_SCHEMA = pa.schema([
    pa.field('window_positions', pa.list_(pa.int32())),
    pa.field('window_scores', pa.list_(pa.float32())),
])
WRITE_BATCH_EVENTS = 5000


def _write_batch(writer, rows):
    writer.write_table(pa.table(
        {name: [row[i] for row in rows] for i, name in enumerate(WINDOW_SCHEMA.names)},
        schema=WINDOW_SCHEMA))


@torch.no_grad()
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Dump the stage-A top-M window '
                                                 '(positions + scores) per event.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', default=os.path.join(TRIPLET_RANK_DIR,
                                                             'candidates_val.parquet'))
    parser.add_argument('--tracks', default=os.path.join(TRIPLET_RANK_DIR,
                                                         'tracks_val.parquet'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--top', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-events', type=int, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    trainer_args = checkpoint['args']
    operating_point = checkpoint['operating_point']
    score_column, tau = operating_point['score_column'], operating_point['tau']

    dataset = TripletRankDataset(
        args.candidates, args.tracks, tau=tau, score_column=score_column,
        mode='eval', norm_stats=checkpoint['norm_stats'], seed=0,
        extra_features=trainer_args.get('extra_features', 'none'),
        weaver_track_blocks=trainer_args.get('weaver_track_blocks', False),
        context_features=trainer_args.get('context_features', False))
    n_events = dataset.table.num_rows if args.max_events is None else \
        min(args.max_events, dataset.table.num_rows)
    loader = DataLoader(Subset(dataset, range(n_events)), batch_size=1,
                        num_workers=args.num_workers,
                        collate_fn=collate_triplet_rank_eval)

    writer = pq.ParquetWriter(args.out, WINDOW_SCHEMA)
    pending = []
    for r, batch in enumerate(tqdm(loader, total=n_events, desc='window',
                                   mininterval=10)):
        n = int(batch['counts'][0])
        if n == 0:
            pending.append(([], []))
        else:
            valid_mask = batch['valid_mask'].to(device)
            scores = model(batch['features'].to(device), valid_mask=valid_mask)[0, :n]
            top = scores.topk(min(args.top, n))
            event_scores = np.asarray(
                dataset.table.candidates[score_column][r].values)
            surviving = np.where(event_scores >= tau)[0]
            assert len(surviving) == n, f'surviving mismatch at row {r}'
            positions = surviving[top.indices.cpu().numpy()]
            pending.append((positions.astype(np.int32).tolist(),
                            top.values.cpu().numpy().astype(np.float32).tolist()))
        if len(pending) >= WRITE_BATCH_EVENTS:
            _write_batch(writer, pending)
            pending = []
    if pending:
        _write_batch(writer, pending)
    writer.close()
    print(f'wrote {args.out}: {n_events} events, top {args.top}')


if __name__ == '__main__':
    main()
