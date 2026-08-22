from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from scripts.python.triplet_failure_taxonomy import _build_model


def main() -> None:
    from train_triplet_reranker import _batch_kwargs
    from utils.triplet_rank_data import (TripletRankDataset,
                                         collate_triplet_rank_eval)

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    model = _build_model(checkpoint, device)
    model_args = checkpoint['args']

    dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=-float('inf'), mode='eval',
        norm_stats=checkpoint['norm_stats'],
        extra_features=model_args['extra_features'],
        context_features=model_args['context_features'],
        vertex_fit=model_args['fit_mode'], from_b_targets=False,
        track32=(model_args['input_mode'] == 'hierarchical'
                 and model_args['track_embed_dim'] == 32))
    if dataset.feature_names != checkpoint['feature_names']:
        raise SystemExit('dataset resolves different feature names than the '
                         'checkpoint')

    loader = DataLoader(dataset, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        collate_fn=collate_triplet_rank_eval)
    event_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            valid_mask = batch['valid_mask'].to(device)
            scores = model(batch['features'].to(device), valid_mask=valid_mask,
                           **_batch_kwargs(batch, device, for_loss=False))
            probabilities = torch.sigmoid(scores).cpu().numpy()
            counts = batch['counts'].numpy()
            for b in range(probabilities.shape[0]):
                event_scores.append(
                    probabilities[b, :int(counts[b])].astype(np.float32))

    table = pa.table({'scores': pa.array([row.tolist() for row in event_scores],
                                         type=pa.list_(pa.float32()))})
    pq.write_table(table, args.output)
    print(f'wrote {len(event_scores)} event score rows to {args.output}')


if __name__ == '__main__':
    main()
