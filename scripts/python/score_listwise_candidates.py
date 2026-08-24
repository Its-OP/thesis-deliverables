from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from weaver.nn.model.ListwiseTripletReranker import ListwiseTripletReranker


def main() -> None:
    from utils.triplet_rank_data import (TripletRankDataset,
                                         collate_listwise_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-list', type=int, default=3072)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu',
                            weights_only=False)
    model_args = checkpoint['args']
    model = ListwiseTripletReranker(
        feature_dim=len(checkpoint['feature_names']),
        hidden_dim=model_args['hidden_dim'],
        num_layers=model_args['num_layers'],
        num_heads=model_args['num_heads'],
        dropout=model_args['dropout']).to(device).eval()
    model.load_state_dict(checkpoint['listwise_reranker_state_dict'])

    dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=args.tau, mode='eval',
        norm_stats=checkpoint['norm_stats'], extra_features='auto',
        context_features=True, vertex_fit='static',
        max_serving_rows=args.max_list)
    if dataset.feature_names != checkpoint['feature_names']:
        raise SystemExit('dataset resolves different feature names than the '
                         'checkpoint')

    loader = DataLoader(dataset, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        collate_fn=collate_listwise_rank)
    event_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=device.type == 'cuda'):
                scores = model(batch['features'].to(device),
                               keys=batch['keys'].to(device),
                               valid_mask=batch['valid_mask'].to(device),
                               filter_logit=batch['filter_logit'].to(device))
            scores = scores.float().cpu().numpy()
            counts = batch['counts'].numpy()
            for b in range(scores.shape[0]):
                event_scores.append(
                    scores[b, :int(counts[b])].astype(np.float32))

    table = pa.table({'scores': pa.array([row.tolist() for row in event_scores],
                                         type=pa.list_(pa.float32()))})
    pq.write_table(table, args.output)
    print(f'wrote {len(event_scores)} event score rows to {args.output}')


if __name__ == '__main__':
    main()
