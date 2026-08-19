import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.triplet_rank_data import (  # noqa: E402
    TripletRankDataset,
    collate_triplet_rank,
    load_norm_stats,
)
from weaver.nn.model.TripletReranker import TripletReranker  # noqa: E402
from weaver.nn.model.VertexFit import FIT_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description='One anomaly-traced training step of the layer-fit model '
                    'on real candidates, to name the op that produces NaN.')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--src-glob', required=True)
    parser.add_argument('--norm-stats', required=True)
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    stats = load_norm_stats(args.norm_stats)
    dataset = TripletRankDataset(
        args.candidates, args.src_glob, tau=args.tau, num_negatives=512,
        mode='train', norm_stats=stats, seed=0, extra_features='auto',
        context_features=True, vertex_fit='layer')
    model = TripletReranker(
        input_mode='flat', feature_names=dataset.feature_names,
        loss_mode='full', trunk_norm='layer', fusion=True,
        vertex_fit_layer=True,
        fit_norm_stats={name: stats[name] for name in FIT_NAMES},
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.autograd.set_detect_anomaly(True)
    rows = dataset.trainable_indices
    generator = np.random.default_rng(0)
    for step in range(args.steps):
        picked = generator.choice(rows, args.batch_size, replace=False)
        batch = collate_triplet_rank([dataset[int(r)] for r in picked])
        fit_inputs = {key: batch[key].to(args.device) for key in batch
                      if key.startswith('fit_') or key == 'primary_vertex'}
        out = model.compute_loss(
            batch['features'].to(args.device),
            batch['pos_mask'].to(args.device),
            batch['valid_mask'].to(args.device),
            filter_logit=batch['filter_logit'].to(args.device),
            fit_inputs=fit_inputs)
        loss = out['total_loss']
        print(f'step {step}: loss {float(loss):.5f}', flush=True)
        if not torch.isfinite(loss):
            print('NON-FINITE FORWARD LOSS — dumping fit channel stats')
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bad = [name for name, parameter in model.named_parameters()
               if parameter.grad is not None
               and not torch.isfinite(parameter.grad).all()]
        if bad:
            print(f'NON-FINITE GRADS at step {step}: {bad[:8]}')
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print('debug run finished')


if __name__ == '__main__':
    main()
