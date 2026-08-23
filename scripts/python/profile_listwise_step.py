from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from scripts.python.bench_listwise_attention import _synthetic_batch
from weaver.nn.model.ListwiseTripletReranker import ListwiseTripletReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--feature-dim', type=int, default=147)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)

    model = ListwiseTripletReranker(feature_dim=args.feature_dim).to(device)
    features, keys, valid_mask, filter_logit, pos_mask = _synthetic_batch(
        args.batch_size, args.length, args.feature_dim, device)

    def step():
        scores = model(features, keys=keys, valid_mask=valid_mask,
                       filter_logit=filter_logit)
        loss = scores[pos_mask].sum() - torch.logsumexp(scores, dim=1).sum()
        (-loss).backward()
        model.zero_grad(set_to_none=True)

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) \
            as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='self_cuda_time_total',
                                    row_limit=15))


if __name__ == '__main__':
    main()
