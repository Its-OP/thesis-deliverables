from __future__ import annotations

import argparse
import time

import torch
from torch.utils.checkpoint import checkpoint

from weaver.nn.model.ListwiseTripletReranker import ListwiseTripletReranker


def _synthetic_batch(batch_size: int, length: int, feature_dim: int, device):
    generator = torch.Generator(device='cpu').manual_seed(7)
    features = torch.randn(batch_size, feature_dim, length,
                           generator=generator).to(device)
    keys = torch.randint(0, 2000, (batch_size, length, 3),
                         generator=generator).to(device)
    valid_mask = torch.ones(batch_size, length, dtype=torch.bool,
                            device=device)
    filter_logit = torch.randn(batch_size, length,
                               generator=generator).to(device)
    pos_mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    pos_mask[:, 0] = True
    return features, keys, valid_mask, filter_logit, pos_mask


def _run(model, batch, steps: int, use_checkpoint: bool) -> tuple[float, float]:
    features, keys, valid_mask, filter_logit, pos_mask = batch
    if use_checkpoint:
        original_blocks = model.blocks

        def forward_with_checkpoint():
            hidden = model.input_projection(features.transpose(1, 2))
            from weaver.nn.model.ListwiseTripletReranker import (
                _MAX_SHARED, shared_track_counts)
            counts = shared_track_counts(keys).long().clamp(0, _MAX_SHARED)
            bias = model.overlap_bias[:, counts].permute(1, 0, 2, 3)
            for block in original_blocks:
                hidden = checkpoint(block, hidden, bias, valid_mask,
                                    use_reentrant=False)
            residual = model.scorer_head(hidden).squeeze(-1)
            return model.fusion_alpha * filter_logit + residual

        forward = forward_with_checkpoint
    else:
        forward = lambda: model(features, keys=keys, valid_mask=valid_mask,
                                filter_logit=filter_logit)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        scores = forward()
        loss = scores[pos_mask].sum() - torch.logsumexp(scores, dim=1).sum()
        (-loss).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--feature-dim', type=int, default=147)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)

    for batch_size in (8, 16, 32):
        for label, kernel, use_checkpoint in (
                ('math', 'math', False),
                ('efficient', 'efficient', False),
                ('checkpoint+eff', 'efficient', True)):
            model = ListwiseTripletReranker(
                feature_dim=args.feature_dim).to(device)
            batch = _synthetic_batch(batch_size, args.length,
                                     args.feature_dim, device)
            backends = ([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
                        if kernel == 'efficient'
                        else [torch.nn.attention.SDPBackend.MATH])
            try:
                with torch.nn.attention.sdpa_kernel(backends):
                    seconds, peak = _run(model, batch, args.steps,
                                         use_checkpoint)
                print(f'bs={batch_size:2d} {label:15s} '
                      f'{seconds:7.3f} s/batch  peak {peak:6.2f} GiB')
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                print(f'bs={batch_size:2d} {label:15s} FAILED: '
                      f'{str(error)[:90]}')
            del model, batch
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
