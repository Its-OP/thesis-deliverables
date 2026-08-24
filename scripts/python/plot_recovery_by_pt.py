from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

K_CURVES = [(1, '#AA3355'), (5, '#77AA44'), (10, '#4466CC')]


def bucket_fractions(pt: np.ndarray, gt_rank: np.ndarray, k: int,
                     edges: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    """pt: (N,); gt_rank: (N,) with NaN for absent; edges: (B+1,).
    Returns per-bucket (fraction with rank <= k, binomial sigma, count).
    Denominator = every reconstructable event in the bucket."""
    fractions = np.full(len(edges) - 1, np.nan)
    sigmas = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    hits_mask = np.nan_to_num(gt_rank, nan=np.inf) <= k
    for b in range(len(edges) - 1):
        in_bucket = (pt >= edges[b]) & (pt < edges[b + 1])
        n = int(in_bucket.sum())
        counts[b] = n
        if n == 0:
            continue
        p = float(hits_mask[in_bucket].mean())
        fractions[b] = p
        sigmas[b] = float(np.sqrt(p * (1.0 - p) / n))
    return fractions, sigmas, counts


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pt-max', type=float, default=20.0)
    parser.add_argument('--bins', type=int, default=40)
    args = parser.parse_args()

    records = pd.read_parquet(args.records)
    pt = records['pt_visible'].to_numpy()
    gt_rank = records['gt_rank'].to_numpy(dtype=np.float64)
    edges = np.linspace(0.0, args.pt_max, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    figure, axis = plt.subplots(figsize=(7, 5))
    for k, color in K_CURVES:
        fractions, sigmas, _ = bucket_fractions(pt, gt_rank, k, edges)
        axis.errorbar(centers, fractions, yerr=sigmas, color=color,
                      marker='o', markersize=3, linewidth=1.2, capsize=2,
                      label=f'Listwise reranker (top-{k})')
    axis.set_xlabel(r'Visible $3\pi$ $p_T$ [GeV]')
    axis.set_ylabel(r'Fraction of GT triplet $\in$ top-$K$')
    axis.set_xlim(0.0, args.pt_max)
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)
    axis.legend(loc='lower right')
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
