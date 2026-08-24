from __future__ import annotations

import numpy as np

from scripts.python.plot_recovery_by_pt import bucket_fractions


def test_bucket_fractions_counts_hits_and_absent_as_misses():
    pt = np.array([0.5, 0.6, 1.5, 1.6])
    gt_rank = np.array([3.0, 20.0, np.nan, 1.0])
    edges = np.array([0.0, 1.0, 2.0])
    fractions, sigmas, counts = bucket_fractions(pt, gt_rank, k=10, edges=edges)
    assert counts.tolist() == [2, 2]
    # Bucket 1: rank 3 hits, rank 20 misses. Bucket 2: NaN (absent) misses,
    # rank 1 hits.
    assert fractions[0] == 0.5 and fractions[1] == 0.5
    assert sigmas[0] == np.sqrt(0.25 / 2)


def test_bucket_fractions_empty_bucket_is_nan():
    fractions, sigmas, counts = bucket_fractions(
        np.array([5.0]), np.array([1.0]), k=5,
        edges=np.array([0.0, 1.0, 10.0]))
    assert counts[0] == 0 and np.isnan(fractions[0])
    assert fractions[1] == 1.0
