from __future__ import annotations

import glob

from scripts.python.compute_eval_metrics import build_gt_lookup

EVAL_DIR = 'data/low-pt/eval'


def test_direct_gt_lookup_matches_dataset_invariants():
    shard = sorted(glob.glob(f'{EVAL_DIR}/*.parquet'))[0]
    lookup = build_gt_lookup([shard])
    assert len(lookup) > 1000
    for key, gt_indices in list(lookup.items())[:200]:
        assert len(key) == 5
        # Every event carries exactly 3 GT tracks (dataset invariant).
        assert len(gt_indices) == 3
        assert all(index >= 0 for index in gt_indices)
