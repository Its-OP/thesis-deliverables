from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from dump_triplet_rerank_window import main as dump_main
from utils.triplet_join import FEATURE_NAMES
from utils.triplet_rank_data import (
    CASCADE_EXTRA_NAMES,
    GBDT_EXTRA_NAMES,
    TripletRankDataset,
    fit_norm_stats,
)
from weaver.nn.model.TripletReranker import TripletReranker

from test_triplet_rank_data import _write_synthetic_artifacts


def _write_stage_a_checkpoint(directory, feature_names, norm_stats):
    torch.manual_seed(0)
    model = TripletReranker(input_mode='flat', feature_names=feature_names,
                            hidden_dim=32, num_residual_blocks=1, dropout=0.0,
                            loss_mode='full')
    checkpoint_path = os.path.join(directory, 'stage_a.pt')
    torch.save({
        'args': {
            'input_mode': 'flat', 'hidden_dim': 32, 'num_residual_blocks': 1,
            'dropout': 0.0, 'num_negatives': 4, 'temperature': 1.0,
            'label_smoothing': 0.0, 'projector_dim': 32, 'loss_mode': 'full',
            'extra_features': 'all', 'weaver_track_blocks': False,
            'context_features': False,
        },
        'feature_names': feature_names,
        'norm_stats': norm_stats,
        'operating_point': {'score_column': 'gbdt6_score', 'tau': 0.0},
        'triplet_reranker_state_dict': model.state_dict(),
        'epoch': 0,
    }, checkpoint_path)
    return checkpoint_path, model


def test_dump_matches_direct_forward_and_truncates(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    feature_names = list(FEATURE_NAMES) + GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES
    norm_stats = fit_norm_stats(cand, tracks, feature_names=feature_names,
                                n_events=2, per_event=10)
    checkpoint_path, model = _write_stage_a_checkpoint(str(tmp_path), feature_names,
                                                       norm_stats)
    out_path = os.path.join(str(tmp_path), 'window.parquet')

    dump_main(['--checkpoint', checkpoint_path, '--candidates', cand,
               '--tracks', tracks, '--out', out_path, '--top', '2',
               '--device', 'cpu', '--num-workers', '0'])

    table = pq.read_table(out_path)
    assert table.num_rows == 2
    dataset = TripletRankDataset(cand, tracks, tau=0.0, mode='eval',
                                 extra_features='all', norm_stats=norm_stats)
    model.eval()
    for r, n_candidates in enumerate([3, 2]):
        positions = np.asarray(table['window_positions'][r].as_py())
        scores = np.asarray(table['window_scores'][r].as_py())
        assert len(positions) == min(2, n_candidates)
        assert len(np.unique(positions)) == len(positions)
        assert all(0 <= p < n_candidates for p in positions)
        assert list(scores) == sorted(scores, reverse=True)
        with torch.no_grad():
            direct = model(dataset[r]['features'].T.unsqueeze(0)).squeeze(0).numpy()
        # Eval items are surviving-ordered, so direct[p] scores candidate position p.
        np.testing.assert_allclose(scores, direct[positions], rtol=0, atol=1e-6)
        assert set(positions) == set(np.argsort(-direct)[:len(positions)])
