from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from utils.triplet_rank_data import (
    LOG1P_MARKERS,
    TripletRankDataset,
    collate_triplet_rank,
    fit_norm_stats,
    load_norm_stats,
    load_track16_params,
    save_norm_stats,
    standardize_features,
    track16_std,
)

_DELIVERABLES = os.path.join(os.path.dirname(__file__), '..')
_AUTO_YAML = os.path.join(_DELIVERABLES, 'data', 'low-pt',
                          'lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml')
_DATA_CONFIG = os.path.join(_DELIVERABLES, 'data', 'low-pt', 'lowpt_tau_trackfinder.yaml')
_SUBSET_DIR = '/Users/oleh/Projects/masters/part/data/low-pt/subset/val'
_DUMP = os.path.join(_DELIVERABLES, 'data', 'low-pt', 'eval', 'perstage_couples_val.parquet')
_SRC_GLOB = '/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet'
_GBDT6 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt_full_P2.joblib')
_GBDT8 = os.path.join(_DELIVERABLES, 'models', 'third_pion_filter_gbdt8_full_P2.joblib')

_HAVE_REAL_VAL = (os.path.exists(_DUMP) and glob.glob(_SRC_GLOB)
                  and os.path.exists(_GBDT6) and os.path.exists(_GBDT8))


def test_load_track16_params_order_and_values():
    params = load_track16_params(_AUTO_YAML)
    assert len(params) == 16
    names = [name for name, _ in params]
    assert names[0] == 'track_px'
    assert names[5] == 'track_charge'
    assert names[15] == 'track_log_relative_pt_error'
    charge = dict(params)['track_charge']
    assert charge['center'] == pytest.approx(1.0)
    assert charge['scale'] == pytest.approx(0.5)
    assert charge['min'] == -5 and charge['max'] == 5


@pytest.mark.skipif(
    not (os.path.exists(_DATA_CONFIG) and glob.glob(f'{_SUBSET_DIR}/*.parquet')),
    reason='data config or subset data not present',
)
def test_track16_std_matches_weaver():
    from torch.utils.data import DataLoader
    from weaver.utils.dataset import SimpleIterDataset

    files = sorted(glob.glob(f'{_SUBSET_DIR}/*.parquet'))
    dataset = SimpleIterDataset(
        {'data': files}, data_config_file=_DATA_CONFIG, for_training=False,
        load_range_and_fraction=((0.0, 1.0), 1.0), fetch_by_files=True,
        fetch_step=len(files), in_memory=False,
    )
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    X, _, _ = next(iter(loader))
    weaver_features = X['pf_features']  # (4, 16, 2100), standardized by weaver

    src = pq.read_table(files[0]).slice(0, 4).to_pylist()
    params = load_track16_params(_AUTO_YAML)
    for b, event in enumerate(src):
        n_tracks = event['event_n_tracks']
        t = lambda key: torch.tensor(event[key], dtype=torch.float32)
        ours = track16_std(
            pt=t('track_pt'), eta=t('track_eta'), phi=t('track_phi'),
            charge=t('track_charge'), dxy_sig=t('track_dxy_significance'),
            dz_sig=t('track_dz_significance'), norm_chi2=t('track_norm_chi2'),
            pt_error=t('track_pt_error'), n_pixel=t('track_n_valid_pixel_hits'),
            dca_sig=t('track_dca_significance'),
            cov_phi_phi=t('track_covariance_phi_phi'),
            cov_lambda_lambda=t('track_covariance_lambda_lambda'),
            params=params,
        )
        assert ours.shape == (n_tracks, 16)
        theirs = weaver_features[b, :, :n_tracks].T
        assert torch.allclose(ours, theirs, atol=1e-4), \
            f'event {b}: max diff {(ours - theirs).abs().max()}'


def test_standardize_features_log1p_and_affine():
    stats = {
        'dz_dist': {'log1p': True, 'center': 0.0, 'scale': 2.0},
        'm_ijk': {'log1p': False, 'center': 1.0, 'scale': 0.5},
    }
    X = torch.tensor([[np.expm1(2.0), 1.5], [-np.expm1(4.0), 0.5]], dtype=torch.float32)
    out = standardize_features(X, ['dz_dist', 'm_ijk'], stats)
    assert out[0, 0] == pytest.approx((2.0 - 0.0) / 2.0)
    assert out[1, 0] == pytest.approx((-4.0 - 0.0) / 2.0)
    assert out[0, 1] == pytest.approx((1.5 - 1.0) / 0.5)
    # clip at +-10
    stats_tight = {'m_ijk': {'log1p': False, 'center': 0.0, 'scale': 1e-6},
                   'dz_dist': {'log1p': False, 'center': 0.0, 'scale': 1.0}}
    clipped = standardize_features(X, ['dz_dist', 'm_ijk'], stats_tight)
    assert clipped[:, 1].abs().max() <= 10.0


def test_log1p_markers_cover_heavy_tails():
    from utils.triplet_join import FEATURE_NAMES
    flagged = [n for n in FEATURE_NAMES if any(m in n for m in LOG1P_MARKERS)]
    for name in ['dz_dist', 'dz_sig_k', 'tk_dz_sig', 'ti_cov_phi_phi', 'lorentz_dot',
                 'dca_sum', 'norm_chi2_k', 'rel_pt_err_k']:
        assert name in flagged
    for name in ['m_ijk', 'dr_min', 'couple_rank', 'is_same_sign', 'ti_eta', 'helicity']:
        assert name not in flagged


@pytest.fixture(scope='module')
def real_artifact(tmp_path_factory):
    if not _HAVE_REAL_VAL:
        pytest.skip('real VAL dump/src/models not present')
    from build_triplet_rank_candidates import main as build_main
    out_dir = str(tmp_path_factory.mktemp('triplet_rank'))
    build_main(['--out-dir', out_dir, '--tag', 'val', '--max-events', '40'])
    return out_dir


def test_fit_norm_stats_round_trip(real_artifact, tmp_path):
    stats = fit_norm_stats(
        os.path.join(real_artifact, 'candidates_val.parquet'),
        os.path.join(real_artifact, 'tracks_val.parquet'),
        n_events=20, per_event=50, seed=0,
    )
    from utils.triplet_join import FEATURE_NAMES
    assert set(stats) == set(FEATURE_NAMES)
    for name, s in stats.items():
        assert np.isfinite(s['center']) and np.isfinite(s['scale']) and s['scale'] > 0
    path = str(tmp_path / 'norm_stats.json')
    save_norm_stats(stats, path)
    assert load_norm_stats(path) == stats


def test_dataset_train_and_eval_items(real_artifact):
    cand = os.path.join(real_artifact, 'candidates_val.parquet')
    tracks = os.path.join(real_artifact, 'tracks_val.parquet')
    train_ds = TripletRankDataset(cand, tracks, tau=0.0, num_negatives=50,
                                  mode='train', seed=0)
    assert len(train_ds.trainable_indices) > 0
    item = train_ds[int(train_ds.trainable_indices[0])]
    n = item['features'].shape[0]
    assert item['features'].shape == (n, 89)
    assert item['pos_mask'].shape == (n,)
    assert item['pos_mask'].sum() >= 1
    assert n <= 50 + 3
    assert torch.isfinite(item['features']).all()

    eval_ds = TripletRankDataset(cand, tracks, tau=0.0, mode='eval', seed=0)
    assert len(eval_ds) == 40
    row = pq.read_table(cand).slice(0, 1).to_pylist()[0]
    ev = eval_ds[0]
    assert ev['features'].shape[0] == row['n_candidates']  # tau=0 keeps all Tier-H
    assert ev['keys'].shape == (row['n_candidates'], 3)
    assert (ev['keys'][:, 0] <= ev['keys'][:, 1]).all()
    assert (ev['keys'][:, 1] <= ev['keys'][:, 2]).all()
    assert ev['pos_mask'].sum() == sum(row['is_gt'])


def test_dataset_tau_masks_candidates(real_artifact):
    cand = os.path.join(real_artifact, 'candidates_val.parquet')
    tracks = os.path.join(real_artifact, 'tracks_val.parquet')
    table = pq.read_table(cand)
    scores = np.asarray(table['gbdt6_score'].combine_chunks()[0].values)
    tau = float(np.quantile(scores, 0.9))
    ds = TripletRankDataset(cand, tracks, tau=tau, mode='eval', seed=0)
    row0 = table.slice(0, 1).to_pylist()[0]
    expected = sum(1 for s in row0['gbdt6_score'] if s >= tau)
    assert ds[0]['features'].shape[0] == expected


def test_collate_pads_and_masks(real_artifact):
    cand = os.path.join(real_artifact, 'candidates_val.parquet')
    tracks = os.path.join(real_artifact, 'tracks_val.parquet')
    ds = TripletRankDataset(cand, tracks, tau=0.0, num_negatives=20, mode='train', seed=0)
    idx = [int(x) for x in ds.trainable_indices[:3]]
    batch = collate_triplet_rank([ds[i] for i in idx])
    B, F, N = batch['features'].shape
    assert B == 3 and F == 89
    assert batch['pos_mask'].shape == (B, N)
    assert batch['valid_mask'].shape == (B, N)
    assert (batch['pos_mask'] & ~batch['valid_mask']).sum() == 0
    assert (batch['features'].transpose(1, 2)[~batch['valid_mask']] == 0).all()
