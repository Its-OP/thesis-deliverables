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

from utils.triplet_join import FEATURE_NAMES, build_track_lorentz
from utils.triplet_rank_data import (
    CASCADE_EXTRA_NAMES,
    GBDT_EXTRA_NAMES,
    LOG1P_MARKERS,
    TripletRankDataset,
    collate_triplet_rank,
    collate_triplet_rank_eval,
    fit_norm_stats,
    load_norm_stats,
    load_track16_params,
    resolve_feature_names,
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


def _write_synthetic_artifacts(directory, with_cascade):
    # Event 0: 5 tracks, 3 candidates (first is GT), one couple. Event 1: 4 tracks,
    # 2 candidates that are BOTH GT decompositions (zero negatives). track_s2[3] of
    # event 0 is NaN so candidate 1 (k=3) exercises the missing-Stage-2 path.
    import pyarrow as pa

    candidates = {
        'n_tracks': [5, 4],
        'n_candidates': [3, 2],
        'cand_i': [[0, 0, 1], [0, 0]],
        'cand_j': [[1, 1, 2], [1, 1]],
        'cand_k': [[2, 3, 4], [2, 3]],
        'couple_rank': [[0, 1, 1], [0, 0]],
        'gbdt6_score': [[0.9, 0.5, 0.2], [0.7, 0.6]],
        'gbdt8_score': [[0.8, 0.4, 0.1], [0.65, 0.55]],
        'is_gt': [[True, False, False], [True, True]],
        'gt_i': [0, 0], 'gt_j': [1, 1], 'gt_k': [2, 2],
        'recon': [True, True],
    }
    if with_cascade:
        nan = float('nan')
        candidates['track_s1'] = [[0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6]]
        candidates['track_s2'] = [[0.5, 0.4, 0.3, nan, 0.2], [0.5, 0.4, 0.3, 0.2]]
        candidates['couple_scores'] = [[1.5, 1.2], [2.0]]
    tracks = {
        'track_pt': [[1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5]],
        'track_eta': [[0.1, -0.2, 0.3, -0.4, 0.5], [0.2, -0.1, 0.4, -0.3]],
        'track_phi': [[0.5, 1.0, -1.0, 2.0, -2.0], [0.3, -0.6, 1.2, -1.5]],
        'track_charge': [[1.0, -1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]],
        'track_dz_significance': [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]],
        'track_dxy_significance': [[1.1, 1.2, 1.3, 1.4, 1.5], [1.1, 1.2, 1.3, 1.4]],
        'track_dca_significance': [[2.1, 2.2, 2.3, 2.4, 2.5], [2.1, 2.2, 2.3, 2.4]],
        'track_n_valid_pixel_hits': [[3.0, 4.0, 3.0, 4.0, 3.0], [4.0, 3.0, 4.0, 3.0]],
        'track_norm_chi2': [[1.0, 1.1, 1.2, 1.3, 1.4], [1.0, 1.1, 1.2, 1.3]],
        'track_pt_error': [[0.01, 0.02, 0.03, 0.04, 0.05], [0.01, 0.02, 0.03, 0.04]],
        'track_covariance_phi_phi': [[1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
                                     [1e-6, 2e-6, 3e-6, 4e-6]],
        'track_covariance_lambda_lambda': [[1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
                                           [1e-6, 2e-6, 3e-6, 4e-6]],
    }
    candidates_path = os.path.join(directory, 'candidates_syn.parquet')
    tracks_path = os.path.join(directory, 'tracks_syn.parquet')
    pq.write_table(pa.table(candidates), candidates_path)
    pq.write_table(pa.table(tracks), tracks_path)
    return candidates_path, tracks_path, candidates


def test_resolve_feature_names_variants(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    with_cascade = TripletRankDataset(cand, tracks, tau=0.0).table
    assert resolve_feature_names(with_cascade, 'none') == list(FEATURE_NAMES)
    assert resolve_feature_names(with_cascade, 'gbdt') == list(FEATURE_NAMES) + GBDT_EXTRA_NAMES
    assert (resolve_feature_names(with_cascade, 'all')
            == list(FEATURE_NAMES) + GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES)
    assert resolve_feature_names(with_cascade, 'auto') == resolve_feature_names(with_cascade, 'all')
    with pytest.raises(ValueError):
        resolve_feature_names(with_cascade, 'bogus')

    bare_dir = tmp_path / 'bare'
    bare_dir.mkdir()
    cand_bare, tracks_bare, _ = _write_synthetic_artifacts(str(bare_dir), with_cascade=False)
    without = TripletRankDataset(cand_bare, tracks_bare, tau=0.0).table
    assert resolve_feature_names(without, 'auto') == list(FEATURE_NAMES) + GBDT_EXTRA_NAMES
    with pytest.raises(ValueError):
        resolve_feature_names(without, 'all')


def test_extra_feature_values_and_nan_pattern(tmp_path):
    cand, tracks, raw = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    ds = TripletRankDataset(cand, tracks, tau=0.0, mode='eval', extra_features='all')
    assert ds.feature_names == list(FEATURE_NAMES) + GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES
    item = ds[0]
    features = item['features']
    assert features.shape == (3, 99)
    base = len(FEATURE_NAMES)
    column = {name: features[:, base + offset]
              for offset, name in enumerate(GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES)}
    torch.testing.assert_close(column['gbdt6_score'],
                               torch.tensor(raw['gbdt6_score'][0]))
    torch.testing.assert_close(column['gbdt8_score'],
                               torch.tensor(raw['gbdt8_score'][0]))
    track_s1 = raw['track_s1'][0]
    torch.testing.assert_close(column['s1_i'],
                               torch.tensor([track_s1[i] for i in raw['cand_i'][0]]))
    torch.testing.assert_close(column['s1_k'],
                               torch.tensor([track_s1[k] for k in raw['cand_k'][0]]))
    # candidate 1 has k=3 whose Stage-2 score is NaN -> raw NaN + flag 0.
    assert torch.isnan(column['s2_k'][1])
    torch.testing.assert_close(column['s2_k_isvalid'], torch.tensor([1.0, 0.0, 1.0]))
    couple_scores = raw['couple_scores'][0]
    torch.testing.assert_close(column['s3_couple'],
                               torch.tensor([couple_scores[c] for c in raw['couple_rank'][0]]))


def test_norm_stats_nan_policy_and_passthrough(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    names = list(FEATURE_NAMES) + GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES
    stats = fit_norm_stats(cand, tracks, feature_names=names, n_events=2, per_event=10)
    assert list(stats) == names
    # NaN-aware fit: s2_k stats come from the finite entries only.
    assert np.isfinite(stats['s2_k']['center']) and np.isfinite(stats['s2_k']['scale'])
    assert stats['s2_k_isvalid'] == {'log1p': False, 'center': 0.0, 'scale': 1.0}

    ds = TripletRankDataset(cand, tracks, tau=0.0, mode='eval', extra_features='all',
                            norm_stats=stats)
    features = ds[0]['features']
    assert torch.isfinite(features).all()  # NaN s2_k standardized to 0
    isvalid = features[:, names.index('s2_k_isvalid')]
    torch.testing.assert_close(isvalid, torch.tensor([1.0, 0.0, 1.0]))
    assert features[1, names.index('s2_k')].item() == 0.0


def test_train_item_zero_negatives_guard(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    ds = TripletRankDataset(cand, tracks, tau=0.0, num_negatives=8, mode='train')
    item = ds[1]  # event 1: both candidates are GT -> zero negatives
    assert item['features'].shape[0] == 2
    assert item['pos_mask'].all()


def test_standardize_features_vectorized_matches_column_loop():
    # The vectorized implementation must reproduce the original per-column loop
    # exactly (values AND NaN handling).
    generator = torch.Generator().manual_seed(3)
    names = ['plain_a', 'heavy_dz', 'plain_b', 'cov_xy', 'flag_isvalid']
    stats = {
        'plain_a': {'log1p': False, 'center': 0.5, 'scale': 2.0},
        'heavy_dz': {'log1p': True, 'center': -1.0, 'scale': 0.25},
        'plain_b': {'log1p': False, 'center': 100.0, 'scale': 1e-3},
        'cov_xy': {'log1p': True, 'center': 0.0, 'scale': 5.0},
        'flag_isvalid': {'log1p': False, 'center': 0.0, 'scale': 1.0},
    }
    X = torch.randn(64, len(names), generator=generator) * 50.0
    X[3, 1] = float('nan')
    X[10, 3] = float('nan')

    reference_columns = []
    for column, name in zip(X.unbind(dim=1), names):
        s = stats[name]
        if s['log1p']:
            column = torch.sign(column) * torch.log1p(column.abs())
        column = torch.clamp((column - s['center']) / s['scale'], -10.0, 10.0)
        reference_columns.append(torch.nan_to_num(column, nan=0.0))
    reference = torch.stack(reference_columns, dim=1)

    torch.testing.assert_close(standardize_features(X, names, stats), reference,
                               rtol=0.0, atol=0.0)


def test_collate_triplet_rank_eval_carries_keys_and_counts(tmp_path):
    cand, tracks, _ = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    ds = TripletRankDataset(cand, tracks, tau=0.0, mode='eval', extra_features='all')
    items = [ds[0], ds[1]]
    batch = collate_triplet_rank_eval(items)
    assert batch['features'].shape == (2, 99, 3)   # event 0 has 3 candidates, event 1 has 2
    assert batch['valid_mask'].tolist() == [[True, True, True], [True, True, False]]
    assert batch['counts'].tolist() == [3, 2]
    assert len(batch['keys']) == 2
    torch.testing.assert_close(batch['keys'][0], items[0]['keys'])
    torch.testing.assert_close(batch['keys'][1], items[1]['keys'])


def _expected_track16_for_event(tracks_path, event_index):
    # Mirrors the dataset's own pt reconstruction (sqrt(px^2+py^2) off the Lorentz
    # vector, not the raw track_pt column) so track16_std sees identical inputs.
    track_row = pq.read_table(tracks_path).slice(event_index, 1).to_pylist()[0]
    column = lambda name: torch.tensor(track_row[name], dtype=torch.float32)
    lorentz = build_track_lorentz(column('track_pt'), column('track_eta'), column('track_phi'))
    reconstructed_pt = torch.sqrt(lorentz[0] ** 2 + lorentz[1] ** 2)
    params = load_track16_params()
    return track16_std(
        pt=reconstructed_pt, eta=column('track_eta'), phi=column('track_phi'),
        charge=column('track_charge'), dxy_sig=column('track_dxy_significance'),
        dz_sig=column('track_dz_significance'), norm_chi2=column('track_norm_chi2'),
        pt_error=column('track_pt_error'), n_pixel=column('track_n_valid_pixel_hits'),
        dca_sig=column('track_dca_significance'),
        cov_phi_phi=column('track_covariance_phi_phi'),
        cov_lambda_lambda=column('track_covariance_lambda_lambda'),
        params=params,
    )


def _feature_columns_with_prefix(feature_names, prefix):
    return [index for index, name in enumerate(feature_names) if name.startswith(prefix)]


def test_weaver_track_blocks_matches_track16_std_and_shields_other_columns(tmp_path):
    candidates_path, tracks_path, raw = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    weaver_dataset = TripletRankDataset(
        candidates_path, tracks_path, tau=0.0, mode='eval', extra_features='all',
        weaver_track_blocks=True)
    baseline_dataset = TripletRankDataset(
        candidates_path, tracks_path, tau=0.0, mode='eval', extra_features='all')
    weaver_item = weaver_dataset[0]
    baseline_item = baseline_dataset[0]

    expected_track16 = _expected_track16_for_event(tracks_path, 0)
    cand_i = torch.tensor(raw['cand_i'][0], dtype=torch.long)
    cand_j = torch.tensor(raw['cand_j'][0], dtype=torch.long)
    cand_k = torch.tensor(raw['cand_k'][0], dtype=torch.long)

    feature_names = weaver_dataset.feature_names
    ti_columns = _feature_columns_with_prefix(feature_names, 'ti_')
    tj_columns = _feature_columns_with_prefix(feature_names, 'tj_')
    tk_columns = _feature_columns_with_prefix(feature_names, 'tk_')
    torch.testing.assert_close(weaver_item['features'][:, ti_columns], expected_track16[cand_i])
    torch.testing.assert_close(weaver_item['features'][:, tj_columns], expected_track16[cand_j])
    torch.testing.assert_close(weaver_item['features'][:, tk_columns], expected_track16[cand_k])

    non_track_columns = [index for index, name in enumerate(feature_names)
                         if not name.startswith(('ti_', 'tj_', 'tk_'))]
    torch.testing.assert_close(weaver_item['features'][:, non_track_columns],
                               baseline_item['features'][:, non_track_columns],
                               rtol=0.0, atol=0.0, equal_nan=True)


def test_weaver_track_blocks_with_norm_stats_keeps_raw_track16_values(tmp_path):
    candidates_path, tracks_path, raw = _write_synthetic_artifacts(str(tmp_path), with_cascade=True)
    names = list(FEATURE_NAMES) + GBDT_EXTRA_NAMES + CASCADE_EXTRA_NAMES
    stats = fit_norm_stats(candidates_path, tracks_path, feature_names=names, n_events=2, per_event=10)

    weaver_dataset = TripletRankDataset(
        candidates_path, tracks_path, tau=0.0, mode='eval', extra_features='all',
        weaver_track_blocks=True, norm_stats=stats)
    baseline_dataset = TripletRankDataset(
        candidates_path, tracks_path, tau=0.0, mode='eval', extra_features='all',
        norm_stats=stats)
    weaver_item = weaver_dataset[0]
    baseline_item = baseline_dataset[0]

    expected_track16 = _expected_track16_for_event(tracks_path, 0)
    cand_i = torch.tensor(raw['cand_i'][0], dtype=torch.long)
    cand_j = torch.tensor(raw['cand_j'][0], dtype=torch.long)
    cand_k = torch.tensor(raw['cand_k'][0], dtype=torch.long)

    feature_names = weaver_dataset.feature_names
    ti_columns = _feature_columns_with_prefix(feature_names, 'ti_')
    tj_columns = _feature_columns_with_prefix(feature_names, 'tj_')
    tk_columns = _feature_columns_with_prefix(feature_names, 'tk_')
    torch.testing.assert_close(weaver_item['features'][:, ti_columns], expected_track16[cand_i])
    torch.testing.assert_close(weaver_item['features'][:, tj_columns], expected_track16[cand_j])
    torch.testing.assert_close(weaver_item['features'][:, tk_columns], expected_track16[cand_k])

    non_track_columns = [index for index, name in enumerate(feature_names)
                         if not name.startswith(('ti_', 'tj_', 'tk_'))]
    torch.testing.assert_close(weaver_item['features'][:, non_track_columns],
                               baseline_item['features'][:, non_track_columns],
                               rtol=0.0, atol=0.0, equal_nan=True)
