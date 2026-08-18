from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'python'))

from utils.triplet_join import FEATURE_NAMES
from utils.triplet_rank_data import (
    BASE_FEATURE_NAMES,
    CASCADE_EXTRA_NAMES,
    CONTEXT_FEATURE_NAMES,
    FILTER_EXTRA_NAMES,
    LOG1P_MARKERS,
    TripletRankDataset,
    _EventTable,
    collate_triplet_rank,
    collate_triplet_rank_eval,
    event_context_features,
    fit_norm_stats,
    load_norm_stats,
    resolve_feature_names,
    save_norm_stats,
    standardize_features,
)
from utils.vertex_fit_features import FIT_NAMES

from triplet_rank_fixture import synthetic_events, write_fixture


def _have_filter():
    from build_triplet_rank_candidates import FILTER_MODEL_GLOB
    return bool(glob.glob(FILTER_MODEL_GLOB))


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    if not _have_filter():
        pytest.skip('sweep filter joblib not present')
    from build_triplet_rank_candidates import main as build_main
    root = tmp_path_factory.mktemp('rank_v2')
    events = synthetic_events(4)
    dump_path, src_glob = write_fixture(root, events, dump_order=[2, 0, 3, 1])
    out_dir = root / 'out'
    build_main(['--role', 'train', '--dump', dump_path, '--src-glob', src_glob,
                '--out-dir', str(out_dir), '--window', '4', '--tail-sample', '2',
                '--top-c', '125'])
    return str(out_dir / 'candidates_train.parquet'), src_glob


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_standardize_features_log1p_and_affine():
    names = ['plain', 'track_dz_thing']
    stats = {'plain': {'log1p': False, 'center': 1.0, 'scale': 2.0},
             'track_dz_thing': {'log1p': True, 'center': 0.0, 'scale': 1.0}}
    X = torch.tensor([[3.0, -np.e + 1.0]])
    out = standardize_features(X, names, stats)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_standardize_maps_nan_to_zero_and_clips():
    names = ['a']
    stats = {'a': {'log1p': False, 'center': 0.0, 'scale': 0.01}}
    X = torch.tensor([[float('nan')], [1e9]])
    out = standardize_features(X, names, stats)
    assert out[0, 0] == 0.0
    assert out[1, 0] == 10.0


def test_log1p_markers_cover_heavy_tails_including_the_fit_block():
    for name in ('fit_res_k', 'fit_lxy_beam', 'fit_sigma_xy', 'fit_arc_i',
                 'fit_dlen_sig_beam', 'fitpv_lxy', 'fit_chi2',
                 'track_dz_significance', 'cov_phi_phi'):
        assert any(marker in name for marker in LOG1P_MARKERS), name
    for name in ('fit_cos_xy_beam', 'fit_mcorr_beam', 'fitpv_cos',
                 'fit_logw_i', 'fit_logdet_a'):
        assert not any(marker in name for marker in LOG1P_MARKERS), name


def test_base_feature_names_are_the_100_wide_champion_layout():
    assert len(BASE_FEATURE_NAMES) == 100
    assert BASE_FEATURE_NAMES[:89] == list(FEATURE_NAMES)
    assert BASE_FEATURE_NAMES[89] == 'poca_max'
    assert BASE_FEATURE_NAMES[-1] == 'n_low_ip_in_cone'


def test_context_names_are_single_filter():
    assert CONTEXT_FEATURE_NAMES == ['ctx_filter_rank_frac', 'ctx_filter_top_gap',
                                     'ctx_filter_z', 'ctx_log_n_surviving',
                                     'ctx_couple_rank_frac']
    assert FILTER_EXTRA_NAMES == ['filter_score']


def test_event_context_features_values():
    arrays = {
        'filter_score': np.array([0.9, 0.5, 0.1, 0.7], dtype=np.float32),
        'couple_rank': np.array([0, 1, 2, 0], dtype=np.int16),
    }
    cascade = {'couple_scores': np.array([0.9, 0.8, 0.7, 0.6, 0.5])}
    surviving = np.array([0, 1, 3])
    context = event_context_features(arrays, cascade, surviving,
                                     np.array([0, 3]))
    # Row 0: top of the surviving list.
    assert context[0, 0] == pytest.approx(1 / 3)          # rank fraction
    assert context[0, 1] == pytest.approx(0.0)            # gap to top
    assert context[1, 0] == pytest.approx(2 / 3)          # 0.7 ranks second
    assert context[1, 1] == pytest.approx(0.2, abs=1e-6)
    assert context[0, 3] == pytest.approx(np.log1p(3))
    assert context[1, 4] == pytest.approx(0 / 5)


# ---------------------------------------------------------------------------
# Artifact-backed dataset behavior
# ---------------------------------------------------------------------------

def test_resolve_feature_names_variants(artifact):
    table = _EventTable(*artifact)
    assert resolve_feature_names(table, 'none') == list(BASE_FEATURE_NAMES)
    assert resolve_feature_names(table, 'filter') == \
        list(BASE_FEATURE_NAMES) + FILTER_EXTRA_NAMES
    assert resolve_feature_names(table, 'all') == \
        list(BASE_FEATURE_NAMES) + FILTER_EXTRA_NAMES + CASCADE_EXTRA_NAMES
    assert resolve_feature_names(table, 'auto') == resolve_feature_names(table, 'all')
    with pytest.raises(ValueError):
        resolve_feature_names(table, 'gbdt')


def test_identity_echo_mismatch_is_rejected(artifact, tmp_path):
    candidates_path, _ = artifact
    events = synthetic_events(4, seed=9)
    for event in events:
        event['src']['event_id'] += 500  # different events entirely
    _, other_glob = write_fixture(tmp_path, events, dump_order=[0, 1, 2, 3])
    with pytest.raises(AssertionError, match='identity echo mismatch'):
        _EventTable(candidates_path, other_glob)


def test_serving_list_is_window_rows_at_or_above_tau(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto')
    arrays = dataset.table.candidate_arrays(0)
    item = dataset[0]
    n_window = int((arrays['row_kind'] == 0).sum())
    assert item['features'].shape[0] == n_window
    # Tightening tau to just above the weakest window score drops that row.
    weakest = arrays['filter_score'][arrays['row_kind'] == 0].min()
    tighter = TripletRankDataset(*artifact, tau=float(weakest) + 1e-6,
                                 mode='eval', extra_features='auto')
    assert tighter[0]['features'].shape[0] == n_window - 1


def test_filter_logit_matches_the_stored_scores(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto')
    arrays = dataset.table.candidate_arrays(0)
    item = dataset[0]
    scores = arrays['filter_score'][arrays['row_kind'] == 0]
    expected = np.log(scores / (1.0 - scores))
    assert item['filter_logit'].numpy() == pytest.approx(expected, rel=1e-5)


def test_train_item_marks_positives_first(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='train',
                                 num_negatives=8, extra_features='auto', seed=3)
    item = dataset[int(dataset.trainable_indices[0])]
    pos = item['pos_mask']
    n_pos = int(pos.sum())
    assert n_pos >= 1
    assert pos[:n_pos].all()
    assert not pos[n_pos:].any()
    assert item['features'].shape[0] == n_pos + 8


def test_tail_weighting_appends_reweighted_tail_rows(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='train',
                                 num_negatives=4, extra_features='auto',
                                 tail_weighting=True, seed=0)
    r = int(dataset.trainable_indices[0])
    arrays = dataset.table.candidate_arrays(r)
    n_tail_stored = int((arrays['row_kind'] == 2).sum())
    item = dataset[r]
    if n_tail_stored == 0:
        assert (item['log_weights'] == 0).all()
        return
    weights = item['log_weights']
    n_selected = weights.shape[0]
    expected = np.log(arrays['n_tail_total'] / n_tail_stored)
    assert weights[-n_tail_stored:].numpy() == pytest.approx(expected, rel=1e-6)
    assert (weights[:n_selected - n_tail_stored] == 0).all()
    assert not item['pos_mask'][-n_tail_stored:].any()


def test_from_b_targets_count_members(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto', from_b_targets=True)
    item = dataset[0]
    arrays = dataset.table.candidate_arrays(0)
    window = arrays['row_kind'] == 0
    # Track 3 is the only from-B track in the fixture.
    expected = ((arrays['cand_i'][window] == 3).astype(int)
                + (arrays['cand_j'][window] == 3).astype(int)
                + (arrays['cand_k'][window] == 3).astype(int))
    assert item['from_b'].numpy().tolist() == expected.tolist()


def test_static_fit_block_appends_fit_names(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='none', vertex_fit='static')
    assert dataset.feature_names[-len(FIT_NAMES):] == list(FIT_NAMES)
    item = dataset[0]
    assert item['features'].shape[1] == 100 + len(FIT_NAMES)
    assert torch.isfinite(item['features']).all()


def test_fit_layer_items_collate(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='train',
                                 num_negatives=4, extra_features='auto',
                                 vertex_fit='layer', seed=0)
    items = [dataset[int(r)] for r in dataset.trainable_indices[:2]]
    batch = collate_triplet_rank(items)
    n_max = batch['features'].shape[2]
    assert batch['fit_reference'].shape == (2, 3, 3, n_max)
    assert batch['fit_quality'].shape == (2, 12, 3, n_max)
    assert batch['fit_mass'].shape == (2, n_max)
    assert batch['primary_vertex'].shape == (2, 3)


def test_trainable_indices_require_a_servable_positive(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='train',
                                 extra_features='auto')
    for r in dataset.trainable_indices:
        arrays = dataset.table.candidate_arrays(int(r))
        window = arrays['row_kind'] == 0
        assert arrays['is_gt'][window].any()


def test_eval_collate_carries_keys_and_counts(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto')
    batch = collate_triplet_rank_eval([dataset[0], dataset[1]])
    assert len(batch['keys']) == 2
    assert batch['counts'].tolist() == [dataset[0]['features'].shape[0],
                                        dataset[1]['features'].shape[0]]
    assert batch['keys'][0].shape[1] == 3


def test_norm_stats_round_trip(artifact, tmp_path):
    stats = fit_norm_stats(*artifact, n_events=4, per_event=6, tau=-np.inf,
                           feature_names=list(BASE_FEATURE_NAMES)
                           + FILTER_EXTRA_NAMES + CASCADE_EXTRA_NAMES,
                           context_features=True, vertex_fit='static')
    expected_names = list(BASE_FEATURE_NAMES) + FILTER_EXTRA_NAMES \
        + CASCADE_EXTRA_NAMES + list(FIT_NAMES) + CONTEXT_FEATURE_NAMES
    assert set(stats) == set(expected_names)
    assert stats['s2_k_isvalid'] == {'log1p': False, 'center': 0.0, 'scale': 1.0}
    path = str(tmp_path / 'stats.json')
    save_norm_stats(stats, path)
    assert load_norm_stats(path) == stats

    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='all', vertex_fit='static',
                                 context_features=True, norm_stats=stats)
    item = dataset[0]
    assert item['features'].shape[1] == len(expected_names)
    assert torch.isfinite(item['features']).all()
    assert item['features'].abs().max() <= 10.0
