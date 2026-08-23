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


def test_max_serving_rows_truncates_at_filter_rank(artifact):
    full = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                              extra_features='auto')
    capped = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                extra_features='auto', max_serving_rows=3)
    n_full = full[0]['features'].shape[0]
    item = capped[0]
    assert item['features'].shape[0] == min(3, n_full)
    # Stored order = filter rank, so the cap keeps the top-scored rows.
    assert torch.equal(item['filter_logit'],
                       full[0]['filter_logit'][:min(3, n_full)])


def test_eval_item_couple_ids_encode_the_stage3_couple(artifact):
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto')
    arrays = dataset.table.candidate_arrays(0)
    serving = arrays['row_kind'] == 0
    expected = (arrays['cand_i'][serving].astype(np.int64) * 4096
                + arrays['cand_j'][serving].astype(np.int64))
    item = dataset[0]
    assert item['couple_ids'].numpy().tolist() == expected.tolist()


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


# ---------------------------------------------------------------------------
# TRACK32 blocks (hierarchical + couple warm-start substrate)
# ---------------------------------------------------------------------------

def test_track32_dataset_swaps_track_blocks_for_weaver_standardized(artifact):
    from utils.track32_features import (TRACK32_VAR_NAMES, TRACK32_YAML_PATH,
                                        compute_track32_raw,
                                        load_track32_params,
                                        standardize_track32)
    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto', track32=True)
    for prefix in ('ti_', 'tj_', 'tk_'):
        block = [name for name in dataset.feature_names
                 if name.startswith(prefix)]
        assert block == [f'{prefix}{var}' for var in TRACK32_VAR_NAMES]

    item = dataset[0]
    arrays = dataset.table.candidate_arrays(0)
    params = load_track32_params(TRACK32_YAML_PATH)
    source = {}
    for name in ['track_pt', 'track_eta', 'track_phi', 'track_charge',
                 'track_dxy_significance', 'track_dz_significance',
                 'track_norm_chi2', 'track_pt_error',
                 'track_n_valid_pixel_hits', 'track_dca_significance',
                 'track_covariance_phi_phi', 'track_covariance_lambda_lambda',
                 'track_dxy', 'track_dz', 'track_covariance_dxy_dxy',
                 'track_covariance_dsz_dsz', 'track_covariance_dxy_dsz',
                 'track_covariance_phi_dxy', 'track_n_valid_hits',
                 'track_vertex_x', 'track_vertex_y', 'track_vertex_z']:
        source[name] = np.asarray(dataset.table.tracks[name][0].values)
    for name in ['event_primary_vertex_x', 'event_primary_vertex_y',
                 'event_primary_vertex_z']:
        source[name] = dataset.table.events[name][0]
    source['event_n_pvs'] = dataset.table.track32_extras['event_n_pvs'][0]
    for name in ['event_other_pv_z', 'muon_eta', 'muon_phi', 'muon_dz',
                 'muon_soft_id', 'sv_x', 'sv_y', 'sv_z']:
        source[name] = np.asarray(
            dataset.table.track32_extras[name][0].values)
    expected = standardize_track32(compute_track32_raw(source), params)

    names = dataset.feature_names
    ti_start = names.index('ti_track_px')
    ti_block = item['features'][:, ti_start:ti_start + 32].numpy()
    cand_i = arrays['cand_i']
    for row in range(ti_block.shape[0]):
        np.testing.assert_allclose(ti_block[row], expected[cand_i[row]],
                                   rtol=1e-5, atol=1e-6)


def test_track32_blocks_bypass_dataset_standardization(artifact):
    plain = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                               extra_features='auto', track32=True)
    stats = fit_norm_stats(artifact[0], artifact[1],
                           feature_names=None, n_events=4, seed=0)
    stats.update({name: {'log1p': True, 'center': 5.0, 'scale': 9.0}
                  for name in plain.feature_names
                  if not name.startswith(('ti_', 'tj_', 'tk_'))
                  and name not in stats})
    standardized = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                      extra_features='auto', track32=True,
                                      norm_stats=stats)
    names = plain.feature_names
    ti_start = names.index('ti_track_px')
    raw_item = plain[0]['features'][:, ti_start:ti_start + 32]
    std_item = standardized[0]['features'][:, ti_start:ti_start + 32]
    torch.testing.assert_close(raw_item, std_item)


def test_warm_start_projector_loads_couple_weights(artifact, tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from train_triplet_reranker import _warm_start_projector
    from weaver.nn.model.TripletReranker import TripletReranker

    dataset = TripletRankDataset(*artifact, tau=-np.inf, mode='eval',
                                 extra_features='auto', track32=True)
    model = TripletReranker(input_mode='hierarchical', track_embed_dim=32,
                            projector_dim=32,
                            feature_names=dataset.feature_names, fusion=True)
    generator = torch.Generator().manual_seed(5)
    state = {
        'couple_projector.0.weight': torch.randn(32, 32, generator=generator),
        'couple_projector.0.bias': torch.randn(32, generator=generator),
        'couple_projector.2.weight': torch.randn(32, generator=generator),
        'couple_projector.2.bias': torch.randn(32, generator=generator),
        'other_component.weight': torch.zeros(2, 2),
    }
    checkpoint_path = tmp_path / 'couple_best.pt'
    torch.save({'couple_reranker_state_dict': state}, checkpoint_path)
    _warm_start_projector(model, str(checkpoint_path))
    torch.testing.assert_close(model.track_projector[0].weight,
                               state['couple_projector.0.weight'])
    torch.testing.assert_close(model.track_projector[2].bias,
                               state['couple_projector.2.bias'])
