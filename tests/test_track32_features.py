from __future__ import annotations

import numpy as np
import pytest

from utils.track32_features import (
    TRACK32_PASSTHROUGH,
    TRACK32_VAR_NAMES,
    TRACK32_YAML_PATH,
    compute_track32_raw,
    load_track32_params,
    standardize_track32,
)


def _event(tracks=2):
    return dict(
        track_pt=np.array([1.0, 2.0][:tracks]),
        track_eta=np.array([0.5, -0.3][:tracks]),
        track_phi=np.array([3.0, 0.2][:tracks]),
        track_charge=np.array([1.0, -1.0][:tracks]),
        track_dxy_significance=np.array([0.4, -1.5][:tracks]),
        track_dz_significance=np.array([-2.0, 3.0][:tracks]),
        track_norm_chi2=np.array([1.5, 0.8][:tracks]),
        track_pt_error=np.array([0.02, 0.05][:tracks]),
        track_n_valid_pixel_hits=np.array([4.0, 3.0][:tracks]),
        track_dca_significance=np.array([1.1, 0.9][:tracks]),
        track_covariance_phi_phi=np.array([1e-3, 2e-3][:tracks]),
        track_covariance_lambda_lambda=np.array([2e-3, 3e-3][:tracks]),
        track_dxy=np.array([-0.02, 0.03][:tracks]),
        track_dz=np.array([0.30, -0.10][:tracks]),
        track_covariance_dxy_dxy=np.array([1e-4, 2e-4][:tracks]),
        track_covariance_dsz_dsz=np.array([2e-4, 3e-4][:tracks]),
        track_covariance_dxy_dsz=np.array([1e-5, -2e-5][:tracks]),
        track_covariance_phi_dxy=np.array([-1e-6, 2e-6][:tracks]),
        track_n_valid_hits=np.array([12.0, 9.0][:tracks]),
        track_vertex_x=np.array([0.10, -0.05][:tracks]),
        track_vertex_y=np.array([0.05, 0.02][:tracks]),
        track_vertex_z=np.array([1.00, 0.50][:tracks]),
        event_primary_vertex_x=0.01,
        event_primary_vertex_y=-0.02,
        event_primary_vertex_z=0.90,
        event_n_pvs=3.0,
        event_other_pv_z=np.array([1.0, 3.0]),
        muon_eta=np.array([0.45, 5.0]),
        muon_phi=np.array([-3.0, 0.0]),
        muon_dz=np.array([0.25, 9.0]),
        muon_soft_id=np.array([1.0, 0.0]),
        sv_x=np.array([0.10]),
        sv_y=np.array([0.06]),
        sv_z=np.array([1.10]),
    )


def test_params_load_order_and_passthrough():
    params = load_track32_params(TRACK32_YAML_PATH)
    assert list(params) == TRACK32_VAR_NAMES
    assert len(params) == 32
    passthrough = {name for name, entry in params.items()
                   if entry['center'] is None}
    assert passthrough == set(TRACK32_PASSTHROUGH)
    assert params['track_px']['center'] == pytest.approx(-0.006469251122325659)
    assert params['track_px']['scale'] == pytest.approx(1.4233942106180126)
    assert params['track_px']['min'] == -5
    assert params['track_px']['max'] == 5


def test_params_reject_tampered_yaml(tmp_path):
    tampered = tmp_path / 'tampered.auto.yaml'
    tampered.write_text('preprocess: {params: {}}\n')
    with pytest.raises(AssertionError):
        load_track32_params(str(tampered))


def test_raw_formulas_match_hand_computation():
    event = _event()
    raw = compute_track32_raw(event)
    assert raw.shape == (2, 32)
    channel = {name: index for index, name in enumerate(TRACK32_VAR_NAMES)}

    assert raw[0, channel['track_px']] == pytest.approx(1.0 * np.cos(3.0))
    assert raw[0, channel['track_py']] == pytest.approx(1.0 * np.sin(3.0))
    assert raw[0, channel['track_pz']] == pytest.approx(np.sinh(0.5))
    assert raw[1, channel['track_log_dz_significance']] == pytest.approx(
        np.log1p(3.0))
    assert raw[0, channel['track_log_dz_significance']] == pytest.approx(
        -np.log1p(2.0))
    assert raw[0, channel['track_log_relative_pt_error']] == pytest.approx(
        np.log(0.02 / (1.0 + 1e-6)))
    assert raw[0, channel['track_absolute_dxy']] == pytest.approx(0.02)
    assert raw[0, channel['track_log_absolute_covariance_dxy_dsz']] \
        == pytest.approx(np.log(1e-5 + 1e-30))

    # Pileup gap: global z = pv_z + dz; nearest of the two other PVs.
    global_z = 0.90 + 0.30
    assert raw[0, channel['track_pileup_min_gap']] == pytest.approx(
        min(abs(global_z - 1.0), abs(global_z - 3.0)))
    # gap 0.2 < |dz| 0.3 -> flag set
    assert raw[0, channel['track_closer_to_other_pv']] == 1.0
    assert raw[0, channel['track_n_pvs']] == 3.0

    # Only the soft muon (id == 1) participates; phi wraps across +-pi.
    delta_eta = 0.5 - 0.45
    delta_phi = np.mod(3.0 - (-3.0) + np.pi, 2 * np.pi) - np.pi
    assert raw[0, channel['track_muon_min_delta_r']] == pytest.approx(
        np.hypot(delta_eta, delta_phi))
    assert raw[0, channel['track_muon_min_dz_gap']] == pytest.approx(
        abs(0.30 - 0.25))
    assert raw[0, channel['track_has_soft_muon']] == 1.0

    # SV line distance: |(sv - vertex) x direction| with unit direction.
    direction = np.array([np.cos(3.0) / np.cosh(0.5),
                          np.sin(3.0) / np.cosh(0.5), np.tanh(0.5)])
    separation = np.array([0.10 - 0.10, 0.06 - 0.05, 1.10 - 1.00])
    assert raw[0, channel['track_sv_min_line_distance']] == pytest.approx(
        np.linalg.norm(np.cross(separation, direction)))
    assert raw[0, channel['track_has_sv']] == 1.0

    pv_separation = np.array([0.01 - 0.10, -0.02 - 0.05, 0.90 - 1.00])
    assert raw[0, channel['track_pv_line_distance_3d']] == pytest.approx(
        np.linalg.norm(np.cross(pv_separation, direction)))


def test_empty_lists_standardize_to_center():
    event = _event()
    event['event_other_pv_z'] = np.zeros(0)
    event['muon_eta'] = np.zeros(0)
    event['muon_phi'] = np.zeros(0)
    event['muon_dz'] = np.zeros(0)
    event['muon_soft_id'] = np.zeros(0)
    event['sv_x'] = np.zeros(0)
    event['sv_y'] = np.zeros(0)
    event['sv_z'] = np.zeros(0)
    raw = compute_track32_raw(event)
    channel = {name: index for index, name in enumerate(TRACK32_VAR_NAMES)}
    for name in ['track_pileup_min_gap', 'track_muon_min_delta_r',
                 'track_muon_min_dz_gap', 'track_sv_min_line_distance']:
        assert np.isnan(raw[:, channel[name]]).all()
    for name in ['track_closer_to_other_pv', 'track_has_soft_muon',
                 'track_has_sv']:
        assert (raw[:, channel[name]] == 0.0).all()

    params = load_track32_params(TRACK32_YAML_PATH)
    standardized = standardize_track32(raw, params)
    assert standardized.dtype == np.float32
    assert np.isfinite(standardized).all()
    # Weaver maps missing values to 0 AFTER the affine transform: the median.
    for name in ['track_pileup_min_gap', 'track_muon_min_delta_r',
                 'track_muon_min_dz_gap', 'track_sv_min_line_distance']:
        assert (standardized[:, channel[name]] == 0.0).all()


def test_standardize_matches_weaver_convention_and_clips():
    event = _event()
    event['track_dz'] = np.array([1e6, -0.10])
    raw = compute_track32_raw(event)
    params = load_track32_params(TRACK32_YAML_PATH)
    standardized = standardize_track32(raw, params)
    channel = {name: index for index, name in enumerate(TRACK32_VAR_NAMES)}
    assert standardized[0, channel['track_absolute_dz']] == 5.0

    entry = params['track_px']
    expected = np.clip((raw[1, channel['track_px']] - entry['center'])
                       * entry['scale'], entry['min'], entry['max'])
    assert standardized[1, channel['track_px']] == pytest.approx(
        expected, abs=1e-6)
    # Passthrough channels keep their raw value.
    assert standardized[0, channel['track_has_sv']] == 1.0
