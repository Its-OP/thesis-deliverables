from __future__ import annotations

import hashlib
import os

import numpy as np
import yaml

# The 32 standardized pf_features channels the promoted couple reranker was
# trained on, in the exact channel order of its data-config yaml. The couple
# projector's warm-started weights are only meaningful if the triplet dataset
# reproduces these channels bit-for-bit (weaver convention: affine, clip,
# then NaN -> 0).
# The filename hash is weaver's source-config hash; the content digest pins
# the exact resolved standardization constants.
TRACK32_YAML_CONFIG_HASH = '4bb52a63a2023c146396395a612cbe3f'
TRACK32_YAML_MD5 = '0e33178976ba97bfaac8835f59b4cd31'
TRACK32_YAML_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt',
    f'lowpt_tau_trackfinder.{TRACK32_YAML_CONFIG_HASH}.auto.yaml')

TRACK32_VAR_NAMES = [
    'track_px', 'track_py', 'track_pz', 'track_eta', 'track_phi',
    'track_charge', 'track_dxy_significance', 'track_log_dz_significance',
    'track_log_norm_chi2', 'track_log_pt_error', 'track_n_valid_pixel_hits',
    'track_dca_significance', 'track_log_covariance_phi_phi',
    'track_log_covariance_lambda_lambda', 'track_log_pt',
    'track_log_relative_pt_error', 'track_absolute_dxy', 'track_absolute_dz',
    'track_log_covariance_dxy_dxy', 'track_log_covariance_dsz_dsz',
    'track_log_absolute_covariance_dxy_dsz',
    'track_log_absolute_covariance_phi_dxy', 'track_n_valid_hits',
    'track_pileup_min_gap', 'track_closer_to_other_pv', 'track_n_pvs',
    'track_muon_min_delta_r', 'track_muon_min_dz_gap', 'track_has_soft_muon',
    'track_sv_min_line_distance', 'track_has_sv',
    'track_pv_line_distance_3d',
]

TRACK32_PASSTHROUGH = ['track_closer_to_other_pv', 'track_has_soft_muon',
                       'track_has_sv']

TRACK32_SOURCE_TRACK_COLUMNS = [
    'track_pt', 'track_eta', 'track_phi', 'track_charge',
    'track_dxy_significance', 'track_dz_significance', 'track_norm_chi2',
    'track_pt_error', 'track_n_valid_pixel_hits', 'track_dca_significance',
    'track_covariance_phi_phi', 'track_covariance_lambda_lambda',
    'track_dxy', 'track_dz', 'track_covariance_dxy_dxy',
    'track_covariance_dsz_dsz', 'track_covariance_dxy_dsz',
    'track_covariance_phi_dxy', 'track_n_valid_hits',
    'track_vertex_x', 'track_vertex_y', 'track_vertex_z',
]
TRACK32_SOURCE_EVENT_COLUMNS = [
    'event_n_pvs', 'event_other_pv_z', 'muon_eta', 'muon_phi', 'muon_dz',
    'muon_soft_id', 'sv_x', 'sv_y', 'sv_z',
]


def load_track32_params(yaml_path: str) -> dict[str, dict]:
    with open(yaml_path, 'rb') as fh:
        content = fh.read()
    digest = hashlib.md5(content).hexdigest()
    assert digest == TRACK32_YAML_MD5, (
        f'track32 yaml digest {digest} does not match the couple stage\'s '
        f'training config {TRACK32_YAML_MD5}: warm-started projector weights '
        f'would see differently standardized channels')
    config = yaml.safe_load(content)
    variables = [entry if isinstance(entry, str) else entry[0]
                 for entry in config['inputs']['pf_features']['vars']]
    assert variables == TRACK32_VAR_NAMES, \
        f'pf_features channel order changed: {variables}'
    raw_params = config['preprocess']['params']
    params = {}
    for name in TRACK32_VAR_NAMES:
        entry = raw_params[name]
        params[name] = dict(center=entry['center'], scale=entry['scale'],
                            min=entry['min'], max=entry['max'])
    passthrough = {name for name, entry in params.items()
                   if entry['center'] is None}
    assert passthrough == set(TRACK32_PASSTHROUGH), \
        f'unexpected passthrough channels: {sorted(passthrough)}'
    return params


def _min_over_or_nan(distances: np.ndarray, count: int) -> np.ndarray:
    """distances: (T, count). Returns (T,), NaN where count == 0."""
    if count == 0:
        return np.full(distances.shape[0], np.nan)
    return distances.min(axis=1)


def _line_distance(point: np.ndarray, vertex: np.ndarray,
                   direction: np.ndarray) -> np.ndarray:
    """point: (..., 3) broadcastable against vertex/direction (T, ..., 3).
    Returns the perpendicular distance from point to each track line."""
    separation = point - vertex
    cross = np.cross(separation, direction)
    return np.sqrt((cross ** 2).sum(axis=-1))


def compute_track32_raw(event: dict) -> np.ndarray:
    """event: raw per-track arrays (T,), event scalars, and muon/SV/other-PV
    lists for one event. Returns (T, 32) float64 in TRACK32_VAR_NAMES order,
    NaN where a min runs over an empty list (weaver standardizes those to the
    channel median)."""
    pt = np.asarray(event['track_pt'], dtype=np.float64)
    eta = np.asarray(event['track_eta'], dtype=np.float64)
    phi = np.asarray(event['track_phi'], dtype=np.float64)
    dz = np.asarray(event['track_dz'], dtype=np.float64)
    pt_error = np.asarray(event['track_pt_error'], dtype=np.float64)
    dz_significance = np.asarray(event['track_dz_significance'],
                                 dtype=np.float64)
    vertex = np.stack([np.asarray(event[f'track_vertex_{axis}'],
                                  dtype=np.float64)
                       for axis in 'xyz'], axis=-1)
    direction = np.stack([np.cos(phi) / np.cosh(eta),
                          np.sin(phi) / np.cosh(eta),
                          np.tanh(eta)], axis=-1)

    global_z = float(event['event_primary_vertex_z']) + dz
    other_pv_z = np.asarray(event['event_other_pv_z'], dtype=np.float64)
    pileup_min_gap = _min_over_or_nan(
        np.abs(global_z[:, None] - other_pv_z[None, :]), other_pv_z.size)

    soft = np.asarray(event['muon_soft_id'], dtype=np.float64) == 1
    muon_eta = np.asarray(event['muon_eta'], dtype=np.float64)[soft]
    muon_phi = np.asarray(event['muon_phi'], dtype=np.float64)[soft]
    muon_dz = np.asarray(event['muon_dz'], dtype=np.float64)[soft]
    delta_phi = np.mod(phi[:, None] - muon_phi[None, :] + np.pi,
                       2 * np.pi) - np.pi
    muon_min_delta_r = _min_over_or_nan(
        np.sqrt((eta[:, None] - muon_eta[None, :]) ** 2 + delta_phi ** 2),
        muon_eta.size)
    muon_min_dz_gap = _min_over_or_nan(
        np.abs(dz[:, None] - muon_dz[None, :]), muon_dz.size)

    sv = np.stack([np.asarray(event[f'sv_{axis}'], dtype=np.float64)
                   for axis in 'xyz'], axis=-1)
    sv_min_line_distance = _min_over_or_nan(
        _line_distance(sv[None, :, :], vertex[:, None, :],
                       direction[:, None, :]), sv.shape[0])

    primary_vertex = np.array([float(event['event_primary_vertex_x']),
                               float(event['event_primary_vertex_y']),
                               float(event['event_primary_vertex_z'])])
    pv_line_distance = _line_distance(primary_vertex[None, :], vertex,
                                      direction)

    def column(name):
        return np.asarray(event[name], dtype=np.float64)

    ones = np.ones_like(eta)
    channels = [
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        eta,
        phi,
        column('track_charge'),
        column('track_dxy_significance'),
        np.sign(dz_significance) * np.log1p(np.abs(dz_significance)),
        np.log1p(column('track_norm_chi2')),
        np.log(np.maximum(pt_error, 1e-12)),
        column('track_n_valid_pixel_hits'),
        column('track_dca_significance'),
        np.log(np.maximum(column('track_covariance_phi_phi'), 1e-12)),
        np.log(np.maximum(column('track_covariance_lambda_lambda'), 1e-12)),
        np.log(pt + 1e-6),
        np.log(np.maximum(pt_error / (pt + 1e-6), 1e-12)),
        np.abs(column('track_dxy')),
        np.abs(dz),
        np.log(np.maximum(column('track_covariance_dxy_dxy'), 1e-30)),
        np.log(np.maximum(column('track_covariance_dsz_dsz'), 1e-30)),
        np.log(np.abs(column('track_covariance_dxy_dsz')) + 1e-30),
        np.log(np.abs(column('track_covariance_phi_dxy')) + 1e-30),
        column('track_n_valid_hits'),
        pileup_min_gap,
        (pileup_min_gap < np.abs(dz)) * 1.0,
        float(event['event_n_pvs']) * ones,
        muon_min_delta_r,
        muon_min_dz_gap,
        float(muon_eta.size > 0) * ones,
        sv_min_line_distance,
        float(sv.shape[0] > 0) * ones,
        pv_line_distance,
    ]
    return np.stack(channels, axis=-1)


def standardize_track32(raw: np.ndarray, params: dict[str, dict]) -> np.ndarray:
    """raw: (T, 32) in TRACK32_VAR_NAMES order. Returns (T, 32) float32
    standardized exactly like weaver's _finalize_inputs: affine, clip to
    [min, max], then NaN -> 0; passthrough channels only get NaN -> 0."""
    output = np.empty_like(raw)
    for index, name in enumerate(TRACK32_VAR_NAMES):
        entry = params[name]
        values = raw[:, index]
        if entry['center'] is not None:
            values = np.clip((values - entry['center']) * entry['scale'],
                             entry['min'], entry['max'])
        output[:, index] = np.nan_to_num(values)
    return output.astype(np.float32)
