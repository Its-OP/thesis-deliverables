"""K* -> K pi candidate builder for the MiniAOD source (numpy only).

Both K* daughters come from the B decay vertex, and the tau-side muon track
passes within about 0.02 cm of that vertex, so candidates are opposite-sign
track pairs whose tracks both pass close to the muon track. The vertex is
the midpoint of the pair's closest approach (within 0.004 cm of the CMSSW
kinematic-fit vertex at the median on the batch-0 ntuple); svprob is the
one-degree-of-freedom probability that the two tracks cross there, from
their separation and position errors along it.
"""
from __future__ import annotations

import math

import numpy as np

import miniaod_geometry as geometry

KAON_MASS, PION_MASS, KSTAR_MASS = 0.493677, 0.13957039, 0.892
MINIMUM_TRACK_PT = 0.5
MAXIMUM_MUON_DISTANCE = 0.05
MASS_WINDOW = 0.15
MINIMUM_PROBABILITY = 0.01
CANDIDATE_FIELDS = ('trk1_idx', 'trk2_idx', 'vtx_x', 'vtx_y', 'vtx_z', 'svprob', 'fitted_mass_Kpi',
                    'fitted_mass_piK', 'fitted_pt', 'lxy', 'lxyerr', 'pt', 'eta', 'phi', 'charge',
                    'muon_dca', 'pair_dca')
INTEGER_FIELDS = ('trk1_idx', 'trk2_idx', 'charge')
_complementary_error = np.frompyfunc(math.erfc, 1, 1)


def empty_candidates():
    """Returns {CANDIDATE_FIELDS: (0,)}: the result for an event without
    candidates."""
    return {name: np.zeros(0, dtype=np.int64 if name in INTEGER_FIELDS else np.float64) for name in CANDIDATE_FIELDS}


def crossing_probability(separation, variance):
    """separation: (N,) closest-approach distances; variance: (N,) summed
    position variances of the two tracks along the separation. Returns
    (N,) one-degree-of-freedom chi-square probabilities."""
    chi_square = separation ** 2 / np.where(variance > 0, variance, np.inf)
    return _complementary_error(np.sqrt(chi_square / 2.0)).astype(np.float64)


def kstar_candidates(point, pt, eta, phi, charge, cov_dxy_dxy, cov_dsz_dsz, cov_dxy_dsz,
                     muon_point, muon_direction, beam_spot):
    """point: (T, 3) track reference points; pt, eta, phi, charge: (T,)
    track momenta at those points (tracks in decreasing pT); cov_*: (T,)
    track covariance terms; muon_point, muon_direction: (3,) anchor muon
    line; beam_spot: (3,). Returns {CANDIDATE_FIELDS: (C,)}: trk1 is the
    higher-pT track, fitted_mass_Kpi takes trk1 as the kaon."""
    point = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    pt, eta, phi = (np.asarray(value, dtype=np.float64) for value in (pt, eta, phi))
    if not pt.size:
        return empty_candidates()
    direction = geometry.unit_direction(eta, phi)
    muon_distance, _ = geometry.line_distance(point, direction, muon_point, muon_direction)
    near = np.flatnonzero((pt >= MINIMUM_TRACK_PT) & (muon_distance < MAXIMUM_MUON_DISTANCE))
    first, second = np.triu_indices(near.size, k=1)
    first, second = near[first], near[second]
    opposite = np.asarray(charge)[first] != np.asarray(charge)[second]
    first, second = first[opposite], second[opposite]
    swap = pt[second] > pt[first]
    first, second = np.where(swap, second, first), np.where(swap, first, second)
    mass_kaon_pion = geometry.pair_mass(pt[first], eta[first], phi[first], KAON_MASS, pt[second], eta[second], phi[second], PION_MASS)
    mass_pion_kaon = geometry.pair_mass(pt[first], eta[first], phi[first], PION_MASS, pt[second], eta[second], phi[second], KAON_MASS)
    in_window = (np.abs(mass_kaon_pion - KSTAR_MASS) < MASS_WINDOW) | (np.abs(mass_pion_kaon - KSTAR_MASS) < MASS_WINDOW)
    first, second = first[in_window], second[in_window]
    if not first.size:
        return empty_candidates()
    separation, midpoint = geometry.line_distance(point[first], direction[first], point[second], direction[second])
    normal = np.cross(direction[first], direction[second])
    normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
    variance = (geometry.perpendicular_variance(normal, phi[first], eta[first], cov_dxy_dxy[first], cov_dsz_dsz[first], cov_dxy_dsz[first])
                + geometry.perpendicular_variance(normal, phi[second], eta[second], cov_dxy_dxy[second], cov_dsz_dsz[second], cov_dxy_dsz[second]))
    probability = crossing_probability(separation, variance)
    keep = probability >= MINIMUM_PROBABILITY
    first, second, separation, midpoint, probability = (
        value[keep] for value in (first, second, separation, midpoint, probability))
    momentum = pt[:, None] * np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], axis=1)
    total = momentum[first] + momentum[second]
    total_pt = np.hypot(total[:, 0], total[:, 1])
    flight = midpoint[:, :2] - np.asarray(beam_spot, dtype=np.float64)[None, :2]
    lxy = np.linalg.norm(flight, axis=1)
    flight_direction = np.concatenate([flight / np.maximum(lxy, 1e-12)[:, None], np.zeros((lxy.size, 1))], axis=1)
    lxy_variance = 0.25 * (
        geometry.perpendicular_variance(flight_direction, phi[first], eta[first], cov_dxy_dxy[first], cov_dsz_dsz[first], cov_dxy_dsz[first])
        + geometry.perpendicular_variance(flight_direction, phi[second], eta[second], cov_dxy_dxy[second], cov_dsz_dsz[second], cov_dxy_dsz[second]))
    return {
        'trk1_idx': first.astype(np.int64), 'trk2_idx': second.astype(np.int64),
        'vtx_x': midpoint[:, 0], 'vtx_y': midpoint[:, 1], 'vtx_z': midpoint[:, 2], 'svprob': probability,
        'fitted_mass_Kpi': mass_kaon_pion[in_window][keep], 'fitted_mass_piK': mass_pion_kaon[in_window][keep],
        'fitted_pt': total_pt, 'lxy': lxy, 'lxyerr': np.sqrt(np.maximum(lxy_variance, 0.0)),
        'pt': total_pt, 'eta': np.arcsinh(total[:, 2] / np.maximum(total_pt, 1e-12)),
        'phi': np.arctan2(total[:, 1], total[:, 0]), 'charge': np.zeros(first.size, dtype=np.int64),
        'muon_dca': np.maximum(muon_distance[first], muon_distance[second]), 'pair_dca': separation,
    }
