"""Tests for the K* -> K pi candidate builder of the MiniAOD source.

Real data only: the builder is run on the batch-0 ntuple's tracks with the
trigger muon as anchor, and checked against the ntuple's own truth flags
(Track_trackFromKstar) and its CMSSW-fitted K* candidates."""
from __future__ import annotations

import glob
import os
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'convert_to_parquet',
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import miniaod_geometry as geometry  # noqa: E402
import miniaod_kstar as kstar  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NTUPLE_FILES = sorted(glob.glob(os.path.join(ROOT, 'data', 'low-pt', 'abcbatch', 'abcbatch0_2024_job*_ditaus_mc.root')))
ntuple_required = pytest.mark.skipif(len(NTUPLE_FILES) < 4, reason='the four abcbatch ntuple files are not present')
BRANCHES = ['Track_pt', 'Track_eta', 'Track_phi', 'Track_charge', 'Track_vx', 'Track_vy', 'Track_vz',
            'Track_covDxyDxy', 'Track_covDszDsz', 'Track_covDxyDsz', 'Track_trackFromKstar',
            'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_vx', 'Muon_vy', 'Muon_vz', 'BeamSpot_x', 'BeamSpot_y', 'BeamSpot_z',
            'KstarToPiKAll_trk1_idx', 'KstarToPiKAll_trk2_idx', 'KstarToPiKAll_vtx_x', 'KstarToPiKAll_vtx_y',
            'KstarToPiKAll_vtx_z']


@pytest.fixture(scope='module')
def built_events():
    """Per event with tracks and a muon: the built candidates, the event
    arrays and the row."""
    results = []
    for path in NTUPLE_FILES:
        events = uproot.open(path)['Events'].arrays(BRANCHES)
        for row in range(len(events)):
            if len(events['Track_pt'][row]) == 0 or len(events['Muon_pt'][row]) == 0:
                continue
            tracks = {name: ak.to_numpy(events[f'Track_{name}'][row]).astype(np.float64)
                      for name in ('pt', 'eta', 'phi', 'charge', 'covDxyDxy', 'covDszDsz', 'covDxyDsz')}
            point = np.stack([ak.to_numpy(events[f'Track_v{axis}'][row]) for axis in 'xyz'], axis=1).astype(np.float64)
            muon_point = np.array([events[f'Muon_v{axis}'][row][0] for axis in 'xyz'], dtype=np.float64)
            muon_direction = geometry.unit_direction(np.array([events['Muon_eta'][row][0]]), np.array([events['Muon_phi'][row][0]]))[0]
            beam_spot = np.array([events['BeamSpot_x'][row], events['BeamSpot_y'][row], events['BeamSpot_z'][row]], dtype=np.float64)
            candidates = kstar.kstar_candidates(
                point, tracks['pt'], tracks['eta'], tracks['phi'], tracks['charge'],
                tracks['covDxyDxy'], tracks['covDszDsz'], tracks['covDxyDsz'], muon_point, muon_direction, beam_spot)
            results.append((candidates, events, row, tracks))
    return results


@ntuple_required
def test_candidates_are_well_formed(built_events):
    total = 0
    for candidates, _, _, tracks in built_events:
        first, second = candidates['trk1_idx'], candidates['trk2_idx']
        total += first.size
        assert np.all(tracks['charge'][first] != tracks['charge'][second])
        assert np.all(tracks['pt'][first] >= tracks['pt'][second])
        assert np.all(tracks['pt'][second] >= kstar.MINIMUM_TRACK_PT)
        assert np.all((candidates['svprob'] >= kstar.MINIMUM_PROBABILITY) & (candidates['svprob'] <= 1.0))
        in_window = ((np.abs(candidates['fitted_mass_Kpi'] - kstar.KSTAR_MASS) < kstar.MASS_WINDOW)
                     | (np.abs(candidates['fitted_mass_piK'] - kstar.KSTAR_MASS) < kstar.MASS_WINDOW))
        assert np.all(in_window)
        assert np.all(candidates['muon_dca'] < kstar.MAXIMUM_MUON_DISTANCE)
        assert np.all(candidates['charge'] == 0)
        assert np.all(np.isfinite(candidates['lxy'])) and np.all(candidates['lxyerr'] >= 0)
        for name in ('vtx_x', 'vtx_y', 'vtx_z', 'fitted_pt', 'pt', 'eta', 'phi', 'pair_dca'):
            assert candidates[name].shape == first.shape
    assert total > 5000


@ntuple_required
def test_true_kstar_pair_is_among_the_candidates(built_events):
    found = reconstructable = 0
    for candidates, events, row, tracks in built_events:
        daughters = np.flatnonzero(ak.to_numpy(events['Track_trackFromKstar'][row]) == 1)
        if daughters.size != 2 or tracks['pt'][daughters].min() < kstar.MINIMUM_TRACK_PT:
            continue
        if tracks['charge'][daughters[0]] == tracks['charge'][daughters[1]]:
            continue
        reconstructable += 1
        pairs = set(zip(candidates['trk1_idx'].tolist(), candidates['trk2_idx'].tolist()))
        found += int((int(daughters[0]), int(daughters[1])) in pairs or (int(daughters[1]), int(daughters[0])) in pairs)
    assert reconstructable > 100
    assert found / reconstructable > 0.8


@ntuple_required
def test_candidate_vertex_agrees_with_the_cmssw_fit_for_shared_pairs(built_events):
    distances = []
    for candidates, events, row, _ in built_events:
        fitted_first = ak.to_numpy(events['KstarToPiKAll_trk1_idx'][row])
        fitted_second = ak.to_numpy(events['KstarToPiKAll_trk2_idx'][row])
        fitted_vertex = np.stack([ak.to_numpy(events[f'KstarToPiKAll_vtx_{axis}'][row]) for axis in 'xyz'], axis=1)
        lookup = {(int(a), int(b)): position for position, (a, b) in enumerate(zip(fitted_first, fitted_second))}
        for index, pair in enumerate(zip(candidates['trk1_idx'].tolist(), candidates['trk2_idx'].tolist())):
            position = lookup.get(pair, lookup.get(pair[::-1]))
            if position is None:
                continue
            ours = np.array([candidates['vtx_x'][index], candidates['vtx_y'][index], candidates['vtx_z'][index]])
            distances.append(np.linalg.norm(ours - fitted_vertex[position]))
    distances = np.array(distances)
    assert distances.size > 500
    assert np.median(distances) < 0.01
    assert np.percentile(distances, 90) < 0.1


@ntuple_required
def test_no_candidates_without_tracks_near_the_muon():
    point = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    candidates = kstar.kstar_candidates(
        point, np.array([2.0, 1.0]), np.array([0.1, -0.2]), np.array([0.3, 0.9]), np.array([1.0, -1.0]),
        np.full(2, 1e-4), np.full(2, 1e-4), np.zeros(2),
        np.array([5.0, 5.0, 5.0]), np.array([0.0, 0.0, 1.0]), np.zeros(3))
    assert candidates['trk1_idx'].size == 0
    assert set(kstar.CANDIDATE_FIELDS) == set(candidates)
