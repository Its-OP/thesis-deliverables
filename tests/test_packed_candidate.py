"""Tests for the numpy port of pat::PackedCandidate unpacking.

Real data only. Kinematics and the reference point are checked against
generator truth in MiniAOD (the tau decay vertex must lie on the unpacked
pion track lines); hit counts, track multiplicity, covariance terms and the
pT error are checked against the batch-0 ntuple, where CMSSW unpacked the
same quantities for tracks of the same simulation (different events, so the
comparison is made inside matched kinematic and hit-count bins)."""
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
import miniaod_reader as reader  # noqa: E402
import miniaod_truth as truth  # noqa: E402
import packed_candidate as packed  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MINIAOD_DIR = os.path.join(
    ROOT, 'data', 'low-pt', 'miniaod_sample', 'abcbatch1_2024', 'BdToKstarTauTau',
    'crab_abcbatch1_2024_BdToKstarTauTau_20261030_161621', '261030_151701', '0001')
MINIAOD_FILES = sorted(glob.glob(os.path.join(MINIAOD_DIR, 'miniaod_10*.root')))[:30]
NTUPLE_FILES = sorted(glob.glob(os.path.join(ROOT, 'data', 'low-pt', 'abcbatch', 'abcbatch0_2024_job*_ditaus_mc.root')))
miniaod_required = pytest.mark.skipif(len(MINIAOD_FILES) < 30, reason='MiniAOD sample files not present')
ntuple_required = pytest.mark.skipif(len(NTUPLE_FILES) < 4, reason='the four abcbatch ntuple files are not present')
PT_BINS = ((0.5, 0.7), (0.7, 1.0), (1.0, 2.0), (2.0, 5.0))
ETA_BINS = ((0.0, 0.8), (0.8, 1.6), (1.6, 2.4))
HIT_BINS = ((8, 11), (12, 15), (16, 22))
NTUPLE_COVARIANCE = {(0, 0): 'Track_covQopQop', (1, 1): 'Track_covLamLam', (2, 2): 'Track_covPhiPhi',
                     (3, 3): 'Track_covDxyDxy', (4, 4): 'Track_covDszDsz', (2, 3): 'Track_covPhiDxy'}


def event_blocks(products, row):
    """products: reader output; row: event. Returns the pf block as flat numpy
    arrays, the PV positions (V, 3) and the PV ndof (V,)."""
    block = {member: ak.to_numpy(values[row]) for member, values in products['pf'].items()}
    position = np.stack([ak.to_numpy(products['pv'][f'position_.fCoordinates.f{axis}'][row]) for axis in 'XYZ'], axis=1)
    return block, position, ak.to_numpy(products['pv']['ndof_'][row])


@pytest.fixture(scope='module')
def tables():
    return packed.CovarianceTables.load()


@pytest.fixture(scope='module')
def miniaod_tracks(tables):
    """Flat arrays of the packed pion-hypothesis candidates with track
    details, high purity, pT >= 0.5, |eta| <= 3 and from_pv > 0 over the
    fixture files, plus the number of such tracks per event."""
    columns = {name: [] for name in ('pt', 'eta', 'n_hits', 'n_pixel_hits', 'pt_error', 'track_pt')}
    covariance = {element: [] for element in NTUPLE_COVARIANCE}
    per_event = []
    for path in MINIAOD_FILES:
        products = reader.read_split_products(path)
        for row in range(products['n_events']):
            block, pv_position, pv_ndof = event_blocks(products, row)
            unpacked = packed.unpack_candidates(block, pv_position, pv_ndof, tables)
            keep = ((np.abs(unpacked['pdg']) == 211) & unpacked['has_track_details'] & unpacked['high_purity']
                    & (unpacked['pt'] >= 0.5) & (np.abs(unpacked['eta']) <= 3.0) & (unpacked['from_pv'] > 0))
            per_event.append(int(keep.sum()))
            for name in columns:
                columns[name].append(unpacked[name][keep])
            for element in covariance:
                covariance[element].append(unpacked['covariance'][element][keep])
    flat = {name: np.concatenate(parts) for name, parts in columns.items()}
    flat['covariance'] = {element: np.concatenate(parts) for element, parts in covariance.items()}
    flat['per_event'] = np.array(per_event)
    return flat


@pytest.fixture(scope='module')
def ntuple_tracks():
    branches = list(NTUPLE_COVARIANCE.values()) + ['Track_pt', 'Track_eta', 'Track_isPacked', 'Track_nValidHits',
                                                   'Track_nValidPixelHits', 'Track_ptErr']
    data = [uproot.open(path)['Events'].arrays(branches) for path in NTUPLE_FILES]
    flat = {name: np.concatenate([ak.to_numpy(ak.flatten(part[name])) for part in data]) for name in branches}
    keep = (flat['Track_isPacked'] == 1) & (flat['Track_pt'] >= 0.5)
    per_event = np.concatenate([ak.to_numpy(ak.sum((part['Track_isPacked'] == 1) & (part['Track_pt'] >= 0.5), axis=1)) for part in data])
    return {name: values[keep] for name, values in flat.items()}, per_event[per_event > 0]


def matched_bin_ratios(ours, ours_tracks, theirs, their_tracks):
    """Median of |ours| over median of |theirs| inside each (pT, |eta|, hits)
    bin for tracks with exactly four pixel hits. Returns the ratios of the
    bins with at least 40 tracks on both sides."""
    ratios = []
    for hit_low, hit_high in HIT_BINS:
        for pt_low, pt_high in PT_BINS:
            for eta_low, eta_high in ETA_BINS:
                a = np.abs(ours[(ours_tracks['pt'] >= pt_low) & (ours_tracks['pt'] < pt_high)
                                & (np.abs(ours_tracks['eta']) >= eta_low) & (np.abs(ours_tracks['eta']) < eta_high)
                                & (ours_tracks['n_hits'] >= hit_low) & (ours_tracks['n_hits'] <= hit_high)
                                & (ours_tracks['n_pixel_hits'] == 4)])
                b = np.abs(theirs[(their_tracks['Track_pt'] >= pt_low) & (their_tracks['Track_pt'] < pt_high)
                                  & (np.abs(their_tracks['Track_eta']) >= eta_low) & (np.abs(their_tracks['Track_eta']) < eta_high)
                                  & (their_tracks['Track_nValidHits'] >= hit_low) & (their_tracks['Track_nValidHits'] <= hit_high)
                                  & (their_tracks['Track_nValidPixelHits'] == 4)])
                if a.size >= 40 and b.size >= 40 and np.median(b) > 0:
                    ratios.append(np.median(a) / np.median(b))
    return np.array(ratios)


@miniaod_required
def test_unpacked_pion_tracks_pass_through_the_true_tau_decay_vertex(tables):
    distances = []
    for path in MINIAOD_FILES[:8]:
        products = reader.read_split_products(path)
        for row in range(products['n_events']):
            gen = {member: ak.to_numpy(values[row]) for member, values in products['gen'].items()}
            vertex = np.stack([gen[f'vertex_.fCoordinates.f{axis}'] for axis in 'XYZ'], axis=1)
            chain = truth.signal_chain(gen['pdgId_'], gen['status_'], vertex, gen['p4Polar_.fCoordinates.fEta'],
                                       gen['p4Polar_.fCoordinates.fPhi'])
            if chain['tau_pions'].size != 3:
                continue
            block, pv_position, pv_ndof = event_blocks(products, row)
            unpacked = packed.unpack_candidates(block, pv_position, pv_ndof, tables)
            tracks = np.flatnonzero((np.abs(unpacked['pdg']) == 211) & unpacked['has_track_details'])
            pions = chain['tau_pions']
            matched, _ = truth.match_tracks_to_generator(
                unpacked['pt'][tracks], unpacked['eta'][tracks], unpacked['phi'][tracks], unpacked['charge'][tracks],
                gen['p4Polar_.fCoordinates.fPt'][pions], gen['p4Polar_.fCoordinates.fEta'][pions],
                gen['p4Polar_.fCoordinates.fPhi'][pions], np.sign(gen['pdgId_'][pions]))
            found = tracks[matched[matched >= 0]]
            directions = geometry.unit_direction(unpacked['track_eta'][found], unpacked['track_phi'][found])
            offsets = geometry.point_line_offset(unpacked['point'][found], directions, chain['tau_decay_vertex'])
            distances.extend(np.linalg.norm(offsets, axis=1).tolist())
    distances = np.array(distances)
    assert distances.size > 100
    assert np.median(distances) < 0.01
    assert np.percentile(distances, 90) < 0.05


@miniaod_required
def test_reference_point_is_the_closest_approach_to_the_associated_vertex(tables):
    products = reader.read_split_products(MINIAOD_FILES[0])
    block, pv_position, pv_ndof = event_blocks(products, 0)
    unpacked = packed.unpack_candidates(block, pv_position, pv_ndof, tables)
    rows = np.flatnonzero(unpacked['has_track_details'] & (unpacked['pv_key'] >= 0))[:300]
    for row in rows:
        target = pv_position[unpacked['pv_key'][row]]
        assert packed.dxy_to_point(unpacked['point'][[row]], unpacked['track_phi'][[row]], target)[0] == pytest.approx(
            unpacked['dxy_associated'][row], abs=1e-6)
        assert packed.dz_to_point(unpacked['point'][[row]], unpacked['track_phi'][[row]], unpacked['track_eta'][[row]],
                                  target)[0] == pytest.approx(unpacked['dz_associated'][row], abs=1e-6)


@miniaod_required
@ntuple_required
def test_hit_counts_and_multiplicity_match_the_ntuple(miniaod_tracks, ntuple_tracks):
    theirs, their_per_event = ntuple_tracks
    assert np.all(miniaod_tracks['n_hits'] >= miniaod_tracks['n_pixel_hits'])
    np.testing.assert_array_equal(np.percentile(miniaod_tracks['n_pixel_hits'], [10, 50, 90]),
                                  np.percentile(theirs['Track_nValidPixelHits'], [10, 50, 90]))
    # the ntuple's hidden track selection is reproduced by from_pv > 0
    assert np.median(miniaod_tracks['per_event']) == pytest.approx(np.median(their_per_event), rel=0.2)


@miniaod_required
@ntuple_required
@pytest.mark.parametrize('element', sorted(NTUPLE_COVARIANCE))
def test_covariance_matches_the_ntuple_in_matched_bins(miniaod_tracks, ntuple_tracks, element):
    theirs, _ = ntuple_tracks
    ratios = matched_bin_ratios(miniaod_tracks['covariance'][element], miniaod_tracks,
                                theirs[NTUPLE_COVARIANCE[element]], theirs)
    assert ratios.size >= 10
    assert np.median(ratios) == pytest.approx(1.0, abs=0.05)


@miniaod_required
@ntuple_required
def test_pt_error_matches_the_ntuple_in_matched_bins(miniaod_tracks, ntuple_tracks):
    theirs, _ = ntuple_tracks
    ratios = matched_bin_ratios(miniaod_tracks['pt_error'] / miniaod_tracks['track_pt'], miniaod_tracks,
                                theirs['Track_ptErr'] / theirs['Track_pt'], theirs)
    assert ratios.size >= 10
    assert np.median(ratios) == pytest.approx(1.0, abs=0.05)


def test_charge_follows_the_particle_code():
    assert packed.charge(np.array([211, -211, 11, -11, 13, -13, 22, 130, 1, 2])).tolist() == [1, -1, -1, 1, -1, 1, 0, 0, 0, 0]


def test_from_pv_follows_the_cmssw_rule():
    # columns: pdg, pv_key, quality, dz_associated; PV z positions and ndof below
    pdg = np.array([211, 211, 13, 211, 211, 211, 211])
    pv_key = np.array([0, 0, 2, 1, 1, 1, 1])
    quality = np.array([7, 5, 7, 4, 4, 5, 7])
    dz_associated = np.array([0.0, 0.0, 0.0, 0.5, 0.01, 0.0, 0.0])
    pv_z = np.array([0.0, 0.2, 5.0])
    pv_ndof = np.array([50.0, 30.0, 2.0])
    # 0: used in the fit of PV0; 1: associated to PV0; 2: muon; 3: b-tag compatible and closer to PV0
    # (|0.5| > |0.5 + 0.2 - 0|? no -> falls through to quality < 6 -> loose); 4: b-tag compatible, farther -> loose;
    # 5: dz compatible with another PV -> loose; 6: used in the fit of another PV -> none
    assert packed.from_pv(pdg, pv_key, quality, dz_associated, pv_z, pv_ndof).tolist() == [3, 2, 2, 1, 1, 1, 0]
    # a b-tag compatible track farther from its own vertex than from PV0 counts as tight
    assert packed.from_pv(np.array([211]), np.array([1]), np.array([4]), np.array([-0.15]), pv_z, pv_ndof).tolist() == [2]
    # a track fitted to a vertex with fewer than four degrees of freedom is not removed
    assert packed.from_pv(np.array([211]), np.array([2]), np.array([7]), np.array([0.0]), pv_z, pv_ndof).tolist() == [1]
