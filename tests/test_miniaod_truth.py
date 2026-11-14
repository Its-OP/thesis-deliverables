"""Tests for the generator-truth helpers of the MiniAOD source: the signal
chain found from shared production vertices, the K* daughters recovered from
the packed generator collection, and the track <-> generator matching.

Real data only: MiniAOD files of abcbatch1 (CRAB subdirectory 0001) and the
batch-0 ntuple. Expectations come from the production's own generator
configuration (pion filter pT >= 0.5, |eta| <= 2.5; tau-muon filter
pT >= 5.8, |eta| <= 1.55; K*0 -> K pi) and from the ntuple's stored matches."""
from __future__ import annotations

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
import miniaod_truth as truth  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MINIAOD_DIR = os.path.join(
    ROOT, 'data', 'low-pt', 'miniaod_sample', 'abcbatch1_2024', 'BdToKstarTauTau',
    'crab_abcbatch1_2024_BdToKstarTauTau_20261030_161621', '261030_151701', '0001')
MINIAOD_FILES = [os.path.join(MINIAOD_DIR, f'miniaod_{number}.root') for number in (1000, 1001, 1003, 1004)]
JOB0 = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch', 'abcbatch0_2024_job0_ditaus_mc.root')
miniaod_required = pytest.mark.skipif(not all(os.path.exists(path) for path in MINIAOD_FILES),
                                      reason='MiniAOD sample files not present')
job0_required = pytest.mark.skipif(not os.path.exists(JOB0), reason='abcbatch job0 file not present')

GEN = 'recoGenParticles_prunedGenParticles__PAT.obj.m_state.'
PACKED = 'patPackedGenParticles_packedGenParticles__PAT.obj.'


@pytest.fixture(scope='module')
def generator_events():
    """List of (pruned dict, packed dict) per event over the fixture files."""
    events = []
    for path in MINIAOD_FILES:
        data = uproot.open(path)['Events'].arrays(
            [GEN + name for name in ('pdgId_', 'status_', 'p4Polar_.fCoordinates.fPt', 'p4Polar_.fCoordinates.fEta',
                                     'p4Polar_.fCoordinates.fPhi', 'p4Polar_.fCoordinates.fM',
                                     'vertex_.fCoordinates.fX', 'vertex_.fCoordinates.fY', 'vertex_.fCoordinates.fZ')]
            + [PACKED + name for name in ('pdgId_', 'packedPt_', 'packedY_', 'packedPhi_', 'packedM_')])
        for row in range(len(data)):
            pruned = {
                'pdg': ak.to_numpy(data[GEN + 'pdgId_'][row]), 'status': ak.to_numpy(data[GEN + 'status_'][row]),
                'pt': ak.to_numpy(data[GEN + 'p4Polar_.fCoordinates.fPt'][row]),
                'eta': ak.to_numpy(data[GEN + 'p4Polar_.fCoordinates.fEta'][row]),
                'phi': ak.to_numpy(data[GEN + 'p4Polar_.fCoordinates.fPhi'][row]),
                'mass': ak.to_numpy(data[GEN + 'p4Polar_.fCoordinates.fM'][row]),
                'vertex': np.stack([ak.to_numpy(data[GEN + f'vertex_.fCoordinates.f{axis}'][row]) for axis in 'XYZ'], axis=1),
            }
            packed = truth.unpack_packed_generator(
                ak.to_numpy(data[PACKED + 'pdgId_'][row]), ak.to_numpy(data[PACKED + 'packedPt_'][row]),
                ak.to_numpy(data[PACKED + 'packedY_'][row]), ak.to_numpy(data[PACKED + 'packedPhi_'][row]),
                ak.to_numpy(data[PACKED + 'packedM_'][row]))
            events.append((pruned, packed))
    return events


@miniaod_required
def test_signal_chain_obeys_the_generator_filters(generator_events):
    chains = [truth.signal_chain(pruned['pdg'], pruned['status'], pruned['vertex'], pruned['eta'], pruned['phi'])
              for pruned, _ in generator_events]
    assert len(chains) >= 30
    complete = [chain for chain in chains if chain['tau_pions'].size == 3 and chain['tau_muon'] >= 0]
    assert len(complete) >= 0.95 * len(chains)
    for (pruned, _), chain in zip(generator_events, chains):
        if chain['tau_pions'].size != 3 or chain['tau_muon'] < 0:
            continue
        pions, muon = chain['tau_pions'], chain['tau_muon']
        assert np.all(np.abs(pruned['pdg'][pions]) == 211)
        assert np.all(pruned['pt'][pions] >= 0.5) and np.all(np.abs(pruned['eta'][pions]) <= 2.5)
        assert abs(int(np.sum(np.sign(pruned['pdg'][pions])))) == 1
        assert abs(pruned['pdg'][muon]) == 13
        assert pruned['pt'][muon] >= 5.8 and abs(pruned['eta'][muon]) <= 1.55
        assert chain['hadronic_tau'] >= 0 and chain['muonic_tau'] >= 0 and chain['hadronic_tau'] != chain['muonic_tau']
        # tau- has pdgId +15: the hadronic tau carries the charge of its three pions
        assert np.sign(pruned['pdg'][chain['hadronic_tau']]) == -np.sum(np.sign(pruned['pdg'][pions]))
        # both taus are produced at the B decay vertex, the pions are displaced from it
        np.testing.assert_allclose(pruned['vertex'][chain['muonic_tau']], chain['b_decay_vertex'], atol=1e-5)
        assert np.linalg.norm(chain['tau_decay_vertex'] - chain['b_decay_vertex']) > 0
        assert chain['b_meson'] >= 0 and abs(pruned['pdg'][chain['b_meson']]) == 511
        assert chain['true_primary_vertex'] is not None


@miniaod_required
def test_signal_b_meson_flies_from_the_primary_vertex_to_its_decay_vertex(generator_events):
    """The event holds a second b hadron, often another B0: the signal one
    is the B0 whose momentum points from the primary vertex to the vertex
    where the taus and the K* are produced."""
    checked = several_b_mesons = 0
    for pruned, _ in generator_events:
        chain = truth.signal_chain(pruned['pdg'], pruned['status'], pruned['vertex'], pruned['eta'], pruned['phi'])
        if chain['b_meson'] < 0 or chain['b_decay_vertex'] is None:
            continue
        flight = chain['b_decay_vertex'] - chain['true_primary_vertex']
        if np.linalg.norm(flight) < 1e-3:
            continue
        checked += 1
        several_b_mesons += int(np.sum(np.abs(pruned['pdg']) == 511) > 1)
        for row in (chain['b_meson'], chain['b_meson_origin']):
            assert abs(pruned['pdg'][row]) == 511
            direction = geometry.unit_direction(pruned['eta'][[row]], pruned['phi'][[row]])[0]
            assert float(direction @ flight) / np.linalg.norm(flight) > 0.999
        assert chain['b_meson_origin'] <= chain['b_meson']
        np.testing.assert_allclose(pruned['vertex'][chain['b_meson_origin']], chain['true_primary_vertex'], atol=1e-5)
    assert checked >= 30
    assert several_b_mesons >= 3


@miniaod_required
def test_kstar_daughters_reconstruct_the_kstar(generator_events):
    found = 0
    with_kstar = 0
    for pruned, packed in generator_events:
        chain = truth.signal_chain(pruned['pdg'], pruned['status'], pruned['vertex'], pruned['eta'], pruned['phi'])
        if chain['kstar'] < 0:
            continue
        with_kstar += 1
        kstar = chain['kstar']
        assert abs(pruned['pdg'][kstar]) == 313
        daughters = truth.kstar_daughters(pruned['pt'][kstar], pruned['eta'][kstar], pruned['phi'][kstar], packed)
        if daughters is None:
            continue
        found += 1
        kaon, pion = daughters
        assert abs(packed['pdg'][kaon]) == 321 and abs(packed['pdg'][pion]) == 211
        assert np.sign(packed['pdg'][kaon]) != np.sign(packed['pdg'][pion])
        mass = geometry.pair_mass(packed['pt'][kaon], packed['eta'][kaon], packed['phi'][kaon], 0.493677,
                                  packed['pt'][pion], packed['eta'][pion], packed['phi'][pion], 0.13957)
        # the pair carries the generated (off-shell) K* mass
        assert mass == pytest.approx(pruned['mass'][kstar], abs=0.02)
    assert with_kstar >= 0.9 * len(generator_events)
    assert found >= 0.85 * with_kstar


@job0_required
def test_tight_matching_agrees_with_the_ntuple_on_its_tight_matches():
    events = uproot.open(JOB0)['Events'].arrays(
        ['Track_pt', 'Track_eta', 'Track_phi', 'Track_charge', 'GenPart_pdgId', 'GenPart_status', 'GenPart_pt',
         'GenPart_eta', 'GenPart_phi', 'GenPart_gentrackIdx', 'GenPart_gentrackMatchDr', 'GenPart_gentrackMatchDPt'],
        entry_stop=150)
    agree = total = 0
    for row in range(len(events)):
        track_pt = ak.to_numpy(events['Track_pt'][row])
        if not track_pt.size:
            continue
        pdg = ak.to_numpy(events['GenPart_pdgId'][row])
        candidates = np.flatnonzero((ak.to_numpy(events['GenPart_status'][row]) == 1) & np.isin(np.abs(pdg), (211, 321)))
        matched, delta_r = truth.match_tracks_to_generator(
            track_pt, ak.to_numpy(events['Track_eta'][row]), ak.to_numpy(events['Track_phi'][row]),
            ak.to_numpy(events['Track_charge'][row]),
            ak.to_numpy(events['GenPart_pt'][row])[candidates], ak.to_numpy(events['GenPart_eta'][row])[candidates],
            ak.to_numpy(events['GenPart_phi'][row])[candidates], np.sign(pdg[candidates]))
        assigned = matched[matched >= 0]
        assert assigned.size == np.unique(assigned).size
        assert np.all(delta_r[matched >= 0] < 0.05)
        ntuple_track = ak.to_numpy(events['GenPart_gentrackIdx'][row])[candidates]
        ntuple_delta_r = ak.to_numpy(events['GenPart_gentrackMatchDr'][row])[candidates]
        ntuple_delta_pt = ak.to_numpy(events['GenPart_gentrackMatchDPt'][row])[candidates]
        tight = (ntuple_track >= 0) & (ntuple_delta_r < 0.02) & (ntuple_delta_pt < 0.2)
        agree += int(np.sum(matched[tight] == ntuple_track[tight]))
        total += int(tight.sum())
    assert total > 1000
    assert agree / total > 0.99


@job0_required
def test_matching_respects_the_window_and_charge():
    track_pt = np.array([1.0, 1.0, 2.0])
    track_eta = np.array([0.0, 0.01, 1.0])
    track_phi = np.array([0.0, 0.0, 3.14])
    track_charge = np.array([1, -1, 1])
    # first particle: nearest same-charge track; second: nothing inside deltaR; third: pT outside the window
    matched, delta_r = truth.match_tracks_to_generator(
        track_pt, track_eta, track_phi, track_charge,
        np.array([1.05, 1.0, 5.0]), np.array([0.005, 0.5, 1.0]), np.array([0.0, 0.0, -3.14]), np.array([-1, 1, 1]))
    assert matched.tolist() == [1, -1, -1]
    assert delta_r[0] == pytest.approx(0.005)
