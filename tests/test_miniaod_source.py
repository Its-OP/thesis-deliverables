"""Tests for the MiniAOD adapter: a MiniAOD file presented as an Events
tree with the collaborator ntuple's branch names.

Real data only: MiniAOD files of abcbatch1 (CRAB subdirectory 0001). Needs
ROOT (the unsplit MiniAOD products are read through it); skipped without.
Expected rates come from the survey of the same 639 files
(reports/abcbatch_ntuple_survey_20261104.md, section 11.13)."""
from __future__ import annotations

import glob
import os
import sys

import awkward as ak
import numpy as np
import pytest

pytest.importorskip('ROOT')

SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'convert_to_parquet',
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import convert_root_to_parquet as converter  # noqa: E402
import miniaod_geometry as geometry  # noqa: E402
import miniaod_reader as reader  # noqa: E402
import miniaod_source as source  # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MINIAOD_DIR = os.path.join(
    ROOT_DIR, 'data', 'low-pt', 'miniaod_sample', 'abcbatch1_2024', 'BdToKstarTauTau',
    'crab_abcbatch1_2024_BdToKstarTauTau_20261030_161621', '261030_151701', '0001')
MINIAOD_FILES = sorted(glob.glob(os.path.join(MINIAOD_DIR, 'miniaod_10*.root')))[:14]
pytestmark = pytest.mark.skipif(len(MINIAOD_FILES) < 14, reason='MiniAOD sample files not present')
MU6_IP6_BIT = source.MUON_TRIGGER_PATHS.index('HLT_Mu6_Barrel_L1HP7_IP6')


@pytest.fixture(scope='module')
def trees():
    return [source.MiniAODTree(path) for path in MINIAOD_FILES]


@pytest.fixture(scope='module')
def events(trees):
    return ak.concatenate([tree.arrays(tree.keys()) for tree in trees if tree.num_entries])


def _flat(events, name):
    return ak.to_numpy(ak.flatten(events[name]))


def test_tree_exposes_every_branch_the_converter_reads(trees):
    keys = set(trees[0].keys())
    needed = set(converter.build_track_branch_map(converter.EXTENDED_TRACK_COLUMNS)) | {'Track_pdgId'}
    needed |= set(converter.MUON_BRANCH_MAP) | set(converter.SV_BRANCH_MAP) | set(converter.EXTENDED_EVENT_BRANCHES)
    needed |= {'PV_x', 'PV_y', 'PV_z', 'run', 'event', 'luminosityBlock'}
    for branch_map, _ in converter.OPTIONAL_BLOCKS.values():
        needed |= set(branch_map)
    assert needed <= keys, sorted(needed - keys)
    assert set(converter.resolve_optional_branches(trees[0])) == set(converter.OPTIONAL_BLOCKS)
    assert sum(tree.num_entries for tree in trees) >= 60


def test_arrays_window_is_a_slice_of_the_full_read(trees):
    tree = next(tree for tree in trees if tree.num_entries >= 3)
    full = tree.arrays(['Track_pt', 'run', 'Muon_pt'])
    window = tree.arrays(['Track_pt', 'run', 'Muon_pt'], entry_start=1, entry_stop=3)
    assert len(full) == tree.num_entries and len(window) == 2
    assert ak.to_list(window['Track_pt']) == ak.to_list(full['Track_pt'][1:3])
    assert ak.to_list(window['run']) == ak.to_list(full['run'][1:3])


def test_track_table_selection_order_and_lengths(events):
    track_branches = [name for name in events.fields if name.startswith('Track_')]
    counts = ak.num(events['Track_pt'])
    for name in track_branches:
        assert ak.all(ak.num(events[name]) == counts), name
    assert float(np.median(ak.to_numpy(counts))) > 200
    assert _flat(events, 'Track_pt').min() >= source.TRACK_MINIMUM_PT
    assert np.abs(_flat(events, 'Track_eta')).max() <= source.TRACK_MAXIMUM_ABS_ETA
    assert np.all(np.abs(_flat(events, 'Track_pdgId')) == 211)
    assert np.all(_flat(events, 'Track_charge') == np.sign(_flat(events, 'Track_pdgId')))
    assert ak.all(events['Track_pt'][:, 1:] <= events['Track_pt'][:, :-1])
    assert np.all(_flat(events, 'Track_isPacked') + _flat(events, 'Track_isLostTrk') == 1)
    lost_share = _flat(events, 'Track_isLostTrk').mean()
    assert 0.01 < lost_share < 0.2


def test_track_columns_follow_the_ntuple_formulas(events):
    dxy, significance = _flat(events, 'Track_dxy'), _flat(events, 'Track_dxyS')
    np.testing.assert_allclose(significance, dxy / np.sqrt(_flat(events, 'Track_covDxyDxy')), rtol=1e-9)
    dca = _flat(events, 'Track_DCASig')
    assert np.all((dca >= 0) | (dca == -1))
    # impact parameters with respect to PV[0]: the primary-vertex tracks sit at zero
    from_pv = _flat(events, 'Track_fromPV')
    assert set(np.unique(from_pv).tolist()) <= {0, 1, 2, 3}
    used_in_fit = from_pv == 3
    assert np.median(np.abs(_flat(events, 'Track_dz')[used_in_fit])) < 0.02
    assert np.median(np.abs(_flat(events, 'Track_dz')[from_pv == 0])) > 0.5
    above_cutoff = _flat(events, 'Track_pt') >= 0.5
    assert 0.2 < (from_pv[above_cutoff] > 0).mean() < 0.6
    for name in ('Track_covLamDxy', 'Track_covQopDxy', 'Track_isMatchedToMuon', 'Track_isMatchedToEle'):
        assert not _flat(events, name).any()
    for name in ('Track_covQopQop', 'Track_covLamLam', 'Track_covPhiPhi', 'Track_covDxyDxy', 'Track_covDszDsz',
                 'Track_ptErr'):
        assert np.all(_flat(events, name) > 0), name
    pixel, hits = _flat(events, 'Track_nValidPixelHits'), _flat(events, 'Track_nValidHits')
    assert np.all(hits >= pixel) and np.median(pixel) >= 3


def test_truth_labels_are_tight_and_one_to_one(events):
    selected = 0
    for row in range(len(events)):
        from_tau = ak.to_numpy(events['Track_trackFromTau'][row]) == 1
        from_kstar = ak.to_numpy(events['Track_trackFromKstar'][row]) == 1
        gen_index = ak.to_numpy(events['Track_trackgenIdx'][row])
        flavour = ak.to_numpy(events['Track_trackgenFlav'][row])
        delta_r = ak.to_numpy(events['Track_genMatchDr'][row])
        gen_track = ak.to_numpy(events['GenPart_gentrackIdx'][row])
        gen_pdg = ak.to_numpy(events['GenPart_pdgId'][row])
        assert from_tau.sum() <= 3 and from_kstar.sum() <= 2 and not np.any(from_tau & from_kstar)
        np.testing.assert_array_equal(ak.to_numpy(events['Track_trackFromB'][row]) == 1, from_tau | from_kstar)
        labelled = np.flatnonzero(from_tau | from_kstar)
        np.testing.assert_array_equal(np.flatnonzero(gen_index >= 0), labelled)
        np.testing.assert_array_equal(gen_track[gen_index[labelled]], labelled)
        assert np.all(np.abs(gen_pdg[gen_index[from_tau]]) == 211) and np.all(flavour[from_tau] == 15)
        assert np.all(np.isin(np.abs(gen_pdg[gen_index[from_kstar]]), (211, 321))) and np.all(flavour[from_kstar] == 313)
        assert np.all((delta_r[labelled] >= 0) & (delta_r[labelled] < 0.05)) and np.all(delta_r[~(from_tau | from_kstar)] == -1)
        track_pt = ak.to_numpy(events['Track_pt'][row])
        gen_pt = ak.to_numpy(events['GenPart_pt'][row])
        assert np.all(np.abs(track_pt[labelled] / gen_pt[gen_index[labelled]] - 1.0) < 0.5)
        np.testing.assert_allclose(ak.to_numpy(events['GenPart_gentrackMatchDPt'][row])[gen_index[labelled]],
                                   np.abs(track_pt[labelled] - gen_pt[gen_index[labelled]]) / gen_pt[gen_index[labelled]],
                                   rtol=1e-6)
        selected += int(np.sum(from_tau & (track_pt >= 0.5)) == 3)
    # survey: all three tau pions tightly matched above 0.5 GeV in 78.8 % of events
    assert 0.6 < selected / len(events) < 0.95


def test_generator_table_carries_the_signal_chain(events):
    complete = 0
    for row in range(len(events)):
        pdg = ak.to_numpy(events['GenPart_pdgId'][row])
        mother = ak.to_numpy(events['GenPart_genPartIdxMother'][row])
        status = ak.to_numpy(events['GenPart_status'][row])
        assert mother.max() < pdg.size and mother.min() >= -1
        tau_pions = np.flatnonzero((np.abs(pdg) == 211) & (status == 1) & (mother >= 0)
                                   & (np.abs(pdg[np.clip(mother, 0, None)]) == 15))
        if tau_pions.size != 3:
            continue
        complete += 1
        tau = mother[tau_pions[0]]
        assert np.all(mother[tau_pions] == tau) and abs(pdg[mother[tau]]) == 511
        kstar_daughters = np.flatnonzero((mother >= 0) & (np.abs(pdg[np.clip(mother, 0, None)]) == 313))
        if kstar_daughters.size:
            assert sorted(np.abs(pdg[kstar_daughters]).tolist()) == [211, 321]
            assert abs(pdg[mother[mother[kstar_daughters[0]]]]) == 511
    assert complete >= 0.95 * len(events)


def test_muon_block_holds_the_pf_muons_with_trigger_and_truth_flags(events):
    counts = ak.num(events['Muon_pt'])
    for name in [name for name in events.fields if name.startswith('Muon_')]:
        assert ak.all(ak.num(events[name]) == counts), name
    assert _flat(events, 'Muon_pt').min() >= source.MUON_MINIMUM_PT
    assert np.abs(_flat(events, 'Muon_eta')).max() <= source.MUON_MAXIMUM_ABS_ETA
    assert ak.all(events['Muon_pt'][:, 1:] <= events['Muon_pt'][:, :-1])
    assert np.all(_flat(events, 'Muon_softId') == 1) and np.all(_flat(events, 'Muon_isPFcand') == 1)
    assert np.all(_flat(events, 'Muon_mediumId') == -1) and np.all(_flat(events, 'Muon_pfRelIso03_all') == -1)
    anchors = ak.to_numpy(ak.sum(events['Muon_isAnchor'], axis=1))
    np.testing.assert_array_equal(anchors, (ak.to_numpy(counts) > 0).astype(int))
    with_tau_muon = ak.to_numpy(ak.sum(events['Muon_muonFromTau'], axis=1))
    assert with_tau_muon.max() <= 1
    # survey: the tau muon is a PF muon in 96 % of events
    assert with_tau_muon.mean() > 0.85
    fired_ip6 = ak.to_numpy(events['Trigger_firedIP6']) == 1
    has_triggering = ak.to_numpy(ak.any(events['Muon_isTriggering'] == 1, axis=1))
    assert np.all(fired_ip6[has_triggering])
    assert has_triggering[fired_ip6].mean() > 0.85
    mu6 = (_flat(events, 'Muon_triggerBits') >> MU6_IP6_BIT) & 1 == 1
    assert mu6.sum() >= 5 and np.median(_flat(events, 'Muon_pt')[mu6]) >= 6.0
    assert np.abs(_flat(events, 'Muon_eta')[mu6]).max() < 1.1
    triggering = _flat(events, 'Muon_isTriggering') == 1
    # ntuple trigger muons: ip3d median 0.056 cm, sip3d median 22
    assert 0.02 < np.median(_flat(events, 'Muon_ip3d')[triggering]) < 0.15
    assert 8 < np.median(_flat(events, 'Muon_sip3d')[triggering]) < 60
    np.testing.assert_allclose(_flat(events, 'Muon_dzErr') / _flat(events, 'Muon_dxyErr') > 0.5, True)


def test_trigger_bits_match_the_survey_rates(events):
    bits = ak.to_numpy(events['Trigger_bits'])
    np.testing.assert_array_equal(ak.to_numpy(events['Trigger_firedAny']) == 1, bits != 0)
    np.testing.assert_array_equal(ak.to_numpy(events['Trigger_firedIP6']) == 1, (bits & source.IP6_PATH_MASK) != 0)
    # survey: any single-muon barrel path 58.3 %, Mu6_Barrel_L1HP7_IP6 24.3 %
    assert 0.4 < (bits != 0).mean() < 0.78
    assert 0.1 < ((bits >> MU6_IP6_BIT) & 1).mean() < 0.42
    assert np.all(ak.to_numpy(events['Pileup_nTrueInt']) > 10)


def test_trigger_path_lookup_is_by_name(trees):
    names = reader.hlt_path_names(MINIAOD_FILES[0])
    lookup = source.trigger_path_lookup(names)
    assert len(lookup) == len(source.MUON_TRIGGER_PATHS) == 13
    for index, name in zip(lookup, source.MUON_TRIGGER_PATHS):
        assert names[index].rsplit('_v', 1)[0] == name
    with pytest.raises(ValueError, match='HLT_Mu6_Barrel_L1HP7_IP6'):
        source.trigger_path_lookup([name for name in names if not name.startswith('HLT_Mu6_Barrel_L1HP7_IP6')])


def test_vertex_blocks_follow_the_ntuple_conventions(events):
    n_listed = ak.to_numpy(ak.num(events['PVtx_vz']))
    n_all, n_good = ak.to_numpy(events['PV_npvs']), ak.to_numpy(events['PV_npvsGood'])
    assert np.all(n_good <= n_all) and np.all((n_listed == n_good) | (n_listed == n_good + 1))
    np.testing.assert_allclose(ak.to_numpy(events['PVtx_vz'][:, 0]), ak.to_numpy(events['PV_z']))
    np.testing.assert_allclose(ak.to_numpy(events['PVtx_score'][:, 0]), ak.to_numpy(events['PV_score']))
    assert np.all(ak.to_numpy(ak.num(events['OtherPV_z'])) == np.minimum(n_all - 1, 3))
    assert ak.all(ak.num(events['OtherPV_score']) == ak.num(events['OtherPV_z']))
    assert np.all(_flat(events, 'PVtx_ndof')[_flat(events, 'PVtx_ndof') <= 4].size <= len(events))
    assert np.all(_flat(events, 'PVtx_covZZ') > 0) and np.all(_flat(events, 'PVtx_ntracks') == 0)
    pv_index = _flat(events, 'Track_pvIdx')
    listed_per_track = np.repeat(n_listed, ak.to_numpy(ak.num(events['Track_pt'])))
    assert pv_index.min() >= -1 and np.all(pv_index < listed_per_track)
    used_in_fit = _flat(events, 'Track_fromPV') == 3
    assert np.all(pv_index[used_in_fit] == 0)
    assert np.all(ak.to_numpy(events['PV_sumpt2']) > 0)
    beam = np.stack([ak.to_numpy(events[f'BeamSpot_{axis}']) for axis in 'xyz'], axis=1)
    assert np.all(np.hypot(beam[:, 0], beam[:, 1]) < 0.5)


def test_secondary_vertices_follow_the_nanoaod_selection(events):
    counts = ak.num(events['SV_x'])
    for name in [name for name in events.fields if name.startswith('SV_')]:
        assert ak.all(ak.num(events[name]) == counts), name
    assert ak.sum(counts) > 50
    assert _flat(events, 'SV_dlenSig').min() > source.SV_MINIMUM_SIGNIFICANCE
    angle = _flat(events, 'SV_pAngle')
    # NanoAOD measures the angle from the vertex back to PV[0]: collinear flight is pi
    assert np.all((angle >= 0) & (angle <= np.pi + 1e-6)) and np.median(angle) > 3.0
    assert np.all(_flat(events, 'SV_ntracks') >= 2)
    # ntuple: median flight-distance error dlen / dlenSig of 0.03 cm
    error = _flat(events, 'SV_dlen') / _flat(events, 'SV_dlenSig')
    assert 0.01 < np.median(error) < 0.08


def test_kstar_candidates_reference_the_track_table(events):
    found = reconstructable = 0
    for row in range(len(events)):
        n_tracks = len(events['Track_pt'][row])
        first = ak.to_numpy(events['KstarToPiKAll_trk1_idx'][row])
        second = ak.to_numpy(events['KstarToPiKAll_trk2_idx'][row])
        for name in [name for name in events.fields if name.startswith('KstarToPiKAll_')]:
            assert len(events[name][row]) == first.size, name
        selected = ak.to_numpy(events['KstarToPiK_trk1_idx'][row])
        assert selected.size == min(first.size, 1)
        if not first.size:
            continue
        assert first.max() < n_tracks and second.max() < n_tracks and min(first.min(), second.min()) >= 0
        leading = int(np.argmax(ak.to_numpy(events['KstarToPiKAll_pt'][row])))
        assert selected[0] == first[leading] and events['KstarToPiK_trk2_idx'][row][0] == second[leading]
        daughters = np.flatnonzero(ak.to_numpy(events['Track_trackFromKstar'][row]) == 1)
        track_pt = ak.to_numpy(events['Track_pt'][row])
        if daughters.size != 2 or track_pt[daughters].min() < 0.5:
            continue
        reconstructable += 1
        pairs = set(zip(first.tolist(), second.tolist()))
        found += int(tuple(daughters.tolist()) in pairs or tuple(daughters[::-1].tolist()) in pairs)
    assert reconstructable >= 20
    assert found / reconstructable > 0.8


def test_anchor_muon_defines_the_track_dz_reference(events):
    for row in range(len(events)):
        anchor = np.flatnonzero(ak.to_numpy(events['Muon_isAnchor'][row]) == 1)
        offset = ak.to_numpy(events['Track_dzTrg'][row])
        if not anchor.size:
            assert not offset.any()
            continue
        np.testing.assert_allclose(offset, ak.to_numpy(events['Track_vz'][row]) - events['Muon_vz'][row][anchor[0]],
                                   atol=1e-6)
