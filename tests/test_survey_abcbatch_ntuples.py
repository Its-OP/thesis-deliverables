"""Tests for scripts/python/survey_abcbatch_ntuples.py.

Fixture = the real job0 ntuple of the abcbatch0_2024 production (303 events,
symlinked under data/low-pt/abcbatch/). Anchors are hand-verified facts of
specific entries; a disagreement means re-derive from the file, never adjust.
"""
import os
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch',
                       'abcbatch0_2024_job0_ditaus_mc.root')
REFERENCE = os.path.join(ROOT, 'data', 'thesis_files',
                         'step_MINI_10_nano_ditaus_mc.root')

data_required = pytest.mark.skipif(not os.path.exists(FIXTURE),
                                   reason='abcbatch job0 ntuple not present')
reference_required = pytest.mark.skipif(
    not os.path.exists(REFERENCE), reason='reference raw nano not present')

from scripts.python import survey_abcbatch_ntuples as survey  # noqa: E402


@pytest.fixture(scope='module')
def events():
    return survey.load_events(FIXTURE, entry_stop=100)


def gen_arrays(events, entry):
    return (ak.to_numpy(events['GenPart_pdgId'][entry]),
            ak.to_numpy(events['GenPart_genPartIdxMother'][entry]),
            ak.to_numpy(events['GenPart_status'][entry]))


def test_wilson_interval():
    low, high = survey.wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.237, 0.763), abs=1e-3)
    assert survey.wilson_interval(0, 10)[0] == 0.0
    assert survey.wilson_interval(10, 10)[1] == 1.0
    assert survey.wilson_interval(0, 0) == (0.0, 0.0)


@data_required
@reference_required
class TestInventory:
    def test_classify_branches_partitions_all_branches(self):
        branch_names = set(uproot.open(FIXTURE)['Events'].keys())
        reference_names = set(uproot.open(REFERENCE)['Events'].keys())
        classes = survey.classify_branches(
            branch_names, reference_names, survey.merge_kept_branches(),
            survey.converter_branches())
        assert set(classes) == branch_names
        assert set(classes.values()) <= {'converter', 'merge_kept',
                                         'reference_not_kept', 'new'}
        assert classes['Track_pt'] == 'converter'
        assert classes['Track_covQopQop'] == 'converter'
        assert classes['Track_trackgenIdx'] == 'reference_not_kept'
        assert classes['ABCNetTau_score'] == 'new'
        assert classes['KstarCutflow_nBest'] == 'new'

    def test_inventory_counts_new_families(self):
        families = survey.inventory(uproot.open(FIXTURE)['Events'], REFERENCE)
        assert families['ABCNetTau']['n'] == 11
        assert set(families['ABCNetTau']['classes']) == {'new'}
        assert families['BToKstarTauTau']['n'] == 57
        assert 'nTrack' in families['Track']['classes']['reference_not_kept'] \
            or 'nTrack' in families['Track']['classes']['merge_kept']
        assert families['ABCNetTau']['new_dtypes']['ABCNetTau_score'] == 'float[]'


@data_required
class TestGenWalks:
    def test_walk_to_ancestor_entry_1(self, events):
        gen_pdg, gen_mother, _ = gen_arrays(events, 1)
        for start in (11, 12, 13):
            assert survey.walk_to_ancestor(gen_pdg, gen_mother, start, {15}) == 10
        assert survey.walk_to_ancestor(gen_pdg, gen_mother, 10, {15}) == 10
        assert survey.walk_to_ancestor(gen_pdg, gen_mother, 150, {15}) == -1
        assert survey.walk_to_ancestor(gen_pdg, gen_mother, -1, {15}) == -1

    def test_top_b_ancestor_entry_1(self, events):
        gen_pdg, gen_mother, _ = gen_arrays(events, 1)
        assert survey.top_b_ancestor(gen_pdg, gen_mother, 10) == 1
        assert abs(int(gen_pdg[1])) == 511
        assert ak.to_numpy(events['GenPart_vz'][1])[1] == pytest.approx(0.0551, abs=1e-3)

    def test_gen_tau_pions_and_reco_entries_0_and_1(self, events):
        pions_0 = survey.gen_tau_pions(*gen_arrays(events, 0))
        assert len(pions_0) == 3
        assert np.all(ak.to_numpy(events['GenPart_gentrackIdx'][0])[pions_0] < 0)
        pions_1 = survey.gen_tau_pions(*gen_arrays(events, 1))
        assert len(pions_1) == 3
        matched = ak.to_numpy(events['GenPart_gentrackIdx'][1])[pions_1]
        assert sorted(int(index) for index in matched) == [0, 2, 141]


def track_arrays(events, entry):
    return (ak.to_numpy(events['Track_pt'][entry]),
            ak.to_numpy(events['Track_pdgId'][entry]),
            ak.to_numpy(events['Track_trackFromTau'][entry]))


def abcnet_arrays(events, entry):
    return (ak.to_numpy(events['ABCNetTau_trk_idx'][entry]),
            ak.to_numpy(events['ABCNetTau_score'][entry]),
            ak.to_numpy(events['ABCNetTau_kstar_idx'][entry]))


def kstar_rows_for(events, entry):
    selected_trk1 = ak.to_numpy(events['KstarToPiK_trk1_idx'][entry])
    selected_trk2 = ak.to_numpy(events['KstarToPiK_trk2_idx'][entry])
    return survey.kstar_rows(
        ak.to_numpy(events['KstarToPiKAll_trk1_idx'][entry]),
        ak.to_numpy(events['KstarToPiKAll_trk2_idx'][entry]),
        ak.to_numpy(events['Track_trackFromKstar'][entry]),
        int(selected_trk1[0]) if selected_trk1.size else None,
        int(selected_trk2[0]) if selected_trk2.size else None)


@data_required
class TestSelectionAndLabels:
    def test_converter_clean_mask_drops_the_dirty_job0_events(self):
        mask = survey.converter_clean_mask(FIXTURE)
        assert mask.shape == (303,) and mask.dtype == bool
        assert np.flatnonzero(~mask).tolist() == [44, 74]
        assert not survey.converter_clean_mask(FIXTURE, entry_stop=50)[44]

    def test_event_record_quality_dropped(self, events):
        dropped = survey.event_record(events, 44, converter_clean=False)
        assert dropped['bucket'] == 'quality_dropped' and dropped['selected'] is False
        assert dropped['gt_indices'] is None
        assert dropped['labels'] is None and dropped['abcnet'] is None
        assert dropped['n_tracks'] > 0 and dropped['pv'] is not None
        kept = survey.event_record(events, 44)
        assert kept['bucket'] == 'selected'
        summary = survey.summarize([dropped, kept])
        assert summary['funnel']['buckets']['quality_dropped'] == 1
        assert summary['funnel']['n_track_events_raw'] == 2
        assert summary['funnel']['n_track_events'] == 1
        assert summary['pv']['track_events']['n'] == 1

    def test_select_event_buckets_first_100(self, events):
        buckets = [survey.select_event(*track_arrays(events, entry))[0]
                   for entry in range(100)]
        assert buckets[0] == 'no_tracks'
        assert buckets[1] == 'selected'
        assert buckets[32] == '3_gt_below_cutoff'
        assert buckets[99] == 'n_gt_2'
        assert set(buckets) <= set(survey.FUNNEL_BUCKETS)
        assert sum(buckets.count(name) for name in survey.FUNNEL_BUCKETS) == 100

    def test_select_event_gt_indices_entry_1_and_32(self, events):
        _, gt_indices = survey.select_event(*track_arrays(events, 1))
        assert gt_indices.tolist() == [0, 2, 141]
        _, below = survey.select_event(*track_arrays(events, 32))
        assert below is not None and len(below) == 3
        assert sorted(ak.to_numpy(events['Track_pt'][32])[below].tolist()) \
            == pytest.approx([0.496, 1.67, 3.41], abs=0.01)

    def test_select_event_counts_gt_after_the_pt_cut_like_the_converter(self):
        """The converter drops sub-cutoff tracks first and then requires
        exactly three GT pions: a four-GT event whose fourth GT pion is below
        the cutoff is kept. job58 holds exactly one such event."""
        job58 = os.path.join(os.path.dirname(FIXTURE), 'abcbatch0_2024_job58_ditaus_mc.root')
        if not os.path.exists(job58):
            pytest.skip('abcbatch job58 ntuple not present')
        arrays = uproot.open(job58)['Events'].arrays(
            ['Track_pt', 'Track_pdgId', 'Track_trackFromTau'])
        flagged = arrays['Track_trackFromTau'] == 1
        four_gt = ak.to_numpy(ak.sum(flagged, axis=1) == 4)
        three_above = ak.to_numpy(ak.sum(flagged & (arrays['Track_pt'] >= 0.5), axis=1) == 3)
        entries = np.flatnonzero(four_gt & three_above)
        assert entries.tolist() and len(entries) == 1
        entry = int(entries[0])
        bucket, gt_indices = survey.select_event(
            ak.to_numpy(arrays['Track_pt'][entry]), ak.to_numpy(arrays['Track_pdgId'][entry]),
            ak.to_numpy(arrays['Track_trackFromTau'][entry]))
        assert bucket == 'selected'
        assert len(gt_indices) == 3
        assert np.all(ak.to_numpy(arrays['Track_pt'][entry])[gt_indices] >= 0.5)

    def test_label_consistency_entry_1(self, events):
        gen_pdg, gen_mother, _ = gen_arrays(events, 1)
        result = survey.label_consistency(
            ak.to_numpy(events['Track_trackgenIdx'][1]),
            ak.to_numpy(events['Track_trackFromTau'][1]), gen_pdg, gen_mother)
        assert result == {'gt_walks': 3, 'gt_no_walk': 0, 'walk_not_gt': 0}


@data_required
class TestKstarAndAbcnet:
    def test_kstar_rows_entries_1_and_5(self, events):
        rows_1 = kstar_rows_for(events, 1)
        assert rows_1['true_rows'] == []
        selected = rows_1['selected_row']
        assert ak.to_numpy(events['KstarToPiKAll_trk1_idx'][1])[selected] == 8
        assert ak.to_numpy(events['KstarToPiKAll_trk2_idx'][1])[selected] == 57
        assert rows_1['selected_is_true'] is False
        rows_5 = kstar_rows_for(events, 5)
        assert rows_5['true_rows'] == [0]
        assert rows_5['selected_row'] == 0
        assert rows_5['selected_is_true'] is True

    def test_kstar_rows_without_selected_kstar_entry_0(self, events):
        rows_0 = kstar_rows_for(events, 0)
        assert rows_0 == {'true_rows': [], 'selected_row': None,
                          'selected_is_true': None}

    def test_abcnet_lists_entry_1(self, events):
        trk_idx, _, kstar_idx = abcnet_arrays(events, 1)
        structure = survey.abcnet_lists(trk_idx, kstar_idx)
        assert structure['n_rows'] == 758
        assert structure['n_lists'] == 21
        assert int(kstar_idx.max()) == 21
        assert int(events['nKstarToPiKAll'][1]) == 22
        assert max(structure['rows_per_list']) == 39
        assert structure['n_missing_track'] == 18

    def test_abcnet_rows_address_track_table_first_100(self, events):
        check = survey.abcnet_index_check(events)
        assert check['max_abs_delta_pt'] == 0.0
        assert check['max_abs_delta_eta'] == 0.0
        assert check['all_kstar_idx_in_range'] is True
        assert check['n_rows'] > 0

    def test_list_ranks_entries_39_and_5(self, events):
        gt_39 = survey.select_event(*track_arrays(events, 39))[1]
        assert gt_39.tolist() == [5, 8, 207]
        true_row_39 = kstar_rows_for(events, 39)['true_rows'][0]
        assert true_row_39 == 2
        ranks_39 = survey.list_ranks(*abcnet_arrays(events, 39), true_row_39, gt_39)
        assert ranks_39['ranks'].tolist() == [24, 25, 20]
        assert ranks_39['list_size'] == 40
        assert ranks_39['n_above_wp'] == 6
        assert np.all(ranks_39['gt_scores'] < 1e-3)
        assert survey.hit_at_k(ranks_39['ranks'], 50) is True
        assert survey.hit_at_k(ranks_39['ranks'], 20) is False
        gt_5 = survey.select_event(*track_arrays(events, 5))[1]
        ranks_5 = survey.list_ranks(*abcnet_arrays(events, 5), 0, gt_5)
        assert ranks_5['ranks'].tolist() == [-1, -1, 39]
        assert ranks_5['list_size'] == 49
        assert survey.hit_at_k(ranks_5['ranks'], 50) is False


def builder_arrays(events, entry):
    return tuple(ak.to_numpy(events[f'TauTo3Pi_idx{member}'][entry])
                 for member in (1, 2, 3))


def b_arrays(events, entry):
    names = ('BToKstarTauTau_pi1_idx', 'BToKstarTauTau_pi2_idx',
             'BToKstarTauTau_pi3_idx', 'BToKstarTauTau_tau_idx',
             'BToKstarTauTau_kstar_idx')
    return tuple(ak.to_numpy(events[name][entry]) for name in names)


@data_required
class TestBuilders:
    @pytest.mark.parametrize('entry, expected_rows, expected_vprob',
                             [(5, [3], 0.0895), (39, [4], 0.9529), (1, [], None)])
    def test_sorted_triplet_hit(self, events, entry, expected_rows, expected_vprob):
        gt_triplet = survey.select_event(*track_arrays(events, entry))[1]
        idx1, idx2, idx3 = builder_arrays(events, entry)
        hits = survey.sorted_triplet_hit(idx1, idx2, idx3, gt_triplet)
        assert hits.dtype == bool and hits.size == idx1.size
        assert np.flatnonzero(hits).tolist() == expected_rows
        if expected_rows:
            vprob = ak.to_numpy(events['TauTo3Pi_vprob'][entry])
            assert vprob[expected_rows[0]] == pytest.approx(expected_vprob, abs=1e-3)
        permuted = survey.sorted_triplet_hit(idx1, idx2, idx3, gt_triplet[::-1])
        assert np.array_equal(permuted, hits)

    def test_builder_candidates_entry_5(self, events):
        gt_triplet = survey.select_event(*track_arrays(events, 5))[1]
        result = survey.builder_candidates(*builder_arrays(events, 5), gt_triplet)
        assert result['n_rows'] == int(events['nTauTo3Pi'][5])
        assert result['hit_rows'] == [3]
        assert 3 <= result['pool_size'] <= 50
        assert result['stored_sorted'] is True

    def test_withmatch_identical_and_mcmatch_zero_first_100(self, events):
        check = survey.withmatch_check(events)
        assert check['all_counts_equal'] is True
        assert check['all_indices_equal'] is True
        assert check['mcmatch_values'] == [0]
        assert check['n_rows'] > 0

    @pytest.mark.parametrize('entry, expected_hit_rows', [(1, 0), (5, 36), (39, 29)])
    def test_b_candidates(self, events, entry, expected_hit_rows):
        gt_triplet = survey.select_event(*track_arrays(events, entry))[1]
        result = survey.b_candidates(*b_arrays(events, entry), *builder_arrays(events, entry),
                                     int(events['nKstarToPiKAll'][entry]), gt_triplet)
        assert result['n_rows'] == int(events['nBToKstarTauTau'][entry])
        assert result['n_hit_rows'] == expected_hit_rows
        assert result['index_consistent'] is True
        assert result['kstar_in_range'] is True
        if entry == 1:
            assert result['n_rows'] == 153
            assert result['n_unique_triplets'] == 8


def truth_for(events, entry):
    gen_pdg, gen_mother, gen_status = gen_arrays(events, entry)
    gen_vertex = np.stack([ak.to_numpy(events[f'GenPart_v{axis}'][entry])
                           for axis in 'xyz'], axis=1)
    return survey.gen_truth(gen_pdg, gen_mother, gen_status,
                            ak.to_numpy(events['GenPart_pt'][entry]), gen_vertex)


def pv_arrays(events, entry):
    pv_xyz = np.array([float(events[f'PV_{axis}'][entry]) for axis in 'xyz'])
    pvtx_xyz = np.stack([ak.to_numpy(events[f'PVtx_v{axis}'][entry])
                         for axis in 'xyz'], axis=1)
    return pv_xyz, pvtx_xyz, ak.to_numpy(events['PVtx_score'][entry])


@data_required
class TestTruth:
    def test_gen_truth_entry_1(self, events):
        truth = truth_for(events, 1)
        assert truth['tau_index'] == 10
        assert truth['b_index'] == 1
        assert sorted(truth['pion_indices'].tolist()) == [11, 12, 13]
        assert truth['tau_vertex'][2] == pytest.approx(-0.2230, abs=1e-3)
        assert truth['true_pv'][2] == pytest.approx(0.0551, abs=1e-3)
        assert truth['decay_length_3d'] > 0.0
        assert truth['decay_length_xy'] > 0.0
        assert truth['tau_pt'] > 0.0

    def test_gen_truth_entry_0_has_pions_but_no_tracks(self, events):
        truth = truth_for(events, 0)
        assert len(truth['pion_indices']) == 3
        assert truth['b_index'] >= 0

    def test_pv_truth_entries_1_and_55(self, events):
        result_1 = survey.pv_truth(*pv_arrays(events, 1), truth_for(events, 1)['true_pv'])
        assert result_1['stored_distance_3d'] == pytest.approx(0.0015, abs=1e-4)
        assert result_1['nearest_index'] == 0
        assert result_1['nearest_distance'] <= result_1['stored_distance_3d'] + 1e-9
        result_55 = survey.pv_truth(*pv_arrays(events, 55), truth_for(events, 55)['true_pv'])
        assert result_55['stored_distance_3d'] == pytest.approx(5.3166, abs=1e-3)
        assert result_55['nearest_index'] == 2
        assert result_55['nearest_distance'] == pytest.approx(0.0029, abs=1e-3)
        assert result_55['nearest_score_rank'] == 2
        assert abs(result_55['stored_dz']) > 1.0

    @pytest.mark.parametrize('entry, expected', [
        (1, {'kstar_daughter': 1, 'matched_to_muon': 0, 'b_chain': 0, 'other_gen': 0, 'unmatched': 2}),
        (5, {'kstar_daughter': 2, 'matched_to_muon': 0, 'b_chain': 0, 'other_gen': 3, 'unmatched': 2}),
        (39, {'kstar_daughter': 2, 'matched_to_muon': 0, 'b_chain': 0, 'other_gen': 3, 'unmatched': 0}),
    ])
    def test_impostor_taxonomy_wide_window(self, events, entry, expected):
        gt_triplet = survey.select_event(*track_arrays(events, entry))[1]
        event = survey.event_tracks(events, entry)
        muon_gen_idx = ak.to_numpy(events['Muon_muonGenIdx'][entry])
        wide = survey.impostor_taxonomy(event, gt_triplet, muon_gen_idx,
                                        survey.IMPOSTOR_WINDOWS['wide'])
        assert wide['categories'] == expected
        assert wide['n_window'] == sum(expected.values())
        tight = survey.impostor_taxonomy(event, gt_triplet, muon_gen_idx,
                                         survey.IMPOSTOR_WINDOWS['tight'])
        assert tight['n_window'] <= wide['n_window']

    @pytest.mark.parametrize('entry', [1, 5, 16, 39])
    def test_proxy_muon_truth_picks_the_truth_muon(self, events, entry):
        event = survey.event_tracks(events, entry)
        result = survey.proxy_muon_truth(
            event, ak.to_numpy(events['Muon_muonFromTau'][entry]))
        assert result['picked'] == 0
        assert result['picked_is_tau'] is True
        assert result['n_tau_muons'] == 1
        assert result['n_other_muons'] == 0
        assert all(result['tau_cut_flags'].values())
        if entry == 1:
            assert result['tau_relations']['dz_gap'] == pytest.approx(0.0461, abs=1e-3)
            assert result['tau_relations']['mass'] == pytest.approx(2.0769, abs=1e-3)
        if entry == 5:
            assert result['tau_relations']['dz_gap'] == pytest.approx(0.1433, abs=1e-3)

    def test_four_gt_selected_event_uses_only_the_three_selected_gt(self):
        job58 = os.path.join(os.path.dirname(FIXTURE), 'abcbatch0_2024_job58_ditaus_mc.root')
        if not os.path.exists(job58):
            pytest.skip('abcbatch job58 ntuple not present')
        job58_events = survey.load_events(job58)
        flagged = job58_events['Track_trackFromTau'] == 1
        entries = np.flatnonzero(ak.to_numpy(
            (ak.sum(flagged, axis=1) == 4)
            & (ak.sum(flagged & (job58_events['Track_pt'] >= 0.5), axis=1) == 3)))
        entry = int(entries[0])
        record = survey.event_record(job58_events, entry)
        assert record['selected'] is True
        event = survey.event_tracks(job58_events, entry, record['gt_indices'])
        assert int(event['track_label'].sum()) == 3
        summary = survey.summarize([record])
        assert summary['funnel']['n_selected'] == 1
        assert summary['muon']['events_with_truth_muon']['total'] == 1

    def test_proxy_muon_truth_without_muons_entry_0(self, events):
        result = survey.proxy_muon_truth(
            survey.event_tracks(events, 0), ak.to_numpy(events['Muon_muonFromTau'][0]))
        assert result['picked'] == -1
        assert result['n_tau_muons'] == 0

    def test_kstar_cutflow_entries_0_and_1(self, events):
        assert set(survey.kstar_cutflow(events, 0).values()) == {0}
        cutflow_1 = survey.kstar_cutflow(events, 1)
        assert cutflow_1['nBest'] == 1
        assert cutflow_1['nTrigMuons'] == 1


@data_required
class TestSummary:
    def test_event_record_entry_1(self, events):
        record = survey.event_record(events, 1)
        assert record['bucket'] == 'selected' and record['selected'] is True
        assert record['gt_indices'] == [0, 2, 141]
        assert record['n_tracks'] == 391
        assert record['builder']['hit'] is False
        assert record['b']['n_hit_rows'] == 0
        assert record['kstar']['selected_is_true'] is False
        assert record['abcnet']['lists']['true']['exists'] is False
        assert record['abcnet']['lists']['selected']['exists'] is True
        assert record['abcnet']['gt_in_any_list'] == [True, True, False]
        assert record['pv']['stored_distance_3d'] == pytest.approx(0.0015, abs=1e-4)
        assert record['impostors']['wide']['n_window'] == 3
        assert record['muon']['picked'] == 0
        assert record['truth']['b_index'] == 1

    def test_event_record_entry_0_without_tracks(self, events):
        record = survey.event_record(events, 0)
        assert record['bucket'] == 'no_tracks' and record['selected'] is False
        assert record['gt_indices'] is None
        assert record['abcnet'] is None and record['builder'] is None
        assert record['pv'] is not None
        assert record['truth']['n_gen_pions'] == 3
        assert record['truth']['n_gen_pions_reco'] == 0

    def test_summarize_first_100_is_consistent_and_serializable(self, events):
        import json
        records = [survey.event_record(events, entry) for entry in range(100)]
        summary = survey.summarize(records)
        assert summary['n_events'] == 100
        assert sum(summary['funnel']['buckets'].values()) == 100
        n_selected = sum(record['selected'] for record in records)
        assert summary['funnel']['n_selected'] == n_selected
        assert summary['builder']['hit']['total'] == n_selected
        assert 0.0 <= summary['builder']['hit']['rate'] <= 1.0
        for list_type in ('true', 'selected'):
            recall = summary['abcnet']['recall'][list_type]
            assert recall['conditional']['50']['total'] <= n_selected
            assert recall['unconditional']['50']['total'] == n_selected
        assert 0.0 <= summary['abcnet']['auc']['all_rows'] <= 1.0
        json.dumps(summary)

    def test_run_writes_outputs(self, tmp_path):
        output_json = tmp_path / 'survey.json'
        output_markdown = tmp_path / 'survey.md'
        summary = survey.run(FIXTURE, str(output_json), str(output_markdown),
                             REFERENCE if os.path.exists(REFERENCE) else None)
        assert output_json.exists() and output_markdown.exists()
        assert list(summary['per_file']) == ['abcbatch0_2024_job0_ditaus_mc.root']
        pooled = summary['pooled']
        assert pooled['n_events'] == 303
        assert pooled['funnel']['buckets']['quality_dropped'] == 2
        assert pooled['funnel']['n_selected'] == 59
        assert pooled['builder']['hit']['successes'] == 39
        assert summary['withmatch']['abcbatch0_2024_job0_ditaus_mc.root']['mcmatch_values'] == [0]
        text = output_markdown.read_text()
        assert '## Funnel' in text and '## ABCNet' in text and '## Builder' in text
        assert text.count('\n| selected |') == 1
        assert '### Builder hit by tercile (pooled)' in text


class TestCondorTruthModule:
    def test_helpers_are_the_condor_module(self):
        if survey.CONVERTER_DIR not in sys.path:
            sys.path.insert(0, survey.CONVERTER_DIR)
        import ntuple_truth
        assert survey.select_event is ntuple_truth.select_event
        assert survey.gen_truth is ntuple_truth.gen_truth
        assert survey.kstar_rows is ntuple_truth.kstar_rows
        assert survey.sorted_triplet_hit is ntuple_truth.sorted_triplet_hit
        assert survey.abcnet_lists is ntuple_truth.abcnet_lists
        assert survey.walk_to_ancestor is ntuple_truth.walk_to_ancestor
        assert survey.FUNNEL_BUCKETS == ntuple_truth.FUNNEL_BUCKETS
        assert survey.PT_CUTOFF == ntuple_truth.PT_CUTOFF == 0.5

    @data_required
    def test_select_event_pt_cutoff_keyword_entry_32(self, events):
        bucket, _ = survey.select_event(*track_arrays(events, 32))
        assert bucket == '3_gt_below_cutoff'
        bucket, gt_indices = survey.select_event(*track_arrays(events, 32), pt_cutoff=0.45)
        assert bucket == 'selected'
        assert gt_indices.size == 3
