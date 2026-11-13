"""Tests for scripts/condor/convert_to_parquet/extract_baseline_blocks.py.

Real data only: the abcbatch job0 file (303 entries) and its three local
siblings. Anchors are hand-verified facts of specific entries (survey
2026-11-04); a disagreement means re-derive from the file, never adjust."""
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

import convert_root_to_parquet as converter  # noqa: E402
import extract_baseline_blocks as extractor  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ABCBATCH_DIR = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch')
PATTERN = 'abcbatch*_job*_ditaus_mc.root'
JOB0 = os.path.join(ABCBATCH_DIR, 'abcbatch0_2024_job0_ditaus_mc.root')
FOUR_FILES = sorted(glob.glob(os.path.join(ABCBATCH_DIR, PATTERN)))
IDENTITY = ('event_run', 'event_id', 'event_luminosity_block', 'source_batch_id', 'source_microbatch_id')

job0_required = pytest.mark.skipif(not os.path.exists(JOB0), reason='abcbatch job0 file not present')
four_files_required = pytest.mark.skipif(len(FOUR_FILES) != 4, reason='the four abcbatch files are not all present')


@pytest.fixture(scope='module')
def job0_baseline_100(tmp_path_factory):
    output = tmp_path_factory.mktemp('baseline_100') / 'baseline_batch0_job0.parquet'
    assert extractor.extract_file(JOB0, str(output), entry_stop=100, chunk_size=40) == 100
    return ak.from_parquet(str(output))


@pytest.fixture(scope='module')
def job0_baseline_full(tmp_path_factory):
    output = tmp_path_factory.mktemp('baseline_full') / 'baseline_batch0_job0.parquet'
    assert extractor.extract_file(JOB0, str(output), chunk_size=100) == 303
    return ak.from_parquet(str(output))


def keys_of(rows):
    return list(zip(*(ak.to_numpy(rows[name]).tolist() for name in IDENTITY)))


@job0_required
def test_row_count_and_identity(job0_baseline_100, job0_baseline_full):
    rows = job0_baseline_100
    assert len(rows) == 100
    assert ak.to_numpy(rows['source_batch_id']).tolist() == [0] * 100
    assert ak.to_numpy(rows['source_microbatch_id']).tolist() == list(range(100))
    assert ak.to_numpy(rows['event_source_entry']).tolist() == list(range(100))
    assert ak.to_numpy(rows['event_source_job']).tolist() == [0] * 100
    assert int(rows['event_id'][1]) == 15522
    full = job0_baseline_full
    assert len(full) == 303
    triple = lambda index: (int(full['event_run'][index]), int(full['event_id'][index]),  # noqa: E731
                            int(full['event_luminosity_block'][index]))
    assert triple(96) == triple(188) == (1, 3604, 1)
    assert full['source_microbatch_id'][96] != full['source_microbatch_id'][188]
    assert len(set(keys_of(full))) == 303


@job0_required
def test_buckets_first_100(job0_baseline_100, job0_baseline_full):
    rows = job0_baseline_100
    buckets = ak.to_list(rows['event_selection_bucket'])
    assert buckets[0] == 'no_tracks' and buckets[1] == 'selected'
    assert buckets[32] == '3_gt_below_cutoff' and len(rows['gt_root_index'][32]) == 3
    assert buckets[99] == 'n_gt_2' and len(rows['gt_root_index'][99]) == 0
    assert buckets[44] == 'quality_dropped' and buckets[74] == 'quality_dropped'
    clean = ak.to_numpy(rows['event_converter_clean'])
    assert clean[44] == 0 and clean[74] == 0 and clean[1] == 1
    selected = ak.to_numpy(rows['event_is_selected'])
    assert all((bucket == 'selected') == bool(flag) for bucket, flag in zip(buckets, selected))
    assert int(ak.sum(job0_baseline_full['event_is_selected'])) == 59
    assert set(ak.to_list(job0_baseline_full['event_selection_bucket'])) <= set(extractor.FUNNEL_BUCKETS)


@job0_required
def test_entry_1_truth_and_counts(job0_baseline_100):
    row = job0_baseline_100[1]
    assert int(row['event_n_tracks_raw']) == 391
    assert int(row['event_n_tracks']) == 182
    assert ak.to_list(row['gt_root_index']) == [0, 2, 141]
    assert int(row['event_n_gt_flagged']) == 3 and int(row['event_n_gt_above_cutoff']) == 3
    assert int(row['cutflow_nBest']) == 1 and int(row['cutflow_nTrigMuons']) == 1
    assert int(row['gen_tau_index']) == 10 and int(row['gen_b_index']) == 1
    assert abs(int(row['gen_b_pdg_id'])) == 511
    assert float(row['gen_true_pv_z']) == pytest.approx(0.0551, abs=1e-4)
    assert float(row['event_primary_vertex_z']) == pytest.approx(0.0560, abs=1e-4)
    assert sorted(ak.to_list(row['gen_pion_track_root_index'])) == [0, 2, 141]
    assert len(row['gen_pion_index']) == 3 and len(row['gen_pion_pt']) == 3
    assert float(row['gen_decay_length_3d']) > 0 and float(row['gen_tau_pt']) > 0
    assert float(row['gen_tau_vertex_z']) == pytest.approx(-0.2230, abs=1e-3)


@job0_required
def test_entry_0_no_tracks(job0_baseline_100):
    row = job0_baseline_100[0]
    assert ak.to_list(row['event_selection_bucket']) == 'no_tracks'
    assert int(row['event_n_tracks_raw']) == 0 and int(row['event_n_tracks']) == 0
    cutflow_names = [name for name in job0_baseline_100.fields if name.startswith('cutflow_')]
    assert len(cutflow_names) == 11
    assert all(int(row[name]) == 0 for name in cutflow_names)
    assert len(row['gen_pion_index']) == 3 and int(row['gen_b_index']) >= 0
    assert ak.to_list(row['gen_pion_track_root_index']) == [-1, -1, -1]
    assert len(row['kstar_track1_root_index']) == 0 and len(row['abcnet_score']) == 0
    assert int(row['event_kstar_selected_row']) == -1
    assert int(row['event_kstar_selected_track1_root_index']) == -1


@job0_required
def test_entry_1_kstar_and_abcnet(job0_baseline_100):
    rows = job0_baseline_100
    row = rows[1]
    assert len(row['kstar_track1_root_index']) == 22 and int(row['event_n_kstar_all']) == 22
    selected_row = int(row['event_kstar_selected_row'])
    assert selected_row >= 0
    assert {int(row['kstar_track1_root_index'][selected_row]),
            int(row['kstar_track2_root_index'][selected_row])} == {8, 57}
    assert (int(row['event_kstar_selected_track1_root_index']),
            int(row['event_kstar_selected_track2_root_index'])) == (8, 57)
    assert int(row['event_n_kstar_selected']) == 1
    assert ak.to_list(row['kstar_is_true']) == [0] * 22
    assert len(row['abcnet_score']) == 758 and int(row['event_n_abcnet_rows']) == 758
    assert int(row['event_n_abcnet_lists']) == 21
    abcnet_track = ak.to_numpy(row['abcnet_track_root_index'])
    assert int(np.sum(abcnet_track < 0)) == 18
    assert ak.to_numpy(row['abcnet_kstar_row']).max() < 22
    raw = uproot.open(JOB0)['Events'].arrays(['ABCNetTau_score', 'ABCNetTau_dz', 'KstarToPiKAll_vtx_z'],
                                             entry_start=1, entry_stop=2)[0]
    assert np.allclose(ak.to_numpy(row['abcnet_score']), ak.to_numpy(raw['ABCNetTau_score']))
    assert np.allclose(ak.to_numpy(row['abcnet_dz']), ak.to_numpy(raw['ABCNetTau_dz']))
    assert np.allclose(ak.to_numpy(row['kstar_vertex_z']), ak.to_numpy(raw['KstarToPiKAll_vtx_z']))
    for name in ('abcnet_track_root_index', 'abcnet_kstar_row', 'kstar_track1_root_index',
                 'kstar_track2_root_index', 'kstar_charge', 'kstar_is_true'):
        assert ak.to_numpy(ak.flatten(rows[name])).dtype == np.int32, name
    for name in ('abcnet_score', 'abcnet_dz', 'abcnet_dxy', 'kstar_vertex_probability'):
        assert ak.to_numpy(ak.flatten(rows[name])).dtype == np.float32, name


@job0_required
def test_entry_5_kstar_true_row(job0_baseline_100):
    row = job0_baseline_100[5]
    assert int(row['event_id']) == 4807
    assert ak.to_list(row['gt_root_index']) == [2, 6, 120]
    assert int(row['kstar_is_true'][0]) == 1
    assert int(row['event_kstar_selected_row']) == 0


def raw_b_block(entry):
    data = uproot.open(JOB0)['Events'].arrays(
        ['nBToKstarTauTau', 'BToKstarTauTau_tau_idx', 'BToKstarTauTau_kstar_idx',
         'BToKstarTauTau_svprob', 'BToKstarTauTau_fit_mass', 'nTauTo3Pi',
         'TauTo3Pi_vprob', 'TauTo3Pi_rho_mass_best', 'TauTo3Pi_ss_pair_mass'],
        entry_start=entry, entry_stop=entry + 1)[0]
    return {name: ak.to_numpy(data[name]) if data[name].ndim else int(data[name]) for name in data.fields}


@job0_required
def test_entry_1_builder_and_b(job0_baseline_100):
    row = job0_baseline_100[1]
    assert len(row['tau_track1_index']) == 11 and int(row['event_n_tau_candidates']) == 11
    assert ak.to_list(row['tau_is_gt_triplet']) == [0] * 11
    assert int(row['event_tau_gt_row']) == -1
    raw = raw_b_block(1)
    assert int(row['event_n_b_rows']) == raw['nBToKstarTauTau'] == 153
    assert int(row['event_n_b_unique_triplets']) == 8
    assert int(row['event_n_b_gt_rows']) == 0
    n_b_rows = ak.to_numpy(row['tau_n_b_rows'])
    assert n_b_rows.tolist() == np.bincount(raw['BToKstarTauTau_tau_idx'], minlength=11).tolist()
    assert n_b_rows.sum() == 153
    empty = np.flatnonzero(n_b_rows == 0)
    assert empty.size > 0
    assert ak.to_numpy(row['tau_b_best_kstar_row'])[empty].tolist() == [-1] * empty.size
    assert ak.to_numpy(row['tau_b_best_vertex_probability'])[empty].tolist() == [-1.0] * empty.size
    assert ak.to_numpy(row['tau_b_best_fit_mass'])[empty].tolist() == [-1.0] * empty.size
    best = int(np.argmax(raw['BToKstarTauTau_svprob']))
    assert int(row['event_b_best_tau_row']) == int(raw['BToKstarTauTau_tau_idx'][best])
    assert int(row['event_b_best_kstar_row']) == int(raw['BToKstarTauTau_kstar_idx'][best])
    assert float(row['event_b_best_vertex_probability']) == pytest.approx(float(raw['BToKstarTauTau_svprob'][best]))


@job0_required
def test_entry_5_builder_and_b(job0_baseline_100):
    row = job0_baseline_100[5]
    hits = ak.to_numpy(row['tau_is_gt_triplet'])
    assert np.flatnonzero(hits).tolist() == [3]
    assert int(row['event_tau_gt_row']) == 3
    assert float(row['tau_vertex_probability'][3]) == pytest.approx(0.0895, abs=1e-3)
    assert int(row['tau_n_b_rows'][3]) == 36
    assert int(row['event_n_b_gt_rows']) == 36
    raw = raw_b_block(5)
    mask = raw['BToKstarTauTau_tau_idx'] == 3
    assert float(row['tau_b_best_vertex_probability'][3]) == pytest.approx(float(raw['BToKstarTauTau_svprob'][mask].max()))
    assert int(row['event_n_b_rows']) == raw['nBToKstarTauTau']
    assert np.allclose(ak.to_numpy(row['tau_rho_mass_best']), raw['TauTo3Pi_rho_mass_best'])
    assert np.allclose(ak.to_numpy(row['tau_same_sign_pair_mass']), raw['TauTo3Pi_ss_pair_mass'])


@job0_required
def test_entry_39_builder_and_b(job0_baseline_100):
    row = job0_baseline_100[39]
    assert ak.to_list(row['gt_root_index']) == [5, 8, 207]
    assert np.flatnonzero(ak.to_numpy(row['tau_is_gt_triplet'])).tolist() == [4]
    assert float(row['tau_vertex_probability'][4]) == pytest.approx(0.9529, abs=1e-3)
    assert int(row['tau_n_b_rows'][4]) == 29
    assert int(row['event_n_b_gt_rows']) == 29


@job0_required
def test_aggregate_b_rows_numpy():
    raw = raw_b_block(5)
    tau_idx = raw['BToKstarTauTau_tau_idx'].astype(np.int64)
    kstar_idx = raw['BToKstarTauTau_kstar_idx'].astype(np.int64)
    svprob = raw['BToKstarTauTau_svprob'].astype(np.float64)
    fit_mass = raw['BToKstarTauTau_fit_mass'].astype(np.float64)
    n_tau = raw['nTauTo3Pi']
    per_row, event = extractor.aggregate_b_rows(tau_idx, kstar_idx, svprob, fit_mass, n_tau)
    assert per_row['n_rows'].sum() == tau_idx.size == event['n_rows']
    assert event['n_unique_triplets'] == np.unique(tau_idx).size
    for tau_row in range(n_tau):
        rows = np.flatnonzero(tau_idx == tau_row)
        if rows.size == 0:
            assert per_row['best_kstar_row'][tau_row] == -1
            assert per_row['best_vertex_probability'][tau_row] == -1.0
            continue
        best = rows[np.argmax(svprob[rows])]  # argmax -> first occurrence on ties
        assert per_row['best_kstar_row'][tau_row] == kstar_idx[best]
        assert per_row['best_vertex_probability'][tau_row] == pytest.approx(svprob[best])
        assert per_row['best_fit_mass'][tau_row] == pytest.approx(fit_mass[best])
    empty_row, empty_event = extractor.aggregate_b_rows(
        np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0), 3)
    assert empty_row['n_rows'].tolist() == [0, 0, 0] and empty_event['best_tau_row'] == -1


@job0_required
def test_aggregate_b_rows_ignores_non_finite_svprob(job0_baseline_full):
    raw = raw_b_block(131)
    svprob = raw['BToKstarTauTau_svprob'].astype(np.float64)
    assert np.isnan(svprob).any(), 'entry 131 is the NaN-svprob anchor'
    tau_idx = raw['BToKstarTauTau_tau_idx'].astype(np.int64)
    kstar_idx = raw['BToKstarTauTau_kstar_idx'].astype(np.int64)
    fit_mass = raw['BToKstarTauTau_fit_mass'].astype(np.float64)
    per_row, event = extractor.aggregate_b_rows(tau_idx, kstar_idx, svprob, fit_mass, raw['nTauTo3Pi'])
    best = int(np.nanargmax(svprob))
    assert event['best_vertex_probability'] == pytest.approx(float(np.nanmax(svprob)))
    assert event['best_tau_row'] == int(tau_idx[best]) and event['best_kstar_row'] == int(kstar_idx[best])
    assert per_row['n_rows'].sum() == tau_idx.size
    for tau_row in range(raw['nTauTo3Pi']):
        rows = np.flatnonzero((tau_idx == tau_row) & np.isfinite(svprob))
        if rows.size == 0:
            assert per_row['best_vertex_probability'][tau_row] == -1.0
            continue
        assert np.isfinite(per_row['best_vertex_probability'][tau_row])
        assert per_row['best_vertex_probability'][tau_row] == pytest.approx(float(svprob[rows].max()))
    written = job0_baseline_full[131]
    assert np.isfinite(float(written['event_b_best_vertex_probability']))
    assert np.all(np.isfinite(ak.to_numpy(job0_baseline_full['event_b_best_vertex_probability'])))


@job0_required
def test_run_rejects_duplicate_batch_job_names(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    for name in ('abcbatch0_2024_job0_ditaus_mc.root', 'abcbatch0_2024_job0_ditaus_mc_1.root'):
        os.symlink(os.path.abspath(JOB0), source_dir / name)
    with pytest.raises(ValueError, match='job'):
        extractor.run(str(source_dir), 'abcbatch*_job*.root', str(tmp_path / 'out'), entry_stop=5)
    assert not os.path.exists(tmp_path / 'out')


@job0_required
def test_extract_file_writes_atomically(tmp_path):
    output = tmp_path / 'baseline_batch0_job0.parquet'
    stale = tmp_path / 'baseline_batch0_job0.parquet.tmp'
    stale.write_bytes(b'truncated by a killed job')
    assert extractor.extract_file(JOB0, str(output), entry_stop=20, chunk_size=10) == 20
    assert output.exists() and not stale.exists()
    assert sorted(os.listdir(tmp_path)) == ['baseline_batch0_job0.parquet']
    assert len(ak.from_parquet(str(output))) == 20


def test_output_name_for():
    assert extractor.output_name_for('/x/abcbatch0_2024_job305_ditaus_mc.root') == 'baseline_batch0_job305.parquet'
    assert extractor.output_name_for('/x/abcbatch7_2024_job4_ditaus_mc.root') == 'baseline_batch7_job4.parquet'
    assert extractor.output_name_for('/x/nan_covariance_fixture.root') == 'baseline_nan_covariance_fixture.parquet'


@pytest.fixture(scope='module')
def four_file_baseline(tmp_path_factory):
    if len(FOUR_FILES) != 4:
        pytest.skip('the four abcbatch files are not all present')
    output_dir = tmp_path_factory.mktemp('baseline_four')
    written = extractor.run(ABCBATCH_DIR, PATTERN, str(output_dir))
    return output_dir, written


@four_files_required
def test_four_files_counts(four_file_baseline):
    output_dir, written = four_file_baseline
    names = sorted(os.listdir(output_dir))
    assert names == ['baseline_batch0_job0.parquet', 'baseline_batch0_job305.parquet',
                     'baseline_batch0_job58.parquet', 'baseline_batch0_job67.parquet']
    assert [rows for _, rows in written] == [303, 351, 326, 326]
    rows = ak.concatenate([ak.from_parquet(os.path.join(output_dir, name)) for name in names])
    assert len(rows) == 1306
    assert int(ak.sum(rows['event_is_selected'])) == 277
    assert len(set(keys_of(rows))) == 1306
    assert set(ak.to_numpy(rows['event_source_job']).tolist()) == {0, 58, 67, 305}


@four_files_required
def test_join_with_model_parquet(four_file_baseline, tmp_path_factory):
    output_dir, _ = four_file_baseline
    baseline = ak.concatenate([ak.from_parquet(os.path.join(output_dir, name))
                               for name in sorted(os.listdir(output_dir))])
    model_dir = tmp_path_factory.mktemp('model_four')
    converter.convert(input_dir=ABCBATCH_DIR, output_dir=str(model_dir), train_events=0, val_events=0,
                      train_files=1, val_files=1, pt_cutoff=0.5, required_gt_pions=3, chunk_size=200,
                      output_columns=None, extended=True, pattern=PATTERN, val_subdir='model',
                      eval_only=True, shard_events=5000)
    model = ak.from_parquet(os.path.join(model_dir, 'model', 'model_000.parquet'))
    assert len(model) == 277
    baseline_index = {key: position for position, key in enumerate(keys_of(baseline))}
    assert len(baseline_index) == len(baseline)
    for position, key in enumerate(keys_of(model)):
        baseline_row = baseline[baseline_index[key]]
        assert ak.to_list(baseline_row['event_selection_bucket']) == 'selected'
        assert int(baseline_row['event_n_tracks']) == int(model['event_n_tracks'][position])
        labels = ak.to_numpy(model['track_label_from_tau'][position])
        gt_from_model = ak.to_numpy(model['track_root_index'][position])[labels == 1].tolist()
        assert ak.to_list(baseline_row['gt_root_index']) == gt_from_model


@four_files_required
def test_stripes_partition_files_and_skip(tmp_path):
    first = extractor.run(ABCBATCH_DIR, PATTERN, str(tmp_path), entry_stop=5, stripe=0, num_stripes=2)
    second = extractor.run(ABCBATCH_DIR, PATTERN, str(tmp_path), entry_stop=5, stripe=1, num_stripes=2)
    assert [os.path.basename(path) for path, _ in first] == ['baseline_batch0_job0.parquet', 'baseline_batch0_job67.parquet']
    assert [os.path.basename(path) for path, _ in second] == ['baseline_batch0_job58.parquet', 'baseline_batch0_job305.parquet']
    assert all(rows == 5 for _, rows in first + second)
    again = extractor.run(ABCBATCH_DIR, PATTERN, str(tmp_path), entry_stop=5, stripe=0, num_stripes=2)
    assert [rows for _, rows in again] == [None, None]
    rewritten = extractor.run(ABCBATCH_DIR, PATTERN, str(tmp_path), entry_stop=5, stripe=0, num_stripes=2,
                              overwrite=True)
    assert [rows for _, rows in rewritten] == [5, 5]


@job0_required
def test_validate_baseline_file(tmp_path):
    output = tmp_path / 'baseline_batch0_job0.parquet'
    extractor.extract_file(JOB0, str(output), entry_stop=50, chunk_size=20)
    extractor.validate_baseline_file(str(output), expected_rows=50)
    with pytest.raises(ValueError, match='rows'):
        extractor.validate_baseline_file(str(output), expected_rows=51)
    rows = ak.from_parquet(str(output))
    broken = ak.Array({**{name: rows[name] for name in rows.fields if name != 'gt_root_index'},
                       'gt_root_index': rows['gt_root_index'][:, :2]})
    broken_path = tmp_path / 'broken.parquet'
    ak.to_parquet(broken, str(broken_path))
    with pytest.raises(ValueError, match='gt_root_index'):
        extractor.validate_baseline_file(str(broken_path), expected_rows=50)
