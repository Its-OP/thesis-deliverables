"""Tests for the abcbatch extensions of the ROOT->parquet converter: job-aware
file ordering, identity injection from the file name, the raw-track index,
the optional truth/vertex/K* blocks and the eval-only shard mode.

Real data only: the abcbatch job0 file (303 entries) and its three local
siblings, plus the two old-production fixtures (no source-id branches /
merged with source ids) for the regression checks."""
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ABCBATCH_DIR = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch')
PATTERN = 'abcbatch*_job*_ditaus_mc.root'
JOB0 = os.path.join(ABCBATCH_DIR, 'abcbatch0_2024_job0_ditaus_mc.root')
FOUR_FILES = sorted(glob.glob(os.path.join(ABCBATCH_DIR, PATTERN)))
NAN_FIXTURE = os.path.join(ROOT, '..', 'part', 'data', 'low-pt', 'nan_covariance_fixture.root')
EXAMPLE_FIXTURE = os.path.join(ROOT, '..', 'part', 'data', 'low-pt', 'example_root.root')

job0_required = pytest.mark.skipif(not os.path.exists(JOB0), reason='abcbatch job0 file not present')
four_files_required = pytest.mark.skipif(len(FOUR_FILES) != 4, reason='the four abcbatch files are not all present')
nan_fixture_required = pytest.mark.skipif(not os.path.exists(NAN_FIXTURE), reason='nan_covariance_fixture.root not present')
example_fixture_required = pytest.mark.skipif(not os.path.exists(EXAMPLE_FIXTURE), reason='example_root.root not present')


@job0_required
def test_discover_root_files_orders_by_job():
    files = converter.discover_root_files(ABCBATCH_DIR, PATTERN)
    jobs = [int(os.path.basename(path).split('_job')[1].split('_')[0]) for path in files]
    assert jobs == sorted(jobs)
    if len(files) == 4:
        assert jobs == [0, 58, 67, 305]


@job0_required
def test_read_event_info_injects_identity_from_filename():
    tree = converter.load_root_tree(JOB0)
    clean = np.ones(100, dtype=bool)
    clean[[44, 74]] = False
    info = converter.read_event_info(tree, clean, 0, 100, filepath=JOB0)
    expected_entries = [entry for entry in range(100) if entry not in (44, 74)]
    assert info['source_batch_id'].tolist() == [0] * 98
    assert info['source_microbatch_id'].tolist() == expected_entries
    assert info['event_source_job'].tolist() == [0] * 98
    assert info['event_source_entry'].tolist() == expected_entries
    assert info['event'][1] == 15522
    for name in ('source_batch_id', 'source_microbatch_id', 'event_source_job', 'event_source_entry'):
        assert info[name].dtype == np.int32, name

    legacy = converter.read_event_info(tree, np.ones(10, dtype=bool), 0, 10)
    for name in ('source_batch_id', 'source_microbatch_id', 'event_source_job', 'event_source_entry'):
        assert legacy[name].tolist() == [-1] * 10, name


@nan_fixture_required
def test_read_event_info_copies_ids_from_branches(tmp_path):
    link = tmp_path / 'abcbatch3_2024_job9_ditaus_mc.root'
    os.symlink(os.path.abspath(NAN_FIXTURE), link)
    tree = converter.load_root_tree(str(link))
    info = converter.read_event_info(tree, np.ones(10, dtype=bool), 0, 10, filepath=str(link))
    assert info['source_batch_id'].tolist() == [2] * 10
    assert info['source_microbatch_id'].tolist() == [140] * 10
    assert info['event_source_job'].tolist() == [140] * 10
    assert info['event_source_entry'].tolist() == [-1] * 10


@pytest.fixture(scope='module')
def job0_track_table():
    """Raw Track_pt / Track_dz / Track_trackFromTau of every job0 entry."""
    tree = uproot.open(JOB0)['Events']
    return tree.arrays(['Track_pt', 'Track_dz', 'Track_trackFromTau'])


@pytest.fixture(scope='module')
def job0_quota_dir(tmp_path_factory):
    """Quota-mode extended conversion of job0 (40 train + 16 val of its 59
    selected events) — the pre-existing code path, with the new columns."""
    output_dir = tmp_path_factory.mktemp('abcbatch_quota')
    converter.convert(
        input_dir=ABCBATCH_DIR, output_dir=str(output_dir),
        train_events=40, val_events=16, train_files=1, val_files=1,
        pt_cutoff=0.5, required_gt_pions=3, chunk_size=100, output_columns=None,
        pattern='abcbatch0_2024_job0_ditaus_mc.root', extended=True,
    )
    return output_dir


def _shard(directory, subdir='train', index=0):
    return ak.from_parquet(os.path.join(directory, subdir, f'{subdir}_{index:03d}.parquet'))


@job0_required
def test_read_and_filter_pion_tracks_carries_root_index():
    tree = converter.load_root_tree(JOB0)
    filtered, clean = converter.read_and_filter_pion_tracks(
        tree, converter.build_track_branch_map(converter.DEFAULT_COLUMNS), 0, 100)
    assert clean.shape == (100,) and not clean[44] and not clean[74]
    root_index = filtered[converter.TRACK_ROOT_INDEX_KEY]
    assert len(root_index) == 98
    entry_1 = ak.to_numpy(root_index[1])
    assert entry_1.tolist() == list(range(391))
    raw_pt = ak.to_numpy(tree.arrays(['Track_pt'], entry_start=1, entry_stop=2)['Track_pt'][0])
    assert np.array_equal(ak.to_numpy(filtered['Track_pt'][1]), raw_pt[entry_1])


@job0_required
def test_track_root_index_round_trip_entry_1(job0_quota_dir, job0_track_table):
    data = _shard(job0_quota_dir)
    row = data[0]
    assert int(row['event_source_entry']) == 1 and int(row['event_id']) == 15522
    root_index = ak.to_numpy(row['track_root_index'])
    assert root_index.dtype == np.int32
    assert int(row['event_n_tracks']) == 182 and root_index.size == 182
    assert np.all(np.diff(root_index) > 0)
    labels = ak.to_numpy(row['track_label_from_tau'])
    assert root_index[labels == 1].tolist() == [0, 2, 141]
    raw_pt = ak.to_numpy(job0_track_table['Track_pt'][1])
    raw_dz = ak.to_numpy(job0_track_table['Track_dz'][1])
    assert np.array_equal(ak.to_numpy(row['track_pt']), raw_pt[root_index].astype(np.float32))
    assert np.array_equal(ak.to_numpy(row['track_dz']), raw_dz[root_index].astype(np.float32))


@job0_required
def test_track_root_index_round_trip_all_rows(job0_quota_dir, job0_track_table):
    for subdir in ('train', 'val'):
        data = _shard(job0_quota_dir, subdir)
        for row in data:
            entry = int(row['event_source_entry'])
            root_index = ak.to_numpy(row['track_root_index'])
            raw_pt = ak.to_numpy(job0_track_table['Track_pt'][entry])
            raw_label = ak.to_numpy(job0_track_table['Track_trackFromTau'][entry])
            assert np.array_equal(ak.to_numpy(row['track_pt']), raw_pt[root_index].astype(np.float32))
            assert np.array_equal(ak.to_numpy(row['track_label_from_tau']), raw_label[root_index].astype(np.int32))
            assert root_index.size == int(row['event_n_tracks'])


@job0_required
def test_validate_track_root_index(job0_quota_dir):
    data = _shard(job0_quota_dir)
    converter.validate_track_root_index(data, 'train_000.parquet')
    broken = ak.Array({**{name: data[name] for name in data.fields if name != 'track_root_index'},
                       'track_root_index': data['track_root_index'][:, ::-1]})
    with pytest.raises(ValueError, match='track_root_index'):
        converter.validate_track_root_index(broken, 'train_000.parquet')
    missing = ak.Array({name: data[name] for name in data.fields if name != 'track_root_index'})
    with pytest.raises(ValueError, match='track_root_index'):
        converter.validate_track_root_index(missing, 'train_000.parquet')


OPTIONAL_BLOCK_NAMES = {'track', 'event_scalar', 'event_jagged', 'pvtx', 'genpart',
                        'muon_truth', 'kstar_all', 'kstar_selected'}


@job0_required
def test_resolve_optional_branches_job0():
    blocks = converter.resolve_optional_branches(converter.load_root_tree(JOB0))
    assert set(blocks) == OPTIONAL_BLOCK_NAMES
    assert blocks['track'] == converter.OPTIONAL_TRACK_BRANCH_MAP
    assert blocks['kstar_all'] == converter.KSTAR_ALL_BRANCH_MAP
    assert len(converter.OPTIONAL_TRACK_BRANCH_MAP) == 4
    assert len(converter.PVTX_BRANCH_MAP) == 13
    assert len(converter.GENPART_BRANCH_MAP) == 12
    assert len(converter.OPTIONAL_MUON_BRANCH_MAP) == 5
    assert len(converter.KSTAR_ALL_BRANCH_MAP) == 15


@nan_fixture_required
def test_resolve_optional_branches_merged_fixture_has_none():
    tree = converter.load_root_tree(NAN_FIXTURE)
    assert converter.resolve_optional_branches(tree) == {}
    assert converter.read_optional_collections(tree, np.ones(5, dtype=bool), 0, 5, {}) == {}


@example_fixture_required
def test_resolve_optional_branches_old_raw_fixture():
    blocks = converter.resolve_optional_branches(converter.load_root_tree(EXAMPLE_FIXTURE))
    assert 'track' in blocks and 'pvtx' in blocks and 'genpart' in blocks
    assert 'kstar_all' not in blocks


@job0_required
def test_optional_columns_equal_raw_branches(job0_quota_dir):
    data = _shard(job0_quota_dir)
    row = data[0]
    assert int(row['event_source_entry']) == 1
    raw = uproot.open(JOB0)['Events'].arrays(
        ['Track_trackFromKstar', 'Track_trackgenIdx', 'PVtx_vz', 'nPVtx', 'PVtx_ntracks',
         'GenPart_pdgId', 'GenPart_vz', 'Muon_muonFromTau', 'PV_score', 'PV_sumpt2',
         'OtherPV_score', 'OtherPV_z', 'nKstarToPiKAll', 'KstarToPiKAll_trk1_idx',
         'KstarToPiKAll_vtx_z', 'KstarToPiK_trk1_idx', 'KstarToPiK_trk2_idx'],
        entry_start=1, entry_stop=2)[0]
    root_index = ak.to_numpy(row['track_root_index'])
    assert ak.to_numpy(row['track_label_from_kstar']).tolist() == ak.to_numpy(raw['Track_trackFromKstar'])[root_index].tolist()
    assert ak.to_numpy(row['track_gen_index']).tolist() == ak.to_numpy(raw['Track_trackgenIdx'])[root_index].tolist()
    assert len(row['pvtx_z']) == int(raw['nPVtx'])
    assert np.allclose(ak.to_numpy(row['pvtx_z']), ak.to_numpy(raw['PVtx_vz']))
    assert ak.to_numpy(row['pvtx_n_tracks']).tolist() == ak.to_numpy(raw['PVtx_ntracks']).tolist()
    gen_pdg = ak.to_numpy(row['gen_pdg_id'])
    assert gen_pdg.tolist() == ak.to_numpy(raw['GenPart_pdgId']).tolist()
    assert abs(int(gen_pdg[10])) == 15 and abs(int(gen_pdg[1])) == 511
    assert float(row['gen_vertex_z'][1]) == pytest.approx(0.0551, abs=1e-4)
    assert ak.to_numpy(row['muon_label_from_tau']).tolist() == [1]
    assert float(row['event_primary_vertex_score']) == pytest.approx(float(raw['PV_score']))
    assert float(row['event_primary_vertex_sum_pt2']) == pytest.approx(float(raw['PV_sumpt2']))
    assert len(row['event_other_pv_score']) == len(row['event_other_pv_z']) == len(raw['OtherPV_z'])
    assert len(row['kstar_track1_root_index']) == int(raw['nKstarToPiKAll']) == 22
    assert ak.to_numpy(row['kstar_track1_root_index']).tolist() == ak.to_numpy(raw['KstarToPiKAll_trk1_idx']).tolist()
    assert np.allclose(ak.to_numpy(row['kstar_vertex_z']), ak.to_numpy(raw['KstarToPiKAll_vtx_z']))
    assert (int(row['event_kstar_selected_track1_root_index']),
            int(row['event_kstar_selected_track2_root_index'])) == (8, 57)
    for name in ('track_label_from_kstar', 'track_gen_index', 'track_gen_flavour', 'track_is_packed',
                 'pvtx_n_tracks', 'gen_pdg_id', 'gen_status', 'gen_mother_index', 'gen_track_index',
                 'muon_label_from_tau', 'muon_gen_index', 'kstar_track1_root_index', 'kstar_charge'):
        assert ak.to_numpy(ak.flatten(data[name])).dtype == np.int32, name
    for name in ('event_kstar_selected_track1_root_index', 'event_primary_vertex_score'):
        assert name in data.fields
    assert ak.to_numpy(data['event_kstar_selected_track1_root_index']).dtype == np.int32


@job0_required
def test_validate_optional_blocks(job0_quota_dir):
    data = _shard(job0_quota_dir)
    converter.validate_optional_blocks(data, 'train_000.parquet')
    partial = ak.Array({name: data[name] for name in data.fields if name != 'pvtx_cov_zz'})
    with pytest.raises(ValueError, match='pvtx'):
        converter.validate_optional_blocks(partial, 'train_000.parquet')
    none = ak.Array({name: data[name] for name in data.fields
                     if not name.startswith(('pvtx_', 'gen_', 'kstar_', 'muon_label', 'muon_gen',
                                             'track_label_from_kstar', 'track_gen', 'track_is_packed',
                                             'event_primary_vertex_score', 'event_primary_vertex_sum',
                                             'event_other_pv_score', 'event_kstar'))})
    converter.validate_optional_blocks(none, 'train_000.parquet')


@example_fixture_required
def test_old_raw_fixture_extended_gains_present_blocks_only(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    os.symlink(os.path.abspath(EXAMPLE_FIXTURE), source_dir / 'merged_noBKstar_batch1.root')
    output_dir = tmp_path / 'parquet'
    converter.convert(
        input_dir=str(source_dir), output_dir=str(output_dir),
        train_events=8, val_events=4, train_files=1, val_files=1,
        pt_cutoff=0.5, required_gt_pions=3, chunk_size=50, output_columns=None, extended=True,
    )
    data = _shard(str(output_dir))
    assert 'track_label_from_kstar' in data.fields and 'pvtx_z' in data.fields
    assert 'kstar_track1_root_index' not in data.fields
    assert 'track_root_index' in data.fields
    assert ak.to_numpy(data['source_batch_id']).tolist() == [-1] * 8
    assert ak.to_numpy(data['event_source_entry']).tolist() == [-1] * 8


EVAL_ONLY_KWARGS = dict(train_events=0, val_events=0, train_files=1, val_files=1,
                        pt_cutoff=0.5, required_gt_pions=3, chunk_size=100,
                        output_columns=None, extended=True, val_subdir='model',
                        eval_only=True)


@pytest.fixture(scope='module')
def job0_model_dir(tmp_path_factory):
    """Eval-only extended conversion of job0 (59 selected events) into
    50-event shards."""
    output_dir = tmp_path_factory.mktemp('abcbatch_model_job0')
    converter.convert(input_dir=ABCBATCH_DIR, output_dir=str(output_dir),
                      pattern='abcbatch0_2024_job0_ditaus_mc.root', shard_events=50,
                      **EVAL_ONLY_KWARGS)
    return output_dir


@pytest.fixture(scope='module')
def four_file_model_dir(tmp_path_factory):
    if len(FOUR_FILES) != 4:
        pytest.skip('the four abcbatch files are not all present')
    output_dir = tmp_path_factory.mktemp('abcbatch_model_four')
    converter.convert(input_dir=ABCBATCH_DIR, output_dir=str(output_dir),
                      pattern=PATTERN, shard_events=5000, **EVAL_ONLY_KWARGS)
    return output_dir


def _all_rows(directory, subdir='model'):
    files = sorted(glob.glob(os.path.join(directory, subdir, '*.parquet')))
    return files, ak.concatenate([ak.from_parquet(path) for path in files])


@job0_required
def test_eval_only_writes_every_selected_event(job0_model_dir):
    files, rows = _all_rows(job0_model_dir)
    assert [os.path.basename(path) for path in files] == ['model_000.parquet', 'model_001.parquet']
    assert len(ak.from_parquet(files[0])) == 50 and len(ak.from_parquet(files[1])) == 9
    assert len(rows) == 59
    assert not os.path.exists(os.path.join(job0_model_dir, 'train'))
    converter.validate_parquet_output(os.path.join(job0_model_dir, 'model'), 0.5, 3, extended=True)


@job0_required
def test_identity_injected_and_ordered(job0_model_dir):
    _, rows = _all_rows(job0_model_dir)
    microbatch = ak.to_numpy(rows['source_microbatch_id'])
    assert ak.to_numpy(rows['source_batch_id']).tolist() == [0] * 59
    assert np.all(np.diff(microbatch) > 0) and microbatch[0] == 1
    assert not ({44, 74} & set(microbatch.tolist()))
    assert int(rows['event_id'][0]) == 15522
    assert ak.to_numpy(rows['event_source_job']).tolist() == [0] * 59
    assert ak.to_numpy(rows['event_source_entry']).tolist() == microbatch.tolist()


@job0_required
def test_four_files_277_unique_keys(four_file_model_dir):
    files, rows = _all_rows(four_file_model_dir)
    assert [os.path.basename(path) for path in files] == ['model_000.parquet']
    assert len(rows) == 277
    keys = set(zip(*(ak.to_numpy(rows[name]).tolist() for name in (
        'event_run', 'event_id', 'event_luminosity_block', 'source_batch_id', 'source_microbatch_id'))))
    assert len(keys) == 277
    assert set(ak.to_numpy(rows['source_batch_id']).tolist()) == {0}
    jobs = ak.to_numpy(rows['event_source_job'])
    assert set(jobs.tolist()) == {0, 58, 67, 305}
    assert np.all(np.diff(jobs) >= 0)


@nan_fixture_required
def test_nan_fixture_copies_ids_and_skips_absent_blocks(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    os.symlink(os.path.abspath(NAN_FIXTURE), source_dir / 'nan_covariance_fixture.root')
    output_dir = tmp_path / 'parquet'
    converter.convert(input_dir=str(source_dir), output_dir=str(output_dir),
                      pattern='nan_covariance_fixture.root', shard_events=1000, **EVAL_ONLY_KWARGS)
    _, rows = _all_rows(str(output_dir))
    assert len(rows) > 0
    assert set(ak.to_numpy(rows['source_batch_id']).tolist()) == {2}
    assert set(ak.to_numpy(rows['source_microbatch_id']).tolist()) == {140}
    assert set(ak.to_numpy(rows['event_source_entry']).tolist()) == {-1}
    optional_columns = {column for branch_map, _ in converter.OPTIONAL_BLOCKS.values()
                        for column in branch_map.values()}
    assert not (optional_columns & set(rows.fields))
    assert 'track_root_index' in rows.fields


@job0_required
def test_discover_root_files_recursive(tmp_path):
    nested = tmp_path / 'BdToKstarTauTau' / 'crab_abcbatch1' / '0000'
    nested.mkdir(parents=True)
    os.symlink(os.path.abspath(JOB0), nested / 'abcbatch1_2024_job5_ditaus_mc.root')
    os.symlink(os.path.abspath(JOB0), tmp_path / 'abcbatch1_2024_job0_ditaus_mc.root')
    recursive = converter.discover_root_files(str(tmp_path), '**/abcbatch*_job*_ditaus_mc.root')
    assert [os.path.basename(path) for path in recursive] == [
        'abcbatch1_2024_job0_ditaus_mc.root', 'abcbatch1_2024_job5_ditaus_mc.root']
    flat = converter.discover_root_files(str(tmp_path), 'abcbatch*_job*_ditaus_mc.root')
    assert [os.path.basename(path) for path in flat] == ['abcbatch1_2024_job0_ditaus_mc.root']


@four_files_required
def test_discover_root_files_several_input_dirs(tmp_path):
    first = tmp_path / 'abcbatch0_2024'
    second = tmp_path / 'abcbatch1_2024'
    first.mkdir()
    second.mkdir()
    os.symlink(FOUR_FILES[0], first / os.path.basename(FOUR_FILES[0]))
    os.symlink(FOUR_FILES[1], first / os.path.basename(FOUR_FILES[1]))
    for path in FOUR_FILES[2:]:
        os.symlink(path, second / os.path.basename(path).replace('abcbatch0_', 'abcbatch1_'))
    files = converter.discover_root_files([str(second), str(first)], PATTERN)
    assert [os.path.basename(path) for path in files] == [
        'abcbatch0_2024_job0_ditaus_mc.root', 'abcbatch0_2024_job305_ditaus_mc.root',
        'abcbatch1_2024_job58_ditaus_mc.root', 'abcbatch1_2024_job67_ditaus_mc.root']
    assert converter.discover_root_files(str(first), PATTERN) == files[:2]
    with pytest.raises(FileNotFoundError):
        converter.discover_root_files([str(tmp_path / 'empty_a'), str(tmp_path / 'empty_b')], PATTERN)


@four_files_required
def test_eval_only_over_several_input_dirs(tmp_path):
    first = tmp_path / 'abcbatch0_2024'
    second = tmp_path / 'abcbatch1_2024'
    first.mkdir()
    second.mkdir()
    os.symlink(FOUR_FILES[0], first / os.path.basename(FOUR_FILES[0]))
    os.symlink(FOUR_FILES[1], first / os.path.basename(FOUR_FILES[1]))
    for path in FOUR_FILES[2:]:
        os.symlink(path, second / os.path.basename(path).replace('abcbatch0_', 'abcbatch1_'))
    output_dir = tmp_path / 'parquet'
    converter.convert(input_dir=[str(first), str(second)], output_dir=str(output_dir),
                      pattern=PATTERN, shard_events=5000,
                      **{**EVAL_ONLY_KWARGS, 'val_subdir': 'train'})
    files, rows = _all_rows(str(output_dir), 'train')
    assert [os.path.basename(path) for path in files] == ['train_000.parquet']
    assert len(rows) == 277
    assert sorted(set(zip(ak.to_numpy(rows['source_batch_id']).tolist(),
                          ak.to_numpy(rows['event_source_job']).tolist()))) == [
        (0, 0), (0, 305), (1, 58), (1, 67)]


@example_fixture_required
def test_eval_only_requires_source_ids(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    os.symlink(os.path.abspath(EXAMPLE_FIXTURE), source_dir / 'merged_noBKstar_batch1.root')
    with pytest.raises(ValueError, match='source ids'):
        converter.convert(input_dir=str(source_dir), output_dir=str(tmp_path / 'out'),
                          pattern='merged_noBKstar_batch1.root', shard_events=50, **EVAL_ONLY_KWARGS)


@job0_required
def test_convert_rejects_duplicate_batch_job_names(tmp_path):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    for name in ('abcbatch0_2024_job0_ditaus_mc.root', 'abcbatch0_2024_job0_ditaus_mc_1.root'):
        os.symlink(os.path.abspath(JOB0), source_dir / name)
    with pytest.raises(ValueError, match='job'):
        converter.convert(input_dir=str(source_dir), output_dir=str(tmp_path / 'out'),
                          pattern='abcbatch*_job*.root', shard_events=50, **EVAL_ONLY_KWARGS)
    assert not os.path.exists(tmp_path / 'out')


@job0_required
def test_validate_identity_unique(job0_model_dir, tmp_path):
    files, rows = _all_rows(job0_model_dir)
    converter.validate_identity_unique(files)
    duplicated = ak.concatenate([rows[:1], rows])
    duplicated_path = tmp_path / 'dup.parquet'
    ak.to_parquet(duplicated, str(duplicated_path))
    with pytest.raises(ValueError, match='duplicate'):
        converter.validate_identity_unique([str(duplicated_path)])


@job0_required
def test_build_output_table_emits_source_detail_columns():
    tree = converter.load_root_tree(JOB0)
    info = converter.read_event_info(tree, np.ones(2, dtype=bool), 0, 2, filepath=JOB0)
    track_features = {'track_pt': ak.Array([[], [12.37]])}
    output = converter.build_output_table(track_features, info, np.array([0, 1]))
    assert output['event_source_job'].tolist() == [0, 0]
    assert output['event_source_entry'].tolist() == [0, 1]
    assert output['source_microbatch_id'].tolist() == [0, 1]
    assert output['event_source_job'].dtype == np.int32
    assert output['event_source_entry'].dtype == np.int32
