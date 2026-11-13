"""Tests for the composite event key shared by the abcbatch converter and the
baseline extractor (scripts/condor/convert_to_parquet/event_identity.py).

Real data only: the abcbatch job0 file (303 entries, no source-id branches),
its three local siblings, the old merged-production NaN fixture (source-id
branches present) and the old raw-production example file (no branches, no
job token)."""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pytest
import uproot

SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'convert_to_parquet',
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import event_identity  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ABCBATCH_DIR = os.path.join(ROOT, 'data', 'low-pt', 'abcbatch')
JOB0 = os.path.join(ABCBATCH_DIR, 'abcbatch0_2024_job0_ditaus_mc.root')
FOUR_FILES = sorted(glob.glob(os.path.join(ABCBATCH_DIR, 'abcbatch0_2024_job*_ditaus_mc.root')))
NAN_FIXTURE = os.path.join(ROOT, '..', 'part', 'data', 'low-pt', 'nan_covariance_fixture.root')
EXAMPLE_FIXTURE = os.path.join(ROOT, '..', 'part', 'data', 'low-pt', 'example_root.root')

job0_required = pytest.mark.skipif(not os.path.exists(JOB0), reason='abcbatch job0 file not present')
four_files_required = pytest.mark.skipif(len(FOUR_FILES) != 4, reason='the four abcbatch files are not all present')
nan_fixture_required = pytest.mark.skipif(not os.path.exists(NAN_FIXTURE), reason='nan_covariance_fixture.root not present')
example_fixture_required = pytest.mark.skipif(not os.path.exists(EXAMPLE_FIXTURE), reason='example_root.root not present')


def test_numbers_from_filename():
    assert event_identity.batch_number_from_filename('abcbatch0_2024_job305_ditaus_mc.root') == 0
    assert event_identity.job_number_from_filename('abcbatch0_2024_job305_ditaus_mc.root') == 305
    assert event_identity.batch_number_from_filename('/x/y/abcbatch7_2024_job12_ditaus_mc.root') == 7
    assert event_identity.batch_number_from_filename('merged_noBKstar_batch12.root') == 12
    assert event_identity.job_number_from_filename('merged_noBKstar_batch12.root') is None
    assert event_identity.batch_number_from_filename('job7.root') is None
    assert event_identity.job_number_from_filename('job7.root') == 7
    assert event_identity.file_sort_key('abcbatch0_2024_job305_ditaus_mc.root') == (0, 305)
    assert event_identity.file_sort_key('merged_noBKstar_batch12.root') == (12, 0)
    assert event_identity.file_sort_key('job7.root') == (0, 7)
    assert event_identity.file_sort_key('example_root.root') == (0, 0)
    names = ['abcbatch0_2024_job305_ditaus_mc.root', 'abcbatch0_2024_job58_ditaus_mc.root',
             'abcbatch0_2024_job0_ditaus_mc.root', 'abcbatch0_2024_job67_ditaus_mc.root']
    ordered = sorted(names, key=event_identity.file_sort_key)
    assert [event_identity.job_number_from_filename(name) for name in ordered] == [0, 58, 67, 305]


def test_numbers_from_crab_layout():
    path = ('/eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau/abcbatch1_2024/BdToKstarTauTau/'
            'crab_abcbatch1_2024_BdToKstarTauTau_20261030_161621/261030_151701/0000/miniaod_10.root')
    assert event_identity.batch_number_from_filename(path) == 1
    assert event_identity.job_number_from_filename(path) == 10
    assert event_identity.file_sort_key(path) == (1, 10)
    assert event_identity.job_number_from_filename('/x/abcbatch7_2024/0000/step1_7.root') == 7
    assert event_identity.batch_number_from_filename('/x/abcbatch7_2024/0000/step1_7.root') == 7
    assert event_identity.job_number_from_filename('step_MINI_10_nano_ditaus_mc.root') is None
    assert event_identity.job_number_from_filename('merged_noBKstar_batch12.root') is None
    assert event_identity.job_number_from_filename('nan_covariance_fixture.root') is None
    assert event_identity.batch_number_from_filename('/data/batch3_2024/x.root') is None


@job0_required
def test_source_ids_from_crab_layout(tmp_path):
    nested = tmp_path / 'abcbatch3_2024' / 'BdToKstarTauTau' / 'crab_x' / '0000'
    nested.mkdir(parents=True)
    link = nested / 'miniaod_7.root'
    os.symlink(os.path.abspath(JOB0), link)
    tree = uproot.open(str(link))['Events']
    ids = event_identity.source_ids(tree, str(link), 0, 5)
    assert ids.origin == 'filename'
    assert ids.batch.tolist() == [3] * 5
    assert ids.job.tolist() == [7] * 5
    assert ids.microbatch.tolist() == [700000 + entry for entry in range(5)]


def test_encode_decode_microbatch_id():
    assert event_identity.encode_microbatch_id(305, 42) == 30500042
    encoded = event_identity.encode_microbatch_id(305, np.arange(5))
    assert encoded.dtype == np.int32
    job, entry = event_identity.decode_microbatch_id(encoded)
    assert job.tolist() == [305] * 5
    assert entry.tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        event_identity.encode_microbatch_id(305, event_identity.ENTRIES_PER_JOB_LIMIT)
    with pytest.raises(ValueError):
        event_identity.encode_microbatch_id(21474, 0)
    assert event_identity.encode_microbatch_id(21473, 99999) == 2147399999


@job0_required
def test_source_ids_injected_job0():
    tree = uproot.open(JOB0)['Events']
    ids = event_identity.source_ids(tree, JOB0, 0, 100)
    assert ids.origin == 'filename'
    assert ids.batch.tolist() == [0] * 100
    assert ids.microbatch.tolist() == list(range(100))
    assert ids.job.tolist() == [0] * 100
    assert ids.entry.tolist() == list(range(100))
    assert ids.batch.dtype == np.int32 and ids.microbatch.dtype == np.int32
    later = event_identity.source_ids(tree, JOB0, 100, 200)
    assert later.entry[0] == 100 and later.microbatch[0] == 100 and later.entry.size == 100
    whole = event_identity.source_ids(tree, JOB0)
    assert whole.entry.size == 303 and whole.entry[-1] == 302


@job0_required
def test_identity_columns_dtypes_and_anchor():
    tree = uproot.open(JOB0)['Events']
    columns = event_identity.identity_columns(tree, JOB0, 0, 10)
    assert tuple(columns) == event_identity.IDENTITY_COLUMNS + event_identity.SOURCE_DETAIL_COLUMNS
    expected_dtypes = {'event_run': np.int32, 'event_id': np.int64, 'event_luminosity_block': np.int32,
                       'source_batch_id': np.int32, 'source_microbatch_id': np.int32,
                       'event_source_job': np.int32, 'event_source_entry': np.int32}
    for name, dtype in expected_dtypes.items():
        assert columns[name].dtype == dtype, name
        assert columns[name].shape == (10,)
    assert columns['event_id'][1] == 15522
    assert columns['event_run'][1] == 1
    assert columns['event_source_entry'].tolist() == list(range(10))


@four_files_required
def test_identity_unique_over_four_files():
    tuples = []
    triples = []
    per_file = {}
    for path in FOUR_FILES:
        tree = uproot.open(path)['Events']
        columns = event_identity.identity_columns(tree, path)
        per_file[path] = columns
        tuples.extend(event_identity.identity_tuples(columns))
        triples.extend(zip(columns['event_run'].tolist(), columns['event_id'].tolist(),
                           columns['event_luminosity_block'].tolist()))
    assert len(tuples) == 1306
    assert len(set(tuples)) == 1306
    assert len(set(triples)) == 1287
    job0 = per_file[JOB0]
    assert (int(job0['event_run'][96]), int(job0['event_id'][96]), int(job0['event_luminosity_block'][96])) == (1, 3604, 1)
    assert (int(job0['event_run'][188]), int(job0['event_id'][188]), int(job0['event_luminosity_block'][188])) == (1, 3604, 1)
    assert job0['source_microbatch_id'][96] != job0['source_microbatch_id'][188]
    assert sorted({int(columns['event_source_job'][0]) for columns in per_file.values()}) == [0, 58, 67, 305]


@nan_fixture_required
def test_source_ids_copied_when_branches_exist(tmp_path):
    link = tmp_path / 'abcbatch3_2024_job9_ditaus_mc.root'
    os.symlink(os.path.abspath(NAN_FIXTURE), link)
    tree = uproot.open(str(link))['Events']
    ids = event_identity.source_ids(tree, str(link), 0, 10)
    assert ids.origin == 'branches'
    assert ids.batch.tolist() == [2] * 10
    assert ids.microbatch.tolist() == [140] * 10
    assert ids.job.tolist() == [140] * 10
    assert ids.entry.tolist() == [-1] * 10


@example_fixture_required
def test_source_ids_missing_without_job_token(tmp_path):
    link = tmp_path / 'merged_noBKstar_batch1.root'
    os.symlink(os.path.abspath(EXAMPLE_FIXTURE), link)
    tree = uproot.open(str(link))['Events']
    ids = event_identity.source_ids(tree, str(link), 0, 10)
    assert ids.origin == 'missing'
    for array in (ids.batch, ids.microbatch, ids.job, ids.entry):
        assert array.tolist() == [-1] * 10


@job0_required
def test_assert_unique_identity():
    tree = uproot.open(JOB0)['Events']
    columns = event_identity.identity_columns(tree, JOB0)
    event_identity.assert_unique_identity(columns, context='job0')

    collided = {name: array.copy() for name, array in columns.items()}
    collided['source_microbatch_id'][188] = collided['source_microbatch_id'][96]
    with pytest.raises(ValueError, match='job0 collided'):
        event_identity.assert_unique_identity(collided, context='job0 collided')

    legacy = {name: array.copy() for name, array in columns.items()}
    legacy['source_batch_id'][:] = event_identity.MISSING_SOURCE_ID
    legacy['source_microbatch_id'][:] = event_identity.MISSING_SOURCE_ID
    legacy['event_source_job'][:] = event_identity.MISSING_SOURCE_ID
    legacy['event_source_entry'][:] = event_identity.MISSING_SOURCE_ID
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        event_identity.assert_unique_identity(legacy, context='legacy')
    assert any('legacy' in str(warning.message) for warning in caught)
