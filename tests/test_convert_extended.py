"""Tests for the H6 extended ROOT->parquet conversion (--extended).

Fixture = the real 100-event ``part/data/low-pt/example_root.root`` (15
qualifying events after the 0.5 GeV cutoff + exactly-3-GT filter). Both a
legacy and an extended conversion run once per session into tmp dirs; the
tests assert schema, determinism, and sanitization on the outputs.
"""
from __future__ import annotations

import os
import sys

import awkward as ak
import numpy as np
import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'condor', 'convert_to_parquet',
))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from convert_root_to_parquet import (
    DEFAULT_COLUMNS,
    EXTENDED_TRACK_COLUMNS,
    LEGACY_CLEAN_BRANCHES,
    MUON_BRANCH_MAP,
    MUON_INTEGER_COLUMNS,
    OTHER_TRACK_BRANCH_MAP,
    OTHER_TRACK_PT_FLOOR,
    SV_BRANCH_MAP,
    SV_INTEGER_COLUMNS,
    build_track_branch_map,
    convert,
    sanitize_jagged,
    validate_parquet_output,
)
from validate_parquet_positional_identity import compare_parquet_dirs

FIXTURE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt', 'example_root.root',
))

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason='example_root.root fixture not present',
)

TRAIN_EVENTS = 8
VAL_EVENTS = 4
CONVERT_KWARGS = dict(
    train_events=TRAIN_EVENTS, val_events=VAL_EVENTS,
    train_files=1, val_files=1,
    pt_cutoff=0.5, required_gt_pions=3, chunk_size=50,
)

LEGACY_SCHEMA = {
    'event_primary_vertex_x', 'event_primary_vertex_y', 'event_primary_vertex_z',
    'event_n_tracks', 'event_run', 'event_id', 'event_luminosity_block',
    'source_batch_id', 'source_microbatch_id',
} | set(DEFAULT_COLUMNS)


def _source_dir(tmp_path_factory):
    source_dir = tmp_path_factory.mktemp('root_source')
    os.symlink(FIXTURE, os.path.join(source_dir, 'merged_noBKstar_batch1.root'))
    return str(source_dir)


@pytest.fixture(scope='session')
def legacy_dir(tmp_path_factory):
    output_dir = str(tmp_path_factory.mktemp('parquet_legacy'))
    convert(input_dir=_source_dir(tmp_path_factory), output_dir=output_dir,
            output_columns=None, **CONVERT_KWARGS)
    return output_dir


@pytest.fixture(scope='session')
def extended_dir(tmp_path_factory):
    output_dir = str(tmp_path_factory.mktemp('parquet_extended'))
    convert(input_dir=_source_dir(tmp_path_factory), output_dir=output_dir,
            output_columns=None, extended=True, **CONVERT_KWARGS)
    return output_dir


def _read(output_dir, split):
    return ak.from_parquet(os.path.join(output_dir, split, f'{split}_000.parquet'))


class TestColumnDefinitions:
    def test_extended_track_columns_resolve(self):
        branch_map = build_track_branch_map(EXTENDED_TRACK_COLUMNS)
        assert len(branch_map) == 29
        assert set(DEFAULT_COLUMNS) < set(branch_map.values())

    def test_extended_track_columns_exist_in_fixture(self):
        import uproot
        available = set(uproot.open(FIXTURE)['Events'].keys())
        branch_map = build_track_branch_map(EXTENDED_TRACK_COLUMNS)
        assert set(branch_map) <= available
        assert set(MUON_BRANCH_MAP) <= available
        assert set(SV_BRANCH_MAP) <= available
        assert set(OTHER_TRACK_BRANCH_MAP) <= available

    def test_block_sizes(self):
        assert len(MUON_BRANCH_MAP) == 17
        assert len(SV_BRANCH_MAP) == 10
        assert len(OTHER_TRACK_BRANCH_MAP) == 8

    def test_legacy_clean_branches_are_the_default_columns(self):
        default_map = build_track_branch_map(DEFAULT_COLUMNS)
        assert LEGACY_CLEAN_BRANCHES == set(default_map)


class TestSanitizeJagged:
    def test_replaces_non_finite_and_extreme(self):
        array = ak.Array([[1.0, np.nan, np.inf], [-np.inf, 5e12], []])
        cleaned = sanitize_jagged(array)
        assert ak.to_list(cleaned) == [[1.0, 0.0, 0.0], [0.0, 0.0], []]

    def test_preserves_structure_and_values(self):
        array = ak.Array([[0.5, -2.5], [3.0]])
        assert ak.to_list(sanitize_jagged(array)) == [[0.5, -2.5], [3.0]]


class TestLegacyUnchanged:
    def test_legacy_schema_exact(self, legacy_dir):
        data = _read(legacy_dir, 'train')
        assert set(data.fields) == LEGACY_SCHEMA


class TestExtendedSchema:
    def test_track_block(self, extended_dir):
        data = _read(extended_dir, 'train')
        assert set(EXTENDED_TRACK_COLUMNS) <= set(data.fields)
        for column, kind in [('track_dxy', 'f'), ('track_vertex_x', 'f'),
                             ('track_covariance_dxy_dxy', 'f'),
                             ('track_is_lost', 'i'),
                             ('track_is_matched_to_muon', 'i'),
                             ('track_is_matched_to_ele', 'i'),
                             ('track_label_from_b', 'i')]:
            flat = ak.to_numpy(ak.flatten(data[column]))
            assert flat.dtype.kind == kind, column

    def test_muon_block(self, extended_dir):
        data = _read(extended_dir, 'train')
        muon_columns = set(MUON_BRANCH_MAP.values())
        assert muon_columns <= set(data.fields)
        lengths = [ak.num(data[column], axis=1) for column in sorted(muon_columns)]
        for other in lengths[1:]:
            assert ak.to_list(other) == ak.to_list(lengths[0])
        for column in MUON_INTEGER_COLUMNS:
            assert ak.to_numpy(ak.flatten(data[column])).dtype.kind == 'i', column
        assert ak.to_numpy(ak.flatten(data['muon_pt'])).dtype == np.float32

    def test_sv_block(self, extended_dir):
        data = _read(extended_dir, 'train')
        sv_columns = set(SV_BRANCH_MAP.values())
        assert sv_columns <= set(data.fields)
        assert ak.to_list(ak.num(data['sv_x'], axis=1)) == \
            ak.to_list(ak.num(data['sv_dlen_sig'], axis=1))
        for column in SV_INTEGER_COLUMNS:
            assert ak.to_numpy(ak.flatten(data[column])).dtype.kind == 'i', column

    def test_other_track_block_window(self, extended_dir):
        data = _read(extended_dir, 'train')
        other_columns = set(OTHER_TRACK_BRANCH_MAP.values())
        assert other_columns <= set(data.fields)
        pt = ak.to_numpy(ak.flatten(data['other_track_pt']))
        assert pt.size > 0
        assert (pt >= OTHER_TRACK_PT_FLOOR).all()
        assert (pt < 0.5).all()
        assert ak.to_numpy(
            ak.flatten(data['other_track_label_from_b'])).dtype.kind == 'i'

    def test_event_block(self, extended_dir):
        data = _read(extended_dir, 'train')
        assert ak.to_numpy(data['event_n_pvs']).dtype.kind == 'i'
        assert ak.to_numpy(data['event_n_pvs_good']).dtype.kind == 'i'
        # NanoAOD stores at most 3 other PVs; events with fewer reco PVs
        # legitimately carry shorter lists.
        assert ak.all(ak.num(data['event_other_pv_z'], axis=1) <= 3)

    def test_no_non_finite_in_new_float_columns(self, extended_dir):
        data = _read(extended_dir, 'train')
        new_float_columns = (
            [c for c in EXTENDED_TRACK_COLUMNS if c not in DEFAULT_COLUMNS
             and not c.startswith('track_is') and c != 'track_label_from_b'
             and c != 'track_n_valid_hits']
            + ['muon_pt', 'muon_dz', 'muon_vertex_x', 'muon_pf_rel_iso03',
               'sv_x', 'sv_dlen_sig', 'other_track_pt', 'other_track_dz']
        )
        for column in new_float_columns:
            flat = ak.to_numpy(ak.flatten(data[column]))
            assert np.isfinite(flat).all(), column


class TestDeterminism:
    def test_event_order_identical_to_legacy(self, legacy_dir, extended_dir):
        for split in ('train', 'val'):
            legacy = _read(legacy_dir, split)
            extended = _read(extended_dir, split)
            assert len(legacy) == len(extended)
            for column in ('event_run', 'event_id', 'event_luminosity_block',
                           'event_n_tracks'):
                assert ak.to_list(legacy[column]) == ak.to_list(extended[column])
            assert ak.to_list(legacy['track_pt']) == ak.to_list(extended['track_pt'])


class TestSplitSubdirNaming:
    def test_custom_val_subdir(self, tmp_path_factory):
        output_dir = str(tmp_path_factory.mktemp('parquet_named'))
        convert(input_dir=_source_dir(tmp_path_factory), output_dir=output_dir,
                output_columns=None, val_subdir='eval', **CONVERT_KWARGS)
        assert os.path.exists(
            os.path.join(output_dir, 'train', 'train_000.parquet'))
        assert os.path.exists(
            os.path.join(output_dir, 'eval', 'eval_000.parquet'))
        assert not os.path.exists(os.path.join(output_dir, 'val'))


class TestTestSplitConverter:
    def test_extended_test_split_conversion(self, tmp_path, monkeypatch):
        import convert_test_split_to_parquet as test_split
        output_dir = str(tmp_path / 'test_split_out')
        monkeypatch.setattr(sys, 'argv', [
            'convert_test_split_to_parquet.py',
            '--input', FIXTURE,
            '--output-dir', output_dir,
            '--events-per-shard', '10',
            '--extended',
        ])
        test_split.main()

        data = ak.from_parquet(os.path.join(output_dir, 'test_000.parquet'))
        assert 'event_global_index' in data.fields
        assert 'tauto3pi_n' in data.fields
        assert set(MUON_BRANCH_MAP.values()) <= set(data.fields)
        assert set(OTHER_TRACK_BRANCH_MAP.values()) <= set(data.fields)
        assert 'event_other_pv_z' in data.fields


NAN_FIXTURE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt',
    'nan_covariance_fixture.root',
))
NAN_DIRTY_PARQUET = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'data', 'low-pt',
    'nan_covariance_dirty.parquet',
))
nan_fixture_required = pytest.mark.skipif(
    not (os.path.exists(NAN_FIXTURE) and os.path.exists(NAN_DIRTY_PARQUET)),
    reason='NaN-covariance fixtures not present',
)


@nan_fixture_required
class TestNaNSanitization:
    """Fixture = real production slice (merged_ext_batch2 entries
    4240-4300) whose kept tracks carry source-borne NaN covariances."""

    @pytest.fixture(scope='class')
    def nan_converted_dir(self, tmp_path_factory):
        source_dir = tmp_path_factory.mktemp('nan_root_source')
        os.symlink(NAN_FIXTURE,
                   os.path.join(source_dir, 'merged_ext_batch1.root'))
        output_dir = str(tmp_path_factory.mktemp('nan_parquet'))
        convert(input_dir=str(source_dir), output_dir=output_dir,
                output_columns=None, extended=True,
                pattern='merged_ext_batch*.root',
                train_events=9, val_events=1, train_files=1, val_files=1,
                pt_cutoff=0.5, required_gt_pions=3, chunk_size=50)
        return output_dir

    def test_dirty_fixture_carries_nan(self):
        data = ak.from_parquet(NAN_DIRTY_PARQUET)
        n_bad = sum(
            int((~np.isfinite(ak.to_numpy(ak.flatten(data[c])))).sum())
            for c in ('track_covariance_dsz_dsz', 'track_covariance_dxy_dsz',
                      'track_covariance_dxy_dxy')
        )
        assert n_bad > 0

    def test_new_track_float_columns_sanitized(self, nan_converted_dir):
        from convert_root_to_parquet import NEW_TRACK_FLOAT_COLUMNS
        assert len(NEW_TRACK_FLOAT_COLUMNS) == 11
        for split in ('train', 'val'):
            data = _read(nan_converted_dir, split)
            for column in NEW_TRACK_FLOAT_COLUMNS:
                flat = ak.to_numpy(ak.flatten(data[column]))
                assert np.isfinite(flat).all(), (split, column)
                assert (np.abs(flat) <= 1e10).all(), (split, column)


@nan_fixture_required
class TestPatchScript:
    @pytest.fixture()
    def patched_dir(self, tmp_path, monkeypatch):
        import shutil
        import patch_sanitize_track_columns as patch_module
        shard_dir = tmp_path / 'shards'
        shard_dir.mkdir()
        shutil.copyfile(NAN_DIRTY_PARQUET,
                        shard_dir / 'nan_covariance_dirty.parquet')
        monkeypatch.setattr(sys, 'argv', [
            'patch_sanitize_track_columns.py',
            '--data-dir', str(shard_dir),
            '--pattern', '*.parquet',
        ])
        patch_module.main()
        return shard_dir

    def test_patch_cleans_and_preserves(self, patched_dir):
        from convert_root_to_parquet import NEW_TRACK_FLOAT_COLUMNS
        original = ak.from_parquet(NAN_DIRTY_PARQUET)
        patched = ak.from_parquet(
            os.path.join(patched_dir, 'nan_covariance_dirty.parquet'))
        assert len(patched) == len(original)
        assert patched.fields == original.fields
        for column in patched.fields:
            original_flat = ak.to_numpy(ak.flatten(original[column],
                                                   axis=None))
            patched_flat = ak.to_numpy(ak.flatten(patched[column], axis=None))
            assert original_flat.shape == patched_flat.shape, column
            if column in NEW_TRACK_FLOAT_COLUMNS:
                assert np.isfinite(patched_flat).all(), column
                was_finite = np.isfinite(original_flat)
                assert np.array_equal(original_flat[was_finite],
                                      patched_flat[was_finite]), column
                assert (patched_flat[~was_finite] == 0.0).all(), column
            else:
                assert np.array_equal(original_flat, patched_flat,
                                      equal_nan=True), column

    def test_patch_idempotent(self, patched_dir, monkeypatch):
        import patch_sanitize_track_columns as patch_module
        shard_path = os.path.join(patched_dir, 'nan_covariance_dirty.parquet')
        before = ak.from_parquet(shard_path)
        monkeypatch.setattr(sys, 'argv', [
            'patch_sanitize_track_columns.py',
            '--data-dir', str(patched_dir),
            '--pattern', '*.parquet',
        ])
        patch_module.main()
        after = ak.from_parquet(shard_path)
        assert before.fields == after.fields
        for column in before.fields:
            assert np.array_equal(
                ak.to_numpy(ak.flatten(before[column], axis=None)),
                ak.to_numpy(ak.flatten(after[column], axis=None)),
            ), column


class TestValidators:
    def test_extended_output_validates(self, extended_dir):
        validate_parquet_output(os.path.join(extended_dir, 'train'),
                                pt_cutoff=0.5, required_gt_pions=3,
                                extended=True)

    def test_positional_identity_passes(self, legacy_dir, extended_dir):
        compare_parquet_dirs(os.path.join(legacy_dir, 'train'),
                             os.path.join(extended_dir, 'train'))

    def test_positional_identity_catches_mismatch(self, legacy_dir, extended_dir):
        with pytest.raises(ValueError):
            compare_parquet_dirs(os.path.join(legacy_dir, 'train'),
                                 os.path.join(extended_dir, 'val'))
