"""End-to-end tests of the ROOT -> parquet converter on MiniAOD input.

Real data only: MiniAOD files of abcbatch1 (CRAB subdirectory 0001), read
through the CRAB directory layout the condor jobs walk. Needs ROOT; skipped
without."""
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
import miniaod_source as source  # noqa: E402

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BATCH_DIR = os.path.join(ROOT_DIR, 'data', 'low-pt', 'miniaod_sample', 'abcbatch1_2024')
PATTERN = '**/miniaod_10[01][0-9].root'
INPUT_FILES = sorted(glob.glob(os.path.join(BATCH_DIR, PATTERN), recursive=True))
pytestmark = pytest.mark.skipif(len(INPUT_FILES) < 8, reason='MiniAOD sample files not present')
KWARGS = dict(train_events=0, val_events=0, train_files=1, val_files=1, pt_cutoff=0.5, required_gt_pions=3,
              chunk_size=2500, output_columns=None, extended=True, eval_only=True, shard_events=5000,
              val_subdir='train', shard_prefix='train_b1')
MINIAOD_BLOCKS = ('track_miniaod', 'muon_miniaod', 'event_miniaod', 'kstar_miniaod')


def _convert(output_dir, track_selection):
    converter.convert(input_dir=[BATCH_DIR], output_dir=str(output_dir), pattern=PATTERN,
                      track_selection=track_selection, **KWARGS)
    files = sorted(glob.glob(os.path.join(output_dir, 'train', '*.parquet')))
    return files, ak.concatenate([ak.from_parquet(path) for path in files])


@pytest.fixture(scope='module')
def from_pv_rows(tmp_path_factory):
    return _convert(tmp_path_factory.mktemp('miniaod_from_pv'), 'from_pv')


@pytest.fixture(scope='module')
def all_rows(tmp_path_factory):
    return _convert(tmp_path_factory.mktemp('miniaod_all'), 'all')


def _keys(rows):
    return list(zip(*(ak.to_numpy(rows[name]).tolist() for name in (
        'event_run', 'event_id', 'event_luminosity_block', 'source_batch_id', 'source_microbatch_id'))))


def test_miniaod_run_writes_prefixed_shards_with_every_block(from_pv_rows):
    files, rows = from_pv_rows
    assert [os.path.basename(path) for path in files] == ['train_b1_000.parquet']
    assert len(rows) >= 20
    expected = set(converter.EXTENDED_TRACK_COLUMNS) | set(converter.MUON_BRANCH_MAP.values())
    expected |= set(converter.SV_BRANCH_MAP.values()) | set(converter.OTHER_TRACK_BRANCH_MAP.values())
    for branch_map, _ in converter.OPTIONAL_BLOCKS.values():
        expected |= set(branch_map.values())
    assert expected <= set(rows.fields), sorted(expected - set(rows.fields))
    for name in MINIAOD_BLOCKS:
        assert name in converter.OPTIONAL_BLOCKS
    converter.validate_parquet_output(os.path.dirname(files[0]), 0.5, 3, extended=True)


def test_event_key_comes_from_the_crab_layout(from_pv_rows):
    _, rows = from_pv_rows
    keys = _keys(rows)
    assert len(set(keys)) == len(keys)
    assert set(ak.to_numpy(rows['source_batch_id']).tolist()) == {1}
    jobs = set(ak.to_numpy(rows['event_source_job']).tolist())
    assert jobs <= set(range(1000, 1020)) and len(jobs) >= 5
    np.testing.assert_array_equal(
        ak.to_numpy(rows['source_microbatch_id']),
        ak.to_numpy(rows['event_source_job']) * 100000 + ak.to_numpy(rows['event_source_entry']))


def test_labels_and_new_columns(from_pv_rows):
    _, rows = from_pv_rows
    assert np.all(ak.to_numpy(ak.sum(rows['track_label_from_tau'], axis=1)) == 3)
    labelled = ak.flatten(rows['track_label_from_tau']) == 1
    delta_r = ak.to_numpy(ak.flatten(rows['track_gen_match_delta_r']))
    assert np.all((delta_r[ak.to_numpy(labelled)] >= 0) & (delta_r[ak.to_numpy(labelled)] < 0.05))
    assert np.all(ak.to_numpy(ak.flatten(rows['track_from_pv'])) > 0)
    for column in ('track_from_pv', 'track_pv_index', 'track_gen_association_index', 'muon_trigger_bits',
                   'muon_is_anchor', 'event_trigger_bits', 'event_fired_ip6', 'event_fired_any_muon_path'):
        values = rows[column]
        assert ak.to_numpy(ak.flatten(values) if values.ndim > 1 else values).dtype.kind == 'i', column
    assert np.all(ak.to_numpy(ak.sum(rows['muon_is_anchor'], axis=1)) <= 1)
    bits = ak.to_numpy(rows['event_trigger_bits'])
    np.testing.assert_array_equal(ak.to_numpy(rows['event_fired_ip6']) == 1, (bits & source.IP6_PATH_MASK) != 0)
    assert np.all(ak.to_numpy(rows['event_n_true_interactions']) > 10)
    assert ak.all(ak.num(rows['kstar_muon_dca']) == ak.num(rows['kstar_track1_root_index']))
    assert ak.all(ak.num(rows['track_dz_to_anchor_muon']) == ak.num(rows['track_pt']))


def test_from_pv_keeps_the_primary_vertex_tracks_of_the_all_selection(from_pv_rows, all_rows):
    _, selected = from_pv_rows
    _, everything = all_rows
    position = {key: row for row, key in enumerate(_keys(everything))}
    selected_keys = _keys(selected)
    assert set(selected_keys) <= set(position)
    # dropping the tracks fitted to another vertex removes true pions: fewer events qualify
    assert len(selected) < len(everything)
    for row, key in enumerate(selected_keys):
        reference = everything[position[key]]
        kept = ak.to_numpy(reference['track_from_pv']) > 0
        np.testing.assert_array_equal(ak.to_numpy(selected['track_root_index'][row]),
                                      ak.to_numpy(reference['track_root_index'])[kept])
        np.testing.assert_array_equal(ak.to_numpy(selected['track_pt'][row]), ak.to_numpy(reference['track_pt'])[kept])
    assert ak.sum(ak.num(selected['track_pt'])) < 0.7 * ak.sum(ak.num(everything['track_pt'][[position[key] for key in selected_keys]]))


def _batch_with_truncated_file(directory):
    """Directory abcbatch1_2024/ with links to three real files and a
    truncated copy of a fourth."""
    batch = directory / 'abcbatch1_2024'
    batch.mkdir()
    for path in INPUT_FILES[:3]:
        os.symlink(path, batch / os.path.basename(path))
    with open(INPUT_FILES[3], 'rb') as handle:
        (batch / os.path.basename(INPUT_FILES[3])).write_bytes(handle.read(200_000))
    return batch


def test_unreadable_miniaod_file_is_skipped_and_reported(tmp_path, capsys):
    batch = _batch_with_truncated_file(tmp_path)
    converter.convert(input_dir=[str(batch)], output_dir=str(tmp_path / 'out'), pattern='miniaod_*.root',
                      track_selection='all', max_unreadable_fraction=0.5, **KWARGS)
    printed = capsys.readouterr().out
    assert 'UNREADABLE' in printed and os.path.basename(INPUT_FILES[3]) in printed
    rows = ak.concatenate([ak.from_parquet(path) for path in glob.glob(str(tmp_path / 'out' / 'train' / '*.parquet'))])
    skipped_job = int(os.path.basename(INPUT_FILES[3]).split('_')[1].split('.')[0])
    assert skipped_job not in set(ak.to_numpy(rows['event_source_job']).tolist())


def test_too_many_unreadable_files_fail_the_run(tmp_path):
    batch = _batch_with_truncated_file(tmp_path)
    with pytest.raises(RuntimeError, match='unreadable'):
        converter.convert(input_dir=[str(batch)], output_dir=str(tmp_path / 'out'), pattern='miniaod_*.root',
                          track_selection='all', **KWARGS)
