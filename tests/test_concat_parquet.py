from __future__ import annotations

import os
import shutil

import pyarrow.parquet as pq
import pytest

from scripts.python.concat_parquet import concat_parquet_files, main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, 'data', 's4_tables', 'triplet_filter_eval_gt.parquet')

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE), reason='local s4 table fixture not present')


def _two_copies(tmp_path):
    first = tmp_path / 'part_s0.parquet'
    second = tmp_path / 'part_s1.parquet'
    shutil.copyfile(FIXTURE, first)
    shutil.copyfile(FIXTURE, second)
    return first, second


def test_concat_preserves_rows_schema_and_order(tmp_path):
    _two_copies(tmp_path)
    output = tmp_path / 'merged.parquet'
    total = concat_parquet_files(str(tmp_path / 'part_s*.parquet'), str(output))
    source = pq.read_table(FIXTURE)
    merged = pq.read_table(output)
    assert total == 2 * source.num_rows
    assert merged.num_rows == 2 * source.num_rows
    assert merged.schema.equals(source.schema)
    # pandas equality treats NaN == NaN; pyarrow Table.equals does not.
    source_frame = source.to_pandas().reset_index(drop=True)
    first_half = merged.slice(0, source.num_rows).to_pandas().reset_index(drop=True)
    second_half = merged.slice(source.num_rows).to_pandas().reset_index(drop=True)
    assert first_half.equals(source_frame)
    assert second_half.equals(source_frame)


def test_concat_rejects_empty_glob(tmp_path):
    with pytest.raises(SystemExit):
        concat_parquet_files(str(tmp_path / 'nothing_*.parquet'),
                             str(tmp_path / 'out.parquet'))
    assert not (tmp_path / 'out.parquet').exists()


def test_main_cli_writes_output_and_reports(tmp_path, capsys):
    _two_copies(tmp_path)
    output = tmp_path / 'merged.parquet'
    main([str(tmp_path / 'part_s*.parquet'), str(output)])
    assert pq.read_metadata(output).num_rows == 2 * pq.read_metadata(FIXTURE).num_rows
    assert 'wrote' in capsys.readouterr().out
