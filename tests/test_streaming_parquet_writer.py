from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.python.eval_cascade_pipeline import StreamingParquetWriter

SCHEMA = pa.schema([
    pa.field('event_id', pa.int64()),
    pa.field('values', pa.list_(pa.float32())),
])


def _make_rows(n, offset=0):
    return [{'event_id': offset + i, 'values': [float(i), float(i + 1)]}
            for i in range(n)]


def test_streaming_writer_flushes_and_preserves_rows(tmp_path):
    output_path = str(tmp_path / 'dump.parquet')
    writer = StreamingParquetWriter(output_path, SCHEMA, flush_rows=4)
    for row in _make_rows(10):
        writer.add(row)
    writer.close()

    table = pq.read_table(output_path)
    assert table.num_rows == 10
    assert table.column('event_id').to_pylist() == list(range(10))
    assert pq.ParquetFile(output_path).num_row_groups >= 2


def test_streaming_writer_close_without_rows_writes_empty_file(tmp_path):
    output_path = str(tmp_path / 'empty.parquet')
    writer = StreamingParquetWriter(output_path, SCHEMA, flush_rows=4)
    writer.close()
    table = pq.read_table(output_path)
    assert table.num_rows == 0


def test_streaming_writer_creates_no_file_before_first_flush(tmp_path):
    output_path = tmp_path / 'lazy.parquet'
    writer = StreamingParquetWriter(str(output_path), SCHEMA, flush_rows=100)
    for row in _make_rows(5):
        writer.add(row)
    assert not output_path.exists()
    writer.close()
    assert pq.read_table(str(output_path)).num_rows == 5


def test_streaming_writer_rows_written_counter(tmp_path):
    output_path = str(tmp_path / 'count.parquet')
    writer = StreamingParquetWriter(output_path, SCHEMA, flush_rows=3)
    for row in _make_rows(7):
        writer.add(row)
    assert writer.rows_written + len(writer._buffer) == 7
    writer.close()
    assert writer.rows_written == 7
