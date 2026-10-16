"""Concatenate the parquet files matched by a glob into one zstd parquet file.

Files are taken in sorted path order and copied row group by row group, so a
10-shard stage dump (1-3 GB each) merges without loading any shard whole.
"""
from __future__ import annotations

import argparse
import glob

import pyarrow.parquet as pq


def concat_parquet_files(pattern: str, output_path: str) -> int:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'no files match {pattern}')
    writer = None
    total_rows = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='zstd')
            writer.write_table(table)
            total_rows += table.num_rows
        print(f'{path}: done, cumulative rows {total_rows}')
    if writer is None:
        raise SystemExit(f'files matching {pattern} contain no row groups')
    writer.close()
    print(f'wrote {output_path}: {total_rows} rows from {len(paths)} files')
    return total_rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pattern', help='glob of parquet files, concatenated in sorted order')
    parser.add_argument('output', help='output parquet path')
    args = parser.parse_args(argv)
    concat_parquet_files(args.pattern, args.output)


if __name__ == '__main__':
    main()
