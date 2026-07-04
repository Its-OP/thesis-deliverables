from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from scripts.python.build_triplet_rank_candidates import (
        CANDIDATE_SCHEMA,
        _list_views,
        dedupe_event_candidates,
    )
except ImportError:  # direct-file invocation: scripts/python is sys.path[0]
    from build_triplet_rank_candidates import (
        CANDIDATE_SCHEMA,
        _list_views,
        dedupe_event_candidates,
    )

WRITE_BATCH_EVENTS = 5000


def _write_batch(writer, rows):
    columns = {field.name: [row[field.name] for row in rows] for field in CANDIDATE_SCHEMA}
    writer.write_table(pa.table(columns, schema=CANDIDATE_SCHEMA))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Post-process an existing candidates '
                                                 'parquet into its 3-set-deduped form.')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-events', type=int, default=None)
    parser.add_argument('--tau', type=float, default=0.025128,
                        help='d6 canon threshold for the GT-survival preservation check')
    args = parser.parse_args(argv)

    table = pq.read_table(args.candidates)
    n_events = table.num_rows if args.max_events is None else min(args.max_events,
                                                                  table.num_rows)
    # Bulk per-event numpy views (offset slicing) instead of per-row to_pylist —
    # the conversion, not the dedupe math, dominates otherwise.
    schema = table.schema
    list_names = [name for name in schema.names
                  if pa.types.is_list(schema.field(name).type)]
    scalar_names = [name for name in schema.names if name not in list_names]
    list_views = {name: _list_views(table, name) for name in list_names}
    scalars = {name: table[name].to_pylist() for name in scalar_names}

    rows_before = 0
    rows_after = 0
    gt_survives_before = 0
    gt_survives_after = 0
    writer = pq.ParquetWriter(args.out, CANDIDATE_SCHEMA)
    pending = []
    for r in tqdm(range(n_events), desc='dedupe', mininterval=10):
        row = {name: list_views[name][r] for name in list_names}
        row.update({name: scalars[name][r] for name in scalar_names})
        scores = np.asarray(row['gbdt6_score'], dtype=np.float64)
        is_gt = np.asarray(row['is_gt'], dtype=bool)
        if is_gt.any() and (scores[is_gt] >= args.tau).any():
            gt_survives_before += 1
        rows_before += len(row['cand_i'])

        deduped = dedupe_event_candidates(row)
        rows_after += deduped['n_candidates']
        kept_scores = np.asarray(deduped['gbdt6_score'], dtype=np.float64)
        kept_gt = np.asarray(deduped['is_gt'], dtype=bool)
        if kept_gt.any() and (kept_scores[kept_gt] >= args.tau).any():
            gt_survives_after += 1

        pending.append(deduped)
        if len(pending) >= WRITE_BATCH_EVENTS:
            _write_batch(writer, pending)
            pending = []
    if pending:
        _write_batch(writer, pending)
    writer.close()

    duplicate_fraction = 1.0 - rows_after / max(rows_before, 1)
    print(f'{n_events} events: {rows_before} rows -> {rows_after} '
          f'({duplicate_fraction:.2%} duplicates removed)')
    print(f'GT survival at tau={args.tau}: {gt_survives_before} before, '
          f'{gt_survives_after} after')
    assert gt_survives_before == gt_survives_after, \
        'dedupe changed GT survival — max-gbdt6 keep-rule violated'
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
