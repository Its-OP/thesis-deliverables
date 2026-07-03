from __future__ import annotations

import argparse
import glob
import multiprocessing
import os

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from utils.triplet_join import (
    build_track_lorentz,
    build_triplet_candidates,
    triplet_candidate_features,
)
from build_triplet_filter_table import DUMP, SRC, SRC_COLS, _load

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'low-pt', 'eval', 'triplet_rank')
MAX_CANDIDATES_PER_EVENT = 32768
CHUNK_SIZE = 500

# One row per event; list columns are parallel across the event's Tier-H candidates.
CANDIDATE_SCHEMA = pa.schema([
    pa.field('n_tracks', pa.int32()),
    pa.field('n_candidates', pa.int32()),
    pa.field('cand_i', pa.list_(pa.int16())),
    pa.field('cand_j', pa.list_(pa.int16())),
    pa.field('cand_k', pa.list_(pa.int16())),
    pa.field('couple_rank', pa.list_(pa.int16())),
    pa.field('gbdt6_score', pa.list_(pa.float32())),
    pa.field('gbdt8_score', pa.list_(pa.float32())),
    pa.field('is_gt', pa.list_(pa.bool_())),
    pa.field('gt_i', pa.int16()),
    pa.field('gt_j', pa.int16()),
    pa.field('gt_k', pa.int16()),
    pa.field('recon', pa.bool_()),
])


def load_gbdt_models():
    return {
        'gbdt6': joblib.load(os.path.join(MODELS_DIR, 'third_pion_filter_gbdt_full_P2.joblib')),
        'gbdt8': joblib.load(os.path.join(MODELS_DIR, 'third_pion_filter_gbdt8_full_P2.joblib')),
    }


def _plain_array(column):
    array = column.combine_chunks()
    if isinstance(array, pa.ChunkedArray):
        array = array.chunk(0)
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    return array


def _list_views(table, name):
    # Per-event numpy views over the arrow buffers: no per-value Python objects,
    # memory stays at raw-data size (to_pylist inflates float32 ~8x).
    array = _plain_array(table[name])
    lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False).astype(np.int64)
    offsets = np.zeros(len(array) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    inner = array.flatten()
    if isinstance(inner, pa.ExtensionArray):
        inner = inner.storage
    if pa.types.is_list(inner.type) or pa.types.is_large_list(inner.type):
        flat = inner.flatten().to_numpy(zero_copy_only=False).reshape(-1, 2)
    else:
        flat = inner.to_numpy(zero_copy_only=False)
    return [flat[offsets[r]:offsets[r + 1]] for r in range(len(array))]


def _event_views(dump, src):
    dump_cols = (_list_views(dump, 'stage1_sorted_indices'),
                 _list_views(dump, 'stage3_sorted_couples'))
    src_cols = {name: (src[name].to_numpy(zero_copy_only=False) if name == 'event_n_tracks'
                       else _list_views(src, name)) for name in SRC_COLS}
    return dump_cols, src_cols


def _load_slice(dump_path, src_glob, start, end):
    # Read only the source files overlapping [start, end) so each worker holds
    # its slice, never the full dataset.
    dump = pq.read_table(dump_path, columns=['stage1_sorted_indices', 'stage3_sorted_couples'])
    dump = dump.slice(start, end - start)
    parts, row0 = [], 0
    for path in sorted(glob.glob(src_glob)):
        n_rows = pq.read_metadata(path).num_rows
        lo, hi = max(start, row0), min(end, row0 + n_rows)
        if lo < hi:
            parts.append(pq.read_table(path, columns=SRC_COLS).slice(lo - row0, hi - lo))
        row0 += n_rows
    src = pa.concat_tables(parts)
    assert src.num_rows == end - start, f'src slice {src.num_rows} != {end - start}'
    return dump, src


def event_features(r, dump_cols, src_cols, *, top_c):
    s1, couples_all = dump_cols
    cols = src_cols
    assert len(s1[r]) == cols['event_n_tracks'][r], f'alignment break at row {r}'
    n_tracks = int(cols['event_n_tracks'][r])
    lorentz = build_track_lorentz(torch.tensor(cols['track_pt'][r], dtype=torch.float32),
                                  torch.tensor(cols['track_eta'][r], dtype=torch.float32),
                                  torch.tensor(cols['track_phi'][r], dtype=torch.float32))
    t = lambda key: torch.tensor(cols[key][r], dtype=torch.float32)
    kw = dict(lorentz=lorentz, charge=t('track_charge'), eta=t('track_eta'), phi=t('track_phi'),
              dz=t('track_dz_significance'), dxy_sig=t('track_dxy_significance'),
              dca_sig=t('track_dca_significance'), n_pixel=t('track_n_valid_pixel_hits'),
              norm_chi2=t('track_norm_chi2'), pt_error=t('track_pt_error'),
              cov_phi_phi=t('track_covariance_phi_phi'),
              cov_lambda_lambda=t('track_covariance_lambda_lambda'))

    labels = np.asarray(cols['track_label_from_tau'][r])
    gt = np.where(labels > 0.5)[0]
    couples_np = np.asarray(couples_all[r][:top_c], dtype=np.int64).reshape(-1, 2)
    couples = torch.tensor(couples_np, dtype=torch.long)
    pool = torch.arange(n_tracks, dtype=torch.long)  # P2: entire input track set
    gt_set = set(gt.tolist())
    has_gt_couple = any(set(c).issubset(gt_set) for c in couples_np.tolist())
    reconstructable = gt.size == 3 and has_gt_couple
    gt_sorted = tuple(sorted(gt.tolist())) if reconstructable else None

    # Features/labels and candidate (i, j, k) indices come from two calls that share the
    # same Tier-H enumeration (charge net +-1, m(ijk) <= m_tau, ascending k per couple);
    # the asserts below pin that order equality.
    X, _, is_gt, couple_row = triplet_candidate_features(couples, pool, gt_sorted=gt_sorted, **kw)
    triplets, couple_row_check = build_triplet_candidates(couples, pool, lorentz=lorentz,
                                                          charge=kw['charge'])
    assert torch.equal(couple_row_check, couple_row), f'enumeration order mismatch at row {r}'
    assert torch.equal(triplets[:, 0], couples[couple_row, 0]), f'i mismatch at row {r}'
    assert torch.equal(triplets[:, 1], couples[couple_row, 1]), f'j mismatch at row {r}'
    n_candidates = int(X.shape[0])
    assert n_candidates <= MAX_CANDIDATES_PER_EVENT, f'{n_candidates} candidates at row {r}'
    assert torch.isfinite(X).all(), f'non-finite features at row {r}'

    gt_i, gt_j, gt_k = (gt_sorted if gt_sorted is not None else (-1, -1, -1))
    row = {
        'n_tracks': n_tracks,
        'n_candidates': n_candidates,
        'cand_i': triplets[:, 0].to(torch.int16).tolist(),
        'cand_j': triplets[:, 1].to(torch.int16).tolist(),
        'cand_k': triplets[:, 2].to(torch.int16).tolist(),
        'couple_rank': couple_row.to(torch.int16).tolist(),
        'is_gt': is_gt.tolist(),
        'gt_i': gt_i, 'gt_j': gt_j, 'gt_k': gt_k,
        'recon': bool(reconstructable),
    }
    return row, X.numpy()


def _predict(model, X):
    if not len(X):
        return np.zeros(0, np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def event_candidates(r, dump_cols, src_cols, *, top_c, gbdt_models):
    row, X = event_features(r, dump_cols, src_cols, top_c=top_c)
    row['gbdt6_score'] = _predict(gbdt_models['gbdt6'], X).tolist()
    row['gbdt8_score'] = _predict(gbdt_models['gbdt8'], X).tolist()
    return row


def _empty_row(src_cols, r):
    try:
        n_tracks = int(src_cols['event_n_tracks'][r])
    except Exception:
        n_tracks = 0
    return {'n_tracks': n_tracks, 'n_candidates': 0,
            'cand_i': [], 'cand_j': [], 'cand_k': [], 'couple_rank': [],
            'gbdt6_score': [], 'gbdt8_score': [], 'is_gt': [],
            'gt_i': -1, 'gt_j': -1, 'gt_k': -1, 'recon': False}


def _valid_rows(path):
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return -1


def _write_rows(path, rows):
    columns = {field.name: [row[field.name] for row in rows] for field in CANDIDATE_SCHEMA}
    writer = pq.ParquetWriter(path, CANDIDATE_SCHEMA)
    writer.write_table(pa.table(columns, schema=CANDIDATE_SCHEMA))
    writer.close()


def _process_chunk(chunk_events, offset, dump_cols, src_cols, top_c, gbdt_models):
    rows, feature_blocks, succeeded = [], [], []
    for g in chunk_events:
        try:
            row, X = event_features(int(g) - offset, dump_cols, src_cols, top_c=top_c)
            rows.append(row)
            feature_blocks.append(X)
            succeeded.append(True)
        except Exception as error:
            print(f'event {g} failed, writing empty row: {error!r}', flush=True)
            rows.append(_empty_row(src_cols, int(g) - offset))
            succeeded.append(False)
    if feature_blocks:
        # One predict_proba per model per chunk amortizes sklearn call overhead.
        lengths = [block.shape[0] for block in feature_blocks]
        stacked = np.vstack(feature_blocks)
        for name in ('gbdt6', 'gbdt8'):
            split = iter(np.split(_predict(gbdt_models[name], stacked), np.cumsum(lengths)[:-1]))
            for row, ok in zip(rows, succeeded):
                if ok:
                    row[f'{name}_score'] = next(split).tolist()
    return rows, succeeded.count(False)


def _chunk_grid(events, chunk_size, out_path):
    for start in range(0, len(events), chunk_size):
        chunk_events = events[start:start + chunk_size]
        yield chunk_events, f'{out_path}.chunk{chunk_events[0]:06d}'


def _finalize_if_complete(out_path, n_events):
    if _valid_rows(out_path) == n_events:
        for path in sorted(glob.glob(out_path + '.chunk*')):
            os.remove(path)
        print(f'{out_path} already complete ({n_events} rows)')
        return True
    return False


def _assemble(out_path, chunk_specs):
    missing = [path for path, expected in chunk_specs if _valid_rows(path) != expected]
    if missing:
        raise RuntimeError(f'{len(missing)} incomplete chunks, e.g. {missing[:3]}')
    writer = pq.ParquetWriter(out_path, CANDIDATE_SCHEMA)
    for path, _ in chunk_specs:
        writer.write_table(pq.read_table(path))
    writer.close()
    for path, _ in chunk_specs:
        os.remove(path)


def write_candidates(dump_cols, src_cols, event_range, top_c, gbdt_models, out_path,
                     chunk_size=CHUNK_SIZE):
    events = list(event_range)
    if _finalize_if_complete(out_path, len(events)):
        return

    # Each chunk is written as its own closed parquet file, so a killed run loses at
    # most one chunk and a rerun skips every complete chunk.
    n_failed = 0
    progress = tqdm(total=len(events), desc='candidates', mininterval=30)
    for chunk_events, chunk_path in _chunk_grid(events, chunk_size, out_path):
        if _valid_rows(chunk_path) == len(chunk_events):
            progress.update(len(chunk_events))
            continue
        rows, failed = _process_chunk(chunk_events, 0, dump_cols, src_cols, top_c, gbdt_models)
        n_failed += failed
        _write_rows(chunk_path, rows)
        progress.update(len(chunk_events))
    progress.close()

    _assemble(out_path, [(path, len(ev)) for ev, path in _chunk_grid(events, chunk_size, out_path)])
    if n_failed:
        print(f'WARNING: {n_failed}/{len(events)} events failed and were written empty')
    print(f'wrote {out_path} ({len(events)} rows)')


def _worker_ranges(n, chunk_size, workers):
    n_chunks = (n + chunk_size - 1) // chunk_size
    boundaries = [min(round(w * n_chunks / workers) * chunk_size, n) for w in range(workers + 1)]
    return [(boundaries[w], boundaries[w + 1]) for w in range(workers)
            if boundaries[w] < boundaries[w + 1]]


def _worker_main(dump_path, src_glob, start, end, top_c, chunk_size, out_path, worker_id):
    torch.set_num_threads(1)
    dump, src = _load_slice(dump_path, src_glob, start, end)
    dump_cols, src_cols = _event_views(dump, src)
    gbdt_models = load_gbdt_models()
    events = list(range(start, end))
    n_failed = 0
    for chunk_events, chunk_path in _chunk_grid(events, chunk_size, out_path):
        if _valid_rows(chunk_path) == len(chunk_events):
            continue
        rows, failed = _process_chunk(chunk_events, start, dump_cols, src_cols, top_c, gbdt_models)
        n_failed += failed
        _write_rows(chunk_path, rows)
        print(f'worker {worker_id}: chunk {chunk_events[0]} done '
              f'({chunk_events[-1] - start + 1}/{end - start} events)', flush=True)
    if n_failed:
        print(f'worker {worker_id}: WARNING {n_failed} events failed', flush=True)


def _run_workers(args, n, out_path):
    if _finalize_if_complete(out_path, n):
        return
    os.environ['OMP_NUM_THREADS'] = '1'
    context = multiprocessing.get_context('spawn')
    ranges = _worker_ranges(n, args.chunk_size, args.workers)
    print(f'launching {len(ranges)} workers over {n} events...', flush=True)
    processes = [context.Process(target=_worker_main,
                                 args=(args.dump, args.src_glob, start, end, args.top_c,
                                       args.chunk_size, out_path, w))
                 for w, (start, end) in enumerate(ranges)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    bad = [process.exitcode for process in processes if process.exitcode != 0]
    if bad:
        raise RuntimeError(f'{len(bad)} workers exited nonzero: {bad}')
    _assemble(out_path, [(path, len(ev))
                         for ev, path in _chunk_grid(list(range(n)), args.chunk_size, out_path)])
    print(f'wrote {out_path} ({n} rows)')


def write_tracks(src, n_events, out_path):
    pq.write_table(src.slice(0, n_events), out_path)
    print(f'wrote {out_path} ({n_events} rows)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=DUMP, help='cascade dump (default: VAL)')
    ap.add_argument('--src-glob', default=SRC, help='per-track source parquet glob (default: VAL)')
    ap.add_argument('--tag', default='val', help='artifact suffix: candidates_<tag>.parquet')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--top-c', type=int, default=100)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel worker processes over disjoint chunk-aligned event ranges')
    ap.add_argument('--tracks-only', action='store_true',
                    help='only write the consolidated tracks_<tag>.parquet')
    ap.add_argument('--skip-tracks', action='store_true',
                    help='do not (re)write tracks_<tag>.parquet')
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    print('loading dump + source parquet...', flush=True)
    dump, src, n = _load(args.dump, args.src_glob, args.max_events)
    if not args.skip_tracks:
        write_tracks(src, n, os.path.join(args.out_dir, f'tracks_{args.tag}.parquet'))
    if args.tracks_only:
        return

    out_path = os.path.join(args.out_dir, f'candidates_{args.tag}.parquet')
    if args.workers > 1:
        del dump, src
        _run_workers(args, n, out_path)
        return

    print(f'building event views for {n} events...', flush=True)
    dump_cols, src_cols = _event_views(dump, src)
    gbdt_models = load_gbdt_models()
    write_candidates(dump_cols, src_cols, range(n), args.top_c, gbdt_models, out_path,
                     chunk_size=args.chunk_size)


if __name__ == '__main__':
    main()
