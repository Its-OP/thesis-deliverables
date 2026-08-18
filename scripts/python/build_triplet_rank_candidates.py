from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import traceback

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

try:
    from scripts.python.build_triplet_filter_table import (
        ROLE_DEFAULTS, IDENTITY_COLS, SRC_COLS,
        _event_candidates, _featurize, _h6_inputs_for_event,  # noqa: F401
        _worker_pool, _WORKER_STATE,
        assert_dump_aligned, dump_row_order, identity_keys)
    from scripts.python.eval_triplet_filter_ranking import use_cpu_inference
except ImportError:  # direct-file invocation: scripts/python is sys.path[0]
    from build_triplet_filter_table import (
        ROLE_DEFAULTS, IDENTITY_COLS, SRC_COLS,
        _event_candidates, _featurize, _h6_inputs_for_event,  # noqa: F401
        _worker_pool, _WORKER_STATE,
        assert_dump_aligned, dump_row_order, identity_keys)
    from eval_triplet_filter_ranking import use_cpu_inference

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                       'triplet_rank_v2')
FILTER_MODEL_GLOB = os.path.join(
    MODELS_DIR, 'sweep',
    'sweep_vertex_physics__uniform__balanced__d8__*.joblib')
# vertex (6) + physics (5): the h6s4 champion's feature set; isolation and
# secondary-vertex blocks measured inert and are never computed here.
H6_WIDTH = 11
MAX_CANDIDATES_PER_EVENT = 32768
CHUNK_SIZE = 500
WINDOW = 2048
TAIL_SAMPLE = 512

DUMP_COLS = ['stage1_sorted_indices', 'stage1_scores', 'stage2_sorted_indices',
             'stage2_scores', 'stage3_sorted_couples', 'stage3_couple_scores']

# One row per event, written in SOURCE-SHARD order (the 5-column identity key
# joins the worker-permuted dump back to the shards). Stored candidate rows =
# top-`window` by filter_score, descending (stored position = filter rank),
# then GT rows the window missed (row_kind 1), then a uniform tail sample for
# unbiased full-list losses (row_kind 2, train side only). The gate tau is a
# runtime parameter — scores are stored ungated.
CANDIDATE_SCHEMA = pa.schema([
    pa.field('n_tracks', pa.int32()),
    pa.field('n_tierh', pa.int32()),
    pa.field('n_tail_total', pa.int32()),
    pa.field('cand_i', pa.list_(pa.int16())),
    pa.field('cand_j', pa.list_(pa.int16())),
    pa.field('cand_k', pa.list_(pa.int16())),
    pa.field('couple_rank', pa.list_(pa.int16())),
    pa.field('filter_score', pa.list_(pa.float32())),
    pa.field('is_gt', pa.list_(pa.bool_())),
    pa.field('row_kind', pa.list_(pa.uint8())),
    pa.field('gt_i', pa.int16()),
    pa.field('gt_j', pa.int16()),
    pa.field('gt_k', pa.int16()),
    pa.field('recon', pa.bool_()),
    pa.field('track_s1', pa.list_(pa.float32())),
    pa.field('track_s2', pa.list_(pa.float32())),
    pa.field('couple_scores', pa.list_(pa.float32())),
    pa.field('event_run', pa.int64()),
    pa.field('event_id', pa.int64()),
    pa.field('event_luminosity_block', pa.int64()),
    pa.field('source_batch_id', pa.int64()),
    pa.field('source_microbatch_id', pa.int64()),
])


def load_filter_model(path_glob):
    matches = sorted(glob.glob(path_glob))
    assert len(matches) == 1, \
        f'expected exactly one filter model at {path_glob}, found {matches}'
    model = joblib.load(matches[0])
    width = int(getattr(model, 'n_features_in_', -1))
    assert width == 89 + H6_WIDTH, \
        f'{matches[0]} expects {width} features, the builder produces {89 + H6_WIDTH}'
    return use_cpu_inference(model)


def _window_selection(scores, is_gt, *, window, tail_sample, generator):
    """scores: (M,) filter scores; is_gt: (M,) bool. Returns (row indices,
    row kinds, tail-population size). Window rows come first in descending
    score order, then forced-GT rows the window missed, then the tail sample."""
    order = np.argsort(-scores, kind='stable')
    if window is None:
        return order, np.zeros(len(order), dtype=np.uint8), 0
    head = order[:window]
    outside = order[window:]
    forced = outside[is_gt[outside]]
    tail_pool = outside[~is_gt[outside]]
    take = min(tail_sample, len(tail_pool))
    tail = (tail_pool[generator.choice(len(tail_pool), take, replace=False)]
            if take else tail_pool[:0])
    rows = np.concatenate([head, forced, tail])
    kinds = np.concatenate([np.zeros(len(head), dtype=np.uint8),
                            np.ones(len(forced), dtype=np.uint8),
                            np.full(len(tail), 2, dtype=np.uint8)])
    return rows, kinds, int(len(tail_pool))


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


def _dump_blocks(dump_path, src_glob, max_events):
    """Yields (dump_views, couples, src_cols, identity_rows, n) one source
    shard at a time, with the dump reordered onto the shard by the 5-column
    identity key."""
    shards = sorted(glob.glob(src_glob))
    assert shards, f'no source shards matched {src_glob}'
    dump_table = pq.read_table(dump_path, columns=DUMP_COLS + IDENTITY_COLS)
    src_rows = sum(pq.read_metadata(shard).num_rows for shard in shards)
    assert dump_table.num_rows == src_rows, \
        f'row mismatch: dump {dump_table.num_rows} vs shards {src_rows}'
    dump_keys = identity_keys(dump_table)

    emitted = 0
    for shard in shards:
        src = pq.read_table(shard, columns=SRC_COLS + IDENTITY_COLS)
        n = src.num_rows
        if max_events is not None:
            n = min(n, max_events - emitted)
            src = src.slice(0, n)
        if n == 0:
            break
        order = dump_row_order(dump_keys, identity_keys(src))
        block = dump_table.take(order)
        couples = block['stage3_sorted_couples'].to_pylist()
        src_cols = {name: src[name].to_pylist() for name in SRC_COLS}
        assert_dump_aligned(couples, src_cols['event_n_tracks'])
        dump_views = {name: _list_views(block, name) for name in DUMP_COLS
                      if name != 'stage3_sorted_couples'}
        identity_rows = {name: src[name].to_pylist() for name in IDENTITY_COLS}
        yield dump_views, couples, src_cols, identity_rows, n
        emitted += n
        if max_events is not None and emitted >= max_events:
            break


def event_row(r, dump_views, couples, src_cols, identity_rows, *, top_c, model,
              window, tail_sample, seed):
    candidates = _event_candidates(r, couples, src_cols, top_c)
    is_gt = np.asarray(candidates['is_gt'], dtype=bool)
    n_tierh = len(is_gt)
    assert n_tierh <= MAX_CANDIDATES_PER_EVENT, \
        f'{n_tierh} candidates at row {r}'
    n_tracks = int(src_cols['event_n_tracks'][r])

    features = _featurize(candidates, np.arange(n_tierh), r, src_cols, True,
                          H6_WIDTH)
    scores = (model.predict_proba(features)[:, 1].astype(np.float32)
              if n_tierh else np.zeros(0, np.float32))
    rows, kinds, n_tail_total = _window_selection(
        scores, is_gt, window=window, tail_sample=tail_sample,
        generator=np.random.default_rng(seed))

    triplets = candidates['triplets'].numpy()
    couple_row = candidates['couple_row'].numpy()

    # Frozen-cascade context: stage-1 scores scattered back to track order
    # (full coverage), stage-2 scores NaN outside the stage-1 top-K1, and the
    # kept couples' stage-3 scores indexed by couple_rank.
    track_s1 = np.full(n_tracks, np.nan, dtype=np.float32)
    track_s1[np.asarray(dump_views['stage1_sorted_indices'][r], dtype=np.int64)] = \
        np.asarray(dump_views['stage1_scores'][r], dtype=np.float32)
    assert np.isfinite(track_s1).all(), f'incomplete stage-1 coverage at row {r}'
    track_s2 = np.full(n_tracks, np.nan, dtype=np.float32)
    track_s2[np.asarray(dump_views['stage2_sorted_indices'][r], dtype=np.int64)] = \
        np.asarray(dump_views['stage2_scores'][r], dtype=np.float32)
    couple_scores = np.asarray(dump_views['stage3_couple_scores'][r],
                               dtype=np.float32)[:top_c]
    if n_tierh:
        assert np.isfinite(track_s2[triplets[:, 0]]).all() \
            and np.isfinite(track_s2[triplets[:, 1]]).all(), \
            f'couple member outside the stage-2 set at row {r}'
        assert int(couple_row.max()) < len(couple_scores), \
            f'couple_rank exceeds kept couple scores at row {r}'

    labels = np.asarray(src_cols['track_label_from_tau'][r])
    gt = np.where(labels > 0.5)[0]
    reconstructable = bool(candidates['reconstructable'])
    gt_i, gt_j, gt_k = (sorted(gt.tolist()) if reconstructable
                        else (-1, -1, -1))

    row = {
        'n_tracks': n_tracks,
        'n_tierh': n_tierh,
        'n_tail_total': n_tail_total,
        'cand_i': triplets[rows, 0].astype(np.int16).tolist(),
        'cand_j': triplets[rows, 1].astype(np.int16).tolist(),
        'cand_k': triplets[rows, 2].astype(np.int16).tolist(),
        'couple_rank': couple_row[rows].astype(np.int16).tolist(),
        'filter_score': scores[rows].tolist(),
        'is_gt': is_gt[rows].tolist(),
        'row_kind': kinds.tolist(),
        'gt_i': gt_i, 'gt_j': gt_j, 'gt_k': gt_k,
        'recon': reconstructable,
        'track_s1': track_s1.tolist(),
        'track_s2': track_s2.tolist(),
        'couple_scores': couple_scores.tolist(),
    }
    for name in IDENTITY_COLS:
        row[name] = int(identity_rows[name][r])
    return row


def _empty_row(src_cols, identity_rows, r):
    try:
        n_tracks = int(src_cols['event_n_tracks'][r])
    except Exception:
        n_tracks = 0
    row = {'n_tracks': n_tracks, 'n_tierh': 0, 'n_tail_total': 0,
           'cand_i': [], 'cand_j': [], 'cand_k': [], 'couple_rank': [],
           'filter_score': [], 'is_gt': [], 'row_kind': [],
           'gt_i': -1, 'gt_j': -1, 'gt_k': -1, 'recon': False,
           'track_s1': [], 'track_s2': [], 'couple_scores': []}
    for name in IDENTITY_COLS:
        try:
            row[name] = int(identity_rows[name][r])
        except Exception:
            row[name] = -1
    return row


def _worker_row(r):
    state = _WORKER_STATE
    try:
        return event_row(
            r, state['dump_views'], state['couples'], state['src_cols'],
            state['identity_rows'], top_c=state['top_c'], model=state['model'],
            window=state['window'], tail_sample=state['tail_sample'],
            seed=state['seed'] + state['offset'] + r), True
    except Exception:
        print(f'event {state["offset"] + r} failed, writing empty row:\n'
              f'{traceback.format_exc()}', flush=True)
        return _empty_row(state['src_cols'], state['identity_rows'], r), False


def _valid_rows(path):
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return -1


def _write_rows(path, rows):
    columns = {field.name: [row[field.name] for row in rows]
               for field in CANDIDATE_SCHEMA}
    writer = pq.ParquetWriter(path, CANDIDATE_SCHEMA, compression='zstd')
    writer.write_table(pa.table(columns, schema=CANDIDATE_SCHEMA))
    writer.close()


def _chunk_grid_range(offset, n_events, chunk_size, out_path):
    for start in range(offset, offset + n_events, chunk_size):
        size = min(chunk_size, offset + n_events - start)
        yield start, size, f'{out_path}.chunk{start:06d}'


def _finalize_if_complete(out_path, n_events):
    if _valid_rows(out_path) == n_events:
        for path in sorted(glob.glob(out_path + '.chunk*')):
            os.remove(path)
        print(f'{out_path} already complete ({n_events} rows)')
        return True
    return False


def _assemble(out_path, chunk_specs):
    missing = [path for _, expected, path in chunk_specs
               if _valid_rows(path) != expected]
    if missing:
        raise RuntimeError(f'{len(missing)} incomplete chunks, e.g. {missing[:3]}')
    writer = pq.ParquetWriter(out_path, CANDIDATE_SCHEMA, compression='zstd')
    for _, _, path in chunk_specs:
        writer.write_table(pq.read_table(path))
        # Delete as we go: peak disk stays at artifact + one chunk instead of
        # 2x artifact. A crash mid-assembly recomputes the deleted chunks on
        # rerun (their shards fail the completeness check and rebuild).
        os.remove(path)
    writer.close()


def _git_sha():
    try:
        return subprocess.run(
            ['git', '-C', os.path.dirname(__file__), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(args):
    out_path = os.path.join(args.out_dir, f'candidates_{args.tag}.parquet')
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_filter_model(args.filter_model)
    window = None if args.role == 'eval' else args.window
    tail_sample = 0 if args.role == 'eval' else args.tail_sample

    gt_scores = []
    chunk_specs, offset, n_failed = [], 0, 0
    progress = None
    for dump_views, couples, src_cols, identity_rows, n in _dump_blocks(
            args.dump, args.src_glob, args.max_events):
        if progress is None:
            progress = tqdm(desc=f'candidates[{args.tag}]', mininterval=30)
        # A killed run resumes at shard granularity: a shard whose chunk files
        # are all complete is never refeaturized.
        shard_chunks = list(_chunk_grid_range(offset, n, args.chunk_size,
                                              out_path))
        chunk_specs.extend(shard_chunks)
        if all(_valid_rows(path) == size for _, size, path in shard_chunks):
            progress.update(n)
            offset += n
            for _, _, path in shard_chunks:
                block = pq.read_table(path, columns=['filter_score', 'is_gt'])
                for scores, flags in zip(block['filter_score'].to_pylist(),
                                         block['is_gt'].to_pylist()):
                    gt_scores.extend(s for s, f in zip(scores, flags) if f)
            continue
        state = dict(dump_views=dump_views, couples=couples, src_cols=src_cols,
                     identity_rows=identity_rows, top_c=args.top_c, model=model,
                     window=window, tail_sample=tail_sample, seed=args.seed,
                     offset=offset)
        if args.workers > 1:
            with _worker_pool(args.workers, state) as pool:
                results = list(pool.imap(_worker_row, range(n), chunksize=8))
        else:
            _WORKER_STATE.clear()
            _WORKER_STATE.update(state)
            results = [_worker_row(r) for r in range(n)]
        for row, ok in results:
            n_failed += 0 if ok else 1
            gt_scores.extend(score for score, flag
                             in zip(row['filter_score'], row['is_gt']) if flag)
        rows = [row for row, _ in results]
        for start, size, path in shard_chunks:
            _write_rows(path, rows[start - offset:start - offset + size])
        progress.update(n)
        offset += n
    if progress is not None:
        progress.close()
    total = offset

    _assemble(out_path, chunk_specs)
    if n_failed:
        print(f'WARNING: {n_failed}/{total} events failed and were written empty')
    print(f'wrote {out_path} ({total} rows)')

    filter_path = sorted(glob.glob(args.filter_model))[0]
    manifest = dict(
        git_sha=_git_sha(), role=args.role, dump=os.path.abspath(args.dump),
        src_glob=args.src_glob, n_events=total, n_failed=n_failed,
        filter_model=os.path.abspath(filter_path),
        filter_sha256=_file_sha256(filter_path),
        top_c=args.top_c, window=window, tail_sample=tail_sample,
        h6_width=H6_WIDTH, seed=args.seed,
    )
    manifest_path = os.path.join(args.out_dir, f'build_manifest_{args.tag}.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print(f'wrote {manifest_path}')

    if args.role == 'train':
        scores = np.asarray(gt_scores, dtype=np.float64)
        points = dict(score_column='filter_score',
                      taus={'p99': float(np.quantile(scores, 0.01)),
                            'p95': float(np.quantile(scores, 0.05))},
                      n_gt=int(len(scores)), derived_from=args.tag)
        points_path = os.path.join(args.out_dir, 'operating_points.json')
        with open(points_path, 'w') as handle:
            json.dump(points, handle, indent=2)
        print(f'wrote {points_path}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', choices=sorted(ROLE_DEFAULTS), required=True,
                    help='train stores a top-window + GT + tail sample; eval '
                         'stores the full Tier-H list, gate applied at runtime')
    ap.add_argument('--dump', default=None,
                    help='per-stage couples dump (default: by role)')
    ap.add_argument('--src-glob', default=None,
                    help='per-track source parquet glob (default: by role)')
    ap.add_argument('--tag', default=None, help='artifact suffix (default: role)')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--filter-model', default=FILTER_MODEL_GLOB)
    ap.add_argument('--top-c', type=int, default=125)
    ap.add_argument('--window', type=int, default=WINDOW)
    ap.add_argument('--tail-sample', type=int, default=TAIL_SAMPLE)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    role_dump, role_src = ROLE_DEFAULTS[args.role]
    args.dump = args.dump or role_dump
    args.src_glob = args.src_glob or role_src
    args.tag = args.tag or args.role

    torch.set_num_threads(1)
    out_path = os.path.join(args.out_dir, f'candidates_{args.tag}.parquet')
    expected = args.max_events
    if expected is not None and _finalize_if_complete(out_path, expected):
        return
    build(args)


if __name__ == '__main__':
    main()
