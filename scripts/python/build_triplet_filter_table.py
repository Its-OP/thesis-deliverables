from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from utils.triplet_join import (
    FEATURE_NAMES,
    FEATURE_NAMES_EXTENDED,
    M_TAU_GEV,
    build_track_lorentz,
    build_triplet_candidates,
    triplet_feature_columns,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "low-pt")
DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dumps")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
# Train and eval come from disjoint shard sets, so events never cross sides.
ROLE_DEFAULTS = {
    "train": (os.path.join(DUMP_DIR, "perstage_couples_train.parquet"),
              os.path.join(DATA_DIR, "train", "*.parquet")),
    "eval": (os.path.join(DUMP_DIR, "perstage_couples_eval.parquet"),
             os.path.join(DATA_DIR, "eval", "*.parquet")),
}
POOL = "P2"
# Back-compatible aliases for the inference-side builder, which defaults to the
# eval role.
DUMP, SRC = ROLE_DEFAULTS["eval"]

TRACK_SRC_COLS = ["event_n_tracks", "track_pt", "track_eta", "track_phi", "track_charge",
                  "track_dz_significance", "track_dxy_significance", "track_dca_significance",
                  "track_n_valid_pixel_hits", "track_norm_chi2", "track_pt_error",
                  "track_covariance_phi_phi", "track_covariance_lambda_lambda",
                  "track_label_from_tau"]
# H6 raw inputs: vertex geometry, the raw longitudinal impact parameter, the
# primary vertex, the secondary-vertex collection and the sub-cutoff companions.
H6_SRC_COLS = ["track_vertex_x", "track_vertex_y", "track_vertex_z", "track_dz",
               "event_primary_vertex_x", "event_primary_vertex_y",
               "sv_x", "sv_y", "sv_z", "sv_dlen_sig", "sv_mass",
               "other_track_pt", "other_track_eta", "other_track_phi", "other_track_dz"]
SRC_COLS = TRACK_SRC_COLS + H6_SRC_COLS


def _h6_inputs_for_event(cols, r):
    """cols: per-column lists over events. Returns the h6_inputs dict for event r."""
    def track_tensor(key):
        return torch.tensor(cols[key][r], dtype=torch.float32)

    return dict(
        vertex_x=track_tensor("track_vertex_x"),
        vertex_y=track_tensor("track_vertex_y"),
        vertex_z=track_tensor("track_vertex_z"),
        dz_raw=track_tensor("track_dz"),
        primary_vertex_x=torch.tensor(cols["event_primary_vertex_x"][r], dtype=torch.float32),
        primary_vertex_y=torch.tensor(cols["event_primary_vertex_y"][r], dtype=torch.float32),
        sv_x=track_tensor("sv_x"),
        sv_y=track_tensor("sv_y"),
        sv_z=track_tensor("sv_z"),
        sv_dlen_sig=track_tensor("sv_dlen_sig"),
        sv_mass=track_tensor("sv_mass"),
        other_pt=track_tensor("other_track_pt"),
        other_eta=track_tensor("other_track_eta"),
        other_phi=track_tensor("other_track_phi"),
        other_dz=track_tensor("other_track_dz"),
    )


def _select_hard_negative_rows(is_gt, scores, hard_top, random_count, gen):
    """is_gt, scores: (M,) over Tier-H survivors. Returns the top-scoring
    negatives plus a uniform draw from the rest — the distribution the current
    filter actually confuses, rather than the easy bulk."""
    negatives = np.flatnonzero(~is_gt)
    if len(negatives) == 0:
        return negatives
    ranked = negatives[np.argsort(-scores[negatives], kind="stable")]
    hard = ranked[:hard_top]
    remainder = ranked[hard_top:]
    take = min(random_count, len(remainder))
    sampled = remainder[gen.choice(len(remainder), take, replace=False)] if take else remainder[:0]
    return np.concatenate([hard, sampled])


def _sample_negative_rows(is_gt, couple_row, neg_per_event, neg_mode, gen):
    """is_gt, couple_row: (M,) over Tier-H survivors. Returns the sampled
    negative row indices."""
    negatives = np.flatnonzero(~is_gt)
    if neg_mode == "uniform":
        take = min(neg_per_event, len(negatives))
        return negatives[gen.choice(len(negatives), take, replace=False)] if take else negatives
    if neg_mode != "per_couple":
        raise ValueError(f"unknown neg_mode {neg_mode!r}")
    # Round-robin over the couples that produced the candidates: every couple
    # contributes one negative before any contributes a second, so the hard
    # same-couple confusions are represented rather than drowned out by the
    # couples that happen to survive most often.
    order = gen.permutation(negatives)
    per_couple = {}
    for row in order:
        per_couple.setdefault(int(couple_row[row]), []).append(int(row))
    chosen = []
    groups = [per_couple[key] for key in sorted(per_couple)]
    depth = 0
    while len(chosen) < neg_per_event and any(len(g) > depth for g in groups):
        for group in groups:
            if len(group) > depth:
                chosen.append(group[depth])
                if len(chosen) == neg_per_event:
                    break
        depth += 1
    return np.asarray(chosen, dtype=np.int64)


IDENTITY_COLS = ["event_run", "event_id", "event_luminosity_block",
                 "source_batch_id", "source_microbatch_id"]


def identity_keys(table):
    """table: arrow table carrying IDENTITY_COLS. Returns the per-row key tuples.
    (run, id, lumi) alone collides heavily — 1,019 distinct values over 67,500
    events — but the two source_* columns travel with the data from the
    original parquets and make the 5-tuple unique."""
    columns = [np.asarray(table.column(name)).tolist() for name in IDENTITY_COLS]
    return list(zip(*columns))


def dump_row_order(dump_keys, source_keys):
    """Returns, for each source row in order, the dump row describing the same
    event. Dumps written with several loader workers are permuted relative to
    the shards, and this is what puts them back."""
    position = {}
    for index, key in enumerate(dump_keys):
        if key in position:
            raise ValueError(
                f"duplicate identity key in the dump: {key}. The 5-column key "
                "must be unique for the reorder to be well defined.")
        position[key] = index
    missing = [key for key in source_keys if key not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} source events are absent from the dump, e.g. "
            f"{missing[0]}. The dump and the shards describe different sets.")
    return np.array([position[key] for key in source_keys], dtype=np.int64)


def assert_dump_aligned(couples, n_tracks, sample=512):
    """Couples carry original track indices, so an index at or beyond the
    event's track count proves the dump row and the source row describe
    different events. Dumps written with several loader workers interleave
    shards and are silently misordered, which this catches at the first shard
    instead of thousands of events later."""
    checked = 0
    for row in range(min(sample, len(couples))):
        event_couples = couples[row]
        if not event_couples:
            continue
        highest = max(max(pair) for pair in event_couples)
        if highest >= n_tracks[row]:
            raise ValueError(
                f"dump is not aligned with the source shards: row {row} "
                f"references track {highest} in an event with "
                f"{n_tracks[row]} tracks. Regenerate the per-stage dump with "
                f"--num-workers 0; several loader workers interleave shards "
                f"and the (run, id, lumi) key is not unique enough to "
                f"restore the order.")
        checked += 1
    return checked


def _shard_blocks(dump_path, src_glob, max_events):
    """Yields (dump_couples, src_cols, n_events) one source shard at a time so
    peak memory stays at one shard rather than the whole dataset."""
    shards = sorted(glob.glob(src_glob))
    assert shards, f"no source shards matched {src_glob}"
    dump_file = pq.ParquetFile(dump_path)
    dump_rows = dump_file.metadata.num_rows
    src_rows = sum(pq.read_metadata(shard).num_rows for shard in shards)
    assert dump_rows == src_rows, f"row mismatch {dump_rows} vs {src_rows}"

    # The dump stays in Arrow (a few hundred MB); each shard takes only the
    # rows describing its own events, matched on the 5-column identity key
    # rather than on position.
    dump_table = pq.read_table(
        dump_path, columns=IDENTITY_COLS + ["stage3_sorted_couples"])
    dump_keys = identity_keys(dump_table)

    emitted = 0
    for shard in shards:
        src = pq.read_table(shard, columns=SRC_COLS + IDENTITY_COLS)
        n = src.num_rows
        if max_events is not None:
            n = min(n, max_events - emitted)
            src = src.slice(0, n)
        src_cols = {name: src[name].to_pylist() for name in SRC_COLS}
        order = dump_row_order(dump_keys, identity_keys(src))
        couples = dump_table.take(order)["stage3_sorted_couples"].to_pylist()
        assert_dump_aligned(couples, src_cols["event_n_tracks"])
        yield couples, src_cols, n
        emitted += n
        if max_events is not None and emitted >= max_events:
            break


def _event_candidates(r, couples_all, cols, top_c):
    """Tier-H enumeration only. Features are computed later for the sampled
    rows alone — the H6 cone block costs O(candidates x tracks), so
    featurizing the whole survivor list would dominate the build."""
    n_tracks = cols["event_n_tracks"][r]
    lorentz = build_track_lorentz(torch.tensor(cols["track_pt"][r], dtype=torch.float32),
                                  torch.tensor(cols["track_eta"][r], dtype=torch.float32),
                                  torch.tensor(cols["track_phi"][r], dtype=torch.float32))
    t = lambda key: torch.tensor(cols[key][r], dtype=torch.float32)
    kw = dict(lorentz=lorentz, charge=t("track_charge"), eta=t("track_eta"), phi=t("track_phi"),
              dz=t("track_dz_significance"), dxy_sig=t("track_dxy_significance"),
              dca_sig=t("track_dca_significance"), n_pixel=t("track_n_valid_pixel_hits"),
              norm_chi2=t("track_norm_chi2"), pt_error=t("track_pt_error"),
              cov_phi_phi=t("track_covariance_phi_phi"),
              cov_lambda_lambda=t("track_covariance_lambda_lambda"))
    labels = np.asarray(cols["track_label_from_tau"][r])
    gt = np.where(labels > 0.5)[0]
    couples_np = np.asarray(couples_all[r][:top_c], dtype=np.int64).reshape(-1, 2)
    couples = torch.tensor(couples_np, dtype=torch.long)
    gt_set = set(gt.tolist())
    has_gt_couple = any(set(c).issubset(gt_set) for c in couples_np.tolist())
    pool = torch.arange(n_tracks, dtype=torch.long)  # P2: entire input track set
    reconstructable = gt.size == 3 and gt_set.issubset(set(pool.tolist())) and has_gt_couple
    members = sum((int(c[0]) < n_tracks) + (int(c[1]) < n_tracks) for c in couples_np.tolist())
    n_full = couples.shape[0] * pool.shape[0] - members

    triplets, couple_row = build_triplet_candidates(
        couples, pool, lorentz=lorentz, charge=kw["charge"], eta=kw["eta"],
        phi=kw["phi"], dz=kw["dz"], charge_gate=True, mass_max=M_TAU_GEV)
    if reconstructable:
        target = torch.tensor(sorted(gt.tolist()))
        is_gt = (triplets.sort(dim=1).values == target).all(dim=1)
    else:
        is_gt = torch.zeros(triplets.shape[0], dtype=torch.bool)
    return dict(triplets=triplets, couple_row=couple_row, is_gt=is_gt.numpy(),
                kw=kw, reconstructable=reconstructable, n_full=n_full)


def _featurize(candidates, rows, r, cols, with_h6):
    """rows: row indices into the candidate list. Returns (len(rows), F)."""
    if len(rows) == 0:
        width = len(FEATURE_NAMES_EXTENDED) if with_h6 else len(FEATURE_NAMES)
        return np.zeros((0, width), dtype=np.float32)
    selected = torch.as_tensor(np.asarray(rows), dtype=torch.long)
    triplets = candidates["triplets"][selected]
    h6_inputs = _h6_inputs_for_event(cols, r) if with_h6 else None
    columns = triplet_feature_columns(
        triplets[:, 0], triplets[:, 1], triplets[:, 2],
        candidates["couple_row"][selected], h6_inputs=h6_inputs,
        **candidates["kw"])
    return columns.numpy()


def _to_table(rows, extra, with_h6):
    names = FEATURE_NAMES_EXTENDED if with_h6 else FEATURE_NAMES
    arrays = {names[c]: pa.array(rows[:, c]) for c in range(len(names))}
    arrays.update(extra)
    return pa.table(arrays)


_WORKER_STATE = {}


def _worker_init(state):
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _worker_pool(workers, state):
    """A shard's columns are ~1.5 GB of Python lists. Passing them through
    initargs pickles that to every worker, which costs more than the work
    itself; under fork the children inherit them for free instead."""
    if sys.platform.startswith("linux"):
        _worker_init(state)
        return mp.get_context("fork").Pool(workers)
    return mp.get_context().Pool(
        workers, initializer=_worker_init, initargs=(state,))


def _build_train_event(r):
    state = _WORKER_STATE
    # Per-event seeding keeps the sampling reproducible and independent of how
    # the work is distributed over workers.
    gen = np.random.default_rng((state["seed"], r))
    return _train_rows_for_event(
        r, state["couples"], state["cols"], state["top_c"],
        state["neg_per_event"], state["neg_mode"], gen, state["with_h6"],
        state["hard_model"], state["hard_top"])


def _train_rows_for_event(r, couples, cols, top_c, neg_per_event, neg_mode,
                          gen, with_h6, hard_model, hard_top):
    candidates = _event_candidates(r, couples, cols, top_c)
    is_gt = candidates["is_gt"]
    positives = np.flatnonzero(is_gt)
    if neg_mode == "hard":
        # Legacy features are cheap to compute for the whole survivor list;
        # only the selected rows pay for the H6 block.
        legacy = _featurize(candidates, np.arange(len(is_gt)), r, cols, False)
        scores = hard_model.predict_proba(legacy)[:, 1]
        negatives = _select_hard_negative_rows(
            is_gt, scores, hard_top, neg_per_event - hard_top, gen)
    else:
        negatives = _sample_negative_rows(
            is_gt, candidates["couple_row"].numpy(), neg_per_event, neg_mode, gen)
    blocks = []
    for rows, label in ((positives, 1), (negatives, 0)):
        if len(rows):
            blocks.append((_featurize(candidates, rows, r, cols, with_h6),
                           np.full(len(rows), label, np.int8)))
    return blocks


def build_train(blocks, top_c, neg_per_event, neg_mode, gen, out_path, with_h6,
                hard_model=None, hard_top=30, workers=0, seed=0):
    train_rows, train_label, train_event, n_events = [], [], [], 0
    for couples, cols, n in blocks:
        if workers > 1:
            state = dict(couples=couples, cols=cols, top_c=top_c,
                         neg_per_event=neg_per_event, neg_mode=neg_mode,
                         with_h6=with_h6, hard_model=hard_model,
                         hard_top=hard_top, seed=seed)
            with _worker_pool(workers, state) as pool:
                per_event = list(tqdm(
                    pool.imap(_build_train_event, range(n), chunksize=16),
                    total=n, desc=f"train x{workers} (+{n})"))
        else:
            per_event = [
                _train_rows_for_event(
                    r, couples, cols, top_c, neg_per_event, neg_mode,
                    np.random.default_rng((seed, r)), with_h6, hard_model,
                    hard_top)
                for r in tqdm(range(n), desc=f"train (+{n})")]
        for r, event_blocks in enumerate(per_event):
            for features, labels in event_blocks:
                train_rows.append(features)
                train_label.append(labels)
                # Event key: grouped CV and per-event ranking metrics both
                # need rows from one event to stay together.
                train_event.append(np.full(len(labels), n_events + r, np.int32))
        n_events += n
    rows = np.concatenate(train_rows)
    tbl = _to_table(rows, {"is_gt": pa.array(np.concatenate(train_label)),
                           "event_index": pa.array(np.concatenate(train_event)),
                           "pool": pa.array(np.full(rows.shape[0], POOL))}, with_h6)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # zstd over snappy: these tables are almost entirely float columns and the
    # 300k-event builds are large enough for the difference to matter on disk.
    pq.write_table(tbl, out_path, compression="zstd")
    print(f"wrote {out_path} ({tbl.num_rows} rows from {n_events} events)")


def _eval_rows_for_event(r, couples, cols, top_c, subsample, gen, with_h6):
    candidates = _event_candidates(r, couples, cols, top_c)
    is_gt = candidates["is_gt"]
    gt_row = None
    if candidates["reconstructable"] and is_gt.any():
        gt_row = _featurize(
            candidates, np.flatnonzero(is_gt)[:1], r, cols, with_h6)[0]
    n_h = len(is_gt)
    sub_block, weight = None, None
    if n_h:
        take = min(subsample, n_h)
        idx = gen.choice(n_h, take, replace=False)
        sub_block = _featurize(candidates, idx, r, cols, with_h6)
        weight = np.full(take, n_h / take)
    return dict(gt_row=gt_row, sub_block=sub_block, weight=weight,
                reconstructable=candidates["reconstructable"],
                n_full=candidates["n_full"])


def _build_eval_event(r):
    state = _WORKER_STATE
    gen = np.random.default_rng((state["seed"], r))
    return _eval_rows_for_event(
        r, state["couples"], state["cols"], state["top_c"],
        state["subsample"], gen, state["with_h6"])


def build_eval(blocks, top_c, subsample, gen, out_path, with_h6, workers=0,
               seed=0):
    gt_rows, gt_pool, sub_rows, sub_w, sub_pool = [], [], [], [], []
    meta = {POOL: dict(recon=0, n_full=0, n_events=0)}
    for couples, cols, n in blocks:
        if workers > 1:
            state = dict(couples=couples, cols=cols, top_c=top_c,
                         subsample=subsample, with_h6=with_h6, seed=seed)
            with _worker_pool(workers, state) as pool:
                per_event = list(tqdm(
                    pool.imap(_build_eval_event, range(n), chunksize=16),
                    total=n, desc=f"eval x{workers} (+{n})"))
        else:
            per_event = [
                _eval_rows_for_event(r, couples, cols, top_c, subsample,
                                     np.random.default_rng((seed, r)), with_h6)
                for r in tqdm(range(n), desc=f"eval (+{n})")]
        for event in per_event:
            if event["reconstructable"]:
                meta[POOL]["recon"] += 1
            meta[POOL]["n_full"] += event["n_full"]
            if event["gt_row"] is not None:
                gt_rows.append(event["gt_row"])
                gt_pool.append(POOL)
            if event["sub_block"] is not None:
                sub_rows.append(event["sub_block"])
                sub_w.append(event["weight"])
                sub_pool.append(np.full(len(event["weight"]), POOL))
        meta[POOL]["n_events"] += n

    gt_tbl = _to_table(np.stack(gt_rows), {"pool": pa.array(gt_pool)}, with_h6)
    sub_tbl = _to_table(np.concatenate(sub_rows),
                        {"weight": pa.array(np.concatenate(sub_w)),
                         "pool": pa.array(np.concatenate(sub_pool))}, with_h6)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(gt_tbl, out_path.replace(".parquet", "_gt.parquet"),
                   compression="zstd")
    pq.write_table(sub_tbl, out_path.replace(".parquet", "_sub.parquet"),
                   compression="zstd")
    with open(out_path.replace(".parquet", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out_path.replace('.parquet', '_{gt,sub}.parquet')} + _meta.json "
          f"(gt {gt_tbl.num_rows}, sub {sub_tbl.num_rows})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=sorted(ROLE_DEFAULTS), required=True,
                    help="train builds the classifier table; eval builds the "
                         "gt/sub frontier tables from the held-out shards")
    ap.add_argument("--dump", default=None, help="per-stage couples dump (default: by role)")
    ap.add_argument("--src-glob", default=None, help="per-track source parquet glob (default: by role)")
    ap.add_argument("--top-c", type=int, default=100)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--neg-per-event", type=int, default=80)
    ap.add_argument("--neg-mode", choices=("uniform", "per_couple", "hard"), default="uniform")
    ap.add_argument("--hard-model", default=os.path.join(
        os.path.dirname(__file__), "..", "..", "models",
        "third_pion_filter_gbdt_full_P2.joblib"),
        help="miner for --neg-mode hard: scores the survivor list on the "
             "legacy 89 features and the top ones become negatives")
    ap.add_argument("--hard-top", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0,
                    help="processes over events within a shard (0/1 = serial)")
    ap.add_argument("--subsample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-h6", action="store_true", help="emit the legacy 89 columns only")
    ap.add_argument("--out-dir", default=EVAL_DIR)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    default_dump, default_src = ROLE_DEFAULTS[args.role]
    dump_path = args.dump or default_dump
    src_glob = args.src_glob or default_src
    with_h6 = not args.no_h6
    gen = np.random.default_rng(args.seed)
    blocks = _shard_blocks(dump_path, src_glob, args.max_events)
    print(f"role={args.role} dump={dump_path} src={src_glob} "
          f"neg_mode={args.neg_mode} h6={with_h6}")

    if args.role == "train":
        out_name = args.out_name or f"triplet_filter_train_{args.neg_mode}.parquet"
        hard_model = joblib.load(args.hard_model) if args.neg_mode == "hard" else None
        build_train(blocks, args.top_c, args.neg_per_event, args.neg_mode, gen,
                    os.path.join(args.out_dir, out_name), with_h6,
                    hard_model=hard_model, hard_top=args.hard_top,
                    workers=args.workers, seed=args.seed)
    else:
        out_name = args.out_name or "triplet_filter_eval.parquet"
        build_eval(blocks, args.top_c, args.subsample, gen,
                   os.path.join(args.out_dir, out_name), with_h6,
                   workers=args.workers, seed=args.seed)


if __name__ == "__main__":
    main()
