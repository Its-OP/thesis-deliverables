from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from utils.triplet_join import FEATURE_NAMES, build_track_lorentz, triplet_candidate_features

POOLS = ["P1", "P2"]
TEST_DUMP = os.path.join(os.path.dirname(__file__), "..", "..", "data", "low-pt", "eval", "stage3_dump_test.parquet")
TEST_SRC = "/Users/oleh/Downloads/test_dataset_unzipped/test_*.parquet"
VAL_DUMP = os.path.join(os.path.dirname(__file__), "..", "..", "data", "low-pt", "eval", "perstage_couples_val.parquet")
VAL_SRC = "/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet"

SRC_COLS = ["event_n_tracks", "track_pt", "track_eta", "track_phi", "track_charge",
            "track_dz_significance", "track_dxy_significance", "track_dca_significance",
            "track_n_valid_pixel_hits", "track_norm_chi2", "track_label_from_tau"]


def _load(dump_path, src_glob, max_events):
    dump = pq.read_table(dump_path, columns=["stage1_sorted_indices", "stage3_sorted_couples"])
    src = pa.concat_tables([pq.read_table(s, columns=SRC_COLS) for s in sorted(glob.glob(src_glob))])
    assert dump.num_rows == src.num_rows, f"row mismatch {dump.num_rows} vs {src.num_rows}"
    n = dump.num_rows if max_events is None else min(max_events, dump.num_rows)
    return dump.slice(0, n), src.slice(0, n), n


def _event_features(r, dump_cols, src_cols, top_c):
    s1, couples_all = dump_cols
    cols = src_cols
    assert len(s1[r]) == cols["event_n_tracks"][r]
    n_tracks = cols["event_n_tracks"][r]
    lorentz = build_track_lorentz(torch.tensor(cols["track_pt"][r], dtype=torch.float32),
                                  torch.tensor(cols["track_eta"][r], dtype=torch.float32),
                                  torch.tensor(cols["track_phi"][r], dtype=torch.float32))
    t = lambda key: torch.tensor(cols[key][r], dtype=torch.float32)
    kw = dict(lorentz=lorentz, charge=t("track_charge"), eta=t("track_eta"), phi=t("track_phi"),
              dz=t("track_dz_significance"), dxy_sig=t("track_dxy_significance"),
              dca_sig=t("track_dca_significance"), n_pixel=t("track_n_valid_pixel_hits"),
              norm_chi2=t("track_norm_chi2"))
    labels = np.asarray(cols["track_label_from_tau"][r])
    gt = np.where(labels > 0.5)[0]
    couples_np = np.asarray(couples_all[r][:top_c], dtype=np.int64).reshape(-1, 2)
    couples = torch.tensor(couples_np, dtype=torch.long)
    gt_set = set(gt.tolist())
    has_gt_couple = any(set(c).issubset(gt_set) for c in couples_np.tolist())
    pools = {"P1": torch.tensor(s1[r][:256], dtype=torch.long),
             "P2": torch.arange(n_tracks, dtype=torch.long)}
    out = {}
    for name, pool in pools.items():
        pool_set = set(pool.tolist())
        reconstructable = gt.size == 3 and gt_set.issubset(pool_set) and has_gt_couple
        gt_sorted = tuple(sorted(gt.tolist())) if reconstructable else None
        members = sum((int(c[0]) in pool_set) + (int(c[1]) in pool_set) for c in couples_np)
        n_full = couples.shape[0] * pool.shape[0] - members
        X, _, is_gt, _ = triplet_candidate_features(couples, pool, gt_sorted=gt_sorted, **kw)
        out[name] = dict(X=X.numpy(), is_gt=is_gt.numpy(), reconstructable=reconstructable, n_full=n_full)
    return out


def _to_table(rows, extra):
    arrays = {FEATURE_NAMES[c]: pa.array(rows[:, c]) for c in range(len(FEATURE_NAMES))}
    arrays.update(extra)
    return pa.table(arrays)


def build(mode, dump, src, n_events, top_c, neg_per_event, subsample, seed, out_path):
    s1 = dump["stage1_sorted_indices"].to_pylist()
    couples_all = dump["stage3_sorted_couples"].to_pylist()
    src_cols = {c: src[c].to_pylist() for c in SRC_COLS}
    gen = np.random.default_rng(seed)

    train_rows, train_label, train_pool = [], [], []
    gt_rows, gt_pool, sub_rows, sub_w, sub_pool = [], [], [], [], []
    meta = {p: dict(recon=0, n_full=0, n_events=n_events) for p in POOLS}

    for r in tqdm(range(n_events), desc=mode):
        feats = _event_features(r, (s1, couples_all), src_cols, top_c)
        for p in POOLS:
            f = feats[p]
            X, is_gt = f["X"], f["is_gt"]
            if mode == "train":
                pos = X[is_gt]
                neg_all = X[~is_gt]
                if len(neg_all):
                    take = min(neg_per_event, len(neg_all))
                    neg = neg_all[gen.choice(len(neg_all), take, replace=False)]
                else:
                    neg = neg_all
                for block, lab in ((pos, 1), (neg, 0)):
                    if len(block):
                        train_rows.append(block)
                        train_label.append(np.full(len(block), lab, np.int8))
                        train_pool.append(np.full(len(block), p))
            else:
                if f["reconstructable"]:
                    meta[p]["recon"] += 1
                meta[p]["n_full"] += f["n_full"]
                if f["reconstructable"] and is_gt.any():
                    gt_rows.append(X[is_gt][0])
                    gt_pool.append(p)
                n_h = len(X)
                if n_h:
                    take = min(subsample, n_h)
                    idx = gen.choice(n_h, take, replace=False)
                    sub_rows.append(X[idx])
                    sub_w.append(np.full(take, n_h / take))
                    sub_pool.append(np.full(take, p))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if mode == "train":
        rows = np.concatenate(train_rows)
        tbl = _to_table(rows, {"is_gt": pa.array(np.concatenate(train_label)),
                               "pool": pa.array(np.concatenate(train_pool))})
        pq.write_table(tbl, out_path)
        print(f"wrote {out_path} ({tbl.num_rows} rows)")
    else:
        gt_tbl = _to_table(np.stack(gt_rows), {"pool": pa.array(gt_pool)})
        sub_tbl = _to_table(np.concatenate(sub_rows),
                            {"weight": pa.array(np.concatenate(sub_w)), "pool": pa.array(np.concatenate(sub_pool))})
        pq.write_table(gt_tbl, out_path.replace(".parquet", "_gt.parquet"))
        pq.write_table(sub_tbl, out_path.replace(".parquet", "_sub.parquet"))
        with open(out_path.replace(".parquet", "_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"wrote {out_path.replace('.parquet', '_{gt,sub}.parquet')} + _meta.json "
              f"(gt {gt_tbl.num_rows}, sub {sub_tbl.num_rows})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--top-c", type=int, default=100)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--neg-per-event", type=int, default=80)
    ap.add_argument("--subsample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "low-pt", "eval"))
    args = ap.parse_args()

    if args.split == "val":
        dump_path, src_glob, mode, out = VAL_DUMP, VAL_SRC, "train", "triplet_filter_train.parquet"
    else:
        dump_path, src_glob, mode, out = TEST_DUMP, TEST_SRC, "eval", "triplet_filter_eval.parquet"

    dump, src, n = _load(dump_path, src_glob, args.max_events)
    build(mode, dump, src, n, args.top_c, args.neg_per_event, args.subsample, args.seed,
          os.path.join(args.out_dir, out))


if __name__ == "__main__":
    main()
