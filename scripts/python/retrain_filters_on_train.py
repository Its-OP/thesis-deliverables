from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

from utils.triplet_join import FEATURE_NAMES
from utils.triplet_rank_data import _EventTable, triplet_feature_columns
from utils.triplet_split import write_split

MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models")
FLOORS = [0.99, 0.97, 0.95]


def _models():
    return {
        "gbdt": HistGradientBoostingClassifier(max_leaf_nodes=31, max_depth=6, learning_rate=0.1,
                                               max_iter=200, class_weight="balanced", random_state=0),
        "gbdt8": HistGradientBoostingClassifier(max_leaf_nodes=63, max_depth=8, learning_rate=0.1,
                                                max_iter=300, class_weight="balanced", random_state=0),
    }


def _feature_rows(table, r, sel):
    kw = table.track_kw(r)
    arr = table.candidate_arrays(r)
    i = torch.tensor(arr["cand_i"][sel], dtype=torch.long)
    j = torch.tensor(arr["cand_j"][sel], dtype=torch.long)
    k = torch.tensor(arr["cand_k"][sel], dtype=torch.long)
    rank = torch.tensor(arr["couple_rank"][sel], dtype=torch.long)
    return triplet_feature_columns(i, j, k, rank, **kw).numpy()


def build_train(table, event_idx, neg_per_event, gen):
    rows, labels = [], []
    for r in tqdm(event_idx, desc="train-table"):
        r = int(r)
        is_gt = table.candidate_arrays(r)["is_gt"].astype(bool)
        pos = np.where(is_gt)[0]
        neg = np.where(~is_gt)[0]
        take = min(neg_per_event, len(neg))
        sel_neg = gen.choice(neg, take, replace=False) if take else neg
        sel = np.concatenate([pos, sel_neg]).astype(np.int64)
        if not len(sel):
            continue
        rows.append(_feature_rows(table, r, sel))
        labels.append(is_gt[sel].astype(np.int8))
    return np.concatenate(rows), np.concatenate(labels)


def build_eval(table, event_idx, subsample, gen):
    gt_rows, sub_rows, sub_w = [], [], []
    recon, n_total = 0, 0
    for r in tqdm(event_idx, desc="eval-table"):
        r = int(r)
        is_gt = table.candidate_arrays(r)["is_gt"].astype(bool)
        n_h = len(is_gt)
        if is_gt.any():
            recon += 1
            gt_rows.append(_feature_rows(table, r, np.where(is_gt)[0][:1])[0])
        n_total += n_h
        if n_h:
            take = min(subsample, n_h)
            idx = gen.choice(n_h, take, replace=False)
            sub_rows.append(_feature_rows(table, r, idx))
            sub_w.append(np.full(take, n_h / take))
    return (np.stack(gt_rows), np.concatenate(sub_rows), np.concatenate(sub_w), recon, n_total)


def _factor_at_floor(gt_scores, sub_scores, sub_w, recon, n_total, floor):
    taus = np.unique(np.quantile(sub_scores, np.linspace(0, 1, 400)))
    best_comp = None
    for tau in taus:
        recall = float((gt_scores >= tau).sum()) / recon
        comp = float((sub_w * (sub_scores >= tau)).sum()) / n_total
        if recall >= floor and comp > 0:
            best_comp = comp if best_comp is None else min(best_comp, comp)
    return (1.0 / best_comp) if best_comp else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/low-pt/eval/triplet_rank/candidates_train.parquet")
    ap.add_argument("--tracks", default="data/low-pt/eval/triplet_rank/tracks_train.parquet")
    ap.add_argument("--neg-per-event", type=int, default=80)
    ap.add_argument("--subsample", type=int, default=100)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-json", default="data/low-pt/eval/triplet_filter_train_split.json")
    ap.add_argument("--out-json", default="reports/filter_retrain_train_vs_val.json")
    ap.add_argument("--save-models", action="store_true",
                    help="overwrite the production d6/d8 joblibs with the TRAIN-trained ones "
                         "(backs up originals to models/backup_val_trained/)")
    args = ap.parse_args()

    table = _EventTable(args.candidates, args.tracks)
    train_idx, test_idx = write_split(args.split_json, table.num_rows, frac_train=args.frac_train, seed=args.seed)
    print(f"split: {table.num_rows} events -> {len(train_idx)} train / {len(test_idx)} test")

    gen = np.random.default_rng(args.seed)
    Xtr, ytr = build_train(table, train_idx, args.neg_per_event, gen)
    Xgt, Xsub, sub_w, recon, n_total = build_eval(table, test_idx, args.subsample, gen)
    print(f"train rows {len(ytr)} (pos {int(ytr.sum())}); eval recon {recon}, n_total {n_total}, "
          f"gt {len(Xgt)}, sub {len(Xsub)}")

    # old (VAL-trained) models, loaded read-only; new = retrained on TRAIN.
    old = {"gbdt": joblib.load(os.path.join(MODELS, "third_pion_filter_gbdt_full_P2.joblib")),
           "gbdt8": joblib.load(os.path.join(MODELS, "third_pion_filter_gbdt8_full_P2.joblib"))}
    new = _models()
    for m in new.values():
        m.fit(Xtr, ytr)

    result = {"recon": recon, "n_total": n_total, "train_rows": int(len(ytr))}
    new_gt = {}
    print("\n| model | trained_on | " + " | ".join(f"comp@{f}" for f in FLOORS) + " |")
    print("|---|---|" + "---|" * len(FLOORS))
    for name, models, tag in [("gbdt(d6)", old, "VAL(old)"), ("gbdt(d6)", new, "TRAIN(new)"),
                              ("gbdt8(d8)", old, "VAL(old)"), ("gbdt8(d8)", new, "TRAIN(new)")]:
        key = "gbdt" if "d6" in name else "gbdt8"
        model = models[key]
        gs = model.predict_proba(Xgt)[:, 1]
        ss = model.predict_proba(Xsub)[:, 1]
        if tag == "TRAIN(new)":
            new_gt[key] = gs
        facs = [_factor_at_floor(gs, ss, sub_w, recon, n_total, f) for f in FLOORS]
        result.setdefault(name, {})[tag] = facs
        print(f"| {name} | {tag} | " + " | ".join(f"{x:.1f}x" if x else "-" for x in facs) + " |")

    # tau for each operating point = score threshold on the held-out GT keeping <floor> recall.
    tau = {"d6@0.99": float(np.quantile(new_gt["gbdt"], 0.01)),
           "d8@0.95": float(np.quantile(new_gt["gbdt8"], 0.05))}
    result["tau"] = tau
    print(f"\nnew tau (TRAIN-trained, held-out): d6@0.99={tau['d6@0.99']:.6f}  d8@0.95={tau['d8@0.95']:.6f}")

    if args.save_models:
        backup = os.path.join(MODELS, "backup_val_trained")
        os.makedirs(backup, exist_ok=True)
        for key, fname in [("gbdt", "third_pion_filter_gbdt_full_P2.joblib"),
                           ("gbdt8", "third_pion_filter_gbdt8_full_P2.joblib")]:
            src = os.path.join(MODELS, fname)
            if os.path.exists(src) and not os.path.exists(os.path.join(backup, fname)):
                import shutil
                shutil.copy2(src, os.path.join(backup, fname))
            joblib.dump(new[key], src)
            print(f"saved {fname} (TRAIN-trained); original backed up to {backup}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
