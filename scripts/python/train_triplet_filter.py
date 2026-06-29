from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.triplet_join import FEATURE_NAMES, GATE4_NAMES

POOLS = ["P1", "P2"]
FLOORS = [0.99, 0.97, 0.95]
FEATURESETS = {"rich": FEATURE_NAMES, "gate4": GATE4_NAMES}
BLUE, CRIMSON, GREEN, BLACK = "#4466CC", "#AA3355", "#229977", "#222222"
EVAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "low-pt", "eval")
REPORTS = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _cols(table, names):
    return np.column_stack([table[n].to_numpy() for n in names])


def _models():
    return {
        "tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0),
        "gbdt": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1,
                                               max_iter=200, class_weight="balanced", random_state=0),
    }


def _curve(gt_scores, sub_scores, sub_weight, recon, n_full):
    taus = np.unique(np.quantile(sub_scores, np.linspace(0, 1, 300)))
    taus = np.concatenate([[-0.01], taus, [1.01]])
    points = []
    for tau in taus:
        recall = float((gt_scores >= tau).sum()) / recon
        comp = float((sub_weight * (sub_scores >= tau)).sum()) / n_full
        points.append((recall, comp))
    return points


def _factor_at_floor(points, floor):
    cands = [p for p in points if p[0] >= floor and p[1] > 0]
    return (1.0 / min(c[1] for c in cands)) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(EVAL, "triplet_filter_train.parquet"))
    ap.add_argument("--eval-prefix", default=os.path.join(EVAL, "triplet_filter_eval"))
    ap.add_argument("--manual-json", default=os.path.join(REPORTS, "triplet_join_hs_sweep_metrics.json"))
    ap.add_argument("--out-dir", default=REPORTS)
    args = ap.parse_args()

    train = pq.read_table(args.train)
    gt = pq.read_table(args.eval_prefix + "_gt.parquet")
    sub = pq.read_table(args.eval_prefix + "_sub.parquet")
    with open(args.eval_prefix + "_meta.json") as fh:
        meta = json.load(fh)
    manual = json.load(open(args.manual_json)) if os.path.exists(args.manual_json) else None

    train_pool = train["pool"].to_numpy(zero_copy_only=False)
    gt_pool = gt["pool"].to_numpy(zero_copy_only=False)
    sub_pool = sub["pool"].to_numpy(zero_copy_only=False)
    sub_weight_all = sub["weight"].to_numpy()

    curves = {p: {} for p in POOLS}
    importances, tree_text = {}, {}
    os.makedirs(MODELS, exist_ok=True)

    for pool in POOLS:
        tr_m = train_pool == pool
        y = train["is_gt"].to_numpy()[tr_m]
        gt_m, sub_m = gt_pool == pool, sub_pool == pool
        for fs_name, fs in FEATURESETS.items():
            Xtr = _cols(train, fs)[tr_m]
            Xgt, Xsub = _cols(gt, fs)[gt_m], _cols(sub, fs)[sub_m]
            for m_name, model in _models().items():
                model.fit(Xtr, y)
                gt_s = model.predict_proba(Xgt)[:, 1]
                sub_s = model.predict_proba(Xsub)[:, 1]
                pts = _curve(gt_s, sub_s, sub_weight_all[sub_m],
                             meta[pool]["recon"], meta[pool]["n_full"])
                curves[pool][f"{m_name}-{fs_name}"] = pts
                joblib.dump(model, os.path.join(MODELS, f"third_pion_filter_{m_name}_{fs_name}_{pool}.joblib"))
                if fs_name == "rich" and m_name == "tree":
                    importances[pool] = dict(sorted(zip(fs, model.feature_importances_),
                                                    key=lambda kv: -kv[1]))
                    tree_text[pool] = export_text(model, feature_names=list(fs), max_depth=2)

    _plot(curves, manual, os.path.join(args.out_dir, "triplet_filter_pareto.png"))
    _write_outputs(curves, manual, importances, tree_text, args.out_dir, train.num_rows)


def _manual_front(manual, pool):
    if not manual:
        return []
    return sorted([(d["recall"], d["ratio"]) for d in manual["pareto"][pool]], key=lambda x: x[0])


def _plot(curves, manual, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"tree-rich": (BLUE, "-"), "gbdt-rich": (GREEN, "-"),
              "tree-gate4": (BLUE, "--"), "gbdt-gate4": (GREEN, "--")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pool in zip(axes, POOLS):
        for key, pts in curves[pool].items():
            pts = sorted(pts)
            rec = [p[0] for p in pts]
            fac = [1.0 / p[1] if p[1] > 0 else np.nan for p in pts]
            c, ls = styles[key]
            ax.plot(rec, fac, ls, color=c, label=key, lw=1.8)
        mf = _manual_front(manual, pool)
        if mf:
            ax.plot([r for r, _ in mf], [1.0 / x for _, x in mf], "-o", color=CRIMSON, ms=3, label="manual Pareto")
            ax.plot(mf[-1][0], 1.0 / mf[-1][1], "o", color=BLACK, ms=7, label="Tier-H")
        for f in FLOORS:
            ax.axvline(f, color="grey", ls=":", lw=1)
        ax.set_title(f"pool {pool}")
        ax.set_xlabel("T-Recall(3GT)")
        ax.set_yscale("log")
        ax.set_xlim(0.85, 1.005)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("compression factor (×)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def _write_outputs(curves, manual, importances, tree_text, out_dir, n_train):
    summary = {}
    lines = ["# Learned Soft-Filter — Held-out Test Frontier (tree vs manual)", "",
             f"Training rows: {n_train}. Held-out test; recall exact from GT candidates, compression from subsample.", ""]
    for pool in POOLS:
        lines += [f"## Pool {pool}", "",
                  "| floor | manual× | tree-rich× | gbdt-rich× | tree-gate4× | gbdt-gate4× |",
                  "|---|---|---|---|---|---|"]
        summary[pool] = {}
        for f in FLOORS:
            man = _factor_at_floor([(r, x) for r, x in _manual_front(manual, pool)], f) if manual else None
            row = [f"{f}"]
            row.append(f"{man:.1f}" if man else "—")
            for key in ["tree-rich", "gbdt-rich", "tree-gate4", "gbdt-gate4"]:
                fac = _factor_at_floor(curves[pool][key], f)
                row.append(f"{fac:.1f}" if fac else "—")
                summary[pool].setdefault(key, {})[f] = fac
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(f"**Top features (tree-rich, {pool}):** "
                     + ", ".join(f"{k} {v:.2f}" for k, v in list(importances[pool].items())[:8]))
        lines += ["", "```", tree_text[pool].strip(), "```", ""]
    path_md = os.path.join(out_dir, "triplet_filter_tree_20260628.md")
    with open(path_md, "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(out_dir, "triplet_filter_metrics.json"), "w") as fh:
        json.dump({"summary": summary,
                   "curves": {p: {k: v for k, v in curves[p].items()} for p in POOLS}}, fh, indent=2)
    print(f"wrote {path_md} + triplet_filter_metrics.json")


if __name__ == "__main__":
    main()
