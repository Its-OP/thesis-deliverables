from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.triplet_join import build_track_lorentz, triplet_gate_quantities
from utils.triplet_split import load_split
from scripts.python.eval_triplet_join import _load, DEFAULT_DUMP, DEFAULT_SRC_GLOB, BLUE, CRIMSON, POOLS

# Sweep grids (np.inf = gate off).
TAU_DZ = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
TAU_DR = [0.15, 0.3, 0.5, 0.8, 1.2, 2.0, np.inf]
A1_LO = [0.0, 0.4, 0.6]
A1_HI = [1.3, 1.5, 1.7, np.inf]
TAU_RHO = [0.10, 0.15, 0.30, 0.50, np.inf]
FLOORS = [0.99, 0.97, 0.95]


def _event_pools(n_tracks):
    return {"P2": torch.arange(n_tracks, dtype=torch.long)}


def collect(dump, src, event_indices, top_c, subsample, seed):
    s1 = dump["stage1_sorted_indices"].to_pylist()
    couples_all = dump["stage3_sorted_couples"].to_pylist()
    n_tracks_col = src["event_n_tracks"].to_pylist()
    pt_col, eta_col, phi_col = (src[c].to_pylist() for c in ("track_pt", "track_eta", "track_phi"))
    charge_col, dz_col, label_col = (src[c].to_pylist() for c in
                                     ("track_charge", "track_dz_significance", "track_label_from_tau"))
    generator = torch.Generator().manual_seed(seed)

    store = {p: dict(gt=[], sub=[], weight=[], recon=0, n_full=0) for p in POOLS}

    for r in tqdm([int(x) for x in event_indices], desc="events"):
        assert len(s1[r]) == n_tracks_col[r]
        labels = np.asarray(label_col[r])
        gt = np.where(labels > 0.5)[0]
        n_tracks = n_tracks_col[r]
        lorentz = build_track_lorentz(torch.tensor(pt_col[r], dtype=torch.float32),
                                      torch.tensor(eta_col[r], dtype=torch.float32),
                                      torch.tensor(phi_col[r], dtype=torch.float32))
        charge = torch.tensor(charge_col[r], dtype=torch.float32)
        eta = torch.tensor(eta_col[r], dtype=torch.float32)
        phi = torch.tensor(phi_col[r], dtype=torch.float32)
        dz = torch.tensor(dz_col[r], dtype=torch.float32)
        couples_np = np.asarray(couples_all[r][:top_c], dtype=np.int64).reshape(-1, 2)
        couples = torch.tensor(couples_np, dtype=torch.long)
        gt_set = set(gt.tolist())
        has_three = gt.size == 3
        has_gt_couple = any(set(c).issubset(gt_set) for c in couples_np.tolist())

        for pool_name, pool in _event_pools(n_tracks).items():
            st = store[pool_name]
            pool_set = set(pool.tolist())
            members = sum((int(c[0]) in pool_set) + (int(c[1]) in pool_set) for c in couples_np)
            n_full = couples.shape[0] * pool.shape[0] - members
            st["n_full"] += n_full
            reconstructable = has_three and gt_set.issubset(pool_set) and has_gt_couple
            if reconstructable:
                st["recon"] += 1

            gt_sorted = tuple(sorted(gt.tolist())) if reconstructable else None
            q = triplet_gate_quantities(couples, pool, lorentz=lorentz, charge=charge,
                                        eta=eta, phi=phi, dz=dz, gt_sorted=gt_sorted)
            quad = torch.stack([q["dz_dist"], q["dr_min"], q["m_ijk"], q["rho_dist"]], dim=1)
            n_h = quad.shape[0]
            if n_h == 0:
                continue
            if reconstructable and bool(q["is_gt"].any()):
                st["gt"].append(quad[q["is_gt"]][0].numpy())
            n_sub = min(subsample, n_h)
            idx = torch.randperm(n_h, generator=generator)[:n_sub]
            st["sub"].append(quad[idx].numpy())
            st["weight"].append(np.full(n_sub, n_h / n_sub, dtype=np.float64))

    for p in POOLS:
        st = store[p]
        st["gt"] = np.asarray(st["gt"]) if st["gt"] else np.empty((0, 4))
        st["sub"] = np.concatenate(st["sub"]) if st["sub"] else np.empty((0, 4))
        st["weight"] = np.concatenate(st["weight"]) if st["weight"] else np.empty((0,))
    return store


def _bool_columns(arr):
    # arr columns: 0 dz, 1 dr, 2 m, 3 rho. Returns precomputed per-threshold masks.
    return {
        "dz": [arr[:, 0] <= t for t in TAU_DZ],
        "dr": [arr[:, 1] <= t for t in TAU_DR],
        "lo": [arr[:, 2] >= t for t in A1_LO],
        "hi": [arr[:, 2] <= t for t in A1_HI],
        "rho": [arr[:, 3] <= t for t in TAU_RHO],
    }


def sweep(store):
    out = {}
    for p in POOLS:
        st = store[p]
        gt_cols = _bool_columns(st["gt"])
        sub_cols = _bool_columns(st["sub"])
        weight = st["weight"]
        recon = max(st["recon"], 1)
        n_full = max(st["n_full"], 1)
        points = []
        for ia, ib, ilo, ihi, ir in itertools.product(
                range(len(TAU_DZ)), range(len(TAU_DR)), range(len(A1_LO)),
                range(len(A1_HI)), range(len(TAU_RHO))):
            gt_pass = gt_cols["dz"][ia] & gt_cols["dr"][ib] & gt_cols["lo"][ilo] & gt_cols["hi"][ihi] & gt_cols["rho"][ir]
            sub_pass = sub_cols["dz"][ia] & sub_cols["dr"][ib] & sub_cols["lo"][ilo] & sub_cols["hi"][ihi] & sub_cols["rho"][ir]
            recall = float(gt_pass.sum()) / recon
            ratio = float((weight * sub_pass).sum()) / n_full
            points.append({"dz": TAU_DZ[ia], "dr": TAU_DR[ib], "a1_lo": A1_LO[ilo],
                           "a1_hi": A1_HI[ihi], "rho": TAU_RHO[ir],
                           "recall": recall, "ratio": ratio})
        out[p] = points
    return out


def _pareto(points):
    # Pareto-optimal: no other point has higher recall AND lower ratio.
    pts = sorted(points, key=lambda d: (-d["recall"], d["ratio"]))
    front, best_ratio = [], float("inf")
    for d in pts:
        if d["ratio"] < best_ratio - 1e-12:
            front.append(d)
            best_ratio = d["ratio"]
    return front


def _inf(x):
    return None if x == np.inf else x


def write_outputs(points_by_pool, out_dir, n_events, top_c):
    os.makedirs(out_dir, exist_ok=True)
    serial = {p: [{**d, "dz": _inf(d["dz"]), "dr": _inf(d["dr"]), "a1_hi": _inf(d["a1_hi"]),
                   "rho": _inf(d["rho"])} for d in pts] for p, pts in points_by_pool.items()}
    front = {p: _pareto(pts) for p, pts in points_by_pool.items()}
    json_path = os.path.join(out_dir, "triplet_join_hs_sweep_metrics.json")
    with open(json_path, "w") as fh:
        json.dump({"n_events": n_events, "top_c": top_c,
                   "grid": {p: serial[p] for p in POOLS},
                   "pareto": {p: [{**d, "dz": _inf(d["dz"]), "dr": _inf(d["dr"]),
                                   "a1_hi": _inf(d["a1_hi"]), "rho": _inf(d["rho"])}
                                  for d in front[p]] for p in POOLS}}, fh, indent=2)
    print(f"wrote {json_path}")
    _plot_pareto(points_by_pool, front, os.path.join(out_dir, "triplet_join_hs_pareto.png"))
    _plot_marginal(points_by_pool, os.path.join(out_dir, "triplet_join_hs_marginal.png"))
    _write_md(front, out_dir, n_events, top_c)


def _plot_pareto(points_by_pool, front, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, p in zip(axes, POOLS):
        pts = points_by_pool[p]
        rec = np.array([d["recall"] for d in pts])
        fac = np.array([1.0 / d["ratio"] if d["ratio"] > 0 else np.nan for d in pts])
        ax.scatter(rec, fac, s=6, alpha=0.25, color=BLUE)
        fr = sorted(front[p], key=lambda d: d["recall"])
        ax.plot([d["recall"] for d in fr], [1.0 / d["ratio"] for d in fr], "-o",
                color=CRIMSON, ms=3, label="Pareto envelope")
        for f in FLOORS:
            ax.axvline(f, color="grey", ls=":", lw=1)
        ax.set_title(f"pool {p}")
        ax.set_xlabel("T-Recall(3GT)")
        ax.set_yscale("log")
        ax.set_xlim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("compression factor (×)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def _marginal(points, gate_key, default_check):
    rows = [d for d in points if default_check(d)]
    rows.sort(key=lambda d: d[gate_key] if d[gate_key] != np.inf else 1e9)
    return rows


def _plot_marginal(points_by_pool, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gates = [
        ("dz", lambda d: d["dr"] == np.inf and d["a1_lo"] == 0.0 and d["a1_hi"] == np.inf and d["rho"] == np.inf),
        ("dr", lambda d: d["dz"] == np.inf and d["a1_lo"] == 0.0 and d["a1_hi"] == np.inf and d["rho"] == np.inf),
        ("a1_hi", lambda d: d["dz"] == np.inf and d["dr"] == np.inf and d["a1_lo"] == 0.0 and d["rho"] == np.inf),
        ("rho", lambda d: d["dz"] == np.inf and d["dr"] == np.inf and d["a1_lo"] == 0.0 and d["a1_hi"] == np.inf),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (gate, check) in enumerate(gates):
        for row, p in enumerate(POOLS):
            rows = _marginal(points_by_pool[p], gate, check)
            x = [d[gate] if d[gate] != np.inf else max(TAU_DZ[-2], 6) for d in rows]
            ax = axes[row][col]
            ax.plot(x, [d["recall"] for d in rows], "-o", color=BLUE, label="T-Recall(3GT)")
            ax2 = ax.twinx()
            ax2.plot(x, [d["ratio"] for d in rows], "-s", color=CRIMSON, label="compression ratio")
            ax.set_title(f"{gate} marginal — {p}")
            ax.set_xlabel(f"{gate} threshold")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def _write_md(front, out_dir, n_events, top_c):
    lines = ["# Tier-HS Recall–Compression Frontier", "",
             f"Test events: {n_events} · top-C couples: {top_c} · compression estimated from per-event H-candidate subsample.", ""]
    for p in POOLS:
        lines += [f"## Pool {p}", "",
                  "| floor | recall | compr.× | dz | ΔR | a1_lo | a1_hi | ρ |",
                  "|---|---|---|---|---|---|---|---|"]
        for f in FLOORS:
            cand = [d for d in front[p] if d["recall"] >= f]
            if not cand:
                lines.append(f"| {f} | — none above floor — | | | | | | |")
                continue
            d = min(cand, key=lambda d: d["ratio"])
            fmt = lambda x: "off" if x == np.inf else f"{x:g}"
            lines.append(f"| {f} | {d['recall']:.4f} | {1.0/d['ratio']:.1f} | "
                         f"{fmt(d['dz'])} | {fmt(d['dr'])} | {fmt(d['a1_lo'])} | {fmt(d['a1_hi'])} | {fmt(d['rho'])} |")
        lines.append("")
    path = os.path.join(out_dir, "triplet_join_hs_sweep_20260628.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    parser.add_argument("--src-glob", default=DEFAULT_SRC_GLOB)
    parser.add_argument("--top-c", type=int, default=100)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--split-json", default=None,
                        help="restrict to the held-out test events listed in this split file")
    parser.add_argument("--subsample", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    args = parser.parse_args()

    dump, src, n = _load(args.dump, args.src_glob, args.max_events)
    if args.split_json:
        event_indices = load_split(args.split_json, "test")
        event_indices = event_indices[event_indices < n]
    else:
        event_indices = np.arange(n)
    n_events = len(event_indices)
    store = collect(dump, src, event_indices, args.top_c, args.subsample, args.seed)
    points = sweep(store)
    for p in POOLS:
        h = [d for d in points[p] if d["dz"] == np.inf and d["dr"] == np.inf
             and d["a1_lo"] == 0.0 and d["a1_hi"] == np.inf and d["rho"] == np.inf][0]
        print(f"[{p}] Tier-H sanity (all gates off): recall={h['recall']:.4f} ratio={h['ratio']:.4f} factor={1.0/h['ratio']:.3f}")
    write_outputs(points, args.out_dir, n_events, args.top_c)


if __name__ == "__main__":
    main()
