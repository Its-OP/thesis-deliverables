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

from utils.triplet_join import build_track_lorentz, candidates_for_tier, compression_stats

DEFAULT_DUMP = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "low-pt", "eval", "stage3_dump_test.parquet",
)
DEFAULT_SRC_GLOB = "/Users/oleh/Downloads/test_dataset_unzipped/test_*.parquet"

BLUE = "#4466CC"
CRIMSON = "#AA3355"
GREY = "#888888"
TIERS = ["A0", "H", "HS"]
POOLS = ["P1", "P2"]


def _load(dump_path, src_glob, max_events):
    dump = pq.read_table(dump_path, columns=["stage1_sorted_indices", "stage3_sorted_couples"])
    src = pa.concat_tables([
        pq.read_table(s, columns=["event_n_tracks", "track_pt", "track_eta", "track_phi",
                                  "track_charge", "track_dz_significance", "track_label_from_tau"])
        for s in sorted(glob.glob(src_glob))
    ])
    n = dump.num_rows if max_events is None else min(max_events, dump.num_rows)
    assert dump.num_rows == src.num_rows, f"row mismatch {dump.num_rows} vs {src.num_rows}"
    return dump.slice(0, n), src.slice(0, n), n


def _sorted_rows(triplets):
    return set(map(tuple, triplets.sort(dim=1).values.tolist()))


def evaluate(dump, src, n_events, top_c):
    s1 = dump["stage1_sorted_indices"].to_pylist()
    couples_all = dump["stage3_sorted_couples"].to_pylist()
    n_tracks_col = src["event_n_tracks"].to_pylist()
    pt_col = src["track_pt"].to_pylist()
    eta_col = src["track_eta"].to_pylist()
    phi_col = src["track_phi"].to_pylist()
    charge_col = src["track_charge"].to_pylist()
    dz_col = src["track_dz_significance"].to_pylist()
    label_col = src["track_label_from_tau"].to_pylist()

    # acc[pool][tier] -> dict of running counters; plus per-event lists for the pt curve.
    acc = {p: {t: dict(n_full=0, n_survive=0, recon=0, survived=0, pe=[]) for t in TIERS} for p in POOLS}
    curve = {p: {t: [] for t in TIERS} for p in POOLS}  # (visible_pt, survived_bool)

    for r in tqdm(range(n_events), desc="events"):
        assert len(s1[r]) == n_tracks_col[r], f"alignment break at row {r}"
        labels = np.asarray(label_col[r])
        gt = np.where(labels > 0.5)[0]
        n_tracks = n_tracks_col[r]

        lorentz = build_track_lorentz(
            torch.tensor(pt_col[r], dtype=torch.float32),
            torch.tensor(eta_col[r], dtype=torch.float32),
            torch.tensor(phi_col[r], dtype=torch.float32),
        )
        charge = torch.tensor(charge_col[r], dtype=torch.float32)
        eta = torch.tensor(eta_col[r], dtype=torch.float32)
        phi = torch.tensor(phi_col[r], dtype=torch.float32)
        dz = torch.tensor(dz_col[r], dtype=torch.float32)

        couples_np = np.asarray(couples_all[r][:top_c], dtype=np.int64).reshape(-1, 2)
        couples = torch.tensor(couples_np, dtype=torch.long)
        gt_set = set(gt.tolist())
        gt_sorted = tuple(sorted(gt.tolist()))
        has_three = gt.size == 3
        visible_pt = float(np.asarray(pt_col[r])[gt].sum()) if has_three else None
        has_gt_couple = any(set(c).issubset(gt_set) for c in couples_np.tolist())

        pools = {"P1": torch.tensor(s1[r][:256], dtype=torch.long),
                 "P2": torch.arange(n_tracks, dtype=torch.long)}

        for pool_name, pool in pools.items():
            pool_set = set(pool.tolist())
            members_in_pool = sum((int(c[0]) in pool_set) + (int(c[1]) in pool_set)
                                  for c in couples_np)
            n_full = couples.shape[0] * pool.shape[0] - members_in_pool
            reconstructable = has_three and gt_set.issubset(pool_set) and has_gt_couple

            kw = dict(lorentz=lorentz, charge=charge, eta=eta, phi=phi, dz=dz)
            for tier in TIERS:
                a = acc[pool_name][tier]
                if reconstructable:
                    a["recon"] += 1
                if tier == "A0":
                    # Lossless full cross-join: count analytically, T-Recall == reconstructable.
                    event_n_survive = n_full
                    survived = reconstructable
                else:
                    triplets, _ = candidates_for_tier(tier, couples, pool, **kw)
                    event_n_survive = int(triplets.shape[0])
                    survived = reconstructable and gt_sorted in _sorted_rows(triplets)
                a["n_full"] += n_full
                a["n_survive"] += event_n_survive
                if n_full > 0:
                    a["pe"].append(event_n_survive / n_full)
                if survived:
                    a["survived"] += 1
                if has_three:
                    curve[pool_name][tier].append((visible_pt, bool(survived)))

    return acc, curve


def _table(acc, n_events):
    rows = []
    for pool in POOLS:
        for tier in TIERS:
            a = acc[pool][tier]
            ratio = a["n_survive"] / a["n_full"] if a["n_full"] else 0.0
            pe = np.asarray(a["pe"]) if a["pe"] else np.array([0.0])
            rows.append({
                "pool": pool, "tier": tier,
                "t_recall_total": a["survived"] / n_events,
                "t_recall_recon": a["survived"] / a["recon"] if a["recon"] else 0.0,
                "n_full": a["n_full"], "n_survive": a["n_survive"],
                "compression_ratio": ratio,
                "compression_factor": (1.0 / ratio) if ratio else float("inf"),
                "compression_pe_median": float(np.median(pe)),
                "compression_pe_mean": float(pe.mean()),
                "purity": a["survived"] / a["n_survive"] if a["n_survive"] else 0.0,
            })
    return rows


def _print_table(rows):
    header = ["pool", "tier", "T-Recall(tot)", "T-Recall(3GT)", "N_full", "N_survive",
              "compr.ratio", "compr.×", "compr.med(pe)", "purity"]
    print("  ".join(f"{h:>13}" for h in header))
    for r in rows:
        print("  ".join(f"{v:>13}" for v in [
            r["pool"], r["tier"], f"{r['t_recall_total']:.4f}", f"{r['t_recall_recon']:.4f}",
            r["n_full"], r["n_survive"], f"{r['compression_ratio']:.4g}",
            f"{r['compression_factor']:.4g}", f"{r['compression_pe_median']:.4g}",
            f"{r['purity']:.4g}"]))


def _plot(curve, out_path, pt_max, bin_width):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.arange(0.0, pt_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colours = {"A0": GREY, "H": BLUE, "HS": CRIMSON}
    styles = {"A0": "--", "H": "-", "HS": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, pool in zip(axes, POOLS):
        for tier in TIERS:
            data = curve[pool][tier]
            if not data:
                continue
            pts = np.array([d[0] for d in data])
            hit = np.array([d[1] for d in data], dtype=float)
            idx = np.digitize(pts, edges) - 1
            eff = np.full(centers.shape, np.nan)
            for b in range(len(centers)):
                m = idx == b
                if m.sum() > 0:
                    eff[b] = hit[m].mean()
            ax.plot(centers, eff, styles[tier], color=colours[tier], label=tier, linewidth=2)
        ax.set_title(f"pool {pool}")
        ax.set_xlabel("visible 3π pT [GeV]")
        ax.set_xlim(0, pt_max)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("T-Recall (GT triplet survives JOIN)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default=DEFAULT_DUMP)
    parser.add_argument("--src-glob", default=DEFAULT_SRC_GLOB)
    parser.add_argument("--top-c", type=int, default=100)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
    parser.add_argument("--pt-max", type=float, default=20.0)
    parser.add_argument("--bin-width", type=float, default=1.0)
    args = parser.parse_args()

    dump, src, n_events = _load(args.dump, args.src_glob, args.max_events)
    acc, curve = evaluate(dump, src, n_events, args.top_c)
    rows = _table(acc, n_events)
    _print_table(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "triplet_join_metrics.json")
    with open(json_path, "w") as fh:
        json.dump({"n_events": n_events, "top_c": args.top_c, "rows": rows}, fh, indent=2)
    print(f"wrote {json_path}")
    _plot(curve, os.path.join(args.out_dir, "triplet_join_recall_vs_pt.png"),
          args.pt_max, args.bin_width)


if __name__ == "__main__":
    main()
