# Third-Pion JOIN — Physics-Informed Candidate Generation

**Date:** 2026-06-28 · **Branch:** `third-pion-join` · **Test set:** 43056 events (post-0.5 GeV-cutoff split)

## Goal

Complete each ranked **couple** `(i,j)` into a τ→3π **triplet** `(i,j,k)` by joining it against a pool of individual tracks. The JOIN is **candidate generation only** — it emits a pruned set of `(i,j,k)` candidates for a downstream neural third-pion scorer (out of scope here). Today's pipeline does this as a full Cartesian product with no physics gating; this work measures how far hard/soft physical filters shrink that product without losing the true triplet.

## Approaches

One composable-gate function (`utils/triplet_join.py::build_triplet_candidates`); the five approaches are gate subsets, not separate code paths.

| # | Approach | Gate | Hardness | Efficient-join structure |
|---|---|---|---|---|
| A1 | Charge bucket | net `q_i+q_j+q_k = ±1` | hard, lossless | 2-cell partition by sign (1-D Voronoi) |
| A2 | Vertex cell | `(η,φ,dz_sig)` proximity to couple centroid | soft | grid-hash / k-NN ball (the "Voronoi cells") |
| A3 | Triplet mass | `m(ijk) < m_τ` (and a1 window for HS) | hard upper | O(1) post-filter |
| A4 | ρ(770) anchor | OS sub-pair within `ρ` window | soft, HS only | annulus query on A2's index |
| A5 | Fused cascade | A1→A2→A3(→A4) over one shared index | mixed | charge bucket + one vertex index + O(1) mass |

4-vectors are built from raw `(pt,eta,phi)` under the pion-mass hypothesis (0.13957 GeV); constants and `compute_invariant_mass` reused from `utils/couple_features.py`.

## Tiers and pools

- **A0** — full cross-join, no filter (recall **ceiling**, compression denominator).
- **H** — charge `±1` + `m(ijk) < m_τ` (hard, near-lossless).
- **HS** — H + vertex `(dz=3, ΔR=0.5)` + a1 `[0.6,1.5]` + ρ `0.30` (soft, lossy — starting windows).
- **P1** = Stage-1 top-256 pool · **P2** = entire input track set. Couples = Stage-3 top-100.

## Results (full 43056-event test split)

`T-Recall(tot)` = events whose full GT triplet survives / all events. `T-Recall(3GT)` = same over events with a reconstructable GT triplet in the pool (the join's own loss). `compr.×` = `Σ N_full / Σ N_survive`. cand/evt = mean candidate triplets per event.

| pool | tier | T-Recall(tot) | T-Recall(3GT) | compr.× | compr ratio | cand/evt | purity |
|---|---|---|---|---|---|---|---|
| P1 | A0 | 0.8606 | 1.0000 | 1.0 | 1.0 | 25 386 | 3.4e-5 |
| P1 | **H** | **0.8575** | **0.9963** | **3.81** | 0.2627 | 6 670 | 1.3e-4 |
| P1 | HS | 0.1695 | 0.1969 | 477 | 0.0021 | 53 | 3.2e-3 |
| P2 | A0 | 0.8902 | 1.0000 | 1.0 | 1.0 | 79 421 | 1.1e-5 |
| P2 | **H** | **0.8850** | **0.9941** | **7.76** | 0.1288 | 10 230 | 8.7e-5 |
| P2 | HS | 0.1696 | 0.1905 | 1434 | 0.0007 | 55 | 3.1e-3 |

Per-event compression-ratio median (not `Σ`-aggregated): P1 H = 0.262, P2 H = 0.131, P1 HS = 9.1e-4, P2 HS = 3.2e-4 — tracks the dataset-total ratio closely, so pruning is uniform across events (no heavy tail of weakly-pruned events).

Curve: `reports/triplet_join_recall_vs_pt.png` (T-Recall vs visible-3π pT, A0/H/HS per pool). Raw: `reports/triplet_join_metrics.json`.

## Findings

1. **Joining against the entire input (P2) raises the ceiling +3.0 pts** of T-Recall over Stage-1 top-256 (0.8902 vs 0.8606) — it recovers events whose third pion was filtered by Stage 1. Confirms the wider-pool rationale.
2. **Charge + mass (Tier-H) is the strong, safe operating point:** near-lossless (`T-Recall(3GT)` 0.996 P1 / 0.994 P2) at **3.8× (P1) / 7.8× (P2)** compression — with no spatial gating at all.
3. **Tier-HS default windows over-prune** (recall → 0.19): the ΔR cone and a1/ρ windows are too aggressive for low-pT τ (weak boost → wide cone). They need a recall-floored Pareto sweep, not these defaults.
4. **Tier-H's 0.4–0.6% gap below A0** is reconstructed `m(ijk)` fluctuating over `m_τ` at the kinematic edge under the pion-mass hypothesis; relaxing `mass_max` to ~1.85 GeV recovers it.
5. **Purity stays low even at H** (≈1 GT triplet per ~6–10k candidates) — the JOIN is a recall-preserving pre-filter, not a discriminator; ranking is the downstream scorer's job.

## Reproduce

```bash
# from deliverables/ (env: /opt/miniconda3/envs/part)
python scripts/python/eval_triplet_join.py            # full split
python scripts/python/eval_triplet_join.py --max-events 1500   # smoke
pytest tests/test_triplet_join.py                     # unit + real-data losslessness
```

## Next steps

- **Recall-floored Pareto sweep** for Tier-HS — scan vertex/ΔR/ρ windows, pick the smallest candidate set holding `T-Recall(3GT) ≥ 0.99`. The robust lever is the vertex `dz` gate (pT-independent); ΔR is the recall risk.
- Relax `mass_max` to recover the Tier-H edge loss.
- Downstream **neural third-pion scorer** over the pruned set (separate design pass).

## Limitations

Single test split; pion-mass hypothesis for all tracks; HS windows are un-tuned starting values. Couple-side recall is inherited from Stage 3 (top-100) and bounds T-Recall(tot) from above.
