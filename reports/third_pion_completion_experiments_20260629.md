# Third-Pion Completion — JOIN + Filtration Experiments

**Date:** 2026-06-29 · **Branch:** `third-pion-join` (deliverables repo) · **Test set:** 43 056 events (post-0.5 GeV cutoff)

Internal technical report consolidating the third-pion-completion experiments. Scope is candidate generation only — turning ranked couples into a pruned set of τ→3π triplet candidates. It absorbs the three sub-reports `triplet_join_20260628.md`, `triplet_join_hs_sweep_20260628.md`, `triplet_filter_tree_20260628.md`.

## 1. Overview & problem framing

The cascade ranks **couples** `(i,j)` (pairs of tracks) but never emits **triplets** `(i,j,k)`. The third-pion JOIN completes each couple by attaching a third track `k` drawn from a candidate pool. This is **candidate generation**: the output is a pruned set of `(i,j,k)` candidates that a downstream neural ranker (not built here) will score. Before this work the only "join" was a full Cartesian product (every couple × every pool track) with no physics — tens of thousands of candidates per event.

The governing question: **how small a candidate set can we emit while still retaining the true triplet?** Every experiment below is a point on that recall ↔ compression trade.

Three experiment families, in order:
1. **A0** — full cross-join baseline (no filtering): the recall ceiling and the compression denominator.
2. **Hard cascade (Tier-H)** — lossless physics gates (charge + mass).
3. **Soft tier** — first hand-tuned (Tier-HS sweep), then **learned** (classification trees), each pruning further at a recall cost.

## 2. Data

### 2.1 Splits & sources

Each split has both per-track features (raw detector quantities + GT labels) and a cascade dump (ranked couples + pools). They are joined per event.

| Split | Featured parquet | Cascade dump | Events | Used for |
|---|---|---|---|---|
| TRAIN | `part/data/low-pt/train/train_*.parquet` | — | 300 000 | (cascade training; not used here) |
| VAL | `part/data/low-pt/val/val_*.parquet` | `deliverables/data/low-pt/eval/perstage_couples_val.parquet` | 52 500 | **train** the tree filter |
| TEST | `/Users/oleh/Downloads/test_dataset_unzipped/test_*.parquet` | `deliverables/data/low-pt/eval/stage3_dump_test.parquet` | 43 056 | **eval** everything |

A0 / Tier-H / manual-sweep results are all measured on TEST. The learned filter is trained on VAL and evaluated held-out on TEST.

### 2.2 Cascade outputs consumed

The dump carries indices only (no kinematics):
- `stage1_sorted_indices` — all event tracks ranked by Stage-1; the top 256 form pool **P1**.
- `stage3_sorted_couples` — couples (from Stage-2 top-K2 survivors) ranked by the Stage-3 couple reranker; the top **C** (default 100) seed the join.

### 2.3 Join

Per-track kinematics come from the featured parquet; couples/pools from the dump. They are aligned **by row position** (dump row *i* ↔ source row *i*). Verified over all rows by an independent fingerprint `len(stage1_sorted_indices) == event_n_tracks`: TEST 43056/43056, VAL 52500/52500. The composite key `(event_run, event_id, event_luminosity_block)` collides (only 1019 unique triples in TEST) and is **never** used to join.

### 2.4 Ground truth

`track_label_from_tau` flags the GT tracks. GT couple = any 2-subset of the 3 GT tracks; GT triplet = the 3-set. An event is **reconstructable** when it has exactly 3 GT tracks, all present in the pool, and at least one GT couple within the top-C couples.

## 3. Problem formulation

- **Candidate** = `(i,j,k)` with `(i,j)` a top-C couple and `k` a pool track, `k ∉ {i,j}`.
- **Pools:** **P1** = Stage-1 top-256; **P2** = the entire input track set (bypasses Stage 1). Couples always come from Stage-2/3 (top-C).
- **A0 candidate counts** ≈ `C·N`: P1 ~25 k/event, P2 ~79 k/event.
- 4-vectors are built from raw `(pt, η, φ)` under the **pion-mass hypothesis** (m = 0.13957 GeV) — every track treated as a pion.

## 4. Methods — three filtration approaches

All gates compose; the implementation is one gated function (`utils/triplet_join.build_triplet_candidates`), the approaches are gate subsets.

### 4.1 A0 baseline
Full cross-join, no physics. Defines `N_full` (compression denominator) and the achievable recall ceiling (a triplet is reachable iff it is reconstructable).

### 4.2 Hard cascade (Tier-H)
Two gates that the true triplet can never violate:
- **Charge:** net `q_i+q_j+q_k = ±1` (charge conservation — τ→3π is 2 same-sign + 1 opposite).
- **Mass:** `m(ijk) < m_τ = 1.77693 GeV` (kinematic — the neutrino carries the remainder).

Both are lossless in truth. The only loss is a thin ~0.4–0.6 % of GT triplets whose **reconstructed** mass fluctuates over `m_τ` at the kinematic edge under the pion-mass hypothesis; relaxing `mass_max` to ~1.85 GeV recovers them.

### 4.3 Manual soft tier (Tier-HS) + recall–compression sweep
Four soft gates on Tier-H survivors: vertex `|Δdz_sig|`, `ΔR` cone (min ΔR of `k` to either couple track), a1 mass band, and a ρ(770) anchor (an opposite-sign sub-pair near 0.775 GeV). To map the trade without re-running, a **one-pass instrumented sweep** (`sweep_triplet_join_hs.py`) records each Tier-H candidate's four gate quantities once, then thresholds them **offline** over a grid (≈7·7·12·5). Recall is exact (from the GT candidate); compression is estimated from a per-event subsample of Tier-H candidates (weight `n_h/n_sub`). The reported Pareto envelope is the max compression at each recall.

### 4.4 Learned soft filter (classification trees)
Replace the hand-tuned soft gates with a classifier on per-candidate features, still downstream of the hard charge+mass cascade. Two sklearn models: a depth-6 `DecisionTreeClassifier` (interpretable — splits read as learned cuts) and a `HistGradientBoostingClassifier` (performance). Trained per pool on VAL candidates (all GT candidates as positives + 80 sampled negatives/event, `class_weight="balanced"`), evaluated on TEST. Each emits `P(is_gt)`; sweeping the score threshold traces the recall–compression curve. Two feature sets: **rich** (20) and the **gate4** ablation (the 4 gate quantities only — same information the manual sweep had).

## 5. Features

### 5.1 Gate quantities (4) — also the `gate4` ablation set
`dz_dist` = `|dz_k − ½(dz_i+dz_j)|` · `dr_min` = `min(ΔR(k,i), ΔR(k,j))` · `m_ijk` = triplet invariant mass · `rho_dist` = `min` over opposite-sign sub-pairs of `|m_pair − m_ρ|`.

### 5.2 Rich feature set (20) — all raw-detector-derived (no cheating features)

| group | features |
|---|---|
| gate quantities | `dz_dist`, `dr_min`, `m_ijk`, `rho_dist` |
| couple structure | `couple_rank` (Stage-3 rank of the seeding couple), `is_same_sign`, `m_ij`, `pt_ij` |
| third-track raw | `pt_k`, `abs_eta_k`, `dz_sig_k`, `dxy_sig_k`, `dca_sig_k`, `n_pixel_k`, `norm_chi2_k` |
| Dalitz / pairwise | `m_ik`, `m_jk` (both OS-pair masses), `dr_ij` |
| triplet | `dz_spread` (max \|Δdz\| over the 3 pairs), `pt_ijk` |

Label `is_gt` = candidate equals the event's GT triplet. Trees need no standardization, so raw quantities are fed directly.

## 6. Evaluation metrics

All event-denominated.

- **T-Recall(tot)** = events whose full GT triplet survives / all events. **T-Recall(3GT)** = same over reconstructable events only — isolates the join's own loss.
- **Compression ratio** = `Σ N_survive / Σ N_full` (dataset-total); **factor** = its inverse (×). **Per-event** ratio reported as a median.
- **Purity** = GT triplets / candidates (signal density).
- **Recall–compression frontier** — operating points read at recall floors 0.99 / 0.97 / 0.95.

## 7. Results

### 7.1 A0 ceiling + Tier-H (TEST)

| pool | tier | T-Recall(tot) | T-Recall(3GT) | compr.× | cand/evt | purity |
|---|---|---|---|---|---|---|
| P1 | A0 | 0.8606 | 1.0000 | 1.0 | 25 386 | 3.4e-5 |
| P1 | **H** | 0.8575 | 0.9963 | **3.81** | 6 670 | 1.3e-4 |
| P2 | A0 | 0.8902 | 1.0000 | 1.0 | 79 421 | 1.1e-5 |
| P2 | **H** | 0.8850 | 0.9941 | **7.76** | 10 230 | 8.7e-5 |

P2's ceiling beats P1 by +3.0 pts of T-Recall(tot) — the full pool recovers true thirds that Stage 1 dropped from the top-256. Tier-H (charge+mass, no spatial gating) is near-lossless at 3.8× (P1) / 7.8× (P2).

### 7.2 Manual Tier-HS frontier (TEST)

Compression factor at each recall floor; the winning gate config was consistently `dz off`, loose ΔR (1.2–2.0), a1 upper cut 1.5–1.7, ρ ≤ 0.5.

| floor | P1 compr.× | P2 compr.× |
|---|---|---|
| 0.99 | 4.3 | 8.9 |
| 0.97 | 5.0 | 11.2 |
| 0.95 | 6.1 | 13.7 |

Hand-tuning the soft gates adds only modest compression over Tier-H, and never benefits from a tight ΔR cone (low-pT τ are weakly boosted → wide cone).

### 7.3 Learned filter frontier (held-out TEST) — compression factor (×)

| pool | floor | manual | tree-rich | **gbdt-rich** | tree-gate4 | gbdt-gate4 |
|---|---|---|---|---|---|---|
| P1 | 0.99 | 4.3 | 5.4 | **6.8** | 4.6 | 5.0 |
| P1 | 0.97 | 5.0 | 7.3 | **12.1** | 6.3 | 6.6 |
| P1 | 0.95 | 6.1 | 11.3 | **17.4** | 7.3 | 7.8 |
| P2 | 0.99 | 8.9 | 10.4 | **13.1** | 9.2 | 10.0 |
| P2 | 0.97 | 11.2 | 18.4 | **26.3** | 13.8 | 14.6 |
| P2 | 0.95 | 13.7 | 24.5 | **39.2** | 16.5 | 17.9 |

The GBDT on rich features dominates the manual Pareto at every floor and pool — ~2–3× more compression at matched recall (P2 @0.95: 39.2× vs 13.7×).

### 7.4 Interpretation

Single-tree feature importances (rich): `couple_rank` **0.63–0.64**, `dr_min` 0.15–0.18, `dca_sig_k` 0.07–0.08, `dz_dist` 0.05–0.06, `m_ijk` 0.04, `n_pixel_k` 0.01–0.03. The tree's root split is `couple_rank ≤ 17.5`. The Stage-3 couple rank is by far the strongest signal — a good couple is a strong prior that its completions are real.

## 8. Findings

- **Wider pool helps:** P2 lifts the ceiling +3.0 pts over P1 by recovering Stage-1-dropped thirds.
- **Charge + mass are the free lunch:** near-lossless 3.8×/7.8× with no spatial gating.
- **Hand-tuned soft gates are weak** and prefer dz-off / loose-ΔR; tight collimation cuts cost recall at low pT.
- **The learned filter wins decisively** (~2–3× over manual) and the win is driven by `couple_rank`, which the manual sweep never used.
- **Ablation isolates the cause:** the `gate4` tree (same 4 physics quantities as the manual sweep) beats manual only marginally (P1@0.99 4.6× vs 4.3×). The large gain comes from the richer features (couple rank + third-track quality `dca_sig_k`, `n_pixel_k`), not merely from learning interactions among the physics gates.

## 9. Limitations

- Single VAL→TEST split — no k-fold or second seed (publish-grade would add one).
- Pion-mass hypothesis for all tracks (drives the small Tier-H mass-edge loss).
- Compression is subsample-estimated (exact recall, estimated compression).
- T-Recall(tot) is bounded above by the couple-side recall inherited from the Stage-3 top-100 (≈0.86 P1 / 0.89 P2).
- The manual HS windows were un-tuned starting values, now superseded by the tree.

## 10. Artefacts & reproduce

| kind | path |
|---|---|
| JOIN + features + gates | `utils/triplet_join.py` |
| A0/H/HS ladder | `scripts/python/eval_triplet_join.py` → `reports/triplet_join_metrics.json`, `..._recall_vs_pt.png` |
| Manual sweep | `scripts/python/sweep_triplet_join_hs.py` → `reports/triplet_join_hs_sweep_*` |
| Filter tables | `scripts/python/build_triplet_filter_table.py` → `data/low-pt/eval/triplet_filter_{train,eval_*}.parquet` |
| Filter train/eval | `scripts/python/train_triplet_filter.py` → `reports/triplet_filter_*`, `models/third_pion_filter_*.joblib` |
| Tests | `tests/test_triplet_join.py`, `tests/test_triplet_filter.py` |

```bash
# from deliverables/ (env: /opt/miniconda3/envs/part)
python scripts/python/eval_triplet_join.py                     # A0/H/HS ladder (TEST)
python scripts/python/sweep_triplet_join_hs.py                 # manual recall–compression frontier
python scripts/python/build_triplet_filter_table.py --split val   # tree training table
python scripts/python/build_triplet_filter_table.py --split test  # held-out eval artifact
python scripts/python/train_triplet_filter.py                  # tree + GBDT frontier vs manual
pytest tests/test_triplet_join.py tests/test_triplet_filter.py
```

Build tables (`data/low-pt/eval/triplet_filter_*.parquet`, ~8.6 M rows = per-event candidate subsample × events × 2 pools) are large regenerable intermediates — gitignore.

## 11. Out of scope / next

- Selecting a single operating point from the frontier.
- The downstream **neural third-pion ranker** that scores within the pruned set — and given `couple_rank`'s dominance here, that feature should be an input to it.
