# Learned Soft-Filter — Held-out Test Frontier (tree vs manual)

Training rows: 8622221. Trained on VAL cascade outputs, evaluated on the held-out TEST split (no leakage). Recall exact from GT candidates, compression from per-event subsample. Architecture: hard charge+mass cascade (lossless) → learned classification tree on soft features. Operating point left to choice (frontier).

## Findings

1. **The learned tree dominates the hand-tuned physics Pareto at every recall floor and both pools.** GBDT-rich gives ~2–3× more compression at matched recall: P1 6.8×/12.1×/17.4× vs manual 4.3×/5.0×/6.1× at 0.99/0.97/0.95; P2 13.1×/26.3×/39.2× vs 8.9×/11.2×/13.7×.
2. **The gain comes from a feature the manual sweep never used — `couple_rank`** (Stage-3 reranker rank), which dominates importance (0.63–0.64), far above any physics gate. The single tree's root split is `couple_rank ≤ 17.5`. A good couple is a strong prior that its completions are real.
3. **Controlled ablation confirms it:** the `gate4` tree (only the 4 physics quantities — the same information the manual sweep had) beats manual only marginally (P1@0.99 4.6× vs 4.3×). The large win is the richer features (`couple_rank` + third-track quality `dca_sig_k`, `n_pixel_k`), not merely learning interactions among the physics gates.
4. **Physics gates act loosely, as expected at low pT:** the tree keeps a loose `dr_min` cut, barely uses `dz` (splits at very large values), and only trims near the `m_τ` mass edge — echoing the manual frontier's "dz off, loose ΔR".
5. GBDT > single tree > gate4 models > manual everywhere; both the ensemble and the richer features contribute.

## Pool P1

| floor | manual× | tree-rich× | gbdt-rich× | tree-gate4× | gbdt-gate4× |
|---|---|---|---|---|---|
| 0.99 | 4.3 | 5.4 | 6.8 | 4.6 | 5.0 |
| 0.97 | 5.0 | 7.3 | 12.1 | 6.3 | 6.6 |
| 0.95 | 6.1 | 11.3 | 17.4 | 7.3 | 7.8 |

**Top features (tree-rich, P1):** couple_rank 0.64, dr_min 0.15, dca_sig_k 0.08, dz_dist 0.06, m_ijk 0.04, n_pixel_k 0.01, pt_k 0.01, dz_spread 0.00

```
|--- couple_rank <= 17.50
|   |--- dr_min <= 0.68
|   |   |--- dca_sig_k <= 1.42
|   |   |   |--- truncated branch of depth 4
|   |   |--- dca_sig_k >  1.42
|   |   |   |--- truncated branch of depth 4
|   |--- dr_min >  0.68
|   |   |--- m_ijk <= 1.42
|   |   |   |--- truncated branch of depth 4
|   |   |--- m_ijk >  1.42
|   |   |   |--- truncated branch of depth 4
|--- couple_rank >  17.50
|   |--- dr_min <= 0.53
|   |   |--- dca_sig_k <= 1.68
|   |   |   |--- truncated branch of depth 4
|   |   |--- dca_sig_k >  1.68
|   |   |   |--- truncated branch of depth 4
|   |--- dr_min >  0.53
|   |   |--- dca_sig_k <= 2.18
|   |   |   |--- truncated branch of depth 4
|   |   |--- dca_sig_k >  2.18
|   |   |   |--- truncated branch of depth 4
```

## Pool P2

| floor | manual× | tree-rich× | gbdt-rich× | tree-gate4× | gbdt-gate4× |
|---|---|---|---|---|---|
| 0.99 | 8.9 | 10.4 | 13.1 | 9.2 | 10.0 |
| 0.97 | 11.2 | 18.4 | 26.3 | 13.8 | 14.6 |
| 0.95 | 13.7 | 24.5 | 39.2 | 16.5 | 17.9 |

**Top features (tree-rich, P2):** couple_rank 0.63, dr_min 0.18, dca_sig_k 0.07, dz_dist 0.05, m_ijk 0.04, n_pixel_k 0.03, pt_k 0.01, dz_spread 0.01

```
|--- couple_rank <= 17.50
|   |--- dr_min <= 0.69
|   |   |--- dca_sig_k <= 1.30
|   |   |   |--- truncated branch of depth 4
|   |   |--- dca_sig_k >  1.30
|   |   |   |--- truncated branch of depth 4
|   |--- dr_min >  0.69
|   |   |--- m_ijk <= 1.42
|   |   |   |--- truncated branch of depth 4
|   |   |--- m_ijk >  1.42
|   |   |   |--- truncated branch of depth 4
|--- couple_rank >  17.50
|   |--- dr_min <= 0.53
|   |   |--- dca_sig_k <= 1.67
|   |   |   |--- truncated branch of depth 4
|   |   |--- dca_sig_k >  1.67
|   |   |   |--- truncated branch of depth 4
|   |--- dr_min >  0.53
|   |   |--- m_ijk <= 1.33
|   |   |   |--- truncated branch of depth 4
|   |   |--- m_ijk >  1.33
|   |   |   |--- truncated branch of depth 4
```
