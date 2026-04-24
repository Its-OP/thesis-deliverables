# MIGRATION_GUIDELINE.md — thesis-deliverables

Scope: defines the **limits of the project**. Every `.py`, every `.yaml`, every condor / shell file in the source tree (`/Users/oleh/Projects/masters/part/` + `/Users/oleh/Projects/masters/weaver/weaver/`) is classified as **in scope** (migrates to `thesis-deliverables`) or **out of scope** (stays behind / deletes). Classifications are evidence-backed (import edge, grep hit, config reference).

Purpose: after reading this, migration execution is mechanical — file moves, splits, and rewrites follow from the tables below without further research.

Source tree state: post-commit `7136275` (Cascade Stage 1 BatchNorm pin) on branch `main` of the source monorepo; `deliverables/` repo is empty (only `.git/`).

---

## 1. Project Limits

### 1.1 IN — production 3-stage cascade runtime graph

Three training entrypoints, three frozen-cascade inference wrappers, one shared data-loading harness, one shared training toolbox. Everything listed below is directly on a forward-pass / training-loop code path.

**Stage 1 — TrackPreFilter**

- CLI: `part/train_prefilter.py` (1368 lines)
- Network wrapper: `part/networks/lowpt_tau_TrackPreFilter.py` (145 lines)
- Model: `weaver/weaver/nn/model/TrackPreFilter.py` (1373 lines)
- Direct model deps: `HierarchicalGraphBackbone.cross_set_knn`, `HierarchicalGraphBackbone.cross_set_gather`, `HierarchicalGraphBackbone.euclidean_cross_set_knn` (added on `prefilter-dynamic-knn` branch)
- Conditional deps (enabled by default P1 head + losses): `prefilter_expressiveness.PerFeatureEmbedding`, `prefilter_losses.*`
- Edge-feature path: `HierarchicalGraphBackbone` internally imports `ParticleTransformer.pairwise_lv_fts`

**Stage 2 — CascadeReranker (ParT-style pairwise-bias encoder over Stage-1 top-K₁)**

- CLI: `part/train_cascade.py` (1077 lines)
- Network wrapper: `part/networks/lowpt_tau_CascadeReranker.py` (226 lines)
- Inference wrapper: `weaver/weaver/nn/model/CascadeModel.py` (191 lines)
- Model: `weaver/weaver/nn/model/CascadeReranker.py` (768 lines)
- Direct model deps: `ParticleTransformer.Block`, `ParticleTransformer.Embed`, `ParticleTransformer.PairEmbed`
- BN-pin wrapper: `weaver/weaver/nn/model/force_train_bn.py` (57 lines)
- Loads Stage-1 from `--stage1-checkpoint`; Stage-1 weights frozen inside `CascadeModel`

**Stage 3 — CoupleReranker (per-couple scorer over Stage-2 top-K₂)**

- CLI: `part/train_couple_reranker.py` (1483 lines)
- Network wrapper: `part/networks/lowpt_tau_CoupleReranker.py` (278 lines)
- Inference wrapper: `weaver/weaver/nn/model/CoupleCascadeModel.py` (281 lines)
- Model: `weaver/weaver/nn/model/CoupleReranker.py` (1154 lines, includes `NanSafeBatchNorm1d` + `ResidualBlock`)
- Feature extraction: `part/utils/couple_features.py` (770 lines — exports `COUPLE_FEATURE_DIM*`, `PAIR_PHYSICS_*`)
- Optional SAM/ASAM: `part/utils/sam_optimizer.py` (115 lines)
- Loads cascade from `--cascade-checkpoint` (glob pattern); Stage-1 + Stage-2 frozen inside `CoupleCascadeModel`

**Shared across all three stages**

- `weaver/weaver/nn/model/HierarchicalGraphBackbone.py` (621 lines) — kept as "graph primitives only"; `farthest_point_sampling` + the hierarchical-backbone class are unused in the cascade and drop out on split.
- `weaver/weaver/nn/model/ParticleTransformer.py` (739 lines) — kept as "ParT blocks + `pairwise_lv_fts` only"; dataset-specific heads drop.
- `weaver/weaver/utils/dataset.py` (385 lines) — `SimpleIterDataset`, used by all three `train_*.py`.
- `weaver/weaver/utils/data/config.py` (274 lines) — `DataConfig`, `_md5`; transitively used via `dataset.py`.
- `weaver/weaver/utils/data/preprocess.py` (300 lines) — `AutoStandardizer`, `WeightMaker`, `_apply_selection`, `_build_new_variables`, `_build_weights`; transitively used via `dataset.py`.
- `weaver/weaver/utils/data/fileio.py` — `_read_files`; transitively used.
- `weaver/weaver/utils/data/tools.py` — `_pad`, `_repeat_pad`, `_clip`, `_stack`, `_concat`, `_get_variable_names`, `_eval_expr`; transitively used.
- `weaver/weaver/utils/logger.py` — `_logger`, `warn_n_times`; transitively used.
- `part/utils/optimizers/{__init__.py, soap.py, muon.py}` (971 lines) — `build_optimizer`, `OPTIMIZER_NAMES` (adamw/soap/muon); used by Stages 1 & 2. Stage 3 uses `torch.optim.AdamW` directly but could go through this factory.
- `part/utils/training_utils.py` (1061 lines) — `CheckpointManager`, `MetricsAccumulator`, `CoupleMetricsAccumulator`, `extract_label_from_inputs`, `load_network_module`, `trim_to_max_valid_tracks`, `save_epoch_metrics`, `format_couple_metrics_table`.
- `part/pretrain_backbone.py` (1162 lines) — **only the helper band** (`WarmupThenCosineScheduler`, `WarmupThenPlateauScheduler`, `_TeeStream`, `build_experiment_directory`, `plot_loss_curves`, `save_loss_history`) is live; the pretraining training loop itself is obsolete. Helpers must be extracted.
- `part/utils/set_augmentation.py` (174 lines) — Stage-1 augmentation (optional CLI flag).
- `part/utils/train_augmentation.py` (147 lines) — Stage-3 cov-smear augmentation (optional CLI flag).

### 1.2 OUT — dead code (confirmed drop)

High-confidence: no production importer, no transitive edge from any file in §1.1.

| Category | Files | Evidence |
|---|---|---|
| Legacy Stage-1 architectures | `weaver/.../TauTrackFinder.py`, `TauTrackFinderHead.py`, `TauTrackFinderOC.py`, `TauTrackFinderV2.py`, `TauTrackFinderV3.py`, `ObjectCondensationHead.py`, `EnrichCompactBackbone.py`, `ParallelBackbone.py`, `ParticleNeXt.py`, `ParticleNet.py` + matching `part/networks/lowpt_tau_TrackFinder*.py` | Replaced by TrackPreFilter; zero imports from any file in §1.1 |
| Pretraining | `weaver/.../BackbonePretraining.py`, `weaver/.../hungarian_matcher.py`, `weaver/.../test_hungarian.py`, `part/networks/lowpt_tau_BackbonePretrain.py`, `part/pretrain_backbone.py` (loop body), `part/sanity_check_pretrain.py`, `part/diagnose_pretraining.py`, `part/train_pretrain.sh`, `part/condor/pretrain.sub`, `part/condor/run_pretrain.sh` | Backbone pretraining was a Stage 0 explored earlier then abandoned; cascade does not depend on a pretrained backbone |
| Two-tier experiment (P6) | `weaver/.../TwoTierPreFilter.py`, `part/networks/lowpt_tau_TwoTierPreFilter.py`, `part/tests/test_two_tier_prefilter.py` | Only text reference in `train_prefilter.py` is in comments (two lines); not invoked on the production path |
| Weaver optimizer zoo | `weaver/weaver/utils/nn/optimizer/{__init__.py, radam.py, ranger.py, lookahead.py}`, `weaver/weaver/utils/nn/{__init__.py, metrics.py, tools.py}` | Zero imports from `part/`; `part/utils/optimizers/` is the live factory |
| Weaver utilities off path | `weaver/weaver/utils/{flops_counter.py, import_tools.py, lr_finder.py}`, `weaver/weaver/train.py` | Zero imports from `part/` |
| Legacy network examples | `part/networks/example_PCNN.py`, `example_PFN.py`, `example_ParticleNet.py`, `example_ParticleTransformer.py` | JetClass baselines from upstream ParT repo; not tau pipeline |
| Non-tau benchmark data + scripts | `part/data/{JetClass, QuarkGluon, TopLandscape}/`, `part/train_{JetClass, QuarkGluon, TopLandscape}.sh`, `part/utils/convert_qg_datasets.py`, `part/utils/convert_top_datasets.py`, `part/get_datasets.py`, `part/dataloader.py`, `part/env.sh`, `part/LICENSE`, `part/README_original.md` | Upstream ParT repo residue |
| Legacy TrackFinder training | `part/train_trackfinder.py`, `part/train_trackfinder.sh`, `part/train_trackfinder_v3.sh` | Obsolete pre-cascade training |
| Run / build artefacts | `part/runs/` (empty), `part/build/` (empty), `part/logs/`, `part/experiments/` (~1.8 GB), `part/models/` (~8.3 GB) | Run output; not source. Checkpoints hosted separately |
| Sweep drivers (historical) | `part/sweep_couple_batch{2..6}.sh`, `part/sweep_couple_batch5_continued.sh`, `part/sweep_couple_improvements.sh`, `part/sweep_prefilter.sh`, `part/sweep_prefilter_expressiveness.sh`, `part/sweep_topk2.sh`, `part/eval_couple_reranker.sh` | Research campaign drivers; stay in legacy repo as archive; deliverables repo ships a minimal `cli/eval_cascade.py` instead |
| Diagnostics / reports | `part/diagnostics/` (27 analysis scripts), `part/reports/` (54+ md / parquet / graphics), `part/figures/` (3 static PNGs), `part/notebooks/` (1 .ipynb), `part/thesis_figures/` (5 chapter scripts) | Research archive; thesis LaTeX repo `Its-OP/masters-thesis-latex` references these — keep in legacy repo, do NOT migrate |
| One-off data utilities | `part/utils/{convert_lowpt_tau_dataset.py, dataset_utils.py, split_parquet.py, validate_parquet_quality.py}` | Used to build the parquet dataset once; not on the training loop; keep in legacy repo |
| Legacy top-level shells | `part/train_prefilter.sh`, `part/train_cascade.sh`, `part/train_couple_reranker.sh` | Replaced by `cli/train_*.py` entrypoints in the new repo |

### 1.3 Boundary-crossing interfaces

| Interface | Artefact | Current location | Migration target |
|---|---|---|---|
| Dataset (local) | `train/*.parquet` + `val/*.parquet` | `part/data/low-pt/{train,val}/` on dev machine | **Not versioned**; document paths in new README; user copies/links at runtime |
| Dataset (lxplus) | Same parquets | `/eos/user/o/oprostak/tau_data/parquet_clean/` | Document path in new README |
| Data config (input spec) | `lowpt_tau_trackfinder.yaml` | `part/data/low-pt/lowpt_tau_trackfinder.yaml` | `data/configs/stage1.yaml` (Stages 2 & 3 reuse this; no stage-specific yamls exist) |
| Standardization cache | `lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml` | Committed alongside the yaml | Keep alongside yaml; `md5(yaml)` dictates filename, so changing the yaml invalidates the cache automatically |
| Stage-1 checkpoint | PyTorch state_dict | `part/models/prefilter_best.pt` | Per-stage separate weights on external host (LFS / S3 / zenodo); see §5 open risk |
| Stage-2 checkpoint | Full cascade state_dict + `saved_args` metadata dict | `part/models/cascade_best.pt` | Same |
| Stage-3 checkpoint | Slim `couple_reranker_state_dict` only (cascade reloaded separately) | `part/models/couple_best.pt` | Same |
| HTCondor data prep | `convert_to_parquet.sub`, `merge_batches.sub`, `regenerate_parquet.dag`, wrapper `.sh` scripts | `part/scripts/condor/{convert_to_parquet,merge_root_batches}/` | `condor/` — but see §5 open risk: wrapper `.sh` scripts invoke AFS-hosted Python (`/afs/cern.ch/user/o/oprostak/condor/…/*.py`), not the in-repo `convert_root_to_parquet.py` / `merge_batches.py` |
| GPU-server workflow | Ad-hoc vast.ai + `setup_server.sh` at `/Users/oleh/Projects/masters/setup_server.sh` | Repo root | Migrate `setup_server.sh` to `deliverables/setup_server.sh` (or drop and document the 4 steps in README) |

---

## 2. Per-file disposition table

Verdict vocabulary:
- **keep-verbatim** — copy file unchanged to the target path.
- **keep-split** — migrate, but split into ≥2 smaller files per §3.
- **rewrite** — rewrite from scratch at a new target path (content maps, but stripped/reshaped).
- **rewrite-split** — rewrite + split across multiple new files.
- **drop** — not migrated.
- **extract** — keep only specific symbols from the file; rest drops.

Target paths assume the flat tree from the earlier plan:
```
thesis-deliverables/
├── models/{common, prefilter, stage2_part, stage3_couple, cascade}/
├── data/
├── training/
├── cli/
├── utils/
├── condor/
└── tests/   (migration-only, deleted after)
```

### 2a. `weaver/weaver/` files

| current_path | verdict | target_path | evidence |
|---|---|---|---|
| `nn/model/TrackPreFilter.py` (1373 L) | rewrite-split | `models/prefilter/{model,encoder,message_passing,dynamic_knn,losses_glue}.py` | §3.1 split plan; live Stage-1 model |
| `nn/model/CascadeReranker.py` (768 L) | rewrite-split | `models/stage2_part/{model,pair_builder,losses}.py` | §3.2 split plan; live Stage-2 model |
| `nn/model/CoupleReranker.py` (1154 L) | rewrite-split | `models/stage3_couple/{model,layers,pooling,losses}.py` | §3.3 split plan; live Stage-3 model |
| `nn/model/CascadeModel.py` (191 L) | keep-verbatim | `models/cascade/cascade_model.py` | <300 L; Stage 1+2 inference wrapper |
| `nn/model/CoupleCascadeModel.py` (281 L) | keep-verbatim | `models/cascade/couple_cascade_model.py` | <300 L; Stage 1+2+3 inference wrapper |
| `nn/model/HierarchicalGraphBackbone.py` (621 L) | extract | `models/common/graph.py` | Keep `cross_set_knn`, `cross_set_gather`, `euclidean_cross_set_knn` + the pairwise-feature preprocessing; drop `farthest_point_sampling` + `HierarchicalGraphBackbone` class (no importer) |
| `nn/model/ParticleTransformer.py` (739 L) | extract | `models/stage2_part/attention.py` + `models/common/edge_features.py` | Keep `Block`, `Embed`, `PairEmbed` (Stage-2 imports) → attention.py; `pairwise_lv_fts` → edge_features.py. Drop rest of file (`ParticleTransformer` class itself + its tagging heads + JetClass plumbing) |
| `nn/model/prefilter_expressiveness.py` (371 L) | extract | `models/prefilter/expressiveness.py` | Keep `PerFeatureEmbedding` (P1 — production default). `FeatureGate` (P2), `FiLMHead` (P3), `SoftAttentionAggregator` (P4) drop — P2/P3/P4 are all off by default and no production checkpoint carries them |
| `nn/model/prefilter_losses.py` (265 L) | extract | `models/prefilter/losses.py` | Keep `listwise_ce_loss`, `infonce_in_event`, `logit_adjust_offset`. Drop `object_condensation_loss` (OC head dropped) |
| `nn/model/force_train_bn.py` (57 L) | keep-verbatim | `models/cascade/force_train_bn.py` | Used by `CascadeModel`; <300 L |
| `nn/model/TwoTierPreFilter.py` | drop | — | P6 experiment; production uses single-tier; only one test imports (drops with its test) |
| `nn/model/BackbonePretraining.py` | drop | — | Obsolete Stage-0 pretraining |
| `nn/model/hungarian_matcher.py` | drop | — | Only consumed by BackbonePretraining + TauTrackFinderV3 (both drop) |
| `nn/model/test_hungarian.py` | drop | — | Tests `hungarian_matcher` |
| `nn/model/TauTrackFinder.py` | drop | — | Legacy Stage-1 |
| `nn/model/TauTrackFinderHead.py` | drop | — | Legacy Stage-1 helper |
| `nn/model/TauTrackFinderOC.py` | drop | — | Legacy OC variant |
| `nn/model/TauTrackFinderV2.py` | drop | — | Legacy v2 |
| `nn/model/TauTrackFinderV3.py` | drop | — | Legacy v3 |
| `nn/model/ObjectCondensationHead.py` | drop | — | Unused head |
| `nn/model/ParallelBackbone.py` | drop | — | Only used by V3 |
| `nn/model/ParticleNeXt.py` | drop | — | Only used by EnrichCompactBackbone + ParallelBackbone |
| `nn/model/EnrichCompactBackbone.py` | drop | — | Only used by TauTrackFinder variants |
| `nn/model/ParticleNet.py` | drop | — | JetClass baseline; not tau |
| `nn/model/__init__.py` | drop | — | Empty |
| `utils/dataset.py` (385 L) | rewrite | `data/dataset.py` | Keep `SimpleIterDataset` logic; trim weaver-specific `copy_config` + `torch.utils.data.get_worker_info` plumbing to match the flat tree |
| `utils/data/config.py` (274 L) | keep-verbatim | `data/config_parser.py` | Used transitively by `dataset.py`; self-contained |
| `utils/data/preprocess.py` (300 L) | keep-verbatim | `data/standardization.py` | Contains `AutoStandardizer` + `WeightMaker`; used transitively |
| `utils/data/fileio.py` | keep-verbatim | `data/fileio.py` | `_read_files`; parquet IO |
| `utils/data/tools.py` | keep-verbatim | `data/_tools.py` | Array padding + expr helpers |
| `utils/data/__init__.py` | drop | — | Empty |
| `utils/logger.py` | keep-verbatim | `utils/logger.py` | `_logger`, `warn_n_times`; used repo-wide |
| `utils/__init__.py` | drop | — | Empty |
| `utils/nn/optimizer/radam.py` | drop | — | Unused |
| `utils/nn/optimizer/ranger.py` | drop | — | Unused |
| `utils/nn/optimizer/lookahead.py` | drop | — | Unused |
| `utils/nn/optimizer/__init__.py` | drop | — | Unused |
| `utils/nn/metrics.py` | drop | — | Unused |
| `utils/nn/tools.py` | drop | — | Unused |
| `utils/nn/__init__.py` | drop | — | Unused |
| `utils/flops_counter.py` | drop | — | Unused |
| `utils/import_tools.py` | drop | — | Unused |
| `utils/lr_finder.py` | drop | — | Unused |
| `train.py` (weaver-core CLI) | drop | — | Not used; `part/train_*.py` replace it |

### 2b. `part/` files

| current_path | verdict | target_path | evidence |
|---|---|---|---|
| `train_prefilter.py` (1368 L) | rewrite-split | `cli/train_prefilter.py` (~120 L) + shared `training/trainer.py` / `training/metrics.py` / `training/checkpointing.py` | §3.4 consolidation; ~70 % scaffolding is common with Stages 2 & 3 |
| `train_cascade.py` (1077 L) | rewrite-split | `cli/train_stage2.py` (~120 L) + shared | §3.4 |
| `train_couple_reranker.py` (1483 L) | rewrite-split | `cli/train_stage3.py` (~180 L — retains SAM + BN-calibration tail) + shared | §3.4 |
| `networks/lowpt_tau_TrackPreFilter.py` (145 L) | rewrite | `models/prefilter/factory.py` | Factory function `get_model`; merge into `prefilter/__init__.py` or keep as a thin `factory.py` |
| `networks/lowpt_tau_CascadeReranker.py` (226 L) | rewrite | `models/stage2_part/factory.py` | Factory + Stage-1 kwarg inference helper |
| `networks/lowpt_tau_CoupleReranker.py` (278 L) | rewrite | `models/stage3_couple/factory.py` | Factory + cascade-checkpoint loader |
| `utils/training_utils.py` (1061 L) | rewrite-split | `training/{metrics.py, checkpointing.py, dataset_helpers.py, couple_metrics.py}` | §3.5 split plan |
| `utils/couple_features.py` (770 L) | rewrite-split | `models/stage3_couple/{features,physics_features}.py` | §3.6 split plan |
| `utils/optimizers/__init__.py` (220 L) | keep-verbatim | `training/optimizer.py` | Factory `build_optimizer`; <300 L |
| `utils/optimizers/soap.py` (446 L) | keep-verbatim | `training/optimizers/soap.py` | Self-contained optimizer |
| `utils/optimizers/muon.py` (305 L) | keep-verbatim | `training/optimizers/muon.py` | Self-contained optimizer |
| `utils/sam_optimizer.py` (115 L) | keep-verbatim | `training/optimizers/sam.py` | Stage-3 SAM wrapper |
| `utils/set_augmentation.py` (174 L) | keep-verbatim | `training/augmentation_stage1.py` | Stage-1 optional augmentation |
| `utils/train_augmentation.py` (147 L) | keep-verbatim | `training/augmentation_stage3.py` | Stage-3 optional augmentation |
| `pretrain_backbone.py` (1162 L) | extract | `training/scheduler.py` + `training/experiment_dir.py` | Keep only the helper band (6 helpers, ~200 L total). Pretraining loop body drops |
| `networks/lowpt_tau_BackbonePretrain.py` | drop | — | Obsolete |
| `networks/lowpt_tau_TrackFinder.py` | drop | — | Legacy |
| `networks/lowpt_tau_TrackFinderOC.py` | drop | — | Legacy |
| `networks/lowpt_tau_TrackFinderV2.py` | drop | — | Legacy |
| `networks/lowpt_tau_TrackFinderV3.py` | drop | — | Legacy |
| `networks/lowpt_tau_TwoTierPreFilter.py` | drop | — | Killed P6 experiment |
| `networks/example_PCNN.py` | drop | — | JetClass baseline |
| `networks/example_PFN.py` | drop | — | JetClass baseline |
| `networks/example_ParticleNet.py` | drop | — | JetClass baseline |
| `networks/example_ParticleTransformer.py` | drop | — | JetClass baseline |
| `train_trackfinder.py` | drop | — | Legacy |
| `pretrain_backbone.py` (loop body) | drop | — | Helpers extracted above; body obsolete |
| `sanity_check_pretrain.py` | drop | — | Pretraining diagnostic |
| `diagnose_pretraining.py` | drop | — | Pretraining diagnostic |
| `dataloader.py` | drop | — | Superseded by `weaver.utils.dataset` |
| `get_datasets.py` | drop | — | One-off JetClass downloader |
| `__init__.py` (part/) | drop | — | Empty |
| `utils/convert_lowpt_tau_dataset.py` (508 L) | drop | — | Parquet conversion, run once; source ROOT files stay on lxplus. Document in `condor/README.md` if we need to recreate |
| `utils/convert_qg_datasets.py` | drop | — | JetClass/QG |
| `utils/convert_top_datasets.py` | drop | — | TopLandscape |
| `utils/dataset_utils.py` (219 L) | drop | — | Used only by `convert_lowpt_tau_dataset.py` + data-validation scripts |
| `utils/split_parquet.py` | drop | — | One-off parquet split |
| `utils/validate_parquet_quality.py` (999 L) | drop | — | Dataset QC; run once |

### 2c. Shell scripts, condor, data configs, non-Python artefacts

| current_path | verdict | target_path | evidence |
|---|---|---|---|
| `part/data/low-pt/lowpt_tau_trackfinder.yaml` | keep-verbatim | `data/configs/stage1.yaml` | Only live input-spec yaml. Stages 2 & 3 share it |
| `part/data/low-pt/lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml` | keep-verbatim | `data/configs/stage1.<md5>.auto.yaml` | Standardization cache; MD5-pinned to yaml; regenerate if yaml changes |
| `part/data/low-pt/{train,val,eval,split,subset}/` (parquet) | drop (from git) | Documented path; user-provided | Not versioned |
| `part/data/low-pt/description.tex` | drop | — | LaTeX snippet; belongs in thesis repo |
| `part/data/low-pt/dataset.zip`, `dataset_old.zip`, `example_root.root`, `unzip_dataset.sh` | drop | — | Dataset bundles; not source |
| `part/data/{JetClass,QuarkGluon,TopLandscape}/` | drop | — | Benchmark data, not tau |
| `part/condor/pretrain.sub`, `part/condor/run_pretrain.sh` | drop | — | Pretraining (obsolete) |
| `part/scripts/condor/convert_to_parquet/convert_to_parquet.sub` | keep-verbatim | `condor/convert_to_parquet.sub` | Live data-prep; verify paths at migration time |
| `part/scripts/condor/convert_to_parquet/run_convert.sh` | rewrite | `condor/run_convert.sh` | **Currently invokes AFS-hosted Python**; need to replace with a path inside the deliverables repo (see §5.3) |
| `part/scripts/condor/convert_to_parquet/convert_root_to_parquet.py` | TBD | `condor/convert_root_to_parquet.py` | In-repo copy; may or may not match live AFS copy — verify at migration |
| `part/scripts/condor/merge_root_batches/merge_batches.sub` | keep-verbatim | `condor/merge_batches.sub` | Live data-prep |
| `part/scripts/condor/merge_root_batches/run_merge.sh` | rewrite | `condor/run_merge.sh` | **Currently invokes AFS-hosted Python**; same issue |
| `part/scripts/condor/merge_root_batches/merge_batches.py` | TBD | `condor/merge_batches.py` | Same verification |
| `part/scripts/condor/regenerate_parquet.dag` | keep-verbatim | `condor/regenerate_parquet.dag` | HTCondor DAG descriptor |
| `part/scripts/check_pt_fp16_binning.py` | drop | — | One-off dataset QC |
| `part/scripts/render_experiment_chart.py`, `render_report_pdf.py`, `render_table.py` | drop | — | Report-rendering utilities; thesis repo owns presentation |
| `part/scripts/convert_lowpt_tau.sh` | drop | — | One-off launcher |
| `part/environment.yml`, `part/requirements.txt` | rewrite | repo root `environment.yml` + `requirements.txt` | Current files carry JetClass / legacy deps; slim to cascade-only |
| `part/train_prefilter.sh`, `train_cascade.sh`, `train_couple_reranker.sh` | drop | — | Stand-in CLI; replaced by `cli/train_*.py` thin argparse entrypoints |
| `part/sweep_*.sh` (8 files) | drop | — | Research campaign drivers; stay in legacy repo |
| `part/eval_couple_reranker.sh` | rewrite | `cli/eval_cascade.py` | Minimal end-to-end evaluation entrypoint |
| `part/env.sh`, `train_JetClass.sh`, `train_QuarkGluon.sh`, `train_TopLandscape.sh`, `train_pretrain.sh`, `train_trackfinder{,_v3}.sh` | drop | — | Legacy / non-tau |
| `part/LICENSE`, `part/README_original.md` | drop | — | Upstream ParT repo residue |
| `part/diagnostics/`, `reports/`, `figures/`, `notebooks/`, `thesis_figures/`, `models/`, `experiments/`, `logs/`, `runs/`, `build/` | drop from new repo | — | Stay in legacy repo; referenced by the LaTeX thesis |
| `part/tests/*` (35 files) | see §2d | — | 20 migrate for phased gates, 15 drop |

### 2d. Tests (migration-only — deleted after migration validated)

| test_file | verdict | covers |
|---|---|---|
| `test_track_prefilter.py` | migrate | Stage 1 core |
| `test_track_prefilter_expressiveness_wiring.py` | migrate (slim) | Stage 1 P1 wiring; drop P2/P3/P4 assertions |
| `test_cascade_model.py` | migrate | `CascadeModel` Stage 1→2 plumbing |
| `test_cascade_reranker.py` | migrate | Stage 2 forward + loss |
| `test_cascade_reranker_loader.py` | migrate | `infer_stage1_kwargs` |
| `test_cascade_ema.py` | migrate | Stage 2 EMA |
| `test_couple_cascade_model.py` | migrate | `CoupleCascadeModel` full pipeline |
| `test_couple_reranker.py` | migrate | Stage 3 forward + loss |
| `test_couple_features.py` | migrate | `couple_features.py` (Stage 3) |
| `test_couple_metrics_accumulator.py` | migrate | C@K / RC@K accumulator |
| `test_cross_set_gather.py` | migrate | `graph.cross_set_gather` |
| `test_extended_metrics.py` | migrate | Shared metrics |
| `test_force_train_bn.py` | migrate | `force_train_bn` |
| `test_metrics_accumulator.py` | migrate | Shared metrics accumulator |
| `test_nan_safe_batchnorm.py` | migrate | Stage 3 BN |
| `test_optimizer_factory.py` | migrate | `build_optimizer` |
| `test_prefilter_expressiveness_heads.py` | migrate (slim) | P1 kept; P2/P3/P4 test blocks drop |
| `test_prefilter_losses.py` | migrate | Stage 1 losses |
| `test_set_augmentation.py` | migrate | Stage 1 augmentation |
| `test_train_couple_reranker.py` | migrate | Stage 3 train helpers |
| `test_train_prefilter_checkpoint_criterion.py` | migrate | Stage 1 CLI |
| `test_aggregate_couple_sweep.py` | drop | Diagnostic |
| `test_analyze_topk2_sweep.py` | drop | Diagnostic |
| `test_compute_couple_metrics.py` | drop | Post-hoc eval |
| `test_eval_couple_reranker.py` | drop | Legacy eval script |
| `test_graph_diagnostic.py` | drop | Diagnostic |
| `test_prefilter_confidence_diagnostic.py` | drop | Diagnostic |
| `test_prefilter_perfect_recall_diagnostic.py` | drop | Diagnostic |
| `test_prefilter_stratified_eval.py` | drop | Diagnostic |
| `test_profile_prefilter.py` | drop | `torch.profiler` hook |
| `test_recall_sweep.py` | drop | Sweep analyzer |
| `test_tau_track_finder.py` | drop | Legacy |
| `test_tau_track_finder_head.py` | drop | Legacy |
| `test_tau_track_finder_v3.py` | drop | Legacy |
| `test_two_tier_prefilter.py` | drop | P6 experiment killed |

All migrated tests are deleted after Phase 6 (see §5). The new repo ships with no `tests/` directory in its final form.

---

## 3. Module-split rules

Every kept file over ~300 LOC gets a deterministic split. Each new file's role, approximate size, and inter-file interface are fixed here. No file in the target repo exceeds ~350 LOC.

### 3.1 `TrackPreFilter.py` (1373 L) → `models/prefilter/`

- `models/prefilter/model.py` (~260 L) — `TrackPreFilter` class: `__init__`, `forward`, `compute_loss` dispatch. Configuration dataclass + feature-validation.
- `models/prefilter/encoder.py` (~180 L) — per-feature embedder (`PerFeatureEmbedding` from `prefilter_expressiveness`) + the `track_mlp` block. Input: (B, F, N) features, mask. Output: (B, H, N) track embeddings.
- `models/prefilter/message_passing.py` (~260 L) — kNN round loop, neighbour MLPs, max-pool aggregation, edge-feature computation. Depends on `models/common/graph.cross_set_knn`, `models/common/edge_features.pairwise_lv_fts`.
- `models/prefilter/dynamic_knn.py` (~140 L) — coord projection (`Conv1d`+`BN` to unit sphere), `coord_to_hidden` residual back-projection, `_dynamic_knn_rebuild` helper. Only imported when `dynamic_knn=True`.
- `models/prefilter/losses_glue.py` (~130 L) — thin wrappers around `models/prefilter/losses.py` (which is the rewritten `prefilter_losses.py`): pairwise ranking, contrastive denoising. Temperature schedules.
- `models/prefilter/expressiveness.py` (~150 L) — extracted from `weaver/.../prefilter_expressiveness.py` — **only `PerFeatureEmbedding` (P1)**. P2/P3/P4 classes drop on the extract boundary; no production checkpoint depends on them.
- `models/prefilter/losses.py` (~180 L) — extracted from `weaver/.../prefilter_losses.py` — `listwise_ce_loss`, `infonce_in_event`, `logit_adjust_offset`. `object_condensation_loss` drops.
- `models/prefilter/factory.py` (~80 L) — replaces `networks/lowpt_tau_TrackPreFilter.py`'s `get_model` function. Returns `(model, model_info)`.

State-dict parity: ensure the new `TrackPreFilter(**config)` produces the same `model.state_dict().keys()` as the weaver version. Submodule nesting (`track_mlp.*`, `neighbor_mlps.*.{norm,conv1,conv2}`, etc.) must match verbatim so current checkpoints load.

### 3.2 `CascadeReranker.py` (768 L) → `models/stage2_part/`

- `models/stage2_part/model.py` (~290 L) — `CascadeReranker` class: `__init__`, `forward`, `compute_loss` dispatch (pairwise, lambda_rank, rs_at_k, hybrid_lambda).
- `models/stage2_part/pair_builder.py` (~160 L) — Stage-1 top-K₁ → Stage-2 input repack. Builds `(B, top_k, F)` tensors + pair indices for the pairwise-bias encoder.
- `models/stage2_part/losses.py` (~180 L) — extracted loss blocks: pairwise ranking, LambdaRank, RS@K (per-batch ranking-recall loss), hybrid LambdaRank. Denoising regularizer (off by default).
- `models/stage2_part/attention.py` (~280 L) — extracted from `weaver/.../ParticleTransformer.py` — `Block`, `Embed`, `PairEmbed` + their helpers. The rest of `ParticleTransformer` (1 kLOC of `ParticleTransformer` class + `ParticleTransformerTagger*` heads + `ParticleTransformerPPO` + JetClass plumbing) drops on the extract boundary.
- `models/stage2_part/factory.py` (~120 L) — replaces `networks/lowpt_tau_CascadeReranker.py`. Owns `infer_stage1_kwargs` helper that reads a Stage-1 checkpoint's `state_dict.keys()` and reconstructs the Stage-1 wrapper config.

### 3.3 `CoupleReranker.py` (1154 L) → `models/stage3_couple/`

- `models/stage3_couple/model.py` (~260 L) — `CoupleReranker` class: `__init__`, `forward`, `compute_loss` dispatch (softmax-CE, pairwise, LambdaNDCG2++, ApproxNDCG).
- `models/stage3_couple/layers.py` (~180 L) — `ResidualBlock`, `NanSafeBatchNorm1d`. These two are self-contained custom layers with unit tests, keep together.
- `models/stage3_couple/pooling.py` (~220 L) — couple-embed heads: `concat`, `projected_infersent`, `infersent`, `symmetric`, `bilinear_lrb`, `ft_transformer`, `per_track_tokens`. Controlled by `--couple-embed-mode`.
- `models/stage3_couple/losses.py` (~220 L) — softmax-CE with label smoothing, pairwise, LambdaNDCG2++, ApproxNDCG. Multi-positive variants (`none`, `uniform`, `soft_or`).
- `models/stage3_couple/features.py` (~260 L) — extracted from `utils/couple_features.py` v1 features: 51-dim base couple features + enumeration helpers.
- `models/stage3_couple/physics_features.py` (~280 L) — v2/v3 extras: `--pair-kinematics-v2`, `--pair-physics-v3`, `--pair-physics-signif`. Exports `PAIR_PHYSICS_V3_EXTRA_DIM`, `PAIR_PHYSICS_SIGNIF_EXTRA_DIM`.
- `models/stage3_couple/factory.py` (~180 L) — replaces `networks/lowpt_tau_CoupleReranker.py`. Owns `_build_frozen_cascade` which parses the cascade checkpoint's `saved_args` dict.

### 3.4 Training-script consolidation (`train_prefilter.py` 1368 + `train_cascade.py` 1077 + `train_couple_reranker.py` 1483 = 3928 L → ~1500 L)

Per Track 4: ~70–80 % of scaffolding is shared (data loader, scheduler, checkpoint manager, logging, AMP, gradient clipping, resume). Consolidate into:

- `training/trainer.py` (~320 L) — `Trainer` class. Methods: `fit`, `train_one_epoch`, `validate`, `build_loader`, `save_checkpoint`, `resume`. Parameterised by: `(model, loss_fn, optimizer, scheduler, metrics_class, checkpoint_criterion, ema_config?, sam_config?)`.
- `training/amp.py` (~40 L) — AMP context manager + GradScaler handling.
- `training/scheduler.py` (~210 L) — `WarmupThenCosineScheduler`, `WarmupThenPlateauScheduler` extracted from `pretrain_backbone.py`. Cosine-power support for Stage 3.
- `training/optimizer.py` (~220 L) — `build_optimizer` + OPTIMIZER_NAMES from `part/utils/optimizers/__init__.py`.
- `training/optimizers/{soap.py, muon.py, sam.py}` — keep-verbatim.
- `training/metrics.py` (~280 L) — `MetricsAccumulator` (P@K, R@K, d_prime, percentile). Stage 1 + Stage 2 accumulator.
- `training/couple_metrics.py` (~240 L) — `CoupleMetricsAccumulator` (C@K, RC@K, D@K_tracks, mean_first_gt_rank). Stage 3 accumulator.
- `training/checkpointing.py` (~240 L) — `CheckpointManager` (keep-best-k by criterion; handles three checkpoint layouts: full-model S1/S2, slim S3).
- `training/dataset_helpers.py` (~120 L) — `extract_label_from_inputs`, `trim_to_max_valid_tracks`, `load_network_module` (or inline into trainer if short).
- `training/experiment_dir.py` (~90 L) — `build_experiment_directory`, `_TeeStream`, `save_loss_history`, `plot_loss_curves` extracted from `pretrain_backbone.py`.
- `training/ema.py` (~140 L) — Stage-2 + Stage-3 EMA context managers, BN calibration (`calibrate_reranker_batchnorm`, 200 steps default).
- `training/augmentation/{stage1.py, stage3.py}` — keep-verbatim `set_augmentation.py` + `train_augmentation.py`.
- `cli/train_prefilter.py` (~120 L) — thin argparse + `trainer.fit(...)`. Owns Stage-1-specific flags (feature-embed, dropout, loss-type, temperature/denoising schedules).
- `cli/train_stage2.py` (~120 L) — thin argparse + `trainer.fit(...)`. Owns Stage-2-specific flags (stage1-checkpoint, top-k1, stage2-* arch, loss-mode, ema-decay).
- `cli/train_stage3.py` (~180 L) — thin argparse + `trainer.fit(...)` with the SAM double-pass path guarded behind `--optim asam` and the post-training BN calibration tail. Owns Stage-3-specific flags (cascade-checkpoint, top-k2, couple-* arch, k-values-tracks, k-values-couples, pair-kinematics-v2 / physics-v3 / physics-signif, train-aug).
- `cli/eval_cascade.py` (~150 L) — end-to-end eval replacing `eval_couple_reranker.sh`.

Consolidation ratio: 3928 L → (~1500 L shared `training/` + ~570 L three thin CLIs) = ~2070 L. **47 % reduction.**

### 3.5 `training_utils.py` (1061 L) → split

Mirrors §3.4 targets — `training/metrics.py` + `training/couple_metrics.py` + `training/checkpointing.py` + `training/dataset_helpers.py`. `save_epoch_metrics` and `format_couple_metrics_table` fold into `training/checkpointing.py` and `training/couple_metrics.py` respectively.

### 3.6 `couple_features.py` (770 L) → `models/stage3_couple/features.py` (~260 L) + `physics_features.py` (~280 L) + `couple_constants.py` (~60 L)

`couple_constants.py` exports `COUPLE_FEATURE_DIM`, `COUPLE_FEATURE_DIM_V2`, `COUPLE_FEATURE_DIM_V3`, `PAIR_PHYSICS_V3_EXTRA_DIM`, `PAIR_PHYSICS_SIGNIF_EXTRA_DIM`. This avoids a cycle between `features.py` and `physics_features.py`.

### 3.7 Shared primitives — extraction map

| symbol | source | target |
|---|---|---|
| `cross_set_knn` | `weaver/.../HierarchicalGraphBackbone.py` | `models/common/graph.py` |
| `cross_set_gather` | same | same |
| `euclidean_cross_set_knn` | same (added on `prefilter-dynamic-knn` branch) | same |
| `pairwise_lv_fts` | `weaver/.../ParticleTransformer.py` | `models/common/edge_features.py` |
| `Block`, `Embed`, `PairEmbed` | same | `models/stage2_part/attention.py` |
| `PerFeatureEmbedding` (P1) | `weaver/.../prefilter_expressiveness.py` | `models/prefilter/expressiveness.py` |
| `listwise_ce_loss`, `infonce_in_event`, `logit_adjust_offset` | `weaver/.../prefilter_losses.py` | `models/prefilter/losses.py` |
| `force_train_bn` | `weaver/.../force_train_bn.py` | `models/cascade/force_train_bn.py` |
| `_logger`, `warn_n_times` | `weaver/.../utils/logger.py` | `utils/logger.py` |
| `DataConfig`, `_md5` | `weaver/.../utils/data/config.py` | `data/config_parser.py` |
| `AutoStandardizer`, `WeightMaker`, `_apply_selection`, `_build_new_variables`, `_build_weights` | `weaver/.../utils/data/preprocess.py` | `data/standardization.py` |
| `SimpleIterDataset` | `weaver/.../utils/dataset.py` | `data/dataset.py` |
| `_read_files` | `weaver/.../utils/data/fileio.py` | `data/fileio.py` |
| `_pad`, `_repeat_pad`, `_clip`, `_stack`, `_concat`, `_get_variable_names`, `_eval_expr` | `weaver/.../utils/data/tools.py` | `data/_tools.py` |

---

## 4. External-system coupling

| Artefact | Current location | Migration action |
|---|---|---|
| Parquet dataset (local) | `/Users/oleh/Projects/masters/part/data/low-pt/{train,val}/` | Not migrated. Document expected layout in new README; user mounts or copies at runtime |
| Parquet dataset (lxplus) | `/eos/user/o/oprostak/tau_data/parquet_clean/` | Document in README under "reproducing on lxplus" |
| Parquet dataset (vast.ai) | Ad-hoc per GPU session; seeded via `setup_server.sh` | Keep `setup_server.sh` at repo root or document the four steps in the README |
| ROOT input corpus | CMS central storage / lxplus user AFS | Out of scope. Condor jobs in `condor/` regenerate the parquet from this — document in `condor/README.md` |
| Standardization cache | `part/data/low-pt/lowpt_tau_trackfinder.c8a40f560c44edfe47c8f0fc25230de1.auto.yaml` | Copy alongside the yaml. `weaver/utils/dataset.py` re-MD5s the base yaml at dataset-construction time; if the hash matches the cache suffix, skip regeneration |
| Checkpoint: Stage 1 | `part/models/prefilter_best.pt` | External host (LFS / S3 / zenodo); new repo's README links to it. **One file per stage — do not bundle**, per project convention |
| Checkpoint: Stage 2 | `part/models/cascade_best.pt` | Same |
| Checkpoint: Stage 3 | `part/models/couple_best.pt` | Same |
| HTCondor data-prep Python | `/afs/cern.ch/user/o/oprostak/condor/{convert_root_to_parquet,compress_source_root}/*.py` — **AFS, outside git** | Resolve at migration: either (a) copy AFS contents into `deliverables/condor/` and rewrite `run_*.sh` to invoke the local copy, or (b) leave `.sub` / `.sh` pointing at AFS and document the AFS layout in `condor/README.md`. Prefer (a); mark as open risk §6.3 |
| HTCondor DAG | `part/scripts/condor/regenerate_parquet.dag` | Migrate verbatim |
| TensorBoard logs | `part/experiments/*/tensorboard/` | Out of scope; run-time artefacts |
| Thesis LaTeX repo | `/Users/oleh/Projects/masters/LaTex` (standalone) | Unaffected by this migration |
| Vast.ai session bootstrap | `/Users/oleh/Projects/masters/setup_server.sh` | Copy to `deliverables/setup_server.sh`; update the git-clone URL to `Its-OP/thesis-deliverables` |
| Conda environment | `part/environment.yml` | Slim to cascade-only (drop JetClass / QG / TopLandscape deps). New top-level `environment.yml` |
| Pip requirements | `part/requirements.txt` | Same |

---

## 5. Phased migration order + TDD gates

Leaves-first. Each phase ends with a test-suite gate that must pass before the next phase starts. Tests are the ones migrated in §2d; they are deleted in Phase 7 after the final gate.

### Phase 1 — Bootstrap

- Write repo root: `README.md` (placeholder), `pyproject.toml`, `requirements.txt`, `environment.yml`, `.gitignore`.
- Create empty package directories: `models/{common,prefilter,stage2_part,stage3_couple,cascade}/`, `data/`, `data/configs/`, `training/`, `training/optimizers/`, `training/augmentation/`, `cli/`, `utils/`, `condor/`, `tests/`.
- Add one `__init__.py` per package (no content).
- **Gate:** `pip install -e .` succeeds; `python -c "import models, data, training, cli, utils"` succeeds.

### Phase 2 — Shared primitives (`models/common/` + `utils/` + `data/`)

- Create `utils/logger.py` (verbatim copy).
- Create `data/{_tools.py, fileio.py, config_parser.py, standardization.py, dataset.py}` (verbatim / rewrite per §2a).
- Create `models/common/{graph.py, edge_features.py}` (extract per §3.7).
- Copy `data/configs/stage1.yaml` + `.auto.yaml` from `part/data/low-pt/`.
- **Gate:** migrate and run `tests/test_cross_set_gather.py`. Also a new smoke test `tests/test_dataconfig_loads.py`: instantiate `SimpleIterDataset` pointing at one of the subset parquet files (`part/data/low-pt/subset/train/`) and iterate one batch.

### Phase 3 — Per-stage models

- Split `TrackPreFilter.py` per §3.1 into `models/prefilter/`. Port `prefilter_expressiveness.py` (P1 only) and `prefilter_losses.py`.
- Split `CascadeReranker.py` per §3.2 into `models/stage2_part/`. Extract `Block`, `Embed`, `PairEmbed` from `ParticleTransformer.py` into `models/stage2_part/attention.py`.
- Split `CoupleReranker.py` per §3.3 into `models/stage3_couple/`. Split `couple_features.py` per §3.6.
- For each stage: write a `factory.py` merging the current network wrapper's `get_model` with the split model.
- **State-dict parity check (CRITICAL):** for each stage, load an existing production checkpoint (`part/models/prefilter_best.pt`, `cascade_best.pt`, `couple_best.pt`) with `model.load_state_dict(..., strict=True)` and assert zero missing/unexpected keys. Do this as a test: `tests/test_state_dict_parity.py`.
- **Gate:** all the following pass: `test_track_prefilter.py`, `test_track_prefilter_expressiveness_wiring.py` (P1-only), `test_prefilter_losses.py`, `test_cascade_reranker.py`, `test_cross_set_gather.py`, `test_couple_reranker.py`, `test_couple_features.py`, `test_nan_safe_batchnorm.py`, `test_force_train_bn.py`, `test_prefilter_expressiveness_heads.py` (P1-only), `test_state_dict_parity.py`.

### Phase 4 — Cascade inference wrappers

- Copy `CascadeModel.py` → `models/cascade/cascade_model.py`; copy `CoupleCascadeModel.py` → `models/cascade/couple_cascade_model.py`; copy `force_train_bn.py` → `models/cascade/force_train_bn.py`.
- Update their imports to point at `models.prefilter`, `models.stage2_part`, `models.stage3_couple`.
- **Gate:** `test_cascade_model.py`, `test_cascade_reranker_loader.py`, `test_couple_cascade_model.py` pass. Plus a forward-pass bit-parity test comparing the new `CascadeModel(stage1_checkpoint=...)` output to a frozen reference tensor captured from the current cascade on one deterministic batch.

### Phase 5 — Training harness

- Extract `WarmupThenCosineScheduler`, `WarmupThenPlateauScheduler`, `build_experiment_directory`, `_TeeStream`, `plot_loss_curves`, `save_loss_history` from `part/pretrain_backbone.py` into `training/scheduler.py` + `training/experiment_dir.py`.
- Port `part/utils/optimizers/{__init__, soap, muon}.py` into `training/optimizer.py` + `training/optimizers/`.
- Port `part/utils/sam_optimizer.py` → `training/optimizers/sam.py`.
- Port `part/utils/{set_augmentation, train_augmentation}.py` into `training/augmentation/`.
- Split `part/utils/training_utils.py` per §3.5 into `training/{metrics, couple_metrics, checkpointing, dataset_helpers, ema}.py`.
- Write `training/trainer.py` + `training/amp.py` that parameterises the loop per §3.4.
- **Gate:** `test_optimizer_factory.py`, `test_metrics_accumulator.py`, `test_couple_metrics_accumulator.py`, `test_extended_metrics.py`, `test_set_augmentation.py`, `test_cascade_ema.py` pass.

### Phase 6 — CLI + end-to-end

- Write `cli/train_prefilter.py`, `cli/train_stage2.py`, `cli/train_stage3.py`, `cli/eval_cascade.py` per §3.4.
- **Gate:** `test_train_couple_reranker.py`, `test_train_prefilter_checkpoint_criterion.py` pass. Then a manual smoke — one-epoch training run for each stage on the subset data (`part/data/low-pt/subset/`) — confirms the loss decreases and val-metrics accumulator reports sane numbers.

### Phase 7 — Condor + README + test deletion

- Port `condor/` per §2c and §4. Resolve the AFS-vs-in-repo question (§6.3) and wire `run_*.sh` to the local copy.
- Write `README.md`: install, 3-stage training recipe, evaluation recipe, checkpoint URLs, lxplus reproduction notes.
- Verify the full test suite passes (`pytest tests/`), then delete `tests/` in the same commit with a message documenting the deletion.
- Tag `v0.1-submission` on the deliverables repo.

### Regression checks (run at every phase ≥ 3)

- State-dict key parity against each current production checkpoint.
- Forward-pass bit-parity on a deterministic batch from `part/data/low-pt/subset/val/`.
- Current test suite (scope-appropriate tests from §2d) green.

---

## 6. Open risks + user-input items

### 6.1 Package name for `pyproject.toml`

- Decision needed: `thesis-deliverables` (dash; pip-install name), `tau3pi` (import-friendly), or `cascade`. Does not affect imports (they stay `from models.prefilter import TrackPreFilter`).
- Blast radius: cosmetic, but locked in once published.

### 6.2 `utils/nn/optimizer/*` — drop confirmation

- All three files (`radam`, `ranger`, `lookahead`) are dropped by this plan. `part/utils/optimizers/` supports only adamw / soap / muon.
- Confirm: no current training run uses `--optimizer radam` (etc.); `grep -r --optimizer part/*.sh part/sweep_*.sh` shows only `adamw`, `soap`, `muon` in use.

### 6.3 Condor data-prep `.sh` → AFS Python

- `part/scripts/condor/*/run_*.sh` call `/afs/cern.ch/user/o/oprostak/condor/…/*.py`, not the in-repo `convert_root_to_parquet.py` / `merge_batches.py`.
- Decision needed: (a) copy AFS Python into `deliverables/condor/` and point `run_*.sh` at the in-repo copy; (b) leave the AFS pointer and document layout; (c) verify in-repo copies already match AFS and retire the AFS pointer.
- Blast radius: unresolved = documentation debt in the deliverables repo; at worst, a future user cannot regenerate the parquet dataset from ROOT.

### 6.4 Checkpoint hosting policy

- Three per-stage checkpoints (~few hundred MB total) cannot live in git. Candidates: GitHub Releases, Git LFS, zenodo DOI, institutional S3.
- Per memory convention: save as separate files (`prefilter_best.pt`, `stage2_best.pt`, `couple_best.pt`), not a bundled `cascade_best.pt`.
- Decision needed: hosting choice.

### 6.5 P2/P3/P4 expressiveness heads — confirm drop

- `feature_gate` (P2), `film_head` (P3), `soft_attention_aggregation` (P4) are all off by default in `lowpt_tau_TrackPreFilter.py` and none of the production checkpoints carry them.
- The entire `FeatureGate`, `FiLMHead`, `SoftAttentionAggregator` code in `prefilter_expressiveness.py` drops on extract. If the thesis text discusses an ablation that needs the numbers but not the running code, nothing changes here — we kept the ablation results in `part/reports/`.
- Decision: confirm drop (this guideline assumes yes).

### 6.6 `part/utils/couple_features.py` v1 / v2 / v3 feature branches

- Controlled by `--pair-kinematics-v2`, `--pair-physics-v3`, `--pair-physics-signif`. All three are off by default but the current production Stage-3 checkpoint may have been trained with one of them on.
- Before dropping any v*-specific code, inspect the production Stage-3 checkpoint's `saved_args` dict: `grep 'pair_kinematics_v2\|pair_physics_v3\|pair_physics_signif' part/experiments/<latest-couple-run>/args.yaml`.
- Decision: which v*-extras are on the kept path. Guideline currently assumes **all three kept** (cheap to keep; §3.6 splits features/physics_features anyway).

### 6.7 `TwoTierPreFilter` drop — no resurrection planned?

- P6 was killed in the 2026-04 expressiveness sweep. Guideline drops it entirely (including `test_two_tier_prefilter.py`, `lowpt_tau_TwoTierPreFilter.py`, `weaver/.../TwoTierPreFilter.py`).
- Confirm the architecture is not revisited in the future thesis roadmap.

### 6.8 Stage-2 EMA and BN calibration default

- Stage 2 EMA is off by default (`--ema-decay=0.0`); Stage 3 EMA likewise. The calibration tail in Stage 3 (`--bn-calibration-steps=200`) runs only when EMA is enabled.
- Decision: if the final thesis run uses EMA, confirm the wired-up EMA code is correct for the new `training/ema.py`. If EMA stays off for the final runs, the code paths still migrate but the smoke-test gate in Phase 5 will not exercise them fully — flag in the README.

### 6.9 `setup_server.sh`

- Currently lives at `/Users/oleh/Projects/masters/setup_server.sh`; clones both `part/` and `weaver/` from the monorepo and runs `pip install -e`.
- Decision: migrate as-is (update git-clone URL) or rewrite against the flat new tree.

### 6.10 Thesis-LaTeX cross-references

- The LaTeX thesis at `Its-OP/masters-thesis-latex` (standalone) references code paths in `part/` by name — e.g. "Stage-1 training script `train_prefilter.py`". Once the deliverables repo renames the script to `cli/train_prefilter.py`, any such reference in the thesis goes stale.
- Decision: either rewrite the thesis references once the deliverables repo is frozen, or leave the deliverables repo's CLI names identical to the current `train_prefilter.py` / `train_cascade.py` / `train_couple_reranker.py` for stability.

---

## 7. Verification

The guideline is testable against the current source tree:

```bash
# From /Users/oleh/Projects/masters

# 1. Every file in §2 exists and matches the LOC class
wc -l part/train_prefilter.py part/train_cascade.py part/train_couple_reranker.py  # expect 1368 1077 1483

# 2. Every §1.1 weaver dep is in the import graph of the three train scripts
git grep -l "from weaver" part/train_prefilter.py part/train_cascade.py part/train_couple_reranker.py

# 3. No kept file imports a dropped file
for drop in BackbonePretraining TauTrackFinder ParticleNet ParticleNeXt EnrichCompactBackbone \
            ObjectCondensationHead ParallelBackbone hungarian_matcher TwoTierPreFilter; do
    git grep -l "from weaver.nn.model.${drop}\|import ${drop}" part/ weaver/weaver/nn/model/ | \
        grep -v -E "(${drop}|__pycache__)" || echo "no importers of ${drop} — drop is safe"
done
```

The final acceptance is a green test suite run in the new `thesis-deliverables` repo at the end of Phase 7, followed by the `tests/` deletion commit.
