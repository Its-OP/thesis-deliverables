#!/bin/bash
# =============================================================================
# Canonical full-eval dump of the promoted three-stage stack at the serving
# operating point K1=400 / K2=125 / 200 couples per event (h6s6 T1, h6s8b):
# on the 67,500-event eval shards the promoted stack reaches stage-2 D@125
# 0.9670 and stage-3 C@100 0.9331 / C@125 0.9388. Without --top-k1/--top-k2
# the evaluator falls back to the K values stored in the checkpoints (K1=256).
# Follow with scripts/bash/compute_eval_metrics.sh on the same OUTPUT.
# Extra args pass through, e.g. --max-events 2000 --device cpu.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="part"

DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
VAL_DATA_DIR="data/low-pt/eval/"
STAGE="couples"
STAGE1_WEIGHTS="models/prefilter_best.pt"
STAGE2_WEIGHTS="models/stage2_best.pt"
STAGE3_WEIGHTS="models/couple_reranker_best.pt"
OUTPUT="data/low-pt/eval/pipeline_eval.parquet"
TOP_K1=400
TOP_K2=125
NUM_COUPLES=200
DEVICE="cuda:0"

if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "/opt/miniconda3" ]; then
    CONDA_BASE="/opt/miniconda3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/root/miniconda3" ]; then
    CONDA_BASE="/root/miniconda3"
else
    echo "ERROR: conda not found."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${SCRIPT_DIR}"
python -m scripts.python.eval_cascade_pipeline \
    --stage "${STAGE}" \
    --stage1-weights "${STAGE1_WEIGHTS}" \
    --stage2-weights "${STAGE2_WEIGHTS}" \
    --stage3-weights "${STAGE3_WEIGHTS}" \
    --data-config "${DATA_CONFIG}" \
    --val-data-dir "${VAL_DATA_DIR}" \
    --output "${OUTPUT}" \
    --top-k1 "${TOP_K1}" \
    --top-k2 "${TOP_K2}" \
    --num-couples "${NUM_COUPLES}" \
    --device "${DEVICE}" \
    "$@"
