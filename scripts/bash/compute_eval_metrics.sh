#!/bin/bash
# =============================================================================
# Metrics JSON (R@K / D@K / C@K / RC@K) for a scripts/bash/eval_cascade_pipeline.sh
# dump. GT labels are read straight from the eval shards, so no data config is
# needed (the --data-config flag was dropped from compute_eval_metrics.py).
# Extra args pass through, e.g. --eval-parquet <other dump> --output <json>.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="part"

EVAL_PARQUET="data/low-pt/eval/pipeline_eval.parquet"
VAL_DATA_DIR="data/low-pt/eval/"
OUTPUT="data/low-pt/eval/pipeline_eval_metrics.json"

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
python -m scripts.python.compute_eval_metrics \
    --eval-parquet "${EVAL_PARQUET}" \
    --val-data-dir "${VAL_DATA_DIR}" \
    --output "${OUTPUT}" \
    "$@"
