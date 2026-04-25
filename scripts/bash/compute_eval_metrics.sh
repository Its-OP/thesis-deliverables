#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="part"

EVAL_PARQUET="data/low-pt/eval/pipeline_val.parquet"
VAL_DATA_DIR="data/low-pt/val/"
DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
OUTPUT="data/low-pt/eval/pipeline_val_metrics.json"

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
    --data-config "${DATA_CONFIG}" \
    --output "${OUTPUT}" \
    "$@"
