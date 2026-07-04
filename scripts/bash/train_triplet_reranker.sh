#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash train_triplet_reranker.sh <experiment_name> [extra trainer args...]"
    echo "  WAIT=1 blocks until the training session finishes (used by triplet_pipeline.sh)."
    exit 1
fi
EXPERIMENT_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION_TRAIN="triplet_train"
SESSION_GPU="triplet_gpu"

# util-linux script(1) is mandatory: -e propagates the exit code, -f flushes every
# write so tail -f survives SSH drops (BSD/macOS script lacks both -> server-only).
if ! script --version >/dev/null 2>&1; then
    echo "ERROR: util-linux script(1) required (this launcher is server-only)."
    exit 1
fi

if [ -f /venv/part/bin/activate ]; then
    ACTIVATE="source /venv/part/bin/activate"
elif [ -x /venv/part/bin/python ]; then
    ACTIVATE="export PATH=/venv/part/bin:\${PATH}"   # venv without an activate script
else
    if command -v conda &>/dev/null; then
        CONDA_BASE=$(conda info --base)
    elif [ -d "/opt/miniconda3" ]; then
        CONDA_BASE="/opt/miniconda3"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "/root/miniconda3" ]; then
        CONDA_BASE="/root/miniconda3"
    else
        echo "ERROR: neither /venv/part nor conda found."
        exit 1
    fi
    ACTIVATE="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate part"
fi
eval "${ACTIVATE}"

TRIPLET_RANK_DIR="${SCRIPT_DIR}/data/low-pt/eval/triplet_rank"
python - <<PY
import os
import pyarrow.parquet as pq
directory = "${TRIPLET_RANK_DIR}"
for name in os.environ.get('PREFLIGHT_ARTIFACTS', 'candidates_train tracks_train').split():
    path = os.path.join(directory, name + '.parquet')
    assert os.path.exists(path), f'missing artifact: {path}'
    print(f'{name}: {pq.read_metadata(path).num_rows} rows')
PY

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/triplet_train_${TS}.log"
SENTINEL="${LOG_DIR}/.triplet_train_exit"
RUN_DIR="${SCRIPT_DIR}/experiments/${EXPERIMENT_NAME}_${TS}"
mkdir -p "${RUN_DIR}"
git -C "${SCRIPT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_sha.txt" 2>/dev/null || true
rm -f "${SENTINEL}"

# Keep TRAIN_CMD free of embedded double quotes: it is interpolated into the
# script -c argument below.
TRAIN_CMD="${ACTIVATE} && cd ${SCRIPT_DIR} && python train_triplet_reranker.py \
    --experiment-dir ${RUN_DIR} \
    $*"

if screen -list 2>/dev/null | grep -q "${SESSION_TRAIN}"; then
    echo "Screen '${SESSION_TRAIN}' already running. Kill: screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi
screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' | while read -r s; do screen -S "$s" -X quit 2>/dev/null || true; done || true

screen -dmS "${SESSION_TRAIN}" bash -c "script -efq -c \"${TRAIN_CMD}\" ${LOG_FILE}; echo \$? > ${SENTINEL}; echo '--- finished. Enter to close. ---'; read"
screen -dmS "${SESSION_GPU}" bash -c "watch -n 1 nvidia-smi"

echo "Launched ${SESSION_TRAIN} + ${SESSION_GPU}."
echo "  run dir:   ${RUN_DIR}"
echo "  log:       tail -f ${LOG_FILE}"
echo "  reattach:  screen -r ${SESSION_TRAIN}"
echo "  progress:  grep 'epoch' ${LOG_FILE} | tail"
echo "  expect:    metrics_history.json, checkpoints/best_model.pt, final_eval.json in the run dir"

if [ "${WAIT:-0}" = "1" ]; then
    echo "WAIT=1: blocking on ${SENTINEL}..."
    while [ ! -f "${SENTINEL}" ]; do sleep 30; done
    RC="$(cat "${SENTINEL}")"
    echo "training session finished with exit code ${RC}"
    exit "${RC}"
fi
