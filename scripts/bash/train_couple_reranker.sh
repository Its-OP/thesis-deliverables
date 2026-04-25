#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash train_couple_reranker.sh <experiment_name> [extra args...]"
    exit 1
fi
EXPERIMENT_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_PARQUET_COUNT=$(find "${SCRIPT_DIR}/data/low-pt/train" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_PARQUET_COUNT=$(find "${SCRIPT_DIR}/data/low-pt/val" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_PARQUET_COUNT" -lt 10 ] || [ "$VAL_PARQUET_COUNT" -lt 10 ]; then
    echo "WARNING: ${TRAIN_PARQUET_COUNT} train + ${VAL_PARQUET_COUNT} val parquet files (expected 10+ each)."
fi

SESSION_TRAIN="couple_train"
SESSION_GPU="couple_gpu"
CONDA_ENV_NAME="part"

DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
DATA_DIR="data/low-pt/train/"
VAL_DATA_DIR="data/low-pt/val/"
NETWORK="networks/lowpt_tau_CoupleReranker.py"
CASCADE_CHECKPOINT="models/cascade_best.pt"
TOP_K2=50
MODEL_NAME="${EXPERIMENT_NAME}_CoupleReranker"
EXPERIMENTS_DIR="experiments"
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=5e-4
SCHEDULER="cosine"
DEVICE="cuda:0"
STEPS_PER_EPOCH=500
NUM_WORKERS=10
KEEP_BEST_K=5
LABEL_SMOOTHING=0.10

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

if [ ! -f "${SCRIPT_DIR}/${CASCADE_CHECKPOINT}" ]; then
    echo "ERROR: Cascade checkpoint not found: ${SCRIPT_DIR}/${CASCADE_CHECKPOINT}"
    exit 1
fi

TRAIN_CMD="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && cd ${SCRIPT_DIR} && python -m scripts.python.train_couple_reranker \
    --data-config ${DATA_CONFIG} \
    --data-dir ${DATA_DIR} \
    --val-data-dir ${VAL_DATA_DIR} \
    --network ${NETWORK} \
    --cascade-checkpoint ${CASCADE_CHECKPOINT} \
    --top-k2 ${TOP_K2} \
    --model-name ${MODEL_NAME} \
    --experiments-dir ${EXPERIMENTS_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --steps-per-epoch ${STEPS_PER_EPOCH} \
    --lr ${LEARNING_RATE} \
    --scheduler ${SCHEDULER} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --keep-best-k ${KEEP_BEST_K} \
    --couple-label-smoothing ${LABEL_SMOOTHING} \
    --amp $@"

if screen -list 2>/dev/null | grep -q "${SESSION_TRAIN}"; then
    echo "Screen '${SESSION_TRAIN}' already running. Kill: screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi
screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' | while read -r s; do screen -S "$s" -X quit 2>/dev/null || true; done

screen -dmS "$SESSION_TRAIN" bash -c "${TRAIN_CMD}; echo '--- finished. Enter to close. ---'; read"
screen -dmS "$SESSION_GPU" bash -c "watch -n 1 nvidia-smi"

echo "Launched ${SESSION_TRAIN} + ${SESSION_GPU}. Reattach: screen -r ${SESSION_TRAIN}"
