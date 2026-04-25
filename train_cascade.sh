#!/bin/bash
# =============================================================================
# Training launch script for CascadeModel (Stage 1 → Stage 2).
#
# Starts two screen sessions:
#   - "cascade_train": Training output (train_cascade.py)
#   - "cascade_gpu":   GPU monitoring (nvidia-smi)
#
# Usage:
#   bash train_cascade.sh <experiment_name>
#   bash train_cascade.sh reranker_v1 --top-k1 400 --epochs 80
#   bash train_cascade.sh reranker_v1 --resume experiments/.../checkpoints/checkpoint_epoch_20.pt
#
# The experiment name is used as a prefix:
#   experiments/{name}_Cascade_{timestamp}/
# =============================================================================
set -euo pipefail

# ---- Require experiment name ----
if [ $# -lt 1 ]; then
    echo "Usage: bash train_cascade.sh <experiment_name> [extra args...]"
    echo "Example: bash train_cascade.sh reranker_v1 --top-k1 400"
    exit 1
fi
EXPERIMENT_NAME="$1"
shift

# ---- Check data split ----
SCRIPT_DIR_CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PARQUET_COUNT=$(find "${SCRIPT_DIR_CHECK}/data/low-pt/train" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
VAL_PARQUET_COUNT=$(find "${SCRIPT_DIR_CHECK}/data/low-pt/val" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TRAIN_PARQUET_COUNT" -lt 10 ] || [ "$VAL_PARQUET_COUNT" -lt 10 ]; then
    echo "WARNING: Found ${TRAIN_PARQUET_COUNT} train and ${VAL_PARQUET_COUNT} val parquet files."
fi
echo "Found ${TRAIN_PARQUET_COUNT} train parquet files, ${VAL_PARQUET_COUNT} val parquet files."

# ---- Configuration ----
SESSION_TRAIN="cascade_train"
SESSION_GPU="cascade_gpu"
CONDA_ENV_NAME="part"

DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
DATA_DIR="data/low-pt/train/"
VAL_DATA_DIR="data/low-pt/val/"
NETWORK="networks/lowpt_tau_CascadeReranker.py"
STAGE1_CHECKPOINT="models/prefilter_best.pt"
TOP_K1=256
MODEL_NAME="${EXPERIMENT_NAME}_Cascade"
EXPERIMENTS_DIR="experiments"
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=1e-3
SCHEDULER="cosine"
DEVICE="cuda:0"
STEPS_PER_EPOCH=500
NUM_WORKERS=10
NO_COMPILE=true
KEEP_BEST_K=5

# ---- Parse extra arguments ----
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--no-compile" ]; then
        NO_COMPILE=true
    else
        EXTRA_ARGS="${EXTRA_ARGS:+${EXTRA_ARGS} }${arg}"
    fi
done

# ---- Resolve conda ----
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base)
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "/opt/miniconda3" ]; then
    CONDA_BASE="/opt/miniconda3"
else
    echo "ERROR: conda not found."
    exit 1
fi

CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME}"

# ---- Resolve script directory ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Check Stage 1 checkpoint ----
if [ ! -f "${SCRIPT_DIR}/${STAGE1_CHECKPOINT}" ]; then
    echo "ERROR: Stage 1 checkpoint not found: ${SCRIPT_DIR}/${STAGE1_CHECKPOINT}"
    echo "  Copy your trained pre-filter to models/prefilter_best.pt"
    exit 1
fi

# ---- Build training command ----
TRAIN_CMD="${CONDA_INIT} && cd ${SCRIPT_DIR} && python train_cascade.py \
    --data-config ${DATA_CONFIG} \
    --data-dir ${DATA_DIR} \
    --val-data-dir ${VAL_DATA_DIR} \
    --network ${NETWORK} \
    --stage1-checkpoint ${STAGE1_CHECKPOINT} \
    --top-k1 ${TOP_K1} \
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
    --amp"

if [ "$NO_COMPILE" = true ]; then
    TRAIN_CMD="${TRAIN_CMD} --no-compile"
fi

TRAIN_CMD="${TRAIN_CMD} ${EXTRA_ARGS}"

# ---- GPU monitoring ----
GPU_MONITOR_CMD="watch -n 1 nvidia-smi"

# ---- Clean up existing sessions ----
screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' | while read -r session_id; do
    screen -S "$session_id" -X quit 2>/dev/null || true
done || true

if screen -list | grep -q "${SESSION_TRAIN}"; then
    echo "Screen session '${SESSION_TRAIN}' already exists."
    echo "To reattach:  screen -r ${SESSION_TRAIN}"
    echo "To kill it:   screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi

# ---- Launch ----
echo "============================================"
echo "  Launching cascade training [${EXPERIMENT_NAME}]"
echo "============================================"
echo ""
echo "Experiment:      ${EXPERIMENT_NAME}"
echo "Session:         ${SESSION_TRAIN} (training)"
echo "                 ${SESSION_GPU} (GPU monitor)"
echo "Stage 1:         ${STAGE1_CHECKPOINT}"
echo "Top-K1:          ${TOP_K1}"
echo "Epochs:          ${EPOCHS}"
echo "Steps/epoch:     ${STEPS_PER_EPOCH}"
echo "Batch size:      ${BATCH_SIZE}"
echo "LR:              ${LEARNING_RATE}"
echo "Scheduler:       ${SCHEDULER}"
echo "Device:          ${DEVICE}"
echo "Network:         ${NETWORK}"
echo "AMP:             enabled"
if [ "$NO_COMPILE" = true ]; then
    echo "Compile:         disabled"
else
    echo "Compile:         enabled"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args:      ${EXTRA_ARGS}"
fi
echo ""

screen -dmS "$SESSION_TRAIN" bash -c "${TRAIN_CMD}; echo '--- Training finished. Press Enter to close. ---'; read"
screen -dmS "$SESSION_GPU" bash -c "$GPU_MONITOR_CMD"

echo "Screen sessions created."
echo ""
echo "To view training:   screen -r ${SESSION_TRAIN}"
echo "To view GPU usage:  screen -r ${SESSION_GPU}"
echo "To detach:          Ctrl+A, then D"
echo ""
echo "TensorBoard:"
echo "  ${CONDA_INIT} && tensorboard --logdir ${SCRIPT_DIR}/${EXPERIMENTS_DIR} --bind_all"
echo ""
