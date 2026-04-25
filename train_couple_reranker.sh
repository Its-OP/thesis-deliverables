#!/bin/bash
# =============================================================================
# Training launch script for the per-couple reranker (Stage 3).
#
# Loads a frozen 2-stage cascade from a checkpoint and trains a fresh
# CoupleReranker on per-couple feature vectors enumerated from the ParT
# top-50 (Filter A: m(ij) <= m_tau).
#
# Starts two screen sessions:
#   - "couple_train": Training output (train_couple_reranker.py)
#   - "couple_gpu":   GPU monitoring (nvidia-smi)
#
# Usage:
#   bash train_couple_reranker.sh <experiment_name>
#   bash train_couple_reranker.sh couple_v1 --top-k2 75 --epochs 80
#   bash train_couple_reranker.sh couple_v1 --resume experiments/.../checkpoints/checkpoint_epoch_20.pt
#
# The experiment name is used as a prefix:
#   experiments/{name}_CoupleReranker_{timestamp}/
# =============================================================================
set -euo pipefail

# ---- Require experiment name ----
if [ $# -lt 1 ]; then
    echo "Usage: bash train_couple_reranker.sh <experiment_name> [extra args...]"
    echo "Example: bash train_couple_reranker.sh couple_v1 --top-k2 75"
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
SESSION_TRAIN="couple_train"
SESSION_GPU="couple_gpu"
CONDA_ENV_NAME="part"

# ---- The cascade checkpoint (frozen Stage 1 + Stage 2) ----
# Persistent path that lives next to models/prefilter_best.pt — survives the
# .gitignore on debug_checkpoints/. The actual contents come from the
# `cascade_soap_Cascade_20260406_202001` epoch-55 best checkpoint and contain
# both stage1 (32 keys, dim256 cutoff prefilter) and stage2 (84 keys, ParT)
# weights — no separate prefilter checkpoint is needed.
CASCADE_CHECKPOINT="models/cascade_best.pt"

# =============================================================================
# Primary tunables (override via env vars OR by passing the matching CLI flag
# as an extra argument: `bash train_couple_reranker.sh exp1 --top-k2 75`).
# =============================================================================

# Number of top-ParT tracks per event from which couples are enumerated.
# Defaults to 50; the candidate pool size scales as C(K2, 2). E.g., K2=50 →
# 1225 candidate couples per event before Filter A; K2=75 → 2775; K2=100 → 4950.
TOP_K2="${TOP_K2:-50}"

# CoupleReranker architecture knobs (forwarded only when set as env vars).
COUPLE_HIDDEN_DIM="${COUPLE_HIDDEN_DIM:-256}"
COUPLE_NUM_RESIDUAL_BLOCKS="${COUPLE_NUM_RESIDUAL_BLOCKS:-4}"
COUPLE_DROPOUT="${COUPLE_DROPOUT:-0.1}"

# =============================================================================
# Standard training config (rarely overridden)
# =============================================================================
DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
DATA_DIR="data/low-pt/train/"
VAL_DATA_DIR="data/low-pt/val/"
NETWORK="networks/lowpt_tau_CoupleReranker.py"
MODEL_NAME="${EXPERIMENT_NAME}_CoupleReranker"
EXPERIMENTS_DIR="experiments"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
SCHEDULER="${SCHEDULER:-cosine}"
DEVICE="${DEVICE:-cuda:0}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
NUM_WORKERS="${NUM_WORKERS:-10}"
KEEP_BEST_K="${KEEP_BEST_K:-5}"

# ---- Parse extra arguments (forwarded verbatim to Python) ----
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="${EXTRA_ARGS:+${EXTRA_ARGS} }${arg}"
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

# ---- Check cascade checkpoint exists ----
if [ ! -f "${SCRIPT_DIR}/${CASCADE_CHECKPOINT}" ]; then
    echo "ERROR: Cascade checkpoint not found: ${SCRIPT_DIR}/${CASCADE_CHECKPOINT}"
    echo "  Override with --cascade-checkpoint <path> as an extra argument,"
    echo "  or update the CASCADE_CHECKPOINT variable in this script."
    exit 1
fi

# ---- Build training command ----
TRAIN_CMD="${CONDA_INIT} && cd ${SCRIPT_DIR} && python train_couple_reranker.py \
    --data-config ${DATA_CONFIG} \
    --data-dir ${DATA_DIR} \
    --val-data-dir ${VAL_DATA_DIR} \
    --network ${NETWORK} \
    --cascade-checkpoint ${CASCADE_CHECKPOINT} \
    --top-k2 ${TOP_K2} \
    --couple-hidden-dim ${COUPLE_HIDDEN_DIM} \
    --couple-num-residual-blocks ${COUPLE_NUM_RESIDUAL_BLOCKS} \
    --couple-dropout ${COUPLE_DROPOUT} \
    --model-name ${MODEL_NAME} \
    --experiments-dir ${EXPERIMENTS_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --steps-per-epoch ${STEPS_PER_EPOCH} \
    --lr ${LEARNING_RATE} \
    --scheduler ${SCHEDULER} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --keep-best-k ${KEEP_BEST_K}"

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
echo "  Launching couple reranker training [${EXPERIMENT_NAME}]"
echo "============================================"
echo ""
echo "Experiment:        ${EXPERIMENT_NAME}"
echo "Session:           ${SESSION_TRAIN} (training)"
echo "                   ${SESSION_GPU} (GPU monitor)"
echo "Cascade ckpt:      ${CASCADE_CHECKPOINT}"
echo "Top-K2:            ${TOP_K2}"
echo "Hidden dim:        ${COUPLE_HIDDEN_DIM}"
echo "Residual blocks:   ${COUPLE_NUM_RESIDUAL_BLOCKS}"
echo "Dropout:           ${COUPLE_DROPOUT}"
echo "Epochs:            ${EPOCHS}"
echo "Steps/epoch:       ${STEPS_PER_EPOCH}"
echo "Batch size:        ${BATCH_SIZE}"
echo "LR:                ${LEARNING_RATE}"
echo "Scheduler:         ${SCHEDULER}"
echo "Device:            ${DEVICE}"
echo "Network:           ${NETWORK}"
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra args:        ${EXTRA_ARGS}"
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
