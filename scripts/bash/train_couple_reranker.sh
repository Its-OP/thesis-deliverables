#!/bin/bash
# =============================================================================
# Stage-3 couple reranker training in dump mode, the promoted h6s8b recipe
# (models/couple_reranker_best.pt stored args; canonical full-eval C@100 0.9331
# / C@125 0.9388 at K1=400): stage-3 input dumps of the frozen stage 1 + 2 at
# K1=400, K2=125, batch 768, 20 epochs of lr 2e-3 cosine, amp, 10 workers,
# label smoothing 0.10, full K grids. One epoch = one pass over the dumps.
#
# Dumps come from eval_cascade_pipeline.py --stage part --dump-stage3-inputs
# --top-k1 400 (one pass per train shard, one pass over data/low-pt/eval):
#   data/dumps_k400_frozen/stage3_dump_train_k400_frozen_s<i>.parquet
#   data/dumps_k400_frozen/stage3_dump_eval_k400_frozen.parquet
# The trainer defaults (fp32, lr 5e-4) OOM at batch 768 / K2 125: keep the
# flags below and pass extra args only on top of them.
#
# Usage:
#   bash train_couple_reranker.sh <experiment_name> [extra trainer args...]
#   DUMP_DIR=data/other_dumps bash train_couple_reranker.sh <experiment_name>
# =============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash train_couple_reranker.sh <experiment_name> [extra args...]"
    exit 1
fi
EXPERIMENT_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SESSION_TRAIN="couple_train"
SESSION_GPU="couple_gpu"
CONDA_ENV_NAME="part"

DUMP_DIR="${DUMP_DIR:-data/dumps_k400_frozen}"
TRAIN_DUMP_GLOB="${DUMP_DIR}/stage3_dump_train_k400_frozen_s*.parquet"
VAL_DUMP="${DUMP_DIR}/stage3_dump_eval_k400_frozen.parquet"
TOP_K2=125
MODEL_NAME="${EXPERIMENT_NAME}_CoupleReranker"
EXPERIMENTS_DIR="experiments"
EPOCHS=20
BATCH_SIZE=768
LEARNING_RATE=2e-3
DEVICE="cuda:0"
NUM_WORKERS=10
KEEP_BEST_K=5
LABEL_SMOOTHING=0.10
K_VALUES_TRACKS="30 50 60 70 80 100 125 150 200"
K_VALUES_COUPLES="50 60 75 100 125 200"

case "${DUMP_DIR}" in
    /*) DUMP_PATH="${DUMP_DIR}" ;;
    *)  DUMP_PATH="${SCRIPT_DIR}/${DUMP_DIR}" ;;
esac
if [ ! -d "${DUMP_PATH}" ]; then
    echo "ERROR: dump directory not found: ${DUMP_PATH}"
    echo "  Produce the dumps with: python -m scripts.python.eval_cascade_pipeline --stage part --dump-stage3-inputs --top-k1 400 ..."
    exit 1
fi
TRAIN_DUMP_COUNT=$(find "${DUMP_PATH}" -maxdepth 1 -name "stage3_dump_train_k400_frozen_s*.parquet" | wc -l | tr -d ' ')
if [ "$TRAIN_DUMP_COUNT" -lt 1 ]; then
    echo "ERROR: no train dumps match ${DUMP_PATH}/stage3_dump_train_k400_frozen_s*.parquet"
    echo "  Produce them with: python -m scripts.python.eval_cascade_pipeline --stage part --dump-stage3-inputs --top-k1 400 ..."
    exit 1
fi
if [ ! -f "${DUMP_PATH}/stage3_dump_eval_k400_frozen.parquet" ]; then
    echo "ERROR: eval dump not found: ${DUMP_PATH}/stage3_dump_eval_k400_frozen.parquet"
    exit 1
fi
echo "Found ${TRAIN_DUMP_COUNT} train dump files under ${DUMP_DIR}; eval dump ${VAL_DUMP}."

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

# The train-dump glob stays single-quoted so the trainer receives it unexpanded.
TRAIN_CMD="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV_NAME} && cd ${SCRIPT_DIR} && python -m scripts.python.train_couple_reranker \
    --train-dump '${TRAIN_DUMP_GLOB}' \
    --val-dump ${VAL_DUMP} \
    --top-k2 ${TOP_K2} \
    --model-name ${MODEL_NAME} \
    --experiments-dir ${EXPERIMENTS_DIR} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --device ${DEVICE} \
    --num-workers ${NUM_WORKERS} \
    --keep-best-k ${KEEP_BEST_K} \
    --couple-label-smoothing ${LABEL_SMOOTHING} \
    --k-values-tracks ${K_VALUES_TRACKS} \
    --k-values-couples ${K_VALUES_COUPLES} \
    --amp $@"

if screen -list 2>/dev/null | grep -q "${SESSION_TRAIN}"; then
    echo "Screen '${SESSION_TRAIN}' already running. Kill: screen -S ${SESSION_TRAIN} -X quit"
    exit 1
fi
screen -list 2>/dev/null | grep "\.${SESSION_GPU}" | awk '{print $1}' | while read -r s; do screen -S "$s" -X quit 2>/dev/null || true; done

screen -dmS "$SESSION_TRAIN" bash -c "${TRAIN_CMD}; echo '--- finished. Enter to close. ---'; read"
screen -dmS "$SESSION_GPU" bash -c "watch -n 1 nvidia-smi"

echo "Launched ${SESSION_TRAIN} + ${SESSION_GPU}. Reattach: screen -r ${SESSION_TRAIN}"
