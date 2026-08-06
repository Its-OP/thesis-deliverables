#!/bin/bash
# =============================================================================
# Stage-3 K2 sweep: sequential CoupleReranker trainings over the K2 grid,
# each followed by a full-eval pass (score dump + metrics JSON).
#
# Usage:
#   bash sweep_couple_reranker_k2.sh [extra trainer args...]
#
# Designed for the GPU box: uses the venv python directly when present,
# falling back to whatever `python` resolves to.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

if [ -x /venv/part/bin/python ]; then
    PYTHON=/venv/part/bin/python
else
    PYTHON=python
fi

K2_GRID=(50 60 70 80 100 125 150)
DATA_CONFIG="data/low-pt/lowpt_tau_trackfinder.yaml"
VAL_DATA_DIR="data/low-pt/eval/"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for K2 in "${K2_GRID[@]}"; do
    MODEL_NAME="h6s3_k${K2}_CoupleReranker"
    echo "=== K2=${K2}: training ${MODEL_NAME} ==="
    "${PYTHON}" -m scripts.python.train_couple_reranker \
        --data-config "${DATA_CONFIG}" \
        --data-dir data/low-pt/train/ \
        --val-data-dir "${VAL_DATA_DIR}" \
        --network networks/lowpt_tau_CoupleReranker.py \
        --stage1-checkpoint models/prefilter_best.pt \
        --stage2-checkpoint models/stage2_best.pt \
        --top-k2 "${K2}" \
        --model-name "${MODEL_NAME}" \
        --experiments-dir experiments \
        --epochs 100 --steps-per-epoch 500 --batch-size 16 --lr 5e-4 \
        --couple-label-smoothing 0.10 --amp --num-workers 10 \
        --keep-best-k 5 \
        --k-values-couples 50 60 75 100 125 200 \
        --k-values-tracks 30 50 60 70 80 100 125 150 200 \
        "$@"

    RUN_DIR=$(ls -dt experiments/${MODEL_NAME}_* | head -1)
    CKPT="${RUN_DIR}/checkpoints/best_model_calibrated.pt"
    if [ ! -f "${CKPT}" ]; then
        CKPT="${RUN_DIR}/checkpoints/best_model.pt"
    fi
    echo "=== K2=${K2}: full eval of ${CKPT} ==="
    "${PYTHON}" -m scripts.python.eval_cascade_pipeline \
        --stage couples \
        --stage1-weights models/prefilter_best.pt \
        --stage2-weights models/stage2_best.pt \
        --stage3-weights "${CKPT}" \
        --top-k2 "${K2}" --num-couples 200 --batch-size 64 --num-workers 0 \
        --val-data-dir "${VAL_DATA_DIR}" \
        --data-config "${DATA_CONFIG}" \
        --output "${RUN_DIR}/couples_k${K2}_dump.parquet"
    "${PYTHON}" -m scripts.python.compute_eval_metrics \
        --eval-parquet "${RUN_DIR}/couples_k${K2}_dump.parquet" \
        --val-data-dir "${VAL_DATA_DIR}" \
        --data-config "${DATA_CONFIG}" \
        --output "${RUN_DIR}/couples_k${K2}_metrics.json"
    echo "=== K2=${K2}: done ==="
done

echo "SWEEP_COMPLETE"
