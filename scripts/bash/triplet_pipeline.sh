#!/bin/bash
set -euo pipefail

# Stage-4 triplet reranker: data -> train -> eval, end to end.
#   bash scripts/bash/triplet_pipeline.sh [steps...]
# Default steps: dump_train dump_val gbdt candidates_train candidates_val train eval export
# Recommended detached run: screen -dmS triplet_pipeline bash scripts/bash/triplet_pipeline.sh
# Env knobs: DEVICE WORKERS TOP_C OPERATING_POINT INPUT_MODE LOSS_MODE EPOCHS BATCH_SIZE
#            NUM_NEGATIVES EVAL_EVENTS REBUILD_GBDT SMOKE EXPORT_MODEL EXTRA_TRAIN_ARGS
#            TRAIN_DATA_DIR VAL_DATA_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-8}"
TOP_C="${TOP_C:-100}"
OPERATING_POINT="${OPERATING_POINT:-d6@0.99}"
INPUT_MODE="${INPUT_MODE:-flat}"
LOSS_MODE="${LOSS_MODE:-sampled}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_NEGATIVES="${NUM_NEGATIVES:-50}"
EVAL_EVENTS="${EVAL_EVENTS:-20000}"
REBUILD_GBDT="${REBUILD_GBDT:-0}"
SMOKE="${SMOKE:-0}"
EXPORT_MODEL="${EXPORT_MODEL:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${SCRIPT_DIR}/../part/data/low-pt/train}"
VAL_DATA_DIR="${VAL_DATA_DIR:-${SCRIPT_DIR}/../part/data/low-pt/val}"

if [ -f /venv/part/bin/activate ]; then
    source /venv/part/bin/activate
else
    if command -v conda &>/dev/null; then
        CONDA_BASE=$(conda info --base)
    elif [ -d "/opt/miniconda3" ]; then
        CONDA_BASE="/opt/miniconda3"
    else
        echo "ERROR: neither /venv/part nor conda found."
        exit 1
    fi
    source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate part
fi

EVAL_DIR="${SCRIPT_DIR}/data/low-pt/eval"
RANK_DIR="${EVAL_DIR}/triplet_rank"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${EVAL_DIR}" "${RANK_DIR}" "${LOG_DIR}" "${SCRIPT_DIR}/reports"

SUF=""
MAX_EVENTS_ARGS=()
if [ "${SMOKE}" = "1" ]; then
    SUF="_smoke"
    MAX_EVENTS_ARGS=(--max-events 2000)
    EPOCHS=2
    EVAL_EVENTS=500
fi
DUMP_TRAIN="${EVAL_DIR}/perstage_couples_train${SUF}.parquet"
DUMP_VAL="${EVAL_DIR}/perstage_couples_val${SUF}.parquet"
# Historical VAL dump name: reuse when present, never rename.
if [ ! -f "${DUMP_VAL}" ] && [ -z "${SUF}" ] && [ -f "${EVAL_DIR}/pipeline_val.parquet" ]; then
    DUMP_VAL="${EVAL_DIR}/pipeline_val.parquet"
fi
TAG_TRAIN="train${SUF}"
TAG_VAL="val${SUF}"
NORM_STATS="${RANK_DIR}/norm_stats_train${SUF}.json"
EXPERIMENT_NAME="triplet_trainfull${SUF}"

CURRENT_STEP="init"
CURRENT_LOG="none"
trap 'echo "[PIPELINE] FAILED at step ${CURRENT_STEP} (exit $?); log: ${CURRENT_LOG}"' ERR

run_step() {
    CURRENT_STEP="$1"
    shift
    CURRENT_LOG="${LOG_DIR}/pipeline_${CURRENT_STEP}_$(date +%Y%m%d_%H%M%S).log"
    local t0
    t0=$(date +%s)
    echo "[${CURRENT_STEP}] START ($(date '+%F %T')) -> ${CURRENT_LOG}"
    "$@" 2>&1 | tee "${CURRENT_LOG}"
    echo "[${CURRENT_STEP}] DONE ($(( $(date +%s) - t0 ))s)"
}

_rows() {
    python -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" "$1" 2>/dev/null || echo "-1"
}

_has_column() {
    python -c "import pyarrow.parquet as pq, sys; sys.exit(0 if sys.argv[2] in pq.read_schema(sys.argv[1]).names else 1)" "$1" "$2" 2>/dev/null
}

_concat_incomplete_parts() {
    # A crashed dump leaves <dump>.INCOMPLETE + resumed part files; merge them.
    local dump="$1"
    if [ -f "${dump}.INCOMPLETE" ] && ls "${dump}".part*.parquet >/dev/null 2>&1; then
        echo "concatenating partial dump parts into ${dump}"
        python - "$dump" <<'PY'
import glob, sys
import pyarrow as pa
import pyarrow.parquet as pq
dump = sys.argv[1]
parts = [dump] + sorted(glob.glob(dump + '.part*.parquet'))
pq.write_table(pa.concat_tables([pq.read_table(p) for p in parts]), dump + '.merged')
PY
        mv "${dump}.merged" "${dump}"
        rm -f "${dump}".part*.parquet "${dump}.INCOMPLETE"
    fi
}

step_dump() {
    local dump="$1" data_dir="$2"
    if [ -f "${dump}" ] && [ ! -f "${dump}.INCOMPLETE" ]; then
        echo "skip: ${dump} exists ($(_rows "${dump}") rows)"
        return 0
    fi
    local start_args=()
    if [ -f "${dump}.INCOMPLETE" ]; then
        local next
        next=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['next_start_event'])" "${dump}.INCOMPLETE")
        echo "resuming crashed dump at event ${next}"
        start_args=(--start-event "${next}" --output "${dump}.part2.parquet")
    fi
    python -m scripts.python.eval_cascade_pipeline \
        --stage couples \
        --stage1-weights models/prefilter_best.pt \
        --stage2-weights models/stage2_best.pt \
        --stage3-weights models/couple_reranker_best.pt \
        --num-couples 200 \
        --data-config data/low-pt/lowpt_tau_trackfinder.yaml \
        --val-data-dir "${data_dir}/" \
        --output "${dump}" \
        --device "${DEVICE}" \
        --batch-size 64 \
        "${MAX_EVENTS_ARGS[@]}" \
        "${start_args[@]:+${start_args[@]}}"
    _concat_incomplete_parts "${dump}"
}

step_gbdt() {
    local gbdt6="${SCRIPT_DIR}/models/third_pion_filter_gbdt_full_P2.joblib"
    local gbdt8="${SCRIPT_DIR}/models/third_pion_filter_gbdt8_full_P2.joblib"
    if [ "${REBUILD_GBDT}" != "1" ] && [ -f "${gbdt6}" ] && [ -f "${gbdt8}" ]; then
        echo "skip: shipped GBDT joblibs reused (REBUILD_GBDT=1 forces a rebuild)"
        return 0
    fi
    echo "WARNING: rebuilding the GBDT soft filters INVALIDATES the frozen tau"
    echo "WARNING: operating points; re-derive OPERATING_POINTS before training."
    python -m scripts.python.build_triplet_filter_table \
        --dump "${DUMP_VAL}" --src-glob "${VAL_DATA_DIR}/*.parquet"
    python -m scripts.python.train_triplet_filter
}

step_candidates() {
    local dump="$1" data_dir="$2" tag="$3"
    local out="${RANK_DIR}/candidates_${tag}.parquet"
    local expected
    expected=$(_rows "${dump}")
    if [ -f "${out}" ] && ! _has_column "${out}" 'track_s1'; then
        echo "schema guard: ${out} lacks track_s1 -> ${out}.legacy, rebuilding"
        mv "${out}" "${out}.legacy"
    fi
    if [ -f "${out}" ] && [ "$(_rows "${out}")" = "${expected}" ]; then
        echo "skip: ${out} exists (${expected} rows, cascade columns present)"
        return 0
    fi
    python -m scripts.python.build_triplet_rank_candidates \
        --dump "${dump}" \
        --src-glob "${data_dir}/*.parquet" \
        --tag "${tag}" \
        --top-c "${TOP_C}" \
        --workers "${WORKERS}"
}

step_train() {
    # shellcheck disable=SC2086
    WAIT=1 bash "${SCRIPT_DIR}/scripts/bash/train_triplet_reranker.sh" "${EXPERIMENT_NAME}" \
        --candidates "${RANK_DIR}/candidates_${TAG_TRAIN}.parquet" \
        --tracks "${RANK_DIR}/tracks_${TAG_TRAIN}.parquet" \
        --eval-candidates "${RANK_DIR}/candidates_${TAG_VAL}.parquet" \
        --eval-tracks "${RANK_DIR}/tracks_${TAG_VAL}.parquet" \
        --norm-stats "${NORM_STATS}" \
        --operating-point "${OPERATING_POINT}" \
        --input-mode "${INPUT_MODE}" \
        --loss-mode "${LOSS_MODE}" \
        --num-negatives "${NUM_NEGATIVES}" \
        --eval-events "${EVAL_EVENTS}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        ${EXTRA_TRAIN_ARGS}
}

_latest_checkpoint() {
    ls -td "${SCRIPT_DIR}/experiments/${EXPERIMENT_NAME}"_*/ 2>/dev/null | head -1 | xargs -I{} echo "{}checkpoints/best_model.pt"
}

step_eval() {
    python -m scripts.python.eval_triplet_rank_baselines \
        --candidates "${RANK_DIR}/candidates_${TAG_VAL}.parquet" \
        --out-json "${SCRIPT_DIR}/reports/triplet_rank_baselines_val${SUF}.json" \
        --ceilings-json "${RANK_DIR}/ceilings${SUF}.json"
    local checkpoint
    checkpoint=$(_latest_checkpoint)
    if [ ! -f "${checkpoint}" ]; then
        echo "ERROR: no trained checkpoint under experiments/${EXPERIMENT_NAME}_*"
        return 1
    fi
    python -m scripts.python.eval_triplet_reranker \
        --checkpoint "${checkpoint}" \
        --candidates "${RANK_DIR}/candidates_${TAG_VAL}.parquet" \
        --tracks "${RANK_DIR}/tracks_${TAG_VAL}.parquet" \
        --out-json "${SCRIPT_DIR}/reports/triplet_reranker_eval${SUF}.json" \
        --device "${DEVICE}"
}

step_export() {
    if [ "${EXPORT_MODEL}" != "1" ]; then
        echo "skip: EXPORT_MODEL != 1"
        return 0
    fi
    local checkpoint
    checkpoint=$(_latest_checkpoint)
    cp "${checkpoint}" "${SCRIPT_DIR}/models/triplet_reranker_best.pt"
    echo "exported ${checkpoint} -> models/triplet_reranker_best.pt"
}

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(dump_train dump_val gbdt candidates_train candidates_val train eval export)
fi
echo "[PIPELINE] steps: ${STEPS[*]} | SMOKE=${SMOKE} | device ${DEVICE}"
echo "[PIPELINE] timeline: grep -E 'START|DONE|FAILED' ${LOG_DIR}/pipeline_*.log"

for step in "${STEPS[@]}"; do
    case "${step}" in
        dump_train)       run_step dump_train step_dump "${DUMP_TRAIN}" "${TRAIN_DATA_DIR}" ;;
        dump_val)         run_step dump_val step_dump "${DUMP_VAL}" "${VAL_DATA_DIR}" ;;
        gbdt)             run_step gbdt step_gbdt ;;
        candidates_train) run_step candidates_train step_candidates "${DUMP_TRAIN}" "${TRAIN_DATA_DIR}" "${TAG_TRAIN}" ;;
        candidates_val)   run_step candidates_val step_candidates "${DUMP_VAL}" "${VAL_DATA_DIR}" "${TAG_VAL}" ;;
        train)            run_step train step_train ;;
        eval)             run_step eval step_eval ;;
        export)           run_step export step_export ;;
        *) echo "unknown step '${step}'"; exit 1 ;;
    esac
done
echo "[PIPELINE] all steps complete"
