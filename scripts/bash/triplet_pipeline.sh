#!/bin/bash
set -euo pipefail

# Triplet stage end to end on the promoted K1=400 stack (the h6s9 recipe):
#   dump_eval / dump_train  per-stage couples dumps of the frozen three-stage
#                           cascade (K1=400, K2=125, 200 couples per event);
#                           the train role runs one evaluator pass per shard
#                           (the evaluator loads a whole --val-data-dir at once)
#                           and concatenates the parts with concat_parquet.py
#   candidates_eval / candidates_train
#                           build_triplet_rank_candidates with the MLP gate
#                           (experiments/bprep_gate_mlp, the h6s6 B-prep twin
#                           that replaced the GBDT gate) at top_c=200; the
#                           train role also writes operating_points.json
#   train                   ListwiseTripletReranker with the arm-B champion
#                           recipe (experiments/armB_listwise_v2 stored args;
#                           full-eval T@10 0.6727 at the p99 gate, h6s6)
#   bash scripts/bash/triplet_pipeline.sh [steps...]
# Default steps: dump_eval dump_train candidates_eval candidates_train train
# Recommended detached run: screen -dmS triplet_pipeline bash scripts/bash/triplet_pipeline.sh
# Env knobs: PYTHON DEVICE WORKERS TOP_C TOP_K1 TOP_K2 NUM_COUPLES GATE DUMP_DIR
#            OUT_DIR TRAIN_SHARD_DIR EVAL_SHARD_DIR EPOCHS EVENTS_PER_EPOCH
#            EVAL_EVENTS RUN_NAME EXTRA_TRAIN_ARGS EXPECTED_S3_SHA16 SMOKE
# SMOKE=1 is a wiring check, not a result: it writes into *_smoke directories
# and trains one short epoch. Point TRAIN_SHARD_DIR / EVAL_SHARD_DIR at small
# single-shard copies (e.g. the first 100 events of a shard); the builder
# requires each dump to cover every source event, so a partial dump of a full
# shard cannot be built.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-16}"
TOP_C="${TOP_C:-200}"
TOP_K1="${TOP_K1:-400}"
TOP_K2="${TOP_K2:-125}"
NUM_COUPLES="${NUM_COUPLES:-200}"
GATE="${GATE:-experiments/bprep_gate_mlp/checkpoints/best_model.pt}"
DUMP_DIR="${DUMP_DIR:-data/dumps_k400_v4}"
OUT_DIR="${OUT_DIR:-data/triplet_rank_v4}"
TRAIN_SHARD_DIR="${TRAIN_SHARD_DIR:-data/low-pt/train}"
EVAL_SHARD_DIR="${EVAL_SHARD_DIR:-data/low-pt/eval}"
EPOCHS="${EPOCHS:-30}"
EVENTS_PER_EPOCH="${EVENTS_PER_EPOCH:-90000}"
EVAL_EVENTS="${EVAL_EVENTS:-12000}"
RUN_NAME="${RUN_NAME:-armB_listwise_v4}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# Promoted h6s8b calibrated stage 3; set EXPECTED_S3_SHA16= (empty) to skip the check.
EXPECTED_S3_SHA16="${EXPECTED_S3_SHA16-b5f7e3eda46af590}"
SMOKE="${SMOKE:-0}"

S1=models/prefilter_best.pt
S2=models/stage2_best.pt
S3=models/couple_reranker_best.pt
DATA_CONFIG=data/low-pt/lowpt_tau_trackfinder.yaml

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

if [ -z "${PYTHON:-}" ]; then
    for candidate in /venv/thesis-deliverables/bin/python /venv/part/bin/python /opt/miniconda3/envs/part/bin/python; do
        if [ -x "${candidate}" ]; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [ -z "${PYTHON:-}" ]; then
    if command -v conda &>/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate part
        PYTHON=python
    else
        echo "ERROR: no python found; set PYTHON=/path/to/python"
        exit 1
    fi
fi

if [ "${SMOKE}" = "1" ]; then
    DUMP_DIR="${DUMP_DIR}_smoke"
    OUT_DIR="${OUT_DIR}_smoke"
    EPOCHS=1
    EVENTS_PER_EPOCH=1000
    EVAL_EVENTS=500
fi
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${DUMP_DIR}" "${OUT_DIR}" "${LOG_DIR}"
EVAL_DUMP="${DUMP_DIR}/perstage_couples_eval_k${TOP_K1}.parquet"
TRAIN_DUMP="${DUMP_DIR}/perstage_couples_train_k${TOP_K1}.parquet"
TRAIN_PART_GLOB="${DUMP_DIR}/perstage_couples_train_shard_*_k${TOP_K1}.parquet"

for required in "${S1}" "${S2}" "${S3}" "${GATE}" "${DATA_CONFIG}"; do
    if [ ! -s "${required}" ]; then
        echo "ERROR: missing ${required}"
        exit 1
    fi
done
_sha16() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -c1-16
    else
        shasum -a 256 "$1" | cut -c1-16
    fi
}
if [ -n "${EXPECTED_S3_SHA16}" ]; then
    S3_SHA16="$(_sha16 "${S3}")"
    if [ "${S3_SHA16}" != "${EXPECTED_S3_SHA16}" ]; then
        echo "ERROR: ${S3} sha256 ${S3_SHA16} != ${EXPECTED_S3_SHA16} (promoted h6s8b); set EXPECTED_S3_SHA16= to skip"
        exit 1
    fi
fi
echo "[PIPELINE] HEAD $(git rev-parse --short HEAD 2>/dev/null || echo n/a) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a) | python ${PYTHON}"
echo "[PIPELINE] K1=${TOP_K1} K2=${TOP_K2} couples=${NUM_COUPLES} top_c=${TOP_C} gate=${GATE} dumps=${DUMP_DIR} out=${OUT_DIR}"

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
    "${PYTHON}" -c "import pyarrow.parquet as pq, sys; print(pq.read_metadata(sys.argv[1]).num_rows)" "$1" 2>/dev/null || echo "-1"
}

dump_valid() {
    # A finished dump: present, no resume marker left behind, readable parquet with rows.
    [ -s "$1" ] && [ ! -e "$1.INCOMPLETE" ] && [ "$(_rows "$1")" -gt 0 ]
}

_dump_couples() {
    local shard_dir="$1" output="$2"
    "${PYTHON}" -m scripts.python.eval_cascade_pipeline --stage couples \
        --stage1-weights "${S1}" --stage2-weights "${S2}" --stage3-weights "${S3}" \
        --top-k1 "${TOP_K1}" --top-k2 "${TOP_K2}" --num-couples "${NUM_COUPLES}" \
        --val-data-dir "${shard_dir}" --data-config "${DATA_CONFIG}" \
        --output "${output}" --batch-size 64 --num-workers 0 --device "${DEVICE}"
}

step_dump_eval() {
    if dump_valid "${EVAL_DUMP}"; then
        echo "skip: ${EVAL_DUMP} exists ($(_rows "${EVAL_DUMP}") rows)"
        return 0
    fi
    _dump_couples "${EVAL_SHARD_DIR}" "${EVAL_DUMP}"
    if ! dump_valid "${EVAL_DUMP}"; then
        echo "eval dump invalid: ${EVAL_DUMP}"
        return 1
    fi
    echo "${EVAL_DUMP}: $(_rows "${EVAL_DUMP}") rows"
}

step_dump_train() {
    if dump_valid "${TRAIN_DUMP}"; then
        echo "skip: ${TRAIN_DUMP} exists ($(_rows "${TRAIN_DUMP}") rows)"
        return 0
    fi
    local shard shard_name link_dir part
    for shard in "${TRAIN_SHARD_DIR}"/*.parquet; do
        shard_name="$(basename "${shard}" .parquet)"
        part="${DUMP_DIR}/perstage_couples_train_shard_${shard_name}_k${TOP_K1}.parquet"
        if dump_valid "${part}"; then
            echo "skip: ${part} exists"
        else
            # The evaluator takes a directory: one symlink per shard keeps RAM at one shard.
            link_dir="${DUMP_DIR}/.shard_${shard_name}"
            mkdir -p "${link_dir}"
            ln -sf "$(cd "$(dirname "${shard}")" && pwd)/$(basename "${shard}")" "${link_dir}/$(basename "${shard}")"
            echo "[dump_train] ${shard_name} -> ${part}"
            _dump_couples "${link_dir}" "${part}"
            if ! dump_valid "${part}"; then
                echo "shard dump invalid: ${part}"
                return 1
            fi
        fi
        if [ "${SMOKE}" = "1" ]; then
            break
        fi
    done
    "${PYTHON}" -m scripts.python.concat_parquet "${TRAIN_PART_GLOB}" "${TRAIN_DUMP}"
}

step_candidates() {
    local role="$1" dump="$2" shard_dir="$3"
    local out="${OUT_DIR}/candidates_${role}.parquet"
    if [ -s "${out}" ]; then
        echo "skip: ${out} exists ($(_rows "${out}") rows)"
        return 0
    fi
    if [ ! -s "${dump}" ]; then
        echo "missing dump ${dump}; run dump_${role} first"
        return 1
    fi
    "${PYTHON}" -m scripts.python.build_triplet_rank_candidates --role "${role}" \
        --dump "${dump}" --src-glob "${shard_dir}/*.parquet" \
        --out-dir "${OUT_DIR}" --filter-checkpoint "${GATE}" \
        --top-c "${TOP_C}" --workers "${WORKERS}"
    if [ ! -s "${out}" ]; then
        echo "builder produced no ${out}"
        return 1
    fi
}

step_train() {
    local train_candidates="${OUT_DIR}/candidates_train.parquet"
    local eval_candidates="${OUT_DIR}/candidates_eval.parquet"
    local operating_points="${OUT_DIR}/operating_points.json"
    local required
    for required in "${train_candidates}" "${eval_candidates}" "${operating_points}"; do
        if [ ! -s "${required}" ]; then
            echo "missing ${required}; run the candidates steps first"
            return 1
        fi
    done
    # Champion recipe (armB_listwise_v2 stored args); norm stats are fitted on
    # first use and written next to the candidates.
    "${PYTHON}" train_listwise_reranker.py \
        --candidates "${train_candidates}" --src-glob "${TRAIN_SHARD_DIR}/*.parquet" \
        --eval-candidates "${eval_candidates}" --eval-src-glob "${EVAL_SHARD_DIR}/*.parquet" \
        --operating-points "${operating_points}" --norm-stats "${OUT_DIR}/norm_stats_listwise.json" \
        --gate p99 --eval-gates p99,p95 \
        --hidden-dim 128 --num-layers 4 --num-heads 8 --dropout 0.1 \
        --label-smoothing 0.10 --contrast-weight 0.2 \
        --batch-size 32 --epochs "${EPOCHS}" --lr 3e-4 --weight-decay 0.01 \
        --warmup-fraction 0.05 --cosine-power 2.0 \
        --events-per-epoch "${EVENTS_PER_EPOCH}" --eval-events "${EVAL_EVENTS}" \
        --eval-every 2 --eval-batch-size 4 --max-list 3072 \
        --num-workers 32 --device "${DEVICE}" --run-name "${RUN_NAME}" \
        ${EXTRA_TRAIN_ARGS}
}

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(dump_eval dump_train candidates_eval candidates_train train)
fi
echo "[PIPELINE] steps: ${STEPS[*]} | SMOKE=${SMOKE} | device ${DEVICE}"
echo "[PIPELINE] timeline: grep -E 'START|DONE|FAILED' ${LOG_DIR}/pipeline_*.log"

for step in "${STEPS[@]}"; do
    case "${step}" in
        dump_eval)        run_step dump_eval step_dump_eval ;;
        dump_train)       run_step dump_train step_dump_train ;;
        candidates_eval)  run_step candidates_eval step_candidates eval "${EVAL_DUMP}" "${EVAL_SHARD_DIR}" ;;
        candidates_train) run_step candidates_train step_candidates train "${TRAIN_DUMP}" "${TRAIN_SHARD_DIR}" ;;
        train)            run_step train step_train ;;
        *) echo "unknown step '${step}'"; exit 1 ;;
    esac
done
echo "[PIPELINE] all steps complete"
