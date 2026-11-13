#!/bin/bash
# HTCondor wrapper: build one split (train / eval / test) of the dataset from
# the abcbatch_2024 ntuples. Every qualifying event of the given batches is
# written, no quota, in shards of SHARD_EVENTS.
# Submit with: condor_submit regenerate_abcbatch.sub   (three jobs: one per split)
#
# Arguments: SPLIT "BATCHES" SHARD_EVENTS BATCH_ROOT OUTPUT_DIR [PATTERN]
#   SPLIT         train | eval | test  (subdirectory and shard prefix)
#   BATCHES       space-separated batch numbers, e.g. "1 2 3 4 5 6 7 8"
#   SHARD_EVENTS  events per shard
#   BATCH_ROOT    directory holding abcbatch{B}_2024/
#   OUTPUT_DIR    /eos/user/o/oprostak/tau_data/datasets/parquet
#                 -> $OUTPUT_DIR/$SPLIT/${SPLIT}_NNN.parquet  (OVERWRITES)
#   PATTERN       glob below each batch directory; '**/' descends into the
#                 CRAB layout (abcbatch{B}_2024/BdToKstarTauTau/crab_*/*/0000/)
# Event key: batch = the abcbatch<B> token of the path, job = the job<N>
# token or the trailing _<N>.root number of the file name.
# Environment: REPO_DIR = the thesis-deliverables clone on lxplus.
set -eo pipefail

SPLIT="${1:?split name}"
BATCHES="${2:?batch numbers}"
SHARD_EVENTS="${3:-25000}"
BATCH_ROOT="${4:-/eos/user/v/valukash/new_tau_prod}"
OUTPUT_DIR="${5:-/eos/user/o/oprostak/tau_data/datasets/parquet}"
PATTERN="${6:-**/abcbatch*_job*_ditaus_mc.root}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"

INPUT_DIRS=()
for batch in $BATCHES; do
    INPUT_DIRS+=("$BATCH_ROOT/abcbatch${batch}_2024")
done

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

python3 "$REPO_DIR/scripts/condor/convert_to_parquet/convert_root_to_parquet.py" \
    --input-dir "${INPUT_DIRS[@]}" \
    --pattern "$PATTERN" \
    --extended --eval-only --shard-events "$SHARD_EVENTS" \
    --output-dir "$OUTPUT_DIR" \
    --val-subdir "$SPLIT"
