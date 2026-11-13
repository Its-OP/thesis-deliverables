#!/bin/bash
# HTCondor wrapper: model-input conversion of one abcbatch_2024 batch.
# Submit with: condor_submit convert_abcbatch.sub   (one job per batch 0..9)
#
# Arguments: BATCH BATCH_ROOT OUTPUT_ROOT
#   BATCH        batch number B; input = $BATCH_ROOT/abcbatch${B}_2024
#   BATCH_ROOT   /eos/user/v/valukash/new_tau_prod
#   OUTPUT_ROOT  /eos/user/o/oprostak/tau_data/datasets/parquet/abcbatch_2024
#                (next to datasets/parquet/{train,eval,test})
#                -> $OUTPUT_ROOT/model/batch${B}/batch${B}_NNN.parquet
# Environment: REPO_DIR = the thesis-deliverables clone on lxplus (deploy =
# git pull there); the script imports its siblings event_identity.py and
# ntuple_truth.py from the same directory.
#
# Every qualifying event (clean, 3 GT pions above 0.5 GeV) is written, in
# shards of 5000, with the extended schema the cascade eval reads, the
# optional truth/vertex/K* blocks and the batch/job/entry event key.
set -eo pipefail

BATCH="${1:?batch number}"
BATCH_ROOT="${2:-/eos/user/v/valukash/new_tau_prod}"
OUTPUT_ROOT="${3:-/eos/user/o/oprostak/tau_data/datasets/parquet/abcbatch_2024}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

python3 "$REPO_DIR/scripts/condor/convert_to_parquet/convert_root_to_parquet.py" \
    --input-dir "$BATCH_ROOT/abcbatch${BATCH}_2024" \
    --pattern '**/abcbatch*_job*_ditaus_mc.root' \
    --extended --eval-only --shard-events 5000 \
    --output-dir "$OUTPUT_ROOT/model" \
    --val-subdir "batch${BATCH}"
