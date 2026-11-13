#!/bin/bash
# HTCondor wrapper: regenerate the train/eval parquet shards of the merged
# production with the current converter (extended schema + builder blocks,
# track_root_index, event_source_* columns, copied source ids).
# Submit with: condor_submit convert_to_parquet.sub, or through
# ../regenerate_parquet.dag after the merges.
#
# Environment (all optional):
#   REPO_DIR    the thesis-deliverables clone on lxplus
#   INPUT_DIR   directory of the merged ROOT files
#   PATTERN     glob of the merged files (merged_bld_batch*.root = the
#               re-merge with the K*/TauTo3Pi/BeamSpot branches)
#   OUTPUT_DIR  destination; writes $OUTPUT_DIR/train/train_NNN.parquet and
#               $OUTPUT_DIR/eval/eval_NNN.parquet (300k + 75k events, 10 + 10
#               shards, the converter defaults). The destination shards are
#               OVERWRITTEN: clear it first.
set -eo pipefail

REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"
INPUT_DIR="${INPUT_DIR:-/eos/user/o/oprostak/tau_data/datasets/root/train_eval}"
PATTERN="${PATTERN:-merged_bld_batch*.root}"
OUTPUT_DIR="${OUTPUT_DIR:-/eos/user/o/oprostak/tau_data/datasets/parquet}"

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

python3 "$REPO_DIR/scripts/condor/convert_to_parquet/convert_root_to_parquet.py" \
    --input-dir "$INPUT_DIR" \
    --pattern "$PATTERN" \
    --extended --builder-blocks \
    --output-dir "$OUTPUT_DIR" \
    --train-subdir train \
    --val-subdir eval
