#!/bin/bash
# HTCondor wrapper: model-input conversion of one abcbatch_2024 batch straight
# from its CRAB MiniAOD files (no CMSSW: plain ROOT + uproot + numpy of the
# LCG view). Submit with: condor_submit convert_miniaod.sub (one job per batch)
#
# Arguments: SPLIT BATCH BATCH_ROOT OUTPUT_DIR TRACK_SELECTION
#   SPLIT            train, eval or test: the split directory under OUTPUT_DIR
#   BATCH            batch number B; input = $BATCH_ROOT/abcbatch${B}_2024/**/miniaod_*.root
#   BATCH_ROOT       /eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau
#   OUTPUT_DIR       /eos/user/o/oprostak/tau_data/datasets/parquet_miniaod_v1
#                    -> $OUTPUT_DIR/$SPLIT/${SPLIT}_b${B}_NNN.parquet
#   TRACK_SELECTION  from_pv (tracks not fitted to another primary vertex,
#                    the ntuple's pool) or all
# Environment: REPO_DIR = the thesis-deliverables clone on lxplus (deploy =
# git pull there); LCG_VIEW = the LCG view providing python3, ROOT and uproot.
#
# Every qualifying event (clean, 3 tightly matched tau pions above 0.5 GeV)
# is written in shards of 5000 with the extended schema, the optional truth /
# vertex / K* blocks, the MiniAOD-only blocks and the batch/job/entry key
# (job = the number of miniaod_<N>.root).
set -eo pipefail

SPLIT="${1:?split: train, eval or test}"
BATCH="${2:?batch number}"
BATCH_ROOT="${3:-/eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau}"
OUTPUT_DIR="${4:-/eos/user/o/oprostak/tau_data/datasets/parquet_miniaod_v1}"
TRACK_SELECTION="${5:-from_pv}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"
LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt}"
INPUT_DIR="$BATCH_ROOT/abcbatch${BATCH}_2024"

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source "$LCG_VIEW/setup.sh"
set -u

if ! find "$INPUT_DIR" -name 'miniaod_*.root' -print -quit | grep -q .; then
    echo "no miniaod_*.root under $INPUT_DIR" >&2
    exit 1
fi

python3 "$REPO_DIR/scripts/condor/convert_to_parquet/convert_root_to_parquet.py" \
    --input-dir "$INPUT_DIR" \
    --pattern '**/miniaod_*.root' \
    --extended --eval-only --shard-events 5000 \
    --track-selection "$TRACK_SELECTION" \
    --output-dir "$OUTPUT_DIR" \
    --val-subdir "$SPLIT" \
    --shard-prefix "${SPLIT}_b${BATCH}"
