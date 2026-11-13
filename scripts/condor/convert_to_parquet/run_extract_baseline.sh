#!/bin/bash
# HTCondor wrapper: baseline-block extraction of one abcbatch_2024 batch.
# Submit with: condor_submit extract_baseline.sub   (one job per batch 0..9)
#
# Arguments: BATCH BATCH_ROOT OUTPUT_ROOT [STRIPE N_STRIPES]
#   BATCH        batch number B; input = $BATCH_ROOT/abcbatch${B}_2024
#   BATCH_ROOT   /eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau
#   OUTPUT_ROOT  /eos/user/o/oprostak/tau_data/datasets/parquet/abcbatch_2024
#                (next to datasets/parquet/{train,eval,test})
#                -> $OUTPUT_ROOT/baseline/batch${B}/baseline_batch${B}_job${J}.parquet
#   STRIPE / N_STRIPES  process only files with index % N_STRIPES == STRIPE
#                (default 0 / 1 = every file); to stripe one slow batch,
#                submit it alone with stripe/n_stripes macros.
# Environment: REPO_DIR = the thesis-deliverables clone on lxplus.
#
# One parquet per job file, one row per ROOT entry (ABCNet rows, K*
# candidates, builder triplets + B aggregates, cutflow, GT indices, gen
# truth), keyed like the model parquet. Existing outputs are skipped, so a
# failed job is resubmitted as is.
set -eo pipefail

BATCH="${1:?batch number}"
BATCH_ROOT="${2:-/eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau}"
OUTPUT_ROOT="${3:-/eos/user/o/oprostak/tau_data/datasets/parquet/abcbatch_2024}"
STRIPE="${4:-0}"
N_STRIPES="${5:-1}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

python3 "$REPO_DIR/scripts/condor/convert_to_parquet/extract_baseline_blocks.py" \
    --input-dir "$BATCH_ROOT/abcbatch${BATCH}_2024" \
    --pattern 'abcbatch*_job*_ditaus_mc.root' \
    --output-dir "$OUTPUT_ROOT/baseline/batch${BATCH}" \
    --stripe "$STRIPE" --num-stripes "$N_STRIPES"
