#!/bin/bash
# HTCondor wrapper: merge the nano microbatches of one production batch into
# merged_bld_batch<N>.root (merge_batches.py, extended + builder branch set).
# Submit with: condor_submit merge_batches.sub   (batches 1..10), or through
# ../regenerate_parquet.dag which chains the merges into the conversion.
#
# Argument: BATCH_ID (1..10)
# Environment: REPO_DIR = the thesis-deliverables clone on lxplus.
set -eo pipefail

BATCH_ID="${1:?batch id}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"

# The LCG view setup dereferences variables the minimal condor job
# environment may not define; enable -u only after it.
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
set -u

python3 "$REPO_DIR/scripts/condor/merge_root_batches/merge_batches.py" "$BATCH_ID"
