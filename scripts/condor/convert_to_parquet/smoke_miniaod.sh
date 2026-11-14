#!/bin/bash
# Smoke test of the MiniAOD -> parquet conversion on lxplus, before submitting
# convert_miniaod.sub: converts the files miniaod_100?.root of one CRAB
# subdirectory under the same LCG view as the condor jobs and prints the
# "SMOKE ..." lines followed by the reference lines measured on the same
# files: the counts must be equal, the sums equal to 4 significant digits.
#
# Arguments: INPUT_DIR (default: CRAB subdirectory 0001 of abcbatch1)
# Environment: REPO_DIR, LCG_VIEW as in run_convert_miniaod.sh.
set -eo pipefail

INPUT_DIR="${1:-/eos/cms/store/group/phys_bphys/valukash/BdToKstarTauTau/abcbatch1_2024/BdToKstarTauTau/crab_abcbatch1_2024_BdToKstarTauTau_20261030_161621/261030_151701/0001}"
REPO_DIR="${REPO_DIR:-$HOME/thesis-deliverables}"
LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt}"

source "$LCG_VIEW/setup.sh"
set -u

export MINIAOD_READER_TRACE=1
OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/miniaod_smoke.XXXXXX")"
echo "SMOKE output directory: $OUTPUT_DIR (full log: $OUTPUT_DIR/smoke.log)"
if ! python3 "$REPO_DIR/scripts/condor/convert_to_parquet/smoke_miniaod.py" "$INPUT_DIR" "$OUTPUT_DIR" > "$OUTPUT_DIR/smoke.log" 2>&1; then
    tail -n 30 "$OUTPUT_DIR/smoke.log"
    echo "SMOKE FAILED"
    exit 1
fi
grep '^SMOKE' "$OUTPUT_DIR/smoke.log"
echo "REFERENCE (abcbatch1 0001, ROOT 6.40.04, uproot 5.7.7):"
echo "REFERENCE input: 7 files, 59 events, first miniaod_1000.root"
echo "REFERENCE all: events 45 | tracks 32846 | tracks/event 729.9 | sum track_pt 34551.77 | sum |dxy significance| 167222.5 | kstar candidates 4560 | fired IP6 16 | columns 155"
echo "REFERENCE from_pv: events 32 | tracks 9570 | tracks/event 299.1 | sum track_pt 11057.68 | sum |dxy significance| 110109.4 | kstar candidates 3461 | fired IP6 12 | columns 155"
