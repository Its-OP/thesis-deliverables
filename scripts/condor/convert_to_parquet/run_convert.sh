#!/bin/bash
# HTCondor wrapper for convert_root_to_parquet.py
# Submit with: condor_submit convert_to_parquet.sub

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# Builder schema: needs the merged_bld_ re-merge (K*/TauTo3Pi/BeamSpot
# branches). Writes to a fresh directory so the canonical datasets/parquet
# shards are not overwritten before the positional-identity gate is run.
python3 /afs/cern.ch/user/o/oprostak/condor/convert_root_to_parquet/convert_root_to_parquet.py \
    --input-dir /eos/user/o/oprostak/tau_data/datasets/root/train_eval \
    --pattern 'merged_bld_batch*.root' \
    --extended --builder-blocks \
    --output-dir /eos/user/o/oprostak/tau_data/datasets/parquet_builder \
    --val-subdir eval
