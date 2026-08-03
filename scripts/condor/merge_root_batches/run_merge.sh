#!/bin/bash
# HTCondor wrapper for merge_batches.py
# Submit with: condor_submit merge_batches.sub

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python3 /afs/cern.ch/user/o/oprostak/condor/compress_source_root/merge_batches.py "$1"
