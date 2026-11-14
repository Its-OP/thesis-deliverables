"""Smoke test of the MiniAOD -> parquet conversion: converts the files
miniaod_100?.root of one CRAB subdirectory with both track selections into a
fresh directory and prints the numbers to compare between machines.

Usage: python3 smoke_miniaod.py INPUT_DIR OUTPUT_DIR
"""
import glob
import os
import sys

import awkward as ak
import numpy as np
import uproot

import convert_root_to_parquet as converter

PATTERN = 'miniaod_100?.root'


def summarize(output_dir, track_selection):
    """output_dir: conversion output; track_selection: its selection.
    Prints one line of event / track counts and checksums."""
    files = sorted(glob.glob(os.path.join(output_dir, 'smoke', '*.parquet')))
    rows = ak.concatenate([ak.from_parquet(path) for path in files])
    track_pt = ak.to_numpy(ak.flatten(rows['track_pt'])).astype(np.float64)
    significance = ak.to_numpy(ak.flatten(rows['track_dxy_significance'])).astype(np.float64)
    print(f'SMOKE {track_selection}: events {len(rows)} | tracks {track_pt.size} | '
          f'tracks/event {track_pt.size / len(rows):.1f} | sum track_pt {track_pt.sum():.2f} | '
          f'sum |dxy significance| {np.abs(significance).sum():.1f} | '
          f'kstar candidates {int(ak.sum(ak.num(rows["kstar_pt"])))} | '
          f'fired IP6 {int(ak.sum(rows["event_fired_ip6"]))} | columns {len(rows.fields)}')


def main():
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    import ROOT
    print(f'SMOKE versions: ROOT {ROOT.gROOT.GetVersion()} | uproot {uproot.__version__} | '
          f'awkward {ak.__version__} | numpy {np.__version__}')
    files = converter.discover_root_files(input_dir, PATTERN)
    scanned = sum(uproot.open(path)['Events'].num_entries for path in files)
    print(f'SMOKE input: {len(files)} files, {scanned} events, first {os.path.basename(files[0])}')
    for track_selection in converter.TRACK_SELECTIONS:
        selection_dir = os.path.join(output_dir, track_selection)
        converter.convert(
            input_dir=[input_dir], output_dir=selection_dir, train_events=0, val_events=0, train_files=1,
            val_files=1, pt_cutoff=0.5, required_gt_pions=3, chunk_size=2500, output_columns=None, extended=True,
            pattern=PATTERN, val_subdir='smoke', eval_only=True, shard_events=5000,
            track_selection=track_selection, shard_prefix='smoke', max_unreadable_fraction=0.0)
        summarize(selection_dir, track_selection)
    print('SMOKE OK')


if __name__ == '__main__':
    main()
