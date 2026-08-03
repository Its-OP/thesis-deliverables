"""Merge per-job ROOT files into a single file per batch.

Keeps only the branches needed by convert_root_to_parquet.py --extended:
  - CMS event identifiers: run, event, luminosityBlock
  - Source file identifiers: source_batch_id, source_microbatch_id
  - Primary vertex + pileup context: PV_x/y/z, PV_npvs, PV_npvsGood,
    OtherPV_z
  - Track features: all branches from TRACK_BRANCH_MAP_ALL (incl. the
    dxy/dsz covariance rows, lost-track / lepton-overlap flags, and the
    from-B label)
  - Track_pdgId: used for pion filtering (|pdgId| == 211)
  - Muon collection: the H6.M branch set
  - SV collection: the H6.S branch set

The source_batch_id and source_microbatch_id columns are derived from
the input file path. Together with ``(run, event, luminosityBlock)``
they form a composite key that uniquely identifies each tau candidate
across the full dataset.

source_batch_id: from the batch folder name (``batch_{N1}``)
source_microbatch_id: from the filename (``step_MINI_{N2}_...``)

This reduces output size ~8x compared to keeping all 272 NanoAOD branches.

Usage (via HTCondor):
    python merge_batches.py <batch_id>
"""

import os
import re
import sys
import glob

import numpy as np
import uproot
import awkward as ak

# All branches consumed by the parquet conversion pipeline.
KEEP_BRANCHES = [
    # CMS event identifiers
    'run', 'event', 'luminosityBlock',
    # Primary vertex
    'PV_x', 'PV_y', 'PV_z',
    # Track kinematics
    'Track_pt', 'Track_eta', 'Track_phi', 'Track_mass', 'Track_charge',
    # Track impact parameters
    'Track_dxy', 'Track_dxyS', 'Track_dz', 'Track_dzS', 'Track_dzTrg',
    # Track quality
    'Track_normChi2', 'Track_nValidHits', 'Track_nValidPixelHits',
    'Track_ptErr', 'Track_DCASig',
    # Track covariance matrix (momentum block + dxy/dsz rows)
    'Track_covQopQop', 'Track_covQopLam', 'Track_covQopPhi',
    'Track_covLamLam', 'Track_covLamPhi', 'Track_covPhiPhi',
    'Track_covDxyDxy', 'Track_covDszDsz', 'Track_covDxyDsz',
    'Track_covLamDxy', 'Track_covPhiDxy', 'Track_covQopDxy',
    # Track vertex position
    'Track_vx', 'Track_vy', 'Track_vz',
    # Track quality / lepton-overlap flags
    'Track_isLostTrk', 'Track_isMatchedToMuon', 'Track_isMatchedToEle',
    # Track labels and ID
    'Track_trackFromTau', 'Track_trackFromB', 'Track_pdgId',
    # Pileup context
    'PV_npvs', 'PV_npvsGood', 'OtherPV_z',
    # Muon collection (H6.M)
    'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_charge',
    'Muon_dz', 'Muon_dzErr', 'Muon_dxy', 'Muon_dxyErr',
    'Muon_ip3d', 'Muon_sip3d', 'Muon_softId', 'Muon_mediumId',
    'Muon_isTriggering', 'Muon_pfRelIso03_all',
    'Muon_vx', 'Muon_vy', 'Muon_vz',
    # Secondary-vertex collection (H6.S)
    'SV_x', 'SV_y', 'SV_z', 'SV_dlen', 'SV_dlenSig', 'SV_pAngle',
    'SV_mass', 'SV_ntracks', 'SV_chi2', 'SV_dxySig',
]

# Regex to extract microbatch id from filename:
#   step_MINI_{N2}_nano_ditaus_mc.root → N2
MICROBATCH_RE = re.compile(r'step_MINI_(\d+)_nano_ditaus_mc\.root$')


def parse_microbatch_id(filepath):
    """Extract microbatch id from a source file path.

    Args:
        filepath: Full path to a microbatch ROOT file.

    Returns:
        Integer microbatch id, or -1 if the filename doesn't match.
    """
    match = MICROBATCH_RE.search(filepath)
    return int(match.group(1)) if match else -1


BASE_DIR = "/eos/user/o/oprostak/tau_data"


def merge_batch(batch_id, files, output):
    """files: list of source ROOT file paths, merged in the given order.
    Writes KEEP_BRANCHES (those present) + source id columns to output."""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with uproot.recreate(output) as out_file:
        writer = None

        for filepath in files:
            microbatch_id = parse_microbatch_id(filepath)
            tree = uproot.open(filepath)["Events"]

            # Keep only branches present in this file
            available = set(tree.keys())
            branches_to_read = [b for b in KEEP_BRANCHES if b in available]

            data = tree.arrays(branches_to_read, library="ak")
            n_entries = len(data)

            # Add source identifiers as scalar columns
            data["source_batch_id"] = np.full(n_entries, batch_id, dtype=np.int32)
            data["source_microbatch_id"] = np.full(n_entries, microbatch_id, dtype=np.int32)

            if writer is None:
                writer = out_file.mktree("Events", {
                    field: data[field].type for field in data.fields
                })

            writer.extend({field: data[field] for field in data.fields})
            print(f"  {os.path.basename(filepath)}: {n_entries} entries "
                  f"(microbatch_id={microbatch_id})")

    print(f"batch{batch_id}: done -> {output}")


def main():
    batch_id = int(sys.argv[1])

    # Extended-schema merge output. Distinct filename prefix (merged_ext_) so
    # the skip-if-exists guard can never collide with legacy merged files.
    output = os.path.join(BASE_DIR, "datasets", "root", "train_eval",
                          f"merged_ext_batch{batch_id}.root")

    if os.path.exists(output):
        print(f"batch{batch_id}: already exists, skipping -> {output}")
        sys.exit(0)

    files = sorted(glob.glob(
        f"/eos/cms/store/group/phys_bphys/valukash/mc_signal/"
        f"batch{batch_id}_2024/*.root"
    ))
    files = [f for f in files if "merged_" not in f]

    if not files:
        print(f"batch{batch_id}: no files found, skipping")
        sys.exit(0)

    print(f"batch{batch_id}: merging {len(files)} files...")
    merge_batch(batch_id, files, output)


if __name__ == "__main__":
    main()
