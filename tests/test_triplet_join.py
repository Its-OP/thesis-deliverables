from __future__ import annotations

import glob
import math
import os

import pytest
import torch

from utils.couple_features import M_TAU_GEV, RHO_MASS_GEV
from utils.triplet_join import (
    FEATURE_NAMES,
    GATE4_NAMES,
    PION_MASS_GEV,
    build_track_lorentz,
    build_triplet_candidates,
    candidates_for_tier,
    compression_stats,
    triplet_candidate_features,
    triplet_gate_quantities,
)

# ---------------------------------------------------------------------------
# Synthetic single-event fixture
# ---------------------------------------------------------------------------
# 5 tracks. Charges chosen so a valid tau-like triplet (2 same + 1 opposite)
# exists. Track 4 sits far away in (eta, phi, dz) to exercise spatial gates.
#   idx: 0    1    2    3    4
#   q:  +1   +1   -1   -1   +1


def _event():
    pt = torch.tensor([1.0, 1.2, 0.9, 1.1, 0.8])
    eta = torch.tensor([0.10, 0.15, 0.12, 0.18, 3.00])
    phi = torch.tensor([0.05, 0.10, 0.08, 0.12, 2.50])
    charge = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0])
    dz = torch.tensor([0.20, 0.25, 0.22, 0.28, 9.00])
    lorentz = build_track_lorentz(pt, eta, phi)
    return dict(
        lorentz=lorentz, charge=charge, eta=eta, phi=phi, dz=dz, pt=pt,
        dxy_sig=torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9]),
        dca_sig=torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
        n_pixel=torch.tensor([4.0, 4.0, 3.0, 5.0, 2.0]),
        norm_chi2=torch.tensor([1.0, 1.2, 0.9, 1.1, 2.0]),
        pt_error=torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]),
    )


def _all_couples(num_tracks):
    i, j = torch.triu_indices(num_tracks, num_tracks, offset=1).unbind(0)
    return torch.stack([i, j], dim=1)


# ---------------------------------------------------------------------------
# build_track_lorentz
# ---------------------------------------------------------------------------

def test_build_track_lorentz_components_and_mass():
    pt = torch.tensor([2.0])
    eta = torch.tensor([0.5])
    phi = torch.tensor([0.3])
    p4 = build_track_lorentz(pt, eta, phi)
    assert p4.shape == (4, 1)
    assert torch.allclose(p4[0], pt * torch.cos(phi))
    assert torch.allclose(p4[1], pt * torch.sin(phi))
    assert torch.allclose(p4[2], pt * torch.sinh(eta))
    mass_sq = p4[3] ** 2 - p4[0] ** 2 - p4[1] ** 2 - p4[2] ** 2
    assert torch.allclose(mass_sq.sqrt(), torch.tensor([PION_MASS_GEV]), atol=1e-4)


# ---------------------------------------------------------------------------
# Charge gate (A1)
# ---------------------------------------------------------------------------

def test_charge_gate_only_net_unit_charge_survives():
    ev = _event()
    couples = _all_couples(5)
    pool = torch.arange(5)
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        charge_gate=True, mass_max=None,
    )
    q = ev["charge"]
    net = q[triplets[:, 0]] + q[triplets[:, 1]] + q[triplets[:, 2]]
    assert torch.all(net.abs().round() == 1)


def test_charge_gate_excludes_all_same_sign_triplet():
    ev = _event()
    # (0,1,4) are all +1 -> net +3, must be excluded by the charge gate.
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([4])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        charge_gate=True, mass_max=None,
    )
    assert triplets.shape[0] == 0


def test_charge_gate_keeps_valid_tau_triplet():
    ev = _event()
    # (0,1,2) = (+,+,-) net +1 -> valid.
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        charge_gate=True, mass_max=None,
    )
    assert triplets.tolist() == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Mass gate (A3) — lossless upper cut
# ---------------------------------------------------------------------------

def test_mass_gate_excludes_above_m_tau():
    # Two heavy tracks: triplet mass far exceeds m_tau.
    pt = torch.tensor([50.0, 50.0, 50.0])
    eta = torch.tensor([0.0, 1.5, -1.5])
    phi = torch.tensor([0.0, 2.0, -2.0])
    charge = torch.tensor([1.0, 1.0, -1.0])
    lorentz = build_track_lorentz(pt, eta, phi)
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=lorentz, charge=charge,
        charge_gate=False, mass_max=M_TAU_GEV,
    )
    assert triplets.shape[0] == 0


def test_mass_gate_keeps_below_m_tau():
    ev = _event()  # soft tracks, triplet mass well below m_tau
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        charge_gate=False, mass_max=M_TAU_GEV,
    )
    assert triplets.tolist() == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# No self-pairs
# ---------------------------------------------------------------------------

def test_third_never_equals_couple_member():
    ev = _event()
    couples = _all_couples(5)
    pool = torch.arange(5)
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        charge_gate=False, mass_max=None,
    )
    assert torch.all(triplets[:, 2] != triplets[:, 0])
    assert torch.all(triplets[:, 2] != triplets[:, 1])


# ---------------------------------------------------------------------------
# A0 baseline = full cross-join count
# ---------------------------------------------------------------------------

def test_a0_full_cross_join_count():
    ev = _event()
    couples = _all_couples(5)  # C = 10
    pool = torch.arange(5)     # P = 5
    triplets, couple_row = candidates_for_tier(
        "A0", couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
    )
    # Each couple (i,j) joins against pool minus {i,j} = 3 thirds.
    assert triplets.shape[0] == 10 * 3
    assert couple_row.max().item() == 9


# ---------------------------------------------------------------------------
# Tier-H losslessness + Tier-HS subset
# ---------------------------------------------------------------------------

def test_tier_h_keeps_gt_triplet_present_in_pool():
    ev = _event()
    # GT triplet (0,1,2): couple (0,1) ranked, third 2 in pool.
    couples = torch.tensor([[0, 1]])
    pool = torch.arange(5)
    triplets, _ = candidates_for_tier(
        "H", couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
    )
    rows = set(map(tuple, triplets.tolist()))
    assert (0, 1, 2) in rows


def test_tier_hs_is_subset_of_tier_h():
    ev = _event()
    couples = _all_couples(5)
    pool = torch.arange(5)
    kw = dict(lorentz=ev["lorentz"], charge=ev["charge"],
              eta=ev["eta"], phi=ev["phi"], dz=ev["dz"])
    h, _ = candidates_for_tier("H", couples, pool, **kw)
    hs, _ = candidates_for_tier("HS", couples, pool, **kw)
    h_set = set(map(tuple, h.tolist()))
    hs_set = set(map(tuple, hs.tolist()))
    assert hs.shape[0] <= h.shape[0]
    assert hs_set.issubset(h_set)


# ---------------------------------------------------------------------------
# Spatial gates (A2)
# ---------------------------------------------------------------------------

def test_dz_window_prunes_far_vertex():
    ev = _event()
    # Track 4 has dz=9.0, far from the couple (0,1) centroid (~0.22).
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2, 4])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
        charge_gate=False, mass_max=None, dz_window=1.0,
    )
    thirds = set(triplets[:, 2].tolist())
    assert 2 in thirds and 4 not in thirds


def test_dr_window_prunes_far_cone():
    ev = _event()
    # Track 4 is far in (eta, phi); track 2 is close to the couple (0,1).
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2, 4])
    triplets, _ = build_triplet_candidates(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
        charge_gate=False, mass_max=None, dr_window=0.5,
    )
    thirds = set(triplets[:, 2].tolist())
    assert 2 in thirds and 4 not in thirds


# ---------------------------------------------------------------------------
# Compression stats helper
# ---------------------------------------------------------------------------

def test_compression_stats_ratio():
    n_survive = torch.tensor([3, 0, 5])
    n_full = torch.tensor([10, 10, 10])
    stats = compression_stats(n_survive, n_full)
    assert stats["ratio"] == pytest.approx(8 / 30)
    assert stats["factor"] == pytest.approx(30 / 8)
    assert stats["per_event_mean"] == pytest.approx((0.3 + 0.0 + 0.5) / 3)


# ---------------------------------------------------------------------------
# triplet_gate_quantities (sweep primitive)
# ---------------------------------------------------------------------------

def _np_mass(p4, *cols):
    s = sum(p4[:, c] for c in cols)
    return math.sqrt(max(s[3] ** 2 - s[0] ** 2 - s[1] ** 2 - s[2] ** 2, 0.0))


def test_gate_quantities_dz_and_dr_manual():
    ev = _event()
    couples = torch.tensor([[0, 1]])
    pool = torch.tensor([2])
    q = triplet_gate_quantities(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
    )
    assert q["dz_dist"].shape == (1,)
    # |0.22 - 0.5*(0.20+0.25)| = 0.005
    assert torch.allclose(q["dz_dist"], torch.tensor([0.005]), atol=1e-5)
    # min ΔR((2,0),(2,1)); both = sqrt(0.0013)
    assert torch.allclose(q["dr_min"], torch.tensor([math.sqrt(0.0013)]), atol=1e-5)


def test_gate_quantities_mass_independent():
    ev = _event()
    q = triplet_gate_quantities(
        torch.tensor([[0, 1]]), torch.tensor([2]), lorentz=ev["lorentz"],
        charge=ev["charge"], eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
    )
    expected = _np_mass(ev["lorentz"].numpy(), 0, 1, 2)
    assert torch.allclose(q["m_ijk"], torch.tensor([expected]), atol=1e-5)


def test_gate_quantities_rho_uses_os_subpairs():
    ev = _event()  # charges (+,+,-) for tracks 0,1,2 -> OS pairs (0,2),(1,2)
    q = triplet_gate_quantities(
        torch.tensor([[0, 1]]), torch.tensor([2]), lorentz=ev["lorentz"],
        charge=ev["charge"], eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
    )
    p4 = ev["lorentz"].numpy()
    expected = min(abs(_np_mass(p4, 0, 2) - RHO_MASS_GEV), abs(_np_mass(p4, 1, 2) - RHO_MASS_GEV))
    assert torch.allclose(q["rho_dist"], torch.tensor([expected]), atol=1e-5)


def test_gate_quantities_is_gt_flags_exactly_the_triplet():
    ev = _event()
    q = triplet_gate_quantities(
        torch.tensor([[0, 1]]), torch.arange(5), lorentz=ev["lorentz"],
        charge=ev["charge"], eta=ev["eta"], phi=ev["phi"], dz=ev["dz"],
        gt_sorted=(0, 1, 2),
    )
    assert q["is_gt"].sum().item() == 1


def test_gate_quantities_universe_equals_tier_h():
    ev = _event()
    couples = _all_couples(5)
    pool = torch.arange(5)
    kw = dict(lorentz=ev["lorentz"], charge=ev["charge"], eta=ev["eta"], phi=ev["phi"], dz=ev["dz"])
    q = triplet_gate_quantities(couples, pool, **kw)
    h, _ = candidates_for_tier("H", couples, pool, **kw)
    assert q["m_ijk"].shape[0] == h.shape[0]
    # All-infinite thresholds keep every H candidate.
    survive = (q["dz_dist"] <= float("inf")) & (q["dr_min"] <= float("inf")) \
        & (q["m_ijk"] <= float("inf")) & (q["rho_dist"] <= float("inf"))
    assert int(survive.sum()) == h.shape[0]


# ---------------------------------------------------------------------------
# triplet_candidate_features (rich feature builder)
# ---------------------------------------------------------------------------

def _features(ev, couples, pool, gt_sorted=None):
    return triplet_candidate_features(
        couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
        eta=ev["eta"], phi=ev["phi"], dz=ev["dz"], dxy_sig=ev["dxy_sig"],
        dca_sig=ev["dca_sig"], n_pixel=ev["n_pixel"], norm_chi2=ev["norm_chi2"],
        pt_error=ev["pt_error"], gt_sorted=gt_sorted,
    )


def test_candidate_features_shape_and_gate4_columns():
    ev = _event()
    couples = _all_couples(5)
    pool = torch.arange(5)
    X, names, is_gt, cr = _features(ev, couples, pool)
    assert names == FEATURE_NAMES
    assert X.shape[1] == len(FEATURE_NAMES) == 24
    assert X.shape[0] == cr.shape[0] == is_gt.shape[0]
    # Columns 0:4 reproduce the gate quantities.
    q = triplet_gate_quantities(couples, pool, lorentz=ev["lorentz"], charge=ev["charge"],
                                eta=ev["eta"], phi=ev["phi"], dz=ev["dz"])
    for col, key in enumerate(["dz_dist", "dr_min", "m_ijk", "rho_dist"]):
        assert torch.allclose(X[:, col], q[key], atol=1e-5)


def test_candidate_features_manual_columns():
    ev = _event()
    X, names, _, _ = _features(ev, torch.tensor([[0, 1]]), torch.tensor([2]))
    idx = {n: c for c, n in enumerate(names)}
    assert X[0, idx["is_same_sign"]].item() == 1.0     # charges 0,1 both +1
    assert torch.allclose(X[0, idx["dz_sig_k"]], torch.tensor(0.22), atol=1e-5)
    assert torch.allclose(X[0, idx["abs_eta_k"]], torch.tensor(0.12), atol=1e-5)
    assert torch.allclose(X[0, idx["pt_k"]], torch.tensor(0.9), atol=1e-5)
    assert torch.allclose(X[0, idx["dxy_sig_k"]], torch.tensor(0.7), atol=1e-5)


def test_candidate_features_is_gt():
    ev = _event()
    X, _, is_gt, _ = _features(ev, torch.tensor([[0, 1]]), torch.arange(5), gt_sorted=(0, 1, 2))
    assert is_gt.sum().item() == 1


def test_gate4_names_are_first_four():
    assert GATE4_NAMES == FEATURE_NAMES[:4]
    assert GATE4_NAMES == ["dz_dist", "dr_min", "m_ijk", "rho_dist"]


def test_candidate_features_new_columns_manual():
    ev = _event()
    X, names, _, _ = _features(ev, torch.tensor([[0, 1]]), torch.tensor([2]))
    idx = {n: c for c, n in enumerate(names)}
    # rel_pt_err_k = pt_error[k] / pt_k = 0.03 / 0.9
    assert torch.allclose(X[0, idx["rel_pt_err_k"]], torch.tensor(0.03 / 0.9), atol=1e-5)
    # dr_ik = sqrt((0.10-0.12)^2 + (0.05-0.08)^2); dr_jk = sqrt((0.15-0.12)^2 + (0.10-0.08)^2)
    assert torch.allclose(X[0, idx["dr_ik"]], torch.tensor(math.sqrt(0.0004 + 0.0009)), atol=1e-5)
    assert torch.allclose(X[0, idx["dr_jk"]], torch.tensor(math.sqrt(0.0009 + 0.0004)), atol=1e-5)
    # pt_frac_k = pt_k / pt_ijk, both transverse magnitudes from the 4-vectors
    p4 = ev["lorentz"]
    rows = [0, 1, 2]
    pt_ijk = torch.sqrt(p4[0, rows].sum() ** 2 + p4[1, rows].sum() ** 2)
    assert torch.allclose(X[0, idx["pt_frac_k"]], torch.tensor(0.9) / pt_ijk, atol=1e-5)


# ---------------------------------------------------------------------------
# Real-data integration: Tier-H is lossless on the test split
# ---------------------------------------------------------------------------

_DUMP = "/Users/oleh/Projects/masters/deliverables/data/low-pt/eval/perstage_couples_val.parquet"
_SRC_GLOB = "/Users/oleh/Projects/masters/part/data/low-pt/val/val_*.parquet"


@pytest.mark.skipif(
    not (os.path.exists(_DUMP) and glob.glob(_SRC_GLOB)),
    reason="test-split dump or source parquet not present on this machine",
)
def test_tier_h_lossless_on_real_test_events():
    import numpy as np
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pyarrow.compute as pc

    n_events = 300
    dump = pq.read_table(
        _DUMP, columns=["stage1_sorted_indices", "stage3_sorted_couples"],
    ).slice(0, n_events)
    src = pa.concat_tables(
        [pq.read_table(s, columns=["event_n_tracks", "track_pt", "track_eta",
                                   "track_phi", "track_charge",
                                   "track_dz_significance", "track_label_from_tau"])
         for s in sorted(glob.glob(_SRC_GLOB))]
    ).slice(0, n_events)

    s1 = dump["stage1_sorted_indices"]
    couples_col = dump["stage3_sorted_couples"]
    checked = 0
    for r in range(n_events):
        # Positional alignment guard (see project memory).
        assert len(s1[r].as_py()) == src["event_n_tracks"][r].as_py()
        labels = np.asarray(src["track_label_from_tau"][r].as_py())
        gt = np.where(labels > 0.5)[0]
        if gt.size != 3:
            continue
        pool_np = np.asarray(s1[r].as_py())[:256]
        if not set(gt.tolist()).issubset(set(pool_np.tolist())):
            continue  # third pion outside the Stage-1 pool -> not H's job
        couples_np = np.asarray([c for c in couples_col[r].as_py()])
        # Keep only couples that are a GT pair (so a GT triplet is constructible).
        gt_set = set(gt.tolist())
        gt_couples = [c for c in couples_np if set(c.tolist()).issubset(gt_set)]
        if not gt_couples:
            continue

        pt = torch.tensor(src["track_pt"][r].as_py(), dtype=torch.float32)
        eta = torch.tensor(src["track_eta"][r].as_py(), dtype=torch.float32)
        phi = torch.tensor(src["track_phi"][r].as_py(), dtype=torch.float32)
        charge = torch.tensor(src["track_charge"][r].as_py(), dtype=torch.float32)
        lorentz = build_track_lorentz(pt, eta, phi)
        couples_t = torch.tensor(np.asarray(gt_couples), dtype=torch.long)
        pool_t = torch.tensor(pool_np, dtype=torch.long)
        triplets, _ = candidates_for_tier(
            "H", couples_t, pool_t, lorentz=lorentz, charge=charge,
        )
        rows = set(map(tuple, triplets.tolist()))
        # The full GT triplet, ordered as (couple_i, couple_j, third), survives.
        gt_couple = gt_couples[0]
        third = (gt_set - set(gt_couple.tolist())).pop()
        assert (int(gt_couple[0]), int(gt_couple[1]), third) in rows
        checked += 1
        if checked >= 20:
            break
    assert checked > 0
