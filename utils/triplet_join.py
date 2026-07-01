from __future__ import annotations

import math

import torch

from utils.couple_features import M_TAU_GEV, RHO_MASS_GEV

PION_MASS_GEV = 0.13957

# Tier-HS default windows (starting points; the eval sweep tightens them).
DZ_WINDOW_HS = 3.0
DR_WINDOW_HS = 0.5
A1_LO_GEV = 0.6
A1_HI_GEV = 1.5
RHO_WINDOW_HS = 0.30


def build_track_lorentz(
    pt: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor,
) -> torch.Tensor:
    """pt, eta, phi: each (N,). Returns (4, N) = (px, py, pz, E), pion-mass hypothesis."""
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + PION_MASS_GEV ** 2)
    return torch.stack([px, py, pz, energy], dim=0)


def _mass(lorentz: torch.Tensor, *index_groups: torch.Tensor) -> torch.Tensor:
    summed = sum(lorentz[:, idx] for idx in index_groups)
    m_squared = summed[3] ** 2 - summed[0] ** 2 - summed[1] ** 2 - summed[2] ** 2
    return torch.sqrt(torch.clamp_min(m_squared, 0.0))


def _delta_r_squared(
    eta_a: torch.Tensor, phi_a: torch.Tensor,
    eta_b: torch.Tensor, phi_b: torch.Tensor,
) -> torch.Tensor:
    delta_phi = (phi_a - phi_b + math.pi) % (2 * math.pi) - math.pi
    return (eta_a - eta_b) ** 2 + delta_phi ** 2


def _enumerate(couples: torch.Tensor, pool: torch.Tensor):
    num_couples = couples.shape[0]
    pool_size = pool.shape[0]
    track_i = couples[:, 0].repeat_interleave(pool_size)
    track_j = couples[:, 1].repeat_interleave(pool_size)
    track_k = pool.repeat(num_couples)
    couple_row = torch.arange(num_couples, device=couples.device).repeat_interleave(pool_size)
    base_keep = (track_k != track_i) & (track_k != track_j)
    return track_i, track_j, track_k, couple_row, base_keep


def _dz_dist(dz, i, j, k):
    # dz holds the longitudinal impact-parameter significance dz/sigma_dz per track.
    # Distance of the third track's significance from the couple midpoint: small when
    # all three share one production vertex, large when k comes from elsewhere.
    return (dz[k] - 0.5 * (dz[i] + dz[j])).abs()


def _dr_min(eta, phi, i, j, k):
    dr_i = _delta_r_squared(eta[k], phi[k], eta[i], phi[i])
    dr_j = _delta_r_squared(eta[k], phi[k], eta[j], phi[j])
    return torch.minimum(dr_i, dr_j).sqrt()


def _rho_dist(lorentz, charge, i, j, k):
    big = torch.full((i.shape[0],), float("inf"))
    os_ik = torch.where(charge[i] * charge[k] < 0, (_mass(lorentz, i, k) - RHO_MASS_GEV).abs(), big)
    os_jk = torch.where(charge[j] * charge[k] < 0, (_mass(lorentz, j, k) - RHO_MASS_GEV).abs(), big)
    os_ij = torch.where(charge[i] * charge[j] < 0, (_mass(lorentz, i, j) - RHO_MASS_GEV).abs(), big)
    return torch.minimum(torch.minimum(os_ik, os_jk), os_ij)


def _pt(lorentz, *index_groups):
    summed = sum(lorentz[:, idx] for idx in index_groups)
    return torch.sqrt(summed[0] ** 2 + summed[1] ** 2)


FEATURE_NAMES = [
    "dz_dist", "dr_min", "m_ijk", "rho_dist",
    "couple_rank", "is_same_sign", "m_ij", "pt_ij",
    "pt_k", "abs_eta_k", "dz_sig_k", "dxy_sig_k", "dca_sig_k", "n_pixel_k", "norm_chi2_k", "rel_pt_err_k",
    "m_ik", "m_jk", "dr_ij", "dr_ik", "dr_jk", "pt_frac_k", "dz_spread", "pt_ijk",
]
GATE4_NAMES = FEATURE_NAMES[:4]


def build_triplet_candidates(
    couples: torch.Tensor,
    pool: torch.Tensor,
    *,
    lorentz: torch.Tensor,
    charge: torch.Tensor,
    eta: torch.Tensor | None = None,
    phi: torch.Tensor | None = None,
    dz: torch.Tensor | None = None,
    charge_gate: bool = True,
    mass_max: float | None = M_TAU_GEV,
    dz_window: float | None = None,
    dr_window: float | None = None,
    a1_window: tuple[float, float] | None = None,
    rho_window: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """couples: (C, 2) long. pool: (P,) long. lorentz: (4, N). charge/eta/phi/dz: (N,).

    Returns (triplets (M, 3) long of (i, j, k), couple_row (M,) long).
    """
    track_i, track_j, track_k, couple_row, keep = _enumerate(couples, pool)

    if charge_gate:
        keep &= (charge[track_i] + charge[track_j] + charge[track_k]).abs().round() == 1

    if mass_max is not None:
        keep &= _mass(lorentz, track_i, track_j, track_k) <= mass_max

    if a1_window is not None:
        triplet_mass = _mass(lorentz, track_i, track_j, track_k)
        keep &= (triplet_mass >= a1_window[0]) & (triplet_mass <= a1_window[1])

    if dz_window is not None:
        keep &= _dz_dist(dz, track_i, track_j, track_k) <= dz_window

    if dr_window is not None:
        keep &= _dr_min(eta, phi, track_i, track_j, track_k) <= dr_window

    if rho_window is not None:
        keep &= _rho_dist(lorentz, charge, track_i, track_j, track_k) <= rho_window

    triplets = torch.stack([track_i[keep], track_j[keep], track_k[keep]], dim=1)
    return triplets, couple_row[keep]


def triplet_gate_quantities(
    couples: torch.Tensor,
    pool: torch.Tensor,
    *,
    lorentz: torch.Tensor,
    charge: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    dz: torch.Tensor,
    gt_sorted: tuple[int, int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """couples: (C, 2) long. pool: (P,) long. lorentz: (4, N). charge/eta/phi/dz: (N,).

    Per Tier-H-surviving candidate (charge ±1 and m(ijk) <= m_tau): the four soft-gate
    quantities, is_gt, couple_row. All returned tensors share length M_H.
    """
    track_i, track_j, track_k, couple_row, base = _enumerate(couples, pool)
    h_keep = base.clone()
    h_keep &= (charge[track_i] + charge[track_j] + charge[track_k]).abs().round() == 1
    h_keep &= _mass(lorentz, track_i, track_j, track_k) <= M_TAU_GEV

    i, j, k = track_i[h_keep], track_j[h_keep], track_k[h_keep]
    result = {
        "dz_dist": _dz_dist(dz, i, j, k),
        "dr_min": _dr_min(eta, phi, i, j, k),
        "m_ijk": _mass(lorentz, i, j, k),
        "rho_dist": _rho_dist(lorentz, charge, i, j, k),
        "couple_row": couple_row[h_keep],
    }
    if gt_sorted is not None:
        sorted_rows = torch.stack([i, j, k], dim=1).sort(dim=1).values
        target = torch.tensor(sorted(gt_sorted), device=i.device)
        result["is_gt"] = (sorted_rows == target).all(dim=1)
    else:
        result["is_gt"] = torch.zeros(i.shape[0], dtype=torch.bool, device=i.device)
    return result


def triplet_candidate_features(
    couples: torch.Tensor,
    pool: torch.Tensor,
    *,
    lorentz: torch.Tensor,
    charge: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    dz: torch.Tensor,
    dxy_sig: torch.Tensor,
    dca_sig: torch.Tensor,
    n_pixel: torch.Tensor,
    norm_chi2: torch.Tensor,
    pt_error: torch.Tensor,
    gt_sorted: tuple[int, int, int] | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """couples: (C, 2) long. pool: (P,) long. lorentz: (4, N). per-track inputs: (N,).

    Per Tier-H-surviving candidate: (X (M_H, 24) features in FEATURE_NAMES order,
    FEATURE_NAMES, is_gt (M_H,), couple_row (M_H,)). Columns 0:4 equal triplet_gate_quantities.
    """
    track_i, track_j, track_k, couple_row, base = _enumerate(couples, pool)
    h_keep = base.clone()
    h_keep &= (charge[track_i] + charge[track_j] + charge[track_k]).abs().round() == 1
    h_keep &= _mass(lorentz, track_i, track_j, track_k) <= M_TAU_GEV

    i, j, k = track_i[h_keep], track_j[h_keep], track_k[h_keep]
    cr = couple_row[h_keep]
    dz_ij = (dz[i] - dz[j]).abs()
    dz_ik = (dz[i] - dz[k]).abs()
    dz_jk = (dz[j] - dz[k]).abs()
    pt_k = _pt(lorentz, k)
    pt_ijk = _pt(lorentz, i, j, k)
    columns = [
        _dz_dist(dz, i, j, k),
        _dr_min(eta, phi, i, j, k),
        _mass(lorentz, i, j, k),
        _rho_dist(lorentz, charge, i, j, k),
        cr.float(),
        (charge[i] == charge[j]).float(),
        _mass(lorentz, i, j),
        _pt(lorentz, i, j),
        pt_k,
        eta[k].abs(),
        dz[k],
        dxy_sig[k],
        dca_sig[k],
        n_pixel[k],
        norm_chi2[k],
        pt_error[k] / torch.clamp_min(pt_k, 1e-6),
        _mass(lorentz, i, k),
        _mass(lorentz, j, k),
        _delta_r_squared(eta[i], phi[i], eta[j], phi[j]).sqrt(),
        _delta_r_squared(eta[i], phi[i], eta[k], phi[k]).sqrt(),
        _delta_r_squared(eta[j], phi[j], eta[k], phi[k]).sqrt(),
        pt_k / torch.clamp_min(pt_ijk, 1e-6),
        torch.maximum(torch.maximum(dz_ij, dz_ik), dz_jk),
        pt_ijk,
    ]
    features = torch.stack(columns, dim=1)

    if gt_sorted is not None:
        sorted_rows = torch.stack([i, j, k], dim=1).sort(dim=1).values
        target = torch.tensor(sorted(gt_sorted), device=i.device)
        is_gt = (sorted_rows == target).all(dim=1)
    else:
        is_gt = torch.zeros(i.shape[0], dtype=torch.bool, device=i.device)
    return features, FEATURE_NAMES, is_gt, cr


_TIER_DEFAULTS = {
    "A0": dict(charge_gate=False, mass_max=None),
    "H": dict(charge_gate=True, mass_max=M_TAU_GEV),
    "HS": dict(
        charge_gate=True, mass_max=M_TAU_GEV,
        dz_window=DZ_WINDOW_HS, dr_window=DR_WINDOW_HS,
        a1_window=(A1_LO_GEV, A1_HI_GEV), rho_window=RHO_WINDOW_HS,
    ),
}


def candidates_for_tier(
    tier: str,
    couples: torch.Tensor,
    pool: torch.Tensor,
    *,
    lorentz: torch.Tensor,
    charge: torch.Tensor,
    eta: torch.Tensor | None = None,
    phi: torch.Tensor | None = None,
    dz: torch.Tensor | None = None,
    **override,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tier: one of "A0", "H", "HS". See build_triplet_candidates for tensor shapes."""
    params = dict(_TIER_DEFAULTS[tier])
    params.update(override)
    return build_triplet_candidates(
        couples, pool, lorentz=lorentz, charge=charge,
        eta=eta, phi=phi, dz=dz, **params,
    )


def compression_stats(
    n_survive: torch.Tensor, n_full: torch.Tensor,
) -> dict[str, float]:
    """n_survive, n_full: each (num_events,) candidate-tuple counts."""
    total_survive = float(n_survive.sum())
    total_full = float(n_full.sum())
    valid = n_full > 0
    per_event = (n_survive[valid].float() / n_full[valid].float())
    return {
        "ratio": total_survive / total_full if total_full > 0 else 0.0,
        "factor": total_full / total_survive if total_survive > 0 else float("inf"),
        "per_event_mean": float(per_event.mean()) if valid.any() else 0.0,
        "per_event_median": float(per_event.median()) if valid.any() else 0.0,
    }
