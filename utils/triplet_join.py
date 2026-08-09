from __future__ import annotations

import math

import torch

from utils.couple_features import (
    A1_MASS_GEV,
    A1_WIDTH_GEV,
    M_TAU_GEV,
    RHO_MASS_GEV,
    RHO_SIGMA_GEV,
    RHO_WIDTH_GEV,
)

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


RICH_NAMES = [
    "dz_dist", "dr_min", "m_ijk", "rho_dist",
    "couple_rank", "is_same_sign", "m_ij", "pt_ij",
    "pt_k", "abs_eta_k", "dz_sig_k", "dxy_sig_k", "dca_sig_k", "n_pixel_k", "norm_chi2_k", "rel_pt_err_k",
    "m_ik", "m_jk", "dr_ij", "dr_ik", "dr_jk", "pt_frac_k", "dz_spread", "pt_ijk",
]

# Per-track 16-block: raw analogue of couple_features.py Block 1 (standardization dropped —
# trees are scale-invariant). Emitted for tracks i, j (the couple) and k (the third track).
TRACK16_NAMES = [
    "px", "py", "pz", "eta", "phi", "charge", "dxy_sig", "dz_sig",
    "norm_chi2", "pt_error", "n_pixel", "dca_sig", "cov_phi_phi", "cov_lambda_lambda",
    "pt", "rel_pt_err",
]
TRACK_I_NAMES = [f"ti_{name}" for name in TRACK16_NAMES]
TRACK_J_NAMES = [f"tj_{name}" for name in TRACK16_NAMES]
TRACK_K_NAMES = [f"tk_{name}" for name in TRACK16_NAMES]

# Couple-unit physics: couple_features.py Blocks 2+3(new)+5, treating the couple as a unit
# (no per-track split, no cascade scores). All derived from raw inputs.
COUPLE_UNIT_NAMES = [
    "ln_kt", "ln_z", "ln_dr", "ln_m2", "charge_prod", "dz_diff", "rho_ind", "rho_os",
    "dxy_phi", "lorentz_dot", "cpl_deta", "cpl_dphi", "kalman", "dca_sum", "helicity",
    "logbw_rho", "logbw_a1",
]

FEATURE_NAMES = RICH_NAMES + TRACK_I_NAMES + TRACK_J_NAMES + TRACK_K_NAMES + COUPLE_UNIT_NAMES
GATE4_NAMES = FEATURE_NAMES[:4]

# H6 block: vertex/lifetime, LHCb-style triplet physics, isolation cones and
# secondary-vertex attachment. Appended after the legacy 89 so every existing
# checkpoint keeps its column layout.
H6_VERTEX_NAMES = [
    "poca_max", "poca_mean", "crossing_z_gap_max", "raw_dz_gap_max",
    "lifetime_positive_count", "dxy_sig_spread",
]
H6_PHYSICS_NAMES = [
    "corrected_mass", "signed_ip_k", "min_pt_ijk", "scalar_sum_pt_ijk",
    "n_low_ip_in_cone",
]
H6_ISOLATION_NAMES = [
    "kept_cone_count", "kept_cone_sum_pt", "other_cone_count",
    "other_cone_sum_pt", "cone_min_dr",
]
H6_SV_NAMES = [
    "sv_min_distance", "sv_nearest_dlen_sig", "sv_nearest_mass",
    "sv_pointing_cos", "n_sv_matched", "has_sv",
]
H6_NAMES = H6_VERTEX_NAMES + H6_PHYSICS_NAMES + H6_ISOLATION_NAMES + H6_SV_NAMES
FEATURE_NAMES_EXTENDED = FEATURE_NAMES + H6_NAMES

CONE_DELTA_R_MAX = 0.4
CONE_DZ_MAX = 0.5
COMPANION_MIN_DR_SENTINEL = 0.4
LOW_IP_SIGNIFICANCE_MAX = 2.0
SV_MATCH_RADIUS_CM = 0.5
H6_INPUT_KEYS = (
    "vertex_x", "vertex_y", "vertex_z", "dz_raw",
    "primary_vertex_x", "primary_vertex_y",
    "sv_x", "sv_y", "sv_z", "sv_dlen_sig", "sv_mass",
    "other_pt", "other_eta", "other_phi", "other_dz",
)


def _unit_direction(eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """eta, phi: (M,). Returns (M, 3) unit momentum direction."""
    cosh_eta = torch.cosh(eta)
    return torch.stack([
        torch.cos(phi) / cosh_eta,
        torch.sin(phi) / cosh_eta,
        torch.tanh(eta),
    ], dim=1)


def _closest_approach(
    reference_a: torch.Tensor, direction_a: torch.Tensor,
    reference_b: torch.Tensor, direction_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All inputs (M, 3). Returns (distance (M,), midpoint (M, 3)) of the
    skew-line closest approach; parallel pairs fall back to point-to-line."""
    separation = reference_a - reference_b
    a_squared = (direction_a * direction_a).sum(dim=1, keepdim=True)
    b_squared = (direction_b * direction_b).sum(dim=1, keepdim=True)
    cross_dot = (direction_a * direction_b).sum(dim=1, keepdim=True)
    separation_a = (direction_a * separation).sum(dim=1, keepdim=True)
    separation_b = (direction_b * separation).sum(dim=1, keepdim=True)
    denominator = a_squared * b_squared - cross_dot ** 2
    parallel = denominator < 1e-12
    # torch.where evaluates both branches, so guard the denominators first.
    safe_denominator = torch.where(
        parallel, torch.ones_like(denominator), denominator)
    safe_b_squared = torch.where(
        b_squared > 0, b_squared, torch.ones_like(b_squared))
    parameter_a = torch.where(
        parallel, torch.zeros_like(denominator),
        (cross_dot * separation_b - b_squared * separation_a)
        / safe_denominator)
    parameter_b = torch.where(
        parallel, separation_b / safe_b_squared,
        (a_squared * separation_b - cross_dot * separation_a)
        / safe_denominator)
    closest_a = reference_a + parameter_a * direction_a
    closest_b = reference_b + parameter_b * direction_b
    distance = (closest_a - closest_b).square().sum(dim=1).sqrt()
    return distance, 0.5 * (closest_a + closest_b)


def _transverse_crossing_z_gap(
    reference_a: torch.Tensor, direction_a: torch.Tensor,
    reference_b: torch.Tensor, direction_b: torch.Tensor,
) -> torch.Tensor:
    """All inputs (M, 3). Returns (M,) |z_a - z_b| where the two transverse
    projections cross; NaN when they are transversely parallel."""
    determinant = (direction_a[:, 0] * direction_b[:, 1]
                   - direction_a[:, 1] * direction_b[:, 0])
    degenerate = determinant.abs() < 1e-12
    safe_determinant = torch.where(
        degenerate, torch.ones_like(determinant), determinant)
    offset_x = reference_b[:, 0] - reference_a[:, 0]
    offset_y = reference_b[:, 1] - reference_a[:, 1]
    parameter_a = (offset_x * direction_b[:, 1]
                   - offset_y * direction_b[:, 0]) / safe_determinant
    parameter_b = (offset_x * direction_a[:, 1]
                   - offset_y * direction_a[:, 0]) / safe_determinant
    gap = ((reference_a[:, 2] + parameter_a * direction_a[:, 2])
           - (reference_b[:, 2] + parameter_b * direction_b[:, 2])).abs()
    return torch.where(degenerate, torch.full_like(gap, float("nan")), gap)


def _nan_aware_max(values: torch.Tensor) -> torch.Tensor:
    """values: (M, K). Returns (M,) max ignoring NaN, NaN where all are NaN."""
    finite = ~torch.isnan(values)
    filled = torch.where(finite, values, torch.full_like(values, -float("inf")))
    maximum = filled.max(dim=1).values
    return torch.where(finite.any(dim=1), maximum,
                       torch.full_like(maximum, float("nan")))


def _cone_sums(
    axis_eta: torch.Tensor, axis_phi: torch.Tensor, axis_dz: torch.Tensor,
    track_eta: torch.Tensor, track_phi: torch.Tensor, track_dz: torch.Tensor,
    track_pt: torch.Tensor, excluded: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """axis_*: (M,). track_*: (P,). excluded: (M, P) bool or None.
    Returns (count (M,), sum_pt (M,), min_dr (M,), inside (M, P) bool)."""
    num_candidates = axis_eta.shape[0]
    pool_size = track_eta.shape[0]
    if pool_size == 0:
        zeros = torch.zeros(num_candidates, device=axis_eta.device)
        return (zeros, zeros.clone(),
                torch.full_like(zeros, COMPANION_MIN_DR_SENTINEL),
                torch.zeros(num_candidates, 0, dtype=torch.bool,
                            device=axis_eta.device))
    delta_r = _delta_r_squared(
        axis_eta.unsqueeze(1), axis_phi.unsqueeze(1),
        track_eta.unsqueeze(0), track_phi.unsqueeze(0)).sqrt()
    inside = (delta_r <= CONE_DELTA_R_MAX) & (
        (axis_dz.unsqueeze(1) - track_dz.unsqueeze(0)).abs() <= CONE_DZ_MAX)
    if excluded is not None:
        inside &= ~excluded
    count = inside.sum(dim=1).float()
    sum_pt = (track_pt.unsqueeze(0) * inside).sum(dim=1)
    masked_delta_r = torch.where(
        inside, delta_r, torch.full_like(delta_r, COMPANION_MIN_DR_SENTINEL))
    min_delta_r = masked_delta_r.min(dim=1).values
    return count, sum_pt, min_delta_r, inside


def _compute_h6_columns(
    i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
    *, lorentz: torch.Tensor, eta: torch.Tensor, phi: torch.Tensor,
    dxy_sig: torch.Tensor, h6_inputs: dict,
) -> torch.Tensor:
    """i, j, k: (M,) long. Returns (M, 22) in H6_NAMES order."""
    missing = [key for key in H6_INPUT_KEYS if key not in h6_inputs]
    if missing:
        raise ValueError(f"h6_inputs is missing {missing}")

    members = torch.stack([i, j, k], dim=1)
    reference = torch.stack([h6_inputs["vertex_x"], h6_inputs["vertex_y"],
                             h6_inputs["vertex_z"]], dim=1)
    direction = _unit_direction(eta, phi)

    pair_distances, pair_gaps, midpoints = [], [], []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        index_a, index_b = members[:, first], members[:, second]
        distance, midpoint = _closest_approach(
            reference[index_a], direction[index_a],
            reference[index_b], direction[index_b])
        pair_distances.append(distance)
        midpoints.append(midpoint)
        pair_gaps.append(_transverse_crossing_z_gap(
            reference[index_a], direction[index_a],
            reference[index_b], direction[index_b]))
    poca = torch.stack(pair_distances, dim=1)
    vertex_estimate = torch.stack(midpoints, dim=0).mean(dim=0)

    dz_raw = h6_inputs["dz_raw"]
    member_dz = dz_raw[members]
    raw_dz_gap_max = (member_dz.unsqueeze(2)
                      - member_dz.unsqueeze(1)).abs().amax(dim=(1, 2))

    momentum = torch.stack([
        lorentz[axis][members].sum(dim=1) for axis in range(3)], dim=1)
    momentum_pt = torch.hypot(momentum[:, 0], momentum[:, 1])

    primary_x = h6_inputs["primary_vertex_x"].reshape(())
    primary_y = h6_inputs["primary_vertex_y"].reshape(())
    member_displacement_x = reference[members][:, :, 0] - primary_x
    member_displacement_y = reference[members][:, :, 1] - primary_y
    member_projection = (member_displacement_x * momentum[:, 0:1]
                         + member_displacement_y * momentum[:, 1:2])
    lifetime_positive_count = (member_projection > 0).sum(dim=1).float()
    dxy_sig_spread = dxy_sig[members].std(dim=1, unbiased=False)

    # LHCb corrected mass: the triplet mass corrected for the momentum
    # transverse to the flight direction, which absorbs the unseen neutrino.
    flight_x = vertex_estimate[:, 0] - primary_x
    flight_y = vertex_estimate[:, 1] - primary_y
    flight_pt = torch.hypot(flight_x, flight_y)
    along_flight = (momentum[:, 0] * flight_x + momentum[:, 1] * flight_y) \
        / torch.clamp_min(flight_pt, 1e-12)
    missing_pt = torch.sqrt(torch.clamp_min(
        momentum_pt ** 2 - along_flight ** 2, 0.0))
    mass = _mass(lorentz, i, j, k)
    corrected_mass = torch.sqrt(mass ** 2 + missing_pt ** 2) + missing_pt

    displacement_k_x = reference[k, 0] - primary_x
    displacement_k_y = reference[k, 1] - primary_y
    signed_ip_k = dxy_sig[k].abs() * torch.sign(
        displacement_k_x * momentum[:, 0] + displacement_k_y * momentum[:, 1])

    member_pt = torch.hypot(lorentz[0][members], lorentz[1][members])
    min_pt_ijk = member_pt.amin(dim=1)
    scalar_sum_pt_ijk = member_pt.sum(dim=1)

    axis_eta = torch.asinh(momentum[:, 2] / torch.clamp_min(momentum_pt, 1e-12))
    axis_phi = torch.atan2(momentum[:, 1], momentum[:, 0])
    axis_dz = member_dz.mean(dim=1)

    pool_pt = torch.hypot(lorentz[0], lorentz[1])
    pool_index = torch.arange(eta.shape[0], device=eta.device)
    is_member = (pool_index.view(1, -1) == members.unsqueeze(2)).any(dim=1)
    kept_count, kept_sum_pt, cone_min_dr, kept_inside = _cone_sums(
        axis_eta, axis_phi, axis_dz, eta, phi, dz_raw, pool_pt, is_member)
    n_low_ip_in_cone = (
        kept_inside & (dxy_sig.abs().unsqueeze(0) <= LOW_IP_SIGNIFICANCE_MAX)
    ).sum(dim=1).float()

    other_count, other_sum_pt, _, _ = _cone_sums(
        axis_eta, axis_phi, axis_dz, h6_inputs["other_eta"],
        h6_inputs["other_phi"], h6_inputs["other_dz"],
        h6_inputs["other_pt"], None)

    # Secondary vertices are matched by proximity to the reconstructed triplet
    # vertex, so no slot ordering is involved; "no vertex" is NaN rather than a
    # sentinel, letting the tree learn a per-split direction for it.
    sv_x, sv_y, sv_z = h6_inputs["sv_x"], h6_inputs["sv_y"], h6_inputs["sv_z"]
    num_candidates = i.shape[0]
    if sv_x.numel() == 0:
        nan = torch.full((num_candidates,), float("nan"), device=i.device)
        sv_columns = [nan, nan.clone(), nan.clone(), nan.clone(),
                      torch.zeros_like(nan), torch.zeros_like(nan)]
    else:
        sv_position = torch.stack([sv_x, sv_y, sv_z], dim=1)
        separation = (vertex_estimate.unsqueeze(1)
                      - sv_position.unsqueeze(0)).square().sum(dim=2).sqrt()
        nearest = separation.argmin(dim=1)
        sv_min_distance = separation.gather(1, nearest.unsqueeze(1)).squeeze(1)
        sv_flight_x = sv_x[nearest] - primary_x
        sv_flight_y = sv_y[nearest] - primary_y
        sv_flight_pt = torch.hypot(sv_flight_x, sv_flight_y)
        sv_columns = [
            sv_min_distance,
            h6_inputs["sv_dlen_sig"][nearest],
            h6_inputs["sv_mass"][nearest],
            (sv_flight_x * momentum[:, 0] + sv_flight_y * momentum[:, 1])
            / torch.clamp_min(sv_flight_pt * momentum_pt, 1e-12),
            (separation <= SV_MATCH_RADIUS_CM).sum(dim=1).float(),
            torch.ones(num_candidates, device=i.device),
        ]

    columns = [
        poca.amax(dim=1), poca.mean(dim=1), _nan_aware_max(
            torch.stack(pair_gaps, dim=1)),
        raw_dz_gap_max, lifetime_positive_count, dxy_sig_spread,
        corrected_mass, signed_ip_k, min_pt_ijk, scalar_sum_pt_ijk,
        n_low_ip_in_cone,
        kept_count, kept_sum_pt, other_count, other_sum_pt, cone_min_dr,
        *sv_columns,
    ]
    return torch.stack(columns, dim=1)


def _track16(idx, *, lorentz, charge, eta, phi, dxy_sig, dz, norm_chi2,
             pt_error, n_pixel, dca_sig, cov_phi_phi, cov_lambda_lambda):
    # Raw per-track block in TRACK16_NAMES order for the tracks selected by idx (M,).
    px, py, pz = lorentz[0, idx], lorentz[1, idx], lorentz[2, idx]
    pt = torch.sqrt(px ** 2 + py ** 2)
    columns = [
        px, py, pz, eta[idx], phi[idx], charge[idx], dxy_sig[idx], dz[idx],
        norm_chi2[idx], pt_error[idx], n_pixel[idx], dca_sig[idx],
        cov_phi_phi[idx], cov_lambda_lambda[idx], pt,
        pt_error[idx] / torch.clamp_min(pt, 1e-6),
    ]
    return torch.stack(columns, dim=1)


def _log_bw(m_squared, m_res, width):
    numerator = m_res ** 2 * width ** 2
    denominator = (m_squared - m_res ** 2) ** 2 + m_res ** 2 * width ** 2
    return math.log(numerator) - torch.log(torch.clamp_min(denominator, 1e-20))


def _couple_unit(i, j, *, lorentz, charge, eta, phi, dz, dxy_sig, dca_sig,
                 cov_phi_phi, cov_lambda_lambda):
    # Couple-unit physics in COUPLE_UNIT_NAMES order. Formulas ported from
    # couple_features.build_couple_feature_vector (Blocks 2, 3, v3), computed from raw inputs.
    px_i, py_i, pz_i, e_i = lorentz[0, i], lorentz[1, i], lorentz[2, i], lorentz[3, i]
    px_j, py_j, pz_j, e_j = lorentz[0, j], lorentz[1, j], lorentz[2, j], lorentz[3, j]

    sum_px, sum_py, sum_pz, sum_e = px_i + px_j, py_i + py_j, pz_i + pz_j, e_i + e_j
    m_squared = sum_e ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
    m_ij = torch.sqrt(torch.clamp_min(m_squared, 1e-10))
    ln_m2 = torch.log(torch.clamp_min(m_squared, 1e-10))

    pt_i = torch.sqrt(px_i ** 2 + py_i ** 2 + 1e-10)
    pt_j = torch.sqrt(px_j ** 2 + py_j ** 2 + 1e-10)
    pt_min = torch.minimum(pt_i, pt_j)
    pt_sum = pt_i + pt_j

    delta_eta = eta[i] - eta[j]
    delta_phi = (phi[i] - phi[j] + math.pi) % (2 * math.pi) - math.pi
    delta_r = torch.sqrt(delta_eta ** 2 + delta_phi ** 2 + 1e-10)

    ln_kt = torch.log(torch.clamp_min(pt_min * delta_r, 1e-10))
    ln_z = torch.log(torch.clamp_min(pt_min / torch.clamp_min(pt_sum, 1e-10), 1e-10))
    ln_dr = torch.log(torch.clamp_min(delta_r, 1e-10))

    charge_prod = charge[i] * charge[j]
    dz_diff = (dz[i] - dz[j]).abs()
    rho_ind = torch.exp(-0.5 * ((m_ij - RHO_MASS_GEV) / RHO_SIGMA_GEV) ** 2)
    rho_os = (charge_prod < 0).float() * rho_ind

    dxy_diff = (dxy_sig[i] - dxy_sig[j]).abs()
    sin_half_dphi = torch.abs(torch.sin(delta_phi / 2.0))
    dxy_phi = dxy_diff / torch.clamp_min(2.0 * sin_half_dphi, 0.05)
    lorentz_dot = e_i * e_j - px_i * px_j - py_i * py_j - pz_i * pz_j

    kalman = (torch.log(torch.clamp_min(cov_phi_phi[i], 1e-10))
              + torch.log(torch.clamp_min(cov_phi_phi[j], 1e-10))
              + torch.log(torch.clamp_min(cov_lambda_lambda[i], 1e-10))
              + torch.log(torch.clamp_min(cov_lambda_lambda[j], 1e-10)))
    dca_sum = dca_sig[i] + dca_sig[j]

    sum_p_mag = torch.sqrt(sum_px ** 2 + sum_py ** 2 + sum_pz ** 2 + 1e-10)
    pi_mag = torch.sqrt(px_i ** 2 + py_i ** 2 + pz_i ** 2 + 1e-10)
    helicity = (px_i * sum_px + py_i * sum_py + pz_i * sum_pz) / (pi_mag * sum_p_mag)

    m_squared_safe = torch.clamp_min(m_ij ** 2, 1e-8)
    logbw_rho = _log_bw(m_squared_safe, RHO_MASS_GEV, RHO_WIDTH_GEV)
    logbw_a1 = _log_bw(m_squared_safe, A1_MASS_GEV, A1_WIDTH_GEV)

    columns = [
        ln_kt, ln_z, ln_dr, ln_m2, charge_prod, dz_diff, rho_ind, rho_os,
        dxy_phi, lorentz_dot, delta_eta, delta_phi, kalman, dca_sum, helicity,
        logbw_rho, logbw_a1,
    ]
    return torch.stack(columns, dim=1)


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


def triplet_feature_columns(
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    couple_rank: torch.Tensor,
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
    cov_phi_phi: torch.Tensor,
    cov_lambda_lambda: torch.Tensor,
    h6_inputs: dict | None = None,
) -> torch.Tensor:
    """i, j, k, couple_rank: (M,) long. lorentz: (4, N). per-track inputs: (N,).
    h6_inputs: optional dict of the H6_INPUT_KEYS arrays.

    Returns (M, 89) in FEATURE_NAMES order, or (M, 111) in
    FEATURE_NAMES_EXTENDED order when h6_inputs is given.
    """
    cr = couple_rank
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
    track_kw = dict(lorentz=lorentz, charge=charge, eta=eta, phi=phi, dxy_sig=dxy_sig,
                    dz=dz, norm_chi2=norm_chi2, pt_error=pt_error, n_pixel=n_pixel,
                    dca_sig=dca_sig, cov_phi_phi=cov_phi_phi, cov_lambda_lambda=cov_lambda_lambda)
    blocks = [
        torch.stack(columns, dim=1),
        _track16(i, **track_kw),
        _track16(j, **track_kw),
        _track16(k, **track_kw),
        _couple_unit(i, j, lorentz=lorentz, charge=charge, eta=eta, phi=phi, dz=dz,
                     dxy_sig=dxy_sig, dca_sig=dca_sig, cov_phi_phi=cov_phi_phi,
                     cov_lambda_lambda=cov_lambda_lambda),
    ]
    if h6_inputs is not None:
        blocks.append(_compute_h6_columns(
            i, j, k, lorentz=lorentz, eta=eta, phi=phi, dxy_sig=dxy_sig,
            h6_inputs=h6_inputs))
    return torch.cat(blocks, dim=1)


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
    cov_phi_phi: torch.Tensor,
    cov_lambda_lambda: torch.Tensor,
    gt_sorted: tuple[int, int, int] | None = None,
    h6_inputs: dict | None = None,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """couples: (C, 2) long. pool: (P,) long. lorentz: (4, N). per-track inputs: (N,).
    h6_inputs: optional dict of the H6_INPUT_KEYS arrays.

    Per Tier-H-surviving candidate: (X (M_H, 89 or 111) features, the matching
    name list, is_gt (M_H,), couple_row (M_H,)). Columns 0:4 equal
    triplet_gate_quantities; the RICH_NAMES block (24) is followed by ti/tj/tk
    16-blocks, the couple-unit block (17) and, when h6_inputs is given, the H6
    block (22).
    """
    track_i, track_j, track_k, couple_row, base = _enumerate(couples, pool)
    h_keep = base.clone()
    h_keep &= (charge[track_i] + charge[track_j] + charge[track_k]).abs().round() == 1
    h_keep &= _mass(lorentz, track_i, track_j, track_k) <= M_TAU_GEV

    i, j, k = track_i[h_keep], track_j[h_keep], track_k[h_keep]
    cr = couple_row[h_keep]
    features = triplet_feature_columns(
        i, j, k, cr, lorentz=lorentz, charge=charge, eta=eta, phi=phi, dz=dz,
        dxy_sig=dxy_sig, dca_sig=dca_sig, n_pixel=n_pixel, norm_chi2=norm_chi2,
        pt_error=pt_error, cov_phi_phi=cov_phi_phi, cov_lambda_lambda=cov_lambda_lambda,
        h6_inputs=h6_inputs,
    )
    names = FEATURE_NAMES if h6_inputs is None else FEATURE_NAMES_EXTENDED

    if gt_sorted is not None:
        sorted_rows = torch.stack([i, j, k], dim=1).sort(dim=1).values
        target = torch.tensor(sorted(gt_sorted), device=i.device)
        is_gt = (sorted_rows == target).all(dim=1)
    else:
        is_gt = torch.zeros(i.shape[0], dtype=torch.bool, device=i.device)
    return features, names, is_gt, cr


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
