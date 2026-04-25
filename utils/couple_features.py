"""Per-couple feature extraction shared between the trainer and the diagnostic.

The post-ParT couple reranker (`reports/triplet_reranking/triplet_research_plan_20260408.md`,
direction A) operates on **canonically-ordered, mass-filtered couples** drawn
from the ParT top-50. This module is the single source of truth for:

- enumerating all C(50, 2) = 1225 candidate couples in canonical order
- applying Filter A (the loose `m(ij) <= m_tau` cut, NO charge or mass window)
- building the 51-dimensional per-couple feature vector documented in the plan
- computing the binary GT-couple label (both members are GT pions)

Both `part/diagnostics/couple_count_diagnostic.py` and `part/train_couple_reranker.py`
import from here so the canonical ordering, filter logic, and feature ordering
stay consistent across analysis and training. Never duplicate any of these in
the calling code — the entire point of this module is to keep one source of
truth for the things that must agree across phases of the project.

All operations are pure PyTorch (no numpy intermediates) and operate on
single-event tensors (no batch dimension). The trainer wraps the per-event
output with a padding loop because the surviving-couple count varies per event
under Filter A.
"""
from __future__ import annotations

import math

import torch

# PDG 2024
M_TAU_GEV = 1.77693

# 4 LV pair features + 6 physics pair features
PAIRWISE_PHYSICS_DIM = 10
# m(ij), pT(ij), Δη, Δφ, ΔR
DERIVED_GEOM_DIM = 5
# stage1(i), stage2(i), stage1(j), stage2(j)
CASCADE_SCORE_DIM = 4
# 16 (track_i) + 16 (track_j) + 10 + 5 + 4
COUPLE_FEATURE_DIM = 16 * 2 + PAIRWISE_PHYSICS_DIM + DERIVED_GEOM_DIM + CASCADE_SCORE_DIM
assert COUPLE_FEATURE_DIM == 51

# T2.2: pair-kinematics v2 adds 4 extra per-couple features that are
# standard in CMS/ATLAS tau / jet-tagging rerankers:
#   1. cos(θ_3D)   — opening angle from the 3D momenta
#   2. (m(ij) − m_τ) / σ_m — normalized mass residual, σ_m set to
#      ~0.1 GeV (order-of-magnitude tau mass resolution in CMS low-pT).
#   3. dxy_sig_i · dxy_sig_j — product of 2D impact-parameter
#      significances (concordance of displacement direction; distinct
#      signal from `|Δdxy|`).
#   4. dz_sig_i · dz_sig_j — product of 3D longitudinal impact-parameter
#      significances.
PAIR_KINEMATICS_V2_EXTRA_DIM = 4
COUPLE_FEATURE_DIM_V2 = COUPLE_FEATURE_DIM + PAIR_KINEMATICS_V2_EXTRA_DIM
assert COUPLE_FEATURE_DIM_V2 == 55

# Scale for the mass residual; matches the ρ(770) σ by default so the
# residual stays O(1) and doesn't require per-run retuning.
TAU_MASS_SIGMA_GEV = 0.1

# ρ(770) Gaussian indicator parameters (matches CascadeReranker._compute_extra_pairwise_features)
RHO_MASS_GEV = 0.770
RHO_SIGMA_GEV = 0.075

# Batch-3 H8: pair-physics v3 adds 4 new features that require nonlinear
# closed-form combinations the downstream Conv1d cannot synthesise from
# the existing channels (unlike v2 which could be synthesised):
#   1. Kalman-χ² proxy — log-combined covariance volume from
#      log_cov_phi_phi and log_cov_lambda_lambda of both tracks; measures
#      how precisely the pair's vertex can be constrained.
#   2. DCA-significance sum — sum of 3D closest-approach significances of
#      the two tracks; proxy for overall displacement from the beamline.
#   3. Lab-frame helicity proxy — cos(angle between track-i 3-momentum
#      and the couple sum-momentum); correlated with the cos θ* the
#      CLEO II a1-Dalitz analysis uses as a primary discriminant.
#   4. log BW density for ρ(770) — closed-form log Breit-Wigner for the
#      dominant sub-resonance in a1 → ρπ. The existing Gaussian indicator
#      is bounded; the BW's log form extends the dynamic range at the
#      tails.
#   5. log BW density for a1(1260) — same for the parent a1 resonance.
# (5 features; keeps the "v3" nomenclature consistent with v2's +4.)
PAIR_PHYSICS_V3_EXTRA_DIM = 5
COUPLE_FEATURE_DIM_V3 = COUPLE_FEATURE_DIM + PAIR_PHYSICS_V3_EXTRA_DIM
assert COUPLE_FEATURE_DIM_V3 == 56

# Batch-4 H9: significance-normalised pair features. Complement the
# raw-geometry channels with their covariance-weighted significances:
#   1. mass_residual / σ_m (unitless mass pull from tau window)
#   2. Δφ / σ_Δφ_combined — angular separation in units of its combined
#      Kalman σ (σ²_combined = σ²_φ_i + σ²_φ_j from the log_cov channels).
#   3. pT-balance significance: (pT_i − pT_j) / sqrt(σ²_pT_i + σ²_pT_j)
#      using log_pt_error as the per-track std.
PAIR_PHYSICS_SIGNIF_EXTRA_DIM = 3

A1_MASS_GEV = 1.230
A1_WIDTH_GEV = 0.420
RHO_WIDTH_GEV = 0.149

# Per-track channel indices in the 16-dim standardized feature vector
# (mirrors the YAML `pf_features` ordering in
# `data/low-pt/lowpt_tau_trackfinder.yaml`).
_IDX_LOG_COV_PHI_PHI = 12
_IDX_LOG_COV_LAMBDA_LAMBDA = 13
_IDX_DCA_SIG = 11


# ---------------------------------------------------------------------------
# Charge / standardization helpers
# ---------------------------------------------------------------------------

def recover_raw_charges(charge_channel_standardized: torch.Tensor) -> torch.Tensor:
    """Recover raw track charges in {-1, +1} from the standardized channel.

    The data pipeline standardizes ``track_charge`` with ``center=1.0,
    scale=0.5`` (raw +1 → standardized 0.0, raw -1 → standardized -1.0).
    Inverse: ``raw = standardized / 0.5 + 1.0``.

    Matches the inverse used by ``CascadeReranker._compute_extra_pairwise_features``
    at ``weaver/weaver/nn/model/CascadeReranker.py:294``.
    """
    return charge_channel_standardized / 0.5 + 1.0


# ---------------------------------------------------------------------------
# Couple enumeration + filter
# ---------------------------------------------------------------------------

def enumerate_couples_canonical(
    num_tracks: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all upper-triangular ``(i, j)`` index pairs with ``i < j``.

    "Canonical ordering" for the couple reranker is "lower index first" — the
    lower-rank (in the top-50 list) track is always position ``i``. Since the
    top-50 list is sorted by Stage 2 score in descending order, position 0 is
    the highest-scoring track in the event.

    Returns:
        Two ``(num_tracks * (num_tracks - 1) / 2,)`` tensors ``(upper_i, upper_j)``
        of int64 indices into the top-50 list, with ``upper_i < upper_j`` for
        every pair.
    """
    return torch.triu_indices(num_tracks, num_tracks, offset=1, device=device).unbind(0)


def compute_invariant_mass(
    lorentz_vectors: torch.Tensor,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
) -> torch.Tensor:
    """Compute m(i, j) = sqrt(max(E² − |p|², 0)) for each pair.

    Args:
        lorentz_vectors: ``(4, N)`` tensor of ``(px, py, pz, E)`` per track.
        upper_i, upper_j: index tensors into the track dimension, both shape
            ``(n_pairs,)``.

    Returns:
        ``(n_pairs,)`` tensor of invariant masses (GeV).
    """
    sum_e = lorentz_vectors[3, upper_i] + lorentz_vectors[3, upper_j]
    sum_px = lorentz_vectors[0, upper_i] + lorentz_vectors[0, upper_j]
    sum_py = lorentz_vectors[1, upper_i] + lorentz_vectors[1, upper_j]
    sum_pz = lorentz_vectors[2, upper_i] + lorentz_vectors[2, upper_j]
    m_squared = sum_e ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
    return torch.sqrt(torch.clamp_min(m_squared, 0.0))


def filter_a_mask(
    lorentz_vectors: torch.Tensor,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
    m_tau: float = M_TAU_GEV,
) -> torch.Tensor:
    """Filter A: only the kinematic ``m(i, j) <= m_tau`` constraint.

    No charge cut. No ρ-mass window. The omitted cuts are exactly the biases
    that the previous physics-filter triplet attempts hardcoded
    (``reports/triplet_reranking/triplet_combinatorics.md``).

    Returns:
        ``(n_pairs,)`` boolean tensor — True for couples that pass the filter.
    """
    invariant_mass = compute_invariant_mass(lorentz_vectors, upper_i, upper_j)
    return invariant_mass <= m_tau


# ---------------------------------------------------------------------------
# Per-couple feature computation
# ---------------------------------------------------------------------------

def compute_couple_labels(
    track_labels: torch.Tensor,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
) -> torch.Tensor:
    """A couple is a GT couple iff both of its tracks are GT pions.

    Args:
        track_labels: ``(N,)`` tensor of per-track binary labels (1 = GT).
        upper_i, upper_j: ``(n_pairs,)`` index tensors.

    Returns:
        ``(n_pairs,)`` tensor of couple labels (1 = GT couple, 0 otherwise).
    """
    return (track_labels[upper_i] > 0.5) & (track_labels[upper_j] > 0.5)


def build_couple_feature_vector(
    top50_features: torch.Tensor,
    top50_points: torch.Tensor,
    top50_lorentz: torch.Tensor,
    top50_stage1_scores: torch.Tensor,
    top50_stage2_scores: torch.Tensor,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
) -> torch.Tensor:
    """Build the 51-dim per-couple feature tensor.

    Block ordering (matches `reports/triplet_reranking/triplet_research_plan_20260408.md`
    and the architecture plan):
        - Block 1 (32 dims): per-track features concat — features_i (16) ‖ features_j (16)
        - Block 2 (10 dims): pairwise physics — 4 LV channels + 6 physics channels
        - Block 3 (5 dims): derived geometric — m(ij), pT(ij), Δη, Δφ, ΔR
        - Block 4 (4 dims): cascade context — s1(i), s2(i), s1(j), s2(j)

    Args:
        top50_features: ``(16, 50)`` standardized per-track features.
        top50_points: ``(2, 50)`` raw ``(eta, phi)`` coordinates.
        top50_lorentz: ``(4, 50)`` raw 4-vectors ``(px, py, pz, E)``.
        top50_stage1_scores: ``(50,)`` raw Stage 1 scores.
        top50_stage2_scores: ``(50,)`` raw Stage 2 (ParT) scores.
        upper_i, upper_j: ``(n_pairs,)`` canonical-order index tensors.

    Returns:
        ``(51, n_pairs)`` couple feature tensor.
    """
    # ---- Block 1: per-track concat (32 dims) ----
    features_i = top50_features[:, upper_i]  # (16, n_pairs)
    features_j = top50_features[:, upper_j]  # (16, n_pairs)

    # ---- Block 2: pairwise physics (10 dims) ----
    # Lorentz components per couple member
    px_i, py_i, pz_i, e_i = top50_lorentz[0:4, upper_i]
    px_j, py_j, pz_j, e_j = top50_lorentz[0:4, upper_j]

    # Sum 4-vector → invariant mass
    sum_e = e_i + e_j
    sum_px = px_i + px_j
    sum_py = py_i + py_j
    sum_pz = pz_i + pz_j
    m_squared = sum_e ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
    m_ij = torch.sqrt(torch.clamp_min(m_squared, 1e-10))
    ln_m_squared = torch.log(torch.clamp_min(m_squared, 1e-10))

    # Per-track pT and ΔR (using RAW eta, phi from points)
    pt_i = torch.sqrt(px_i ** 2 + py_i ** 2 + 1e-10)
    pt_j = torch.sqrt(px_j ** 2 + py_j ** 2 + 1e-10)
    pt_min = torch.minimum(pt_i, pt_j)
    pt_sum = pt_i + pt_j

    eta_i_raw = top50_points[0, upper_i]
    eta_j_raw = top50_points[0, upper_j]
    phi_i_raw = top50_points[1, upper_i]
    phi_j_raw = top50_points[1, upper_j]
    delta_eta = eta_i_raw - eta_j_raw
    # Wrap Δφ to (-π, π]
    delta_phi = phi_i_raw - phi_j_raw
    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
    delta_r = torch.sqrt(delta_eta ** 2 + delta_phi ** 2 + 1e-10)

    # Lorentz pair features (4 channels: ln k_T, ln z, ln ΔR, ln m²)
    ln_kt = torch.log(torch.clamp_min(pt_min * delta_r, 1e-10))
    ln_z = torch.log(torch.clamp_min(pt_min / torch.clamp_min(pt_sum, 1e-10), 1e-10))
    ln_dr = torch.log(torch.clamp_min(delta_r, 1e-10))

    # Charge product (recovered from standardized channel 5)
    charge_i = recover_raw_charges(top50_features[5, upper_i])
    charge_j = recover_raw_charges(top50_features[5, upper_j])
    charge_product = charge_i * charge_j

    # Δdz_sig (channel 7 = log_dz_significance, kept standardized — same convention
    # as CascadeReranker._compute_extra_pairwise_features)
    dz_sig_i = top50_features[7, upper_i]
    dz_sig_j = top50_features[7, upper_j]
    dz_diff_abs = (dz_sig_i - dz_sig_j).abs()

    # ρ(770) Gaussian indicator: exp(-(m - 0.770)² / (2 × 0.075²))
    rho_indicator = torch.exp(
        -0.5 * ((m_ij - RHO_MASS_GEV) / RHO_SIGMA_GEV) ** 2,
    )

    # ρ-OS conjunction
    is_opposite_sign = (charge_product < 0).to(rho_indicator.dtype)
    rho_os_indicator = is_opposite_sign * rho_indicator

    # φ-corrected dxy compatibility (channel 6 = dxy_significance)
    dxy_sig_i = top50_features[6, upper_i]
    dxy_sig_j = top50_features[6, upper_j]
    dxy_diff_abs = (dxy_sig_i - dxy_sig_j).abs()
    sin_half_dphi = torch.abs(torch.sin(delta_phi / 2.0))
    dxy_phi_corrected = dxy_diff_abs / torch.clamp_min(2.0 * sin_half_dphi, 0.05)

    # Minkowski dot product (PELICAN-inspired)
    lorentz_dot = e_i * e_j - px_i * px_j - py_i * py_j - pz_i * pz_j

    # Stack the 10 pairwise physics channels
    pairwise_physics = torch.stack(
        [
            ln_kt,                  # 1: ln k_T
            ln_z,                   # 2: ln z
            ln_dr,                  # 3: ln ΔR
            ln_m_squared,           # 4: ln m²
            charge_product,         # 5: q_i × q_j
            dz_diff_abs,            # 6: |Δdz_sig|
            rho_indicator,          # 7: ρ(770) indicator
            rho_os_indicator,       # 8: OS × ρ
            dxy_phi_corrected,      # 9: φ-corrected dxy
            lorentz_dot,            # 10: Lorentz dot
        ],
        dim=0,
    )  # (10, n_pairs)

    # ---- Block 3: derived geometric (5 dims) ----
    pt_ij = torch.sqrt(sum_px ** 2 + sum_py ** 2 + 1e-10)
    derived_geometric = torch.stack(
        [
            m_ij,        # raw mass (not log)
            pt_ij,       # couple transverse momentum
            delta_eta,   # raw Δη
            delta_phi,   # wrapped Δφ
            delta_r,     # raw ΔR (not log)
        ],
        dim=0,
    )  # (5, n_pairs)

    # ---- Block 4: cascade context (4 dims) ----
    cascade_scores = torch.stack(
        [
            top50_stage1_scores[upper_i],
            top50_stage2_scores[upper_i],
            top50_stage1_scores[upper_j],
            top50_stage2_scores[upper_j],
        ],
        dim=0,
    )  # (4, n_pairs)

    # ---- Concatenate all 51 dims ----
    couple_features = torch.cat(
        [
            features_i,           # (16, n_pairs)
            features_j,           # (16, n_pairs)
            pairwise_physics,     # (10, n_pairs)
            derived_geometric,    # (5, n_pairs)
            cascade_scores,       # (4, n_pairs)
        ],
        dim=0,
    )  # (51, n_pairs)
    assert couple_features.shape[0] == COUPLE_FEATURE_DIM, (
        f'Expected {COUPLE_FEATURE_DIM} feature dim, got {couple_features.shape[0]}'
    )
    return couple_features


# ---------------------------------------------------------------------------
# Batched per-event feature builder (for the trainer)
# ---------------------------------------------------------------------------


def build_couple_features_batched(
    top_k2_features: torch.Tensor,
    top_k2_points: torch.Tensor,
    top_k2_lorentz: torch.Tensor,
    top_k2_stage1_scores: torch.Tensor,
    top_k2_stage2_scores: torch.Tensor,
    top_k2_track_labels: torch.Tensor | None = None,
    track_valid_mask: torch.Tensor | None = None,
    m_tau: float = M_TAU_GEV,
    pair_kinematics_v2: bool = False,
    pair_physics_v3: bool = False,
    pair_physics_signif: bool = False,
) -> dict[str, torch.Tensor]:
    """Vectorized batched version of the per-event feature builder.

    Enumerates ALL ``C(K2, 2)`` couples per event (no Filter A pre-filter)
    and produces fixed-shape tensors for the entire batch in one fully-
    vectorized pass — no Python per-event loop. The Filter A condition
    (``m(ij) <= m_tau``) is encoded as a separate boolean mask so the
    trainer can use it for loss masking.

    This is the primary entry point for the trainer. The per-event helper
    ``enumerate_and_featurize_filter_a`` is kept for the diagnostic and for
    unit tests where variable-length output is convenient.

    Args:
        top_k2_features: ``(B, 16, K2)`` standardized per-track features.
        top_k2_points: ``(B, 2, K2)`` raw ``(eta, phi)``.
        top_k2_lorentz: ``(B, 4, K2)`` raw 4-vectors.
        top_k2_stage1_scores: ``(B, K2)`` Stage 1 scores.
        top_k2_stage2_scores: ``(B, K2)`` Stage 2 scores.
        top_k2_track_labels: optional ``(B, K2)`` per-track binary labels.
        track_valid_mask: optional ``(B, K2)`` boolean — True for real
            tracks, False for padding. When provided, padding tracks are
            zeroed out before couple feature computation (preventing
            ``-inf`` cascade scores from producing ``Inf`` in the feature
            vector), and couples involving any padding track are excluded
            from ``filter_a_mask``.
        m_tau: kinematic mass cut (default = PDG τ mass).

    Returns:
        dict with:
            ``couple_features``: ``(B, 51, n_couples)`` per-couple feature
                tensor where ``n_couples = K2 * (K2 - 1) / 2`` (always).
            ``filter_a_mask``: ``(B, n_couples)`` boolean — True for couples
                passing the loose mass cut AND having both tracks valid.
            ``couple_labels``: ``(B, n_couples)`` boolean — True iff both
                tracks of the couple are GT pions. Only present when
                ``top_k2_track_labels`` is provided.
    """
    batch_size, _, k2 = top_k2_features.shape
    device = top_k2_features.device

    # Zero out padding tracks so that -inf cascade scores (and garbage
    # features/lorentz vectors from gathered padding positions) never
    # enter the couple feature computation.
    # Uses torch.where instead of multiplication because -inf * 0 = NaN
    # in IEEE 754 arithmetic.
    if track_valid_mask is not None:
        # valid_mask_3d: (B, 1, K2) for broadcasting against (B, C, K2)
        valid_mask_3d = track_valid_mask.unsqueeze(1)
        zero_2d = torch.zeros(1, device=device, dtype=top_k2_features.dtype)
        top_k2_features = torch.where(valid_mask_3d, top_k2_features, zero_2d)
        top_k2_points = torch.where(valid_mask_3d, top_k2_points, zero_2d)
        top_k2_lorentz = torch.where(valid_mask_3d, top_k2_lorentz, zero_2d)
        # Scores: (B, K2)
        zero_1d = torch.zeros(1, device=device, dtype=top_k2_stage1_scores.dtype)
        top_k2_stage1_scores = torch.where(track_valid_mask, top_k2_stage1_scores, zero_1d)
        top_k2_stage2_scores = torch.where(track_valid_mask, top_k2_stage2_scores, zero_1d)

    # Canonical (i, j) indices, shared by every event in the batch.
    upper_i, upper_j = torch.triu_indices(k2, k2, offset=1, device=device).unbind(0)
    # upper_i, upper_j: (n_couples,) where n_couples = k2*(k2-1)/2

    # ---- Block 1: per-track concat (32 dims) ----
    # Advanced indexing: top_k2_features[:, :, upper_i] → (B, 16, n_couples)
    feat_i = top_k2_features[:, :, upper_i]
    feat_j = top_k2_features[:, :, upper_j]

    # ---- Block 2: pairwise physics (10 dims) ----
    # Per-couple Lorentz components
    px_i = top_k2_lorentz[:, 0, upper_i]
    py_i = top_k2_lorentz[:, 1, upper_i]
    pz_i = top_k2_lorentz[:, 2, upper_i]
    e_i = top_k2_lorentz[:, 3, upper_i]
    px_j = top_k2_lorentz[:, 0, upper_j]
    py_j = top_k2_lorentz[:, 1, upper_j]
    pz_j = top_k2_lorentz[:, 2, upper_j]
    e_j = top_k2_lorentz[:, 3, upper_j]

    # Sum 4-vector and invariant mass
    sum_e = e_i + e_j
    sum_px = px_i + px_j
    sum_py = py_i + py_j
    sum_pz = pz_i + pz_j
    m_squared = sum_e ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
    m_ij = torch.sqrt(torch.clamp_min(m_squared, 1e-10))
    ln_m_squared = torch.log(torch.clamp_min(m_squared, 1e-10))

    # Per-track pT
    pt_i = torch.sqrt(px_i ** 2 + py_i ** 2 + 1e-10)
    pt_j = torch.sqrt(px_j ** 2 + py_j ** 2 + 1e-10)
    pt_min = torch.minimum(pt_i, pt_j)
    pt_sum = pt_i + pt_j

    # Δη, Δφ from raw points
    eta_i = top_k2_points[:, 0, upper_i]
    eta_j = top_k2_points[:, 0, upper_j]
    phi_i = top_k2_points[:, 1, upper_i]
    phi_j = top_k2_points[:, 1, upper_j]
    delta_eta = eta_i - eta_j
    delta_phi = phi_i - phi_j
    delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
    delta_r = torch.sqrt(delta_eta ** 2 + delta_phi ** 2 + 1e-10)

    # Lorentz pair features
    ln_kt = torch.log(torch.clamp_min(pt_min * delta_r, 1e-10))
    ln_z = torch.log(torch.clamp_min(pt_min / torch.clamp_min(pt_sum, 1e-10), 1e-10))
    ln_dr = torch.log(torch.clamp_min(delta_r, 1e-10))

    # Charge product
    charge_i = recover_raw_charges(top_k2_features[:, 5, upper_i])
    charge_j = recover_raw_charges(top_k2_features[:, 5, upper_j])
    charge_product = charge_i * charge_j

    # Δdz_sig (channel 7 = log_dz_significance, kept standardized — same
    # convention as CascadeReranker._compute_extra_pairwise_features)
    dz_sig_i = top_k2_features[:, 7, upper_i]
    dz_sig_j = top_k2_features[:, 7, upper_j]
    dz_diff_abs = (dz_sig_i - dz_sig_j).abs()

    # ρ(770) Gaussian indicator
    rho_indicator = torch.exp(
        -0.5 * ((m_ij - RHO_MASS_GEV) / RHO_SIGMA_GEV) ** 2,
    )
    is_opposite_sign = (charge_product < 0).to(rho_indicator.dtype)
    rho_os_indicator = is_opposite_sign * rho_indicator

    # φ-corrected dxy (channel 6)
    dxy_sig_i = top_k2_features[:, 6, upper_i]
    dxy_sig_j = top_k2_features[:, 6, upper_j]
    dxy_diff_abs = (dxy_sig_i - dxy_sig_j).abs()
    sin_half_dphi = torch.abs(torch.sin(delta_phi / 2.0))
    dxy_phi_corrected = dxy_diff_abs / torch.clamp_min(2.0 * sin_half_dphi, 0.05)

    # Lorentz dot product
    lorentz_dot = e_i * e_j - px_i * px_j - py_i * py_j - pz_i * pz_j

    # Stack pairwise physics → (B, 10, n_couples)
    pairwise_physics = torch.stack(
        [
            ln_kt,
            ln_z,
            ln_dr,
            ln_m_squared,
            charge_product,
            dz_diff_abs,
            rho_indicator,
            rho_os_indicator,
            dxy_phi_corrected,
            lorentz_dot,
        ],
        dim=1,
    )

    # ---- Block 3: derived geometric (5 dims) ----
    pt_ij = torch.sqrt(sum_px ** 2 + sum_py ** 2 + 1e-10)
    derived_geometric = torch.stack(
        [m_ij, pt_ij, delta_eta, delta_phi, delta_r],
        dim=1,
    )

    # ---- Block 4: cascade context (4 dims) ----
    s1_i = top_k2_stage1_scores[:, upper_i]
    s2_i = top_k2_stage2_scores[:, upper_i]
    s1_j = top_k2_stage1_scores[:, upper_j]
    s2_j = top_k2_stage2_scores[:, upper_j]
    cascade_scores = torch.stack([s1_i, s2_i, s1_j, s2_j], dim=1)

    # ---- Concat all 51 dims ----
    couple_features_list = [
        feat_i,
        feat_j,
        pairwise_physics,
        derived_geometric,
        cascade_scores,
    ]

    expected_dim = COUPLE_FEATURE_DIM
    if pair_kinematics_v2:
        # ---- Block 5 (T2.2 v2): 4 extra pair-kinematic features ----
        p_mag_i = torch.sqrt(
            px_i ** 2 + py_i ** 2 + pz_i ** 2 + 1e-10,
        )
        p_mag_j = torch.sqrt(
            px_j ** 2 + py_j ** 2 + pz_j ** 2 + 1e-10,
        )
        # cos(θ_3D): 3D momentum dot / (|p_i| |p_j|)
        cos_opening_angle = (
            (px_i * px_j + py_i * py_j + pz_i * pz_j)
            / (p_mag_i * p_mag_j)
        )
        # Normalized mass residual; small positive when the couple is
        # near the τ mass, negative when below.
        mass_residual_norm = (m_ij - m_tau) / TAU_MASS_SIGMA_GEV
        # Impact-parameter concordance products (channels 6 = dxy_sig,
        # 7 = log_dz_sig in standardized inputs).
        dxy_sig_product = dxy_sig_i * dxy_sig_j
        dz_sig_product = dz_sig_i * dz_sig_j
        pair_v2_extras = torch.stack(
            [
                cos_opening_angle,
                mass_residual_norm,
                dxy_sig_product,
                dz_sig_product,
            ],
            dim=1,
        )
        couple_features_list.append(pair_v2_extras)
        expected_dim = COUPLE_FEATURE_DIM_V2

    if pair_physics_v3:
        # ---- Block 5 (Batch-3 H8): 5 new physics features ----
        # (i) Kalman-χ² proxy — combined log-covariance; standardized
        #     channels are already log-space, so their sum is the log of
        #     the product of the per-track covariance eigenvalues along
        #     (φ, λ=dip). Lower value → tighter vertex constraint.
        log_cov_phi_i = top_k2_features[:, _IDX_LOG_COV_PHI_PHI, upper_i]
        log_cov_phi_j = top_k2_features[:, _IDX_LOG_COV_PHI_PHI, upper_j]
        log_cov_lam_i = top_k2_features[:, _IDX_LOG_COV_LAMBDA_LAMBDA, upper_i]
        log_cov_lam_j = top_k2_features[:, _IDX_LOG_COV_LAMBDA_LAMBDA, upper_j]
        kalman_proxy = (
            log_cov_phi_i + log_cov_phi_j + log_cov_lam_i + log_cov_lam_j
        )
        # (ii) Sum of 3D DCA significances.
        dca_sig_i = top_k2_features[:, _IDX_DCA_SIG, upper_i]
        dca_sig_j = top_k2_features[:, _IDX_DCA_SIG, upper_j]
        dca_sig_sum = dca_sig_i + dca_sig_j
        # (iii) Lab-frame helicity proxy:
        #       cos(angle between p_i and p_i + p_j)
        sum_px_3 = px_i + px_j
        sum_py_3 = py_i + py_j
        sum_pz_3 = pz_i + pz_j
        sum_p_mag = torch.sqrt(
            sum_px_3 ** 2 + sum_py_3 ** 2 + sum_pz_3 ** 2 + 1e-10,
        )
        pi_mag = torch.sqrt(
            px_i ** 2 + py_i ** 2 + pz_i ** 2 + 1e-10,
        )
        helicity_lab = (
            (px_i * sum_px_3 + py_i * sum_py_3 + pz_i * sum_pz_3)
            / (pi_mag * sum_p_mag)
        )
        # (iv, v) log Breit-Wigner densities for ρ(770) and a1(1260).
        # BW density: ρ(m²) = m_res² Γ² / ((m² - m_res²)² + m_res² Γ²)
        # Take log for numerical stability + dynamic range.
        m_squared_safe = torch.clamp_min(
            m_ij ** 2, 1e-8,
        )
        def _log_bw(m_res: float, width: float) -> torch.Tensor:
            num = m_res ** 2 * width ** 2
            den = (m_squared_safe - m_res ** 2) ** 2 + m_res ** 2 * width ** 2
            return math.log(num) - torch.log(torch.clamp_min(den, 1e-20))
        log_bw_rho = _log_bw(RHO_MASS_GEV, RHO_WIDTH_GEV)
        log_bw_a1 = _log_bw(A1_MASS_GEV, A1_WIDTH_GEV)

        pair_v3_extras = torch.stack(
            [
                kalman_proxy,
                dca_sig_sum,
                helicity_lab,
                log_bw_rho,
                log_bw_a1,
            ],
            dim=1,
        )
        couple_features_list.append(pair_v3_extras)
        expected_dim += PAIR_PHYSICS_V3_EXTRA_DIM

    if pair_physics_signif:
        # ---- Block 6 (H9): 3 significance-normalised pair features ----
        # (i) mass pull: |m_ij − m_τ| / σ_m (σ_m = TAU_MASS_SIGMA_GEV).
        mass_pull = (m_ij - m_tau).abs() / TAU_MASS_SIGMA_GEV
        # (ii) Δφ significance. σ²_Δφ ≈ exp(log_cov_phi_i) + exp(log_cov_phi_j);
        # work in standardized space, which keeps the scale O(1).
        log_cov_phi_i_signif = top_k2_features[
            :, _IDX_LOG_COV_PHI_PHI, upper_i,
        ]
        log_cov_phi_j_signif = top_k2_features[
            :, _IDX_LOG_COV_PHI_PHI, upper_j,
        ]
        sigma_phi = torch.sqrt(
            torch.exp(log_cov_phi_i_signif)
            + torch.exp(log_cov_phi_j_signif)
            + 1e-10,
        )
        delta_phi_sig = delta_phi / sigma_phi
        # (iii) pT-balance significance: (pT_i − pT_j) / sqrt(σ²_pT_i + σ²_pT_j)
        # using log_pt_error as per-track std in standardized space.
        log_pt_err_i = top_k2_features[:, 9, upper_i]
        log_pt_err_j = top_k2_features[:, 9, upper_j]
        sigma_pt = torch.sqrt(
            torch.exp(2 * log_pt_err_i)
            + torch.exp(2 * log_pt_err_j)
            + 1e-10,
        )
        pt_balance_sig = (pt_i - pt_j) / sigma_pt

        pair_signif_extras = torch.stack(
            [mass_pull, delta_phi_sig, pt_balance_sig], dim=1,
        )
        couple_features_list.append(pair_signif_extras)
        expected_dim += PAIR_PHYSICS_SIGNIF_EXTRA_DIM

    couple_features = torch.cat(couple_features_list, dim=1)
    assert couple_features.shape[1] == expected_dim

    # Filter A: m(ij) <= m_tau (boolean mask, kept separate from features)
    # Couples involving padding tracks are excluded when track_valid_mask
    # is provided — both tracks must be real AND mass must pass the cut.
    filter_a_mask = m_ij <= m_tau
    if track_valid_mask is not None:
        both_tracks_valid = track_valid_mask[:, upper_i] & track_valid_mask[:, upper_j]
        filter_a_mask = filter_a_mask & both_tracks_valid

    result: dict[str, torch.Tensor] = {
        'couple_features': couple_features,
        'filter_a_mask': filter_a_mask,
    }
    if top_k2_track_labels is not None:
        labels_i = top_k2_track_labels[:, upper_i] > 0.5
        labels_j = top_k2_track_labels[:, upper_j] > 0.5
        result['couple_labels'] = labels_i & labels_j
    return result


# ---------------------------------------------------------------------------
# Convenience: one-call enumerate + filter + featurize for a single event
# ---------------------------------------------------------------------------

def enumerate_and_featurize_filter_a(
    top50_features: torch.Tensor,
    top50_points: torch.Tensor,
    top50_lorentz: torch.Tensor,
    top50_stage1_scores: torch.Tensor,
    top50_stage2_scores: torch.Tensor,
    top50_track_labels: torch.Tensor | None = None,
    m_tau: float = M_TAU_GEV,
) -> dict[str, torch.Tensor]:
    """One-call helper: enumerate canonical couples, apply Filter A, build the
    feature vector and (optionally) the GT-couple labels.

    This is the primary entry point for both the training step and the
    diagnostic. The trainer loops over events in a batch and pads the
    per-event outputs to batch-max C; the diagnostic uses the same per-event
    output for its statistics.

    Args:
        top50_features: ``(16, 50)`` standardized per-track features.
        top50_points: ``(2, 50)`` raw ``(eta, phi)`` coordinates.
        top50_lorentz: ``(4, 50)`` raw 4-vectors.
        top50_stage1_scores: ``(50,)`` Stage 1 scores.
        top50_stage2_scores: ``(50,)`` Stage 2 (ParT) scores.
        top50_track_labels: optional ``(50,)`` per-track GT labels (1 / 0).
            If provided, the returned dict includes ``couple_labels``.
        m_tau: kinematic mass cut (default = PDG τ mass).

    Returns:
        dict with:
            ``upper_i``, ``upper_j``: ``(n_couples,)`` canonical indices into
                the top-50 list, restricted to Filter A survivors.
            ``couple_features``: ``(51, n_couples)`` feature tensor.
            ``couple_labels``: ``(n_couples,)`` boolean GT labels (only present
                if ``top50_track_labels`` was provided).
    """
    num_tracks = top50_features.shape[1]
    device = top50_features.device
    upper_i_full, upper_j_full = enumerate_couples_canonical(num_tracks, device)
    keep_mask = filter_a_mask(top50_lorentz, upper_i_full, upper_j_full, m_tau)
    upper_i = upper_i_full[keep_mask]
    upper_j = upper_j_full[keep_mask]

    couple_features = build_couple_feature_vector(
        top50_features=top50_features,
        top50_points=top50_points,
        top50_lorentz=top50_lorentz,
        top50_stage1_scores=top50_stage1_scores,
        top50_stage2_scores=top50_stage2_scores,
        upper_i=upper_i,
        upper_j=upper_j,
    )

    result: dict[str, torch.Tensor] = {
        'upper_i': upper_i,
        'upper_j': upper_j,
        'couple_features': couple_features,
    }
    if top50_track_labels is not None:
        result['couple_labels'] = compute_couple_labels(
            top50_track_labels, upper_i, upper_j,
        )
    return result
