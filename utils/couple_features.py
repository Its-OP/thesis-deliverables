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

# All 32 standardized pf_features channels of each member enter Block 1
# (widened from the legacy 16 together with the H6 feature extension).
TRACK_EMBED_DIM = 32
# 4 LV pair features + 6 physics pair features
PAIRWISE_PHYSICS_DIM = 10
# m(ij), pT(ij), Δη, Δφ, ΔR
DERIVED_GEOM_DIM = 5
# stage1(i), stage2(i), stage1(j), stage2(j)
CASCADE_SCORE_DIM = 4
# 32 (track_i) + 32 (track_j) + 10 + 5 + 4
COUPLE_FEATURE_DIM = (
    TRACK_EMBED_DIM * 2 + PAIRWISE_PHYSICS_DIM + DERIVED_GEOM_DIM
    + CASCADE_SCORE_DIM
)
assert COUPLE_FEATURE_DIM == 83

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

# H6 couple-level block (S3.1 POCA + S3.4 companion cone + S3.6 SV attach):
#   ln 3D POCA distance, ln transverse displacement of the POCA midpoint
#   from the PV, transverse pointing cosine, companion count / sum-pT /
#   min-ΔR (+ has-companion flag) over the FULL kept-track set, nearest-SV
#   ln distance / dlen_sig / mass (+ has-SV flag).
H6_COUPLE_EXTRA_DIM = 11
COUPLE_FEATURE_DIM_TOTAL = (
    COUPLE_FEATURE_DIM + PAIR_PHYSICS_V3_EXTRA_DIM + H6_COUPLE_EXTRA_DIM
)
assert COUPLE_FEATURE_DIM_TOTAL == 99
COUPLE_REST_DIM = COUPLE_FEATURE_DIM_TOTAL - 2 * TRACK_EMBED_DIM
assert COUPLE_REST_DIM == 35

A1_MASS_GEV = 1.230
A1_WIDTH_GEV = 0.420
RHO_WIDTH_GEV = 0.149

# Per-track channel indices in the standardized feature vector — the legacy
# 16-channel prefix of the YAML `pf_features` ordering is load-bearing
# (`data/low-pt/lowpt_tau_trackfinder.yaml`).
_IDX_LOG_COV_PHI_PHI = 12
_IDX_LOG_COV_LAMBDA_LAMBDA = 13
_IDX_DCA_SIG = 11
_IDX_HAS_SV_FEATURE = 30

# Point-channel indices of the raw transport block
# (`pf_points`: eta, phi, dz, vertex xyz, nearest-other-PV index,
# lifetime disp xy, PV xy, 3 SV slots x (x, y, z, dlen_sig, mass)).
_POINT_IDX_DZ = 2
_POINT_IDX_VERTEX = slice(3, 6)
_POINT_IDX_PRIMARY_VERTEX_X = 9
_POINT_IDX_PRIMARY_VERTEX_Y = 10
_POINT_IDX_SV_BASE = 11
SV_SLOT_COUNT = 3
SV_COORDINATE_SENTINEL = 1e4
REQUIRED_POINT_CHANNELS = _POINT_IDX_SV_BASE + SV_SLOT_COUNT * 5
assert REQUIRED_POINT_CHANNELS == 26

# Companion cone (matches the H6 measurement:
# part/diagnostics/h6_stage3_couple_features.py)
CONE_DELTA_R_MAXIMUM = 0.4
CONE_DZ_WINDOW = 0.5
COMPANION_MIN_DR_SENTINEL = 0.4
COMPANION_CHUNK_TARGET_ELEMENTS = 2 ** 27


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


# ---------------------------------------------------------------------------
# H6 couple-level channels (S3.1 POCA block, S3.4 companion cone, S3.6 SV)
# ---------------------------------------------------------------------------


def _compute_h6_couple_channels(
    top_k2_points: torch.Tensor,
    top_k2_features: torch.Tensor,
    upper_i: torch.Tensor,
    upper_j: torch.Tensor,
    sum_px: torch.Tensor,
    sum_py: torch.Tensor,
    sum_pz: torch.Tensor,
    full_points: torch.Tensor,
    full_lorentz: torch.Tensor,
    full_valid_mask: torch.Tensor,
    member_full_indices: torch.Tensor,
    companion_chunk_elements: int,
) -> torch.Tensor:
    """top_k2_points: (B, >=26, K2). top_k2_features: (B, 32, K2).
    upper_i/upper_j: (C,). sum_px/sum_py/sum_pz: (B, C). full_points:
    (B, >=3, P). full_lorentz: (B, 4, P). full_valid_mask: (B, P).
    member_full_indices: (B, K2). Returns (B, 11, C)."""
    with torch.no_grad():
        batch_size, _, pool_size = full_points.shape
        n_couples = upper_i.shape[0]

        eta = top_k2_points[:, 0, :].float()
        phi = top_k2_points[:, 1, :].float()
        cosh_eta = torch.cosh(eta)
        direction = torch.stack([
            torch.cos(phi) / cosh_eta,
            torch.sin(phi) / cosh_eta,
            torch.tanh(eta),
        ], dim=1)
        reference = top_k2_points[:, _POINT_IDX_VERTEX, :].float()

        direction_i = direction[:, :, upper_i]
        direction_j = direction[:, :, upper_j]
        reference_i = reference[:, :, upper_i]
        reference_j = reference[:, :, upper_j]

        # Skew-line closest approach between the members' linearized tracks;
        # parallel pairs fall back to the point-to-line solution. torch.where
        # evaluates both branches, so denominators are made safe first.
        separation = reference_i - reference_j
        direction_i_squared = (
            direction_i * direction_i).sum(dim=1, keepdim=True)
        direction_j_squared = (
            direction_j * direction_j).sum(dim=1, keepdim=True)
        direction_dot = (direction_i * direction_j).sum(dim=1, keepdim=True)
        separation_dot_i = (direction_i * separation).sum(dim=1, keepdim=True)
        separation_dot_j = (direction_j * separation).sum(dim=1, keepdim=True)
        denominator = (
            direction_i_squared * direction_j_squared - direction_dot ** 2
        )
        parallel = denominator < 1e-12
        safe_denominator = torch.where(
            parallel, torch.ones_like(denominator), denominator)
        safe_direction_j_squared = torch.where(
            direction_j_squared > 0,
            direction_j_squared, torch.ones_like(direction_j_squared))
        parameter_i = torch.where(
            parallel, torch.zeros_like(denominator),
            (direction_dot * separation_dot_j
             - direction_j_squared * separation_dot_i) / safe_denominator)
        parameter_j = torch.where(
            parallel, separation_dot_j / safe_direction_j_squared,
            (direction_i_squared * separation_dot_j
             - direction_dot * separation_dot_i) / safe_denominator)
        closest_i = reference_i + parameter_i * direction_i
        closest_j = reference_j + parameter_j * direction_j
        poca_distance = (
            (closest_i - closest_j).square().sum(dim=1).sqrt()
        )
        midpoint = 0.5 * (closest_i + closest_j)
        ln_poca = torch.log(poca_distance + 1e-6)

        # Transverse displacement of the couple vertex from the stored PV
        # (transverse only — the stored PV z is unreliable in ~68% of
        # events) and its pointing cosine along the couple pT vector.
        primary_x = top_k2_points[:, _POINT_IDX_PRIMARY_VERTEX_X, upper_i]
        primary_y = top_k2_points[:, _POINT_IDX_PRIMARY_VERTEX_Y, upper_i]
        displacement_x = midpoint[:, 0, :] - primary_x
        displacement_y = midpoint[:, 1, :] - primary_y
        displacement = torch.hypot(displacement_x, displacement_y)
        ln_displacement = torch.log(displacement + 1e-6)
        axis_pt = torch.hypot(sum_px, sum_py)
        pointing_cos = (
            (displacement_x * sum_px + displacement_y * sum_py)
            / torch.clamp_min(displacement * axis_pt, 1e-12)
        )

        # Companion cone over the FULL kept-track set, members excluded,
        # chunked over the couple axis to bound (B, chunk, P) intermediates.
        axis_eta = torch.asinh(sum_pz / torch.clamp_min(axis_pt, 1e-12))
        axis_phi = torch.atan2(sum_py, sum_px)
        member_dz_i = top_k2_points[:, _POINT_IDX_DZ, upper_i]
        member_dz_j = top_k2_points[:, _POINT_IDX_DZ, upper_j]
        axis_dz = 0.5 * (member_dz_i + member_dz_j)
        original_index_i = member_full_indices.gather(
            1, upper_i.unsqueeze(0).expand(batch_size, -1))
        original_index_j = member_full_indices.gather(
            1, upper_j.unsqueeze(0).expand(batch_size, -1))

        full_eta = full_points[:, 0, :].float()
        full_phi = full_points[:, 1, :].float()
        full_dz = full_points[:, _POINT_IDX_DZ, :].float()
        full_pt = torch.hypot(
            full_lorentz[:, 0, :].float(), full_lorentz[:, 1, :].float())
        track_index = torch.arange(
            pool_size, device=full_points.device).view(1, 1, pool_size)

        companion_count = torch.empty(
            batch_size, n_couples, device=full_points.device)
        companion_sum_pt = torch.empty_like(companion_count)
        companion_min_dr = torch.empty_like(companion_count)

        chunk_size = max(
            64, min(n_couples,
                    companion_chunk_elements
                    // max(1, batch_size * pool_size)))
        for chunk_start in range(0, n_couples, chunk_size):
            chunk = slice(chunk_start, min(chunk_start + chunk_size,
                                           n_couples))
            delta_eta = (
                full_eta.unsqueeze(1) - axis_eta[:, chunk].unsqueeze(-1))
            delta_phi = (
                full_phi.unsqueeze(1) - axis_phi[:, chunk].unsqueeze(-1))
            delta_phi = (delta_phi + math.pi) % (2 * math.pi) - math.pi
            delta_r = torch.sqrt(delta_eta ** 2 + delta_phi ** 2)
            dz_gap = (
                full_dz.unsqueeze(1) - axis_dz[:, chunk].unsqueeze(-1)
            ).abs()
            in_cone = (
                (delta_r < CONE_DELTA_R_MAXIMUM)
                & (dz_gap < CONE_DZ_WINDOW)
                & full_valid_mask.unsqueeze(1)
                & (track_index != original_index_i[:, chunk].unsqueeze(-1))
                & (track_index != original_index_j[:, chunk].unsqueeze(-1))
            )
            in_cone_float = in_cone.float()
            companion_count[:, chunk] = in_cone_float.sum(dim=-1)
            companion_sum_pt[:, chunk] = (
                in_cone_float * full_pt.unsqueeze(1)).sum(dim=-1)
            masked_delta_r = torch.where(
                in_cone, delta_r, torch.full_like(delta_r, float('inf')))
            companion_min_dr[:, chunk] = masked_delta_r.amin(dim=-1)

        has_companion = (companion_count > 0).float()
        companion_min_dr = torch.where(
            companion_count > 0, companion_min_dr,
            torch.full_like(companion_min_dr, COMPANION_MIN_DR_SENTINEL))

        # Nearest secondary vertex to the couple vertex (POCA midpoint) over
        # the 3 transported SV slots; empty slots sit at the coordinate
        # sentinel and never win the argmin when a real SV exists. All three
        # outputs are gated to 0 for events without any SV.
        slot_distances = []
        slot_dlen_sig = []
        slot_mass = []
        for slot in range(SV_SLOT_COUNT):
            base = _POINT_IDX_SV_BASE + slot * 5
            slot_x = top_k2_points[:, base + 0, upper_i]
            slot_y = top_k2_points[:, base + 1, upper_i]
            slot_z = top_k2_points[:, base + 2, upper_i]
            slot_distances.append(torch.sqrt(
                (midpoint[:, 0, :] - slot_x) ** 2
                + (midpoint[:, 1, :] - slot_y) ** 2
                + (midpoint[:, 2, :] - slot_z) ** 2))
            slot_dlen_sig.append(top_k2_points[:, base + 3, upper_i])
            slot_mass.append(top_k2_points[:, base + 4, upper_i])
        distances = torch.stack(slot_distances, dim=1)
        nearest_slot = distances.argmin(dim=1, keepdim=True)
        nearest_distance = distances.gather(1, nearest_slot).squeeze(1)
        nearest_dlen_sig = torch.stack(slot_dlen_sig, dim=1).gather(
            1, nearest_slot).squeeze(1)
        nearest_mass = torch.stack(slot_mass, dim=1).gather(
            1, nearest_slot).squeeze(1)

        has_sv = top_k2_features[:, _IDX_HAS_SV_FEATURE, upper_i]
        has_sv_gate = has_sv > 0.5
        zero = torch.zeros_like(nearest_distance)
        ln_sv_distance = torch.where(
            has_sv_gate, torch.log(nearest_distance + 1e-6), zero)
        sv_dlen_sig = torch.where(has_sv_gate, nearest_dlen_sig, zero)
        sv_mass = torch.where(has_sv_gate, nearest_mass, zero)
        has_sv_flag = has_sv_gate.float()

        return torch.stack([
            ln_poca,
            ln_displacement,
            pointing_cos,
            companion_count,
            companion_sum_pt,
            companion_min_dr,
            has_companion,
            ln_sv_distance,
            sv_dlen_sig,
            sv_mass,
            has_sv_flag,
        ], dim=1)


# ---------------------------------------------------------------------------
# Batched per-event feature builder (for the trainer)
# ---------------------------------------------------------------------------


def build_couple_features_batched(
    top_k2_features: torch.Tensor,
    top_k2_points: torch.Tensor,
    top_k2_lorentz: torch.Tensor,
    top_k2_stage1_scores: torch.Tensor,
    top_k2_stage2_scores: torch.Tensor,
    *,
    full_points: torch.Tensor,
    full_lorentz: torch.Tensor,
    full_valid_mask: torch.Tensor,
    member_full_indices: torch.Tensor,
    top_k2_track_labels: torch.Tensor | None = None,
    track_valid_mask: torch.Tensor | None = None,
    m_tau: float = M_TAU_GEV,
    companion_chunk_elements: int = COMPANION_CHUNK_TARGET_ELEMENTS,
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
        top_k2_features: ``(B, 32, K2)`` standardized per-track features.
        top_k2_points: ``(B, 26, K2)`` raw point channels.
        top_k2_lorentz: ``(B, 4, K2)`` raw 4-vectors.
        top_k2_stage1_scores: ``(B, K2)`` Stage 1 scores.
        top_k2_stage2_scores: ``(B, K2)`` Stage 2 scores.
        full_points: ``(B, >=3, P)`` raw point channels of the FULL event
            (companion cone candidates).
        full_lorentz: ``(B, 4, P)`` raw 4-vectors of the full event.
        full_valid_mask: ``(B, P)`` boolean over the full event.
        member_full_indices: ``(B, K2)`` original track index of each pool
            member within the full event (for cone self-exclusion).
        top_k2_track_labels: optional ``(B, K2)`` per-track binary labels.
        track_valid_mask: optional ``(B, K2)`` boolean — True for real
            tracks, False for padding. When provided, padding tracks are
            zeroed out before couple feature computation (preventing
            ``-inf`` cascade scores from producing ``Inf`` in the feature
            vector), and couples involving any padding track are excluded
            from ``filter_a_mask``.
        m_tau: kinematic mass cut (default = PDG τ mass).
        companion_chunk_elements: target ``B * chunk * P`` element budget
            for one cone chunk.

    Returns:
        dict with:
            ``couple_features``: ``(B, 99, n_couples)`` per-couple feature
                tensor where ``n_couples = K2 * (K2 - 1) / 2`` (always).
            ``filter_a_mask``: ``(B, n_couples)`` boolean — True for couples
                passing the loose mass cut AND having both tracks valid.
            ``couple_labels``: ``(B, n_couples)`` boolean — True iff both
                tracks of the couple are GT pions. Only present when
                ``top_k2_track_labels`` is provided.
    """
    batch_size, _, k2 = top_k2_features.shape
    device = top_k2_features.device

    if top_k2_points.shape[1] < REQUIRED_POINT_CHANNELS:
        raise ValueError(
            f'couple feature builder needs the {REQUIRED_POINT_CHANNELS}-'
            'channel pf_points transport block, got '
            f'{top_k2_points.shape[1]} point channels.'
        )
    if top_k2_features.shape[1] != TRACK_EMBED_DIM:
        raise ValueError(
            f'couple feature builder expects {TRACK_EMBED_DIM} per-track '
            f'feature channels, got {top_k2_features.shape[1]}.'
        )

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

    # pair-physics v3: 5 extra features always emitted (always on in production).
    log_cov_phi_i = top_k2_features[:, _IDX_LOG_COV_PHI_PHI, upper_i]
    log_cov_phi_j = top_k2_features[:, _IDX_LOG_COV_PHI_PHI, upper_j]
    log_cov_lam_i = top_k2_features[:, _IDX_LOG_COV_LAMBDA_LAMBDA, upper_i]
    log_cov_lam_j = top_k2_features[:, _IDX_LOG_COV_LAMBDA_LAMBDA, upper_j]
    kalman_proxy = log_cov_phi_i + log_cov_phi_j + log_cov_lam_i + log_cov_lam_j
    dca_sig_i = top_k2_features[:, _IDX_DCA_SIG, upper_i]
    dca_sig_j = top_k2_features[:, _IDX_DCA_SIG, upper_j]
    dca_sig_sum = dca_sig_i + dca_sig_j
    sum_px_3 = px_i + px_j
    sum_py_3 = py_i + py_j
    sum_pz_3 = pz_i + pz_j
    sum_p_mag = torch.sqrt(sum_px_3 ** 2 + sum_py_3 ** 2 + sum_pz_3 ** 2 + 1e-10)
    pi_mag = torch.sqrt(px_i ** 2 + py_i ** 2 + pz_i ** 2 + 1e-10)
    helicity_lab = (
        (px_i * sum_px_3 + py_i * sum_py_3 + pz_i * sum_pz_3) / (pi_mag * sum_p_mag)
    )
    # log Breit-Wigner densities: ρ(m²) = m_res² Γ² / ((m² − m_res²)² + m_res² Γ²).
    m_squared_safe = torch.clamp_min(m_ij ** 2, 1e-8)
    def _log_bw(m_res: float, width: float) -> torch.Tensor:
        num = m_res ** 2 * width ** 2
        den = (m_squared_safe - m_res ** 2) ** 2 + m_res ** 2 * width ** 2
        return math.log(num) - torch.log(torch.clamp_min(den, 1e-20))
    log_bw_rho = _log_bw(RHO_MASS_GEV, RHO_WIDTH_GEV)
    log_bw_a1 = _log_bw(A1_MASS_GEV, A1_WIDTH_GEV)
    pair_v3_extras = torch.stack(
        [kalman_proxy, dca_sig_sum, helicity_lab, log_bw_rho, log_bw_a1],
        dim=1,
    )
    couple_features_list.append(pair_v3_extras)

    h6_couple_channels = _compute_h6_couple_channels(
        top_k2_points=top_k2_points,
        top_k2_features=top_k2_features,
        upper_i=upper_i,
        upper_j=upper_j,
        sum_px=sum_px,
        sum_py=sum_py,
        sum_pz=sum_pz,
        full_points=full_points,
        full_lorentz=full_lorentz,
        full_valid_mask=full_valid_mask,
        member_full_indices=member_full_indices,
        companion_chunk_elements=companion_chunk_elements,
    )
    couple_features_list.append(h6_couple_channels)

    couple_features = torch.cat(couple_features_list, dim=1)
    assert couple_features.shape[1] == COUPLE_FEATURE_DIM_TOTAL

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


