"""Training-time input augmentation utilities.

Current augmentations:

* ``apply_cov_smear`` — draws Gaussian noise scaled by the per-track
  log-pT-error channel and adds it to the kinematic channels
  (px, py, pz, eta, phi). Physics interpretation: mimic the detector
  resolution implied by the fitted covariance entries. Used by the
  Batch-3 H12 hypothesis (R1 §8; JetCLR, arXiv:2108.04253).

* ``drop_cov_features`` — zeros out the standardized covariance /
  uncertainty channels so the model cannot read σ directly while the
  smearing is on. Without this the net can "look at the σ" to detect
  that the input was jittered, which defeats the augmentation.

* ``symmetric_kl`` — symmetric-KL consistency term for the R-Drop
  recipe (arXiv:2106.14448). Consumes two per-couple score tensors
  ``s1, s2 ∈ ℝ^{B×C}`` and a validity mask; returns a scalar loss.

These helpers operate on the 16-channel standardized track feature
tensor emitted by the data pipeline. Channel indices mirror the YAML
config ``pf_features`` ordering:

    0  track_px           9  track_log_pt_error
    1  track_py          10  track_n_valid_pixel_hits
    2  track_pz          11  track_dca_significance
    3  track_eta         12  track_log_covariance_phi_phi
    4  track_phi         13  track_log_covariance_lambda_lambda
    5  track_charge      14  track_log_pt
    6  track_dxy_significance          15  track_log_relative_pt_error
    7  track_log_dz_significance
    8  track_log_norm_chi2
"""
from __future__ import annotations

import torch
import torch.nn.functional as functional


# Kinematic channels perturbed by cov-smear.
_KINEMATIC_CHANNELS = (0, 1, 2, 3, 4)  # px, py, pz, eta, phi

# Covariance / uncertainty channels zeroed out by drop_cov_features so
# the model cannot detect the smearing magnitude via the σ features.
_COV_CHANNELS = (9, 11, 12, 13, 15)


def apply_cov_smear(
    features: torch.Tensor,
    *,
    smear_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to the kinematic channels, scaled per-track.

    ``features`` is the standardized track feature tensor with shape
    ``(B, 16, N)`` as produced by the data pipeline. The per-track
    noise σ is ``smear_scale · exp(log_pt_error_standardized)`` —
    higher-uncertainty tracks receive more perturbation, matching the
    cov-aware data augmentation used by JetCLR (arXiv:2108.04253) and
    R-Drop-style consistency regularisation (arXiv:2106.14448).

    The noise is drawn from the standard normal distribution in the
    standardized input space; since the standardized scale differs
    per channel, the effective physical perturbation is not strictly
    calibrated but is a physics-plausible proxy.

    Returns a new tensor; the input is not modified in place.
    """
    if features.dim() != 3 or features.shape[1] < 16:
        raise ValueError(
            f'expected features of shape (B, ≥16, N), got {tuple(features.shape)}',
        )
    noisy = features.clone()
    # Per-track, per-channel scale factor.
    log_pt_err = features[:, 9:10, :]  # (B, 1, N)
    noise_scale = torch.exp(log_pt_err) * smear_scale  # (B, 1, N)
    # One independent noise draw per kinematic channel, shared σ
    # across the 5 kinematic channels (detector resolution correlates
    # across those channels because they all stem from the same
    # track fit).
    kin_shape = (
        features.shape[0],
        len(_KINEMATIC_CHANNELS),
        features.shape[2],
    )
    noise = torch.randn(
        kin_shape,
        device=features.device,
        dtype=features.dtype,
        generator=generator,
    ) * noise_scale
    for offset, channel in enumerate(_KINEMATIC_CHANNELS):
        noisy[:, channel, :] = features[:, channel, :] + noise[:, offset, :]
    return noisy


def drop_cov_features(features: torch.Tensor) -> torch.Tensor:
    """Zero out the covariance / uncertainty channels.

    Used jointly with ``apply_cov_smear`` so the model cannot detect
    the smearing magnitude by reading σ directly. Returns a new tensor.
    """
    if features.dim() != 3 or features.shape[1] < 16:
        raise ValueError(
            f'expected features of shape (B, ≥16, N), got {tuple(features.shape)}',
        )
    out = features.clone()
    for channel in _COV_CHANNELS:
        out[:, channel, :] = 0.0
    return out


def symmetric_kl(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL divergence between two per-couple softmax
    distributions under the given validity mask.

    Inputs: ``scores_a``, ``scores_b`` shape ``(B, C)``; ``mask`` shape
    ``(B, C)`` with 1 on valid positions, 0 on padding. Returns a scalar
    loss that is the mean over events of
    ``0.5 · (KL(p_a || p_b) + KL(p_b || p_a))`` where ``p_a, p_b`` are
    softmax distributions over valid positions only.

    Used for R-Drop consistency on the output couple-rank distribution.
    """
    valid = mask > 0.5
    batch_losses: list[torch.Tensor] = []
    for event_index in range(scores_a.shape[0]):
        event_valid = valid[event_index]
        if not event_valid.any():
            continue
        s_a = scores_a[event_index][event_valid]
        s_b = scores_b[event_index][event_valid]
        log_p_a = functional.log_softmax(s_a, dim=0)
        log_p_b = functional.log_softmax(s_b, dim=0)
        p_a = log_p_a.exp()
        p_b = log_p_b.exp()
        kl_ab = (p_a * (log_p_a - log_p_b)).sum()
        kl_ba = (p_b * (log_p_b - log_p_a)).sum()
        batch_losses.append(0.5 * (kl_ab + kl_ba))
    if not batch_losses:
        return scores_a.sum() * 0.0
    return torch.stack(batch_losses).mean()
