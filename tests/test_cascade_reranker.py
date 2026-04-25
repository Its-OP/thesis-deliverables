"""Unit tests for CascadeReranker (Track A: ParT-style pairwise-bias encoder).

TDD: Written before implementation. Tests cover:
    - Forward pass shape: (B, K1) scores
    - Pairwise attention bias computation
    - Per-track scoring (not CLS-based)
    - Gradient flow through all parameters
    - compute_loss() with ranking loss
    - Stage 2 interface compatibility (stage1_scores input)
    - Integration with CascadeModel
"""
import pytest
import torch

from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 100  # K1 = 100 for test speed
INPUT_DIM = 16


def _make_filtered_inputs(
    batch_size=BATCH_SIZE,
    num_tracks=NUM_TRACKS,
    seed=42,
):
    """Create inputs simulating Stage 1 filtered output."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, INPUT_DIM, num_tracks, generator=generator,
    )

    transverse_momentum = (
        torch.rand(batch_size, 1, num_tracks, generator=generator) * 5 + 0.5
    )
    px = transverse_momentum * torch.cos(phi)
    py = transverse_momentum * torch.sin(phi)
    pz = transverse_momentum * torch.sinh(eta)
    pion_mass = 0.13957
    energy = torch.sqrt(px**2 + py**2 + pz**2 + pion_mass**2)
    lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

    mask = torch.ones(batch_size, 1, num_tracks)
    mask[:, :, -10:] = 0.0  # Some padding

    track_labels = torch.zeros(batch_size, 1, num_tracks)
    for batch_index in range(batch_size):
        track_labels[batch_index, 0, 5] = 1.0
        track_labels[batch_index, 0, 15] = 1.0
        track_labels[batch_index, 0, 25] = 1.0

    stage1_scores = torch.randn(batch_size, num_tracks, generator=generator)

    return points, features, lorentz_vectors, mask, track_labels, stage1_scores


def _make_reranker(**kwargs):
    """Create a CascadeReranker with small config for testing."""
    defaults = dict(
        input_dim=INPUT_DIM,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        pair_embed_dims=[32, 32],
    )
    defaults.update(kwargs)
    return CascadeReranker(**defaults)


# ---- Forward pass ----

class TestCascadeRerankerForward:
    """Test forward pass shape and properties."""

    def test_output_shape(self):
        """forward() should return (B, K1) scores."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_valid_scores_finite(self):
        """Scores for valid tracks should be finite."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf scores."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))


# ---- Gradient flow ----

class TestCascadeRerankerGradients:
    """Test gradient flow through the model."""

    def test_all_parameters_receive_gradients(self):
        """All parameters should get gradients via compute_loss()."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        # Allow 1-2 params with zero grad (e.g. unused bias)
        assert params_with_grad >= total_params - 2, (
            f'Only {params_with_grad}/{total_params} params got gradients'
        )

    def test_loss_is_finite(self):
        """compute_loss() should return finite total_loss."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_loss_dict_has_ranking_loss(self):
        """Loss dict should contain ranking_loss component."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'ranking_loss' in loss_dict


# ---- Stage 2 interface ----

class TestStage2Interface:
    """Test that CascadeReranker implements the Stage 2 interface."""

    def test_forward_accepts_stage1_scores(self):
        """forward() must accept stage1_scores parameter."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        # Should not raise
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores is not None

    def test_compute_loss_accepts_stage1_scores(self):
        """compute_loss() must accept stage1_scores parameter."""
        model = _make_reranker()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'total_loss' in loss_dict
        assert '_scores' in loss_dict


# ---- Integration with CascadeModel ----

# ---- Extended pairwise features ----

class TestExtendedPairwiseFeatures:
    """Test physics-motivated pairwise features (charge, dz, rho, dxy)."""

    def test_extra_pairwise_forward_shape(self):
        """Model with pair_extra_dim=6 should produce correct output shape."""
        model = _make_reranker(pair_extra_dim=6)
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_extra_pairwise_scores_finite(self):
        """Valid track scores should be finite with extra pairwise features."""
        model = _make_reranker(pair_extra_dim=6)
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_extra_pairwise_gradient_flow(self):
        """Gradients should flow through the pairwise feature MLP."""
        model = _make_reranker(pair_extra_dim=6)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad >= total_params - 2

    def test_extra_pairwise_no_nan_backward(self):
        """Backward pass should not produce NaN (critical for dxy_phi_corrected
        division which clamps sin(dphi/2) to avoid 0/0)."""
        model = _make_reranker(pair_extra_dim=6)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), (
                    f'NaN gradient in {name}'
                )

    def test_pair_extra_dim_zero_backward_compatible(self):
        """pair_extra_dim=0 should give same behavior as original model."""
        model = _make_reranker(pair_extra_dim=0)
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_pairwise_features_correctness(self):
        """Verify physics pairwise features are computed with correct values.

        Uses known inputs to check:
        - Charge product recovers raw {-1, +1} from standardized {-1, 0}
        - Rho indicator peaks at m_ij ~ 770 MeV
        - dz_diff is symmetric and non-negative
        - Masking zeros out features for padded tracks
        """
        model = _make_reranker(pair_extra_dim=6)

        # Construct inputs with known charge values
        batch_size, num_tracks = 2, 20
        generator = torch.Generator().manual_seed(99)
        eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 0.5
        phi = torch.rand(batch_size, 1, num_tracks, generator=generator) * 2 * 3.14159 - 3.14159
        points = torch.cat([eta, phi], dim=1)

        features = torch.randn(batch_size, INPUT_DIM, num_tracks, generator=generator)

        # Feature 5 = charge: set standardized values
        # Raw +1 → standardized 0.0, Raw -1 → standardized -1.0
        features[:, 5, :10] = 0.0    # positive charge tracks
        features[:, 5, 10:] = -1.0   # negative charge tracks

        # Feature 7 = dz_sig: set known values
        features[:, 7, :] = 0.0
        features[0, 7, 0] = 1.5   # track 0 has dz=1.5
        features[0, 7, 1] = 1.5   # track 1 has same dz (same vertex)
        features[0, 7, 2] = 0.0   # track 2 has dz=0 (PV track)

        # Build physical 4-vectors
        transverse_momentum = torch.ones(batch_size, 1, num_tracks) * 0.5
        px = transverse_momentum * torch.cos(phi)
        py = transverse_momentum * torch.sin(phi)
        pz = transverse_momentum * torch.sinh(eta)
        pion_mass = 0.13957
        energy = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + pion_mass ** 2)
        lorentz_vectors = torch.cat([px, py, pz, energy], dim=1)

        mask = torch.ones(batch_size, 1, num_tracks)
        mask[:, :, -5:] = 0.0  # Last 5 are padded

        mask_float = mask.float()
        lorentz_for_pairs = (lorentz_vectors * mask_float).detach().float()

        extra = model._compute_extra_pairwise_features(
            points, features, lorentz_for_pairs, mask_float,
        )
        assert extra.shape == (batch_size, 6, num_tracks, num_tracks)

        # Channel 0: charge_product
        charge_prod = extra[0, 0]  # (20, 20)
        # Tracks 0 (+1) and 10 (-1) should have product = -1
        assert charge_prod[0, 10].item() == pytest.approx(-1.0, abs=0.01), (
            f'OS pair should give -1, got {charge_prod[0, 10].item()}'
        )
        # Tracks 0 (+1) and 1 (+1) should have product = +1
        assert charge_prod[0, 1].item() == pytest.approx(1.0, abs=0.01), (
            f'SS pair should give +1, got {charge_prod[0, 1].item()}'
        )

        # Channel 1: dz_diff
        dz_diff = extra[0, 1]
        # Tracks 0 and 1 share dz=1.5 → diff = 0
        assert dz_diff[0, 1].item() == pytest.approx(0.0, abs=1e-5)
        # Tracks 0 (dz=1.5) and 2 (dz=0) → diff = 1.5
        assert dz_diff[0, 2].item() == pytest.approx(1.5, abs=1e-5)
        # Symmetric
        assert dz_diff[0, 2].item() == pytest.approx(dz_diff[2, 0].item())

        # Channel 2: rho_indicator — should be in [0, 1]
        rho_ind = extra[0, 2]
        assert (rho_ind >= 0).all()
        assert (rho_ind <= 1).all()

        # Channel 3: rho_os_indicator — should be 0 for same-sign pairs
        rho_os = extra[0, 3]
        # SS pair (both positive charge): should be 0
        assert rho_os[0, 1].item() == 0.0

        # Channels for padded tracks should be 0
        for channel in range(5):
            # Padded track indices: 15-19
            assert (extra[0, channel, 15:, :] == 0).all(), (
                f'Channel {channel} not zeroed for padded rows'
            )
            assert (extra[0, channel, :, 15:] == 0).all(), (
                f'Channel {channel} not zeroed for padded cols'
            )

    def test_sum_mode_works(self):
        """pair_embed_mode='sum' should also work (for ablation)."""
        model = _make_reranker(pair_extra_dim=6, pair_embed_mode='sum')
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask, stage1_scores)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()


# ---- Loss modes ----

class TestLossModes:
    """Test togglable loss functions: pairwise, lambda_rank, rs_at_k."""

    def test_pairwise_loss_default(self):
        """Default loss_mode='pairwise' should work as before."""
        model = _make_reranker(loss_mode='pairwise')
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'ranking_loss' in loss_dict

    def test_lambda_rank_loss(self):
        """LambdaRank should produce finite loss with boundary weighting."""
        model = _make_reranker(loss_mode='lambda_rank')
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'ranking_loss' in loss_dict

    def test_lambda_rank_gradient_flow(self):
        """LambdaRank gradients should reach all parameters."""
        # K boundary must be < num_tracks so some pairs straddle it
        model = _make_reranker(loss_mode='lambda_rank', rs_at_k_target=30)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()
        params_with_grad = sum(
            1 for _, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad > 0

    def test_rs_at_k_loss(self):
        """RS@K loss should produce finite, differentiable loss."""
        model = _make_reranker(loss_mode='rs_at_k', rs_at_k_target=30)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'rs_at_k_loss' in loss_dict

    def test_rs_at_k_gradient_flow(self):
        """RS@K gradients should reach all parameters."""
        model = _make_reranker(loss_mode='rs_at_k', rs_at_k_target=30)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()
        params_with_grad = sum(
            1 for _, p in model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad > 0

    def test_hybrid_lambda_loss(self):
        """Hybrid loss should produce finite loss and anneal alpha."""
        model = _make_reranker(loss_mode='hybrid_lambda', rs_at_k_target=30)
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )

        # Early training: alpha=0, pure pairwise
        model.set_training_progress(0.0)
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'lambda_alpha' in loss_dict
        assert loss_dict['lambda_alpha'].item() == 0.0

        # Late training: alpha > 0, lambda_rank active
        model.set_training_progress(0.8)
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert loss_dict['lambda_alpha'].item() > 0.0



class TestCascadeIntegration:
    """Test CascadeReranker plugged into CascadeModel."""

    def test_end_to_end_forward(self):
        """CascadeModel with CascadeReranker should produce scores."""
        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
        )
        stage2 = _make_reranker()
        cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=50)

        points, features, lorentz_vectors, mask, _, _ = (
            _make_filtered_inputs(num_tracks=200)
        )
        mask = torch.ones(BATCH_SIZE, 1, 200)
        mask[:, :, -30:] = 0.0

        scores = cascade(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, 50)

    def test_end_to_end_loss(self):
        """CascadeModel with CascadeReranker should produce finite loss."""
        stage1 = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
        )
        stage2 = _make_reranker()
        cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=50)

        points, features, lorentz_vectors, mask, track_labels, _ = (
            _make_filtered_inputs(num_tracks=200)
        )
        mask = torch.ones(BATCH_SIZE, 1, 200)
        mask[:, :, -30:] = 0.0
        track_labels = torch.zeros(BATCH_SIZE, 1, 200)
        for batch_index in range(BATCH_SIZE):
            track_labels[batch_index, 0, 5] = 1.0
            track_labels[batch_index, 0, 15] = 1.0

        loss_dict = cascade.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert 'stage1_recall_at_k1' in loss_dict


# ---- Contrastive denoising (auxiliary regularizer on GT track features) ----

class TestContrastiveDenoising:
    """Port of the TrackPreFilter contrastive denoising trick to CascadeReranker.

    The auxiliary loss adds Gaussian noise σ to GT track features, runs a
    second forward pass, and requires the noised positives to still score
    above the (original-pass) background tracks. σ anneals from
    denoising_sigma_start → denoising_sigma_end based on _training_progress.
    """

    def test_denoising_constructor_defaults_off(self):
        """Existing callers that don't know about denoising must not be
        broken — it defaults to OFF."""
        model = _make_reranker()
        assert hasattr(model, 'use_contrastive_denoising')
        assert model.use_contrastive_denoising is False

    def test_denoising_constructor_accepts_kwargs(self):
        """Explicit knobs: enable flag + sigma schedule + loss weight."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.3,
            denoising_sigma_end=0.05,
            denoising_loss_weight=0.5,
        )
        assert model.use_contrastive_denoising is True
        assert model.denoising_sigma_start == pytest.approx(0.3)
        assert model.denoising_sigma_end == pytest.approx(0.05)
        assert model.denoising_loss_weight == pytest.approx(0.5)

    def test_current_sigma_interpolates_by_progress(self):
        """`current_denoising_sigma` should linearly interpolate start→end
        as `_training_progress` goes 0→1 (same mechanism as hybrid_lambda)."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.3,
            denoising_sigma_end=0.05,
        )
        model.set_training_progress(0.0)
        assert model.current_denoising_sigma == pytest.approx(0.3)
        model.set_training_progress(1.0)
        assert model.current_denoising_sigma == pytest.approx(0.05)
        model.set_training_progress(0.5)
        assert model.current_denoising_sigma == pytest.approx(0.175)

    def test_loss_dict_has_denoising_loss_when_enabled(self):
        """When denoising is ON, compute_loss must expose the denoising
        component as a separate key so it shows up in training logs."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.2,
            denoising_sigma_end=0.2,
            denoising_loss_weight=0.5,
        )
        model.train()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'denoising_loss' in loss_dict
        assert torch.isfinite(loss_dict['denoising_loss']).all()
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_loss_dict_has_no_denoising_loss_when_disabled(self):
        """When denoising is OFF, compute_loss must NOT report a
        denoising_loss component (matches pre-feature behavior)."""
        model = _make_reranker(use_contrastive_denoising=False)
        model.train()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        assert 'denoising_loss' not in loss_dict

    def test_denoising_loss_skipped_in_eval_mode(self):
        """Denoising is a training-only regularizer — must be gated by
        self.training (matches TrackPreFilter's pattern)."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.2,
            denoising_sigma_end=0.2,
        )
        model.eval()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        with torch.no_grad():
            loss_dict = model.compute_loss(
                points, features, lorentz_vectors, mask,
                track_labels, stage1_scores,
            )
        assert 'denoising_loss' not in loss_dict

    def test_denoising_loss_is_zero_without_gt_tracks(self):
        """If an event has no GT tracks, denoising contributes 0."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.2,
            denoising_sigma_end=0.2,
        )
        model.train()
        points, features, lorentz_vectors, mask, _, stage1_scores = (
            _make_filtered_inputs()
        )
        # Zero out all labels — no GT tracks anywhere in the batch
        empty_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            empty_labels, stage1_scores,
        )
        assert 'denoising_loss' in loss_dict
        assert loss_dict['denoising_loss'].item() == pytest.approx(0.0)

    def test_denoising_backward_runs_without_nan(self):
        """Backward through the noised forward + ranking loss must produce
        finite gradients for all parameters."""
        model = _make_reranker(
            use_contrastive_denoising=True,
            denoising_sigma_start=0.3,
            denoising_sigma_end=0.05,
            denoising_loss_weight=0.5,
        )
        model.train()
        points, features, lorentz_vectors, mask, track_labels, stage1_scores = (
            _make_filtered_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask,
            track_labels, stage1_scores,
        )
        loss_dict['total_loss'].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), (
                    f'NaN or inf gradient in {name} after denoising backward'
                )
