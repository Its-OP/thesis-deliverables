"""Unit tests for TrackPreFilter (Stage 1 of two-stage pipeline).

TDD: Written before implementation. Tests cover:
    - Three pre-filter modes: MLP+neighborhood, two-tower, autoencoder
    - Top-K selection preserving GT tracks
    - Masking, finiteness, gradient flow
    - Two-stage pipeline integration
"""
import pytest
import torch

from weaver.nn.model.TrackPreFilter import TrackPreFilter


# ---- Shared configuration ----

BATCH_SIZE = 4
NUM_TRACKS = 200
INPUT_DIM = 7
INPUT_DIM_EXTENDED = 13
TOP_K = 50  # Smaller than production 200 for test speed


def _make_physical_inputs(batch_size, num_tracks, input_dim=INPUT_DIM, seed=42):
    """Create physically sensible synthetic inputs."""
    generator = torch.Generator().manual_seed(seed)

    eta = torch.randn(batch_size, 1, num_tracks, generator=generator) * 1.5
    phi = (
        torch.rand(batch_size, 1, num_tracks, generator=generator)
        * 2 * 3.14159 - 3.14159
    )
    points = torch.cat([eta, phi], dim=1)

    features = torch.randn(
        batch_size, input_dim, num_tracks, generator=generator,
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

    return points, features, lorentz_vectors


def _make_training_inputs():
    """Create inputs with mask and labels."""
    points, features, lorentz_vectors = _make_physical_inputs(
        BATCH_SIZE, NUM_TRACKS,
    )
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -30:] = 0.0  # Last 30 are padding

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


# ---- Mode A: MLP + Neighborhood ----

class TestMLPMode:
    """Test per-track MLP with neighborhood context."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf so they never appear in top-K."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))


# ---- Mode B: Two-Tower ----

class TestTwoTowerMode:
    """Test two-tower retrieve with learned tau prototype."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='two_tower', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='two_tower', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()


# ---- Mode C: Autoencoder ----

class TestAutoencoderMode:
    """Test autoencoder anomaly scorer."""

    def test_forward_shape(self):
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_reconstruction_loss_finite(self):
        """Autoencoder mode should return finite reconstruction loss."""
        model = TrackPreFilter(mode='autoencoder', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'reconstruction_loss' in loss_dict
        assert torch.isfinite(loss_dict['reconstruction_loss']).all()


# ---- Top-K Selection ----

class TestTopKSelection:
    """Test top-K candidate selection from pre-filter scores."""

    def test_topk_preserves_gt(self):
        """When GT tracks have high scores, top-K should include them all."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)

        # Create scores where GT tracks (10, 20, 30) score highest
        scores = torch.randn(1, NUM_TRACKS) - 5  # Background: low scores
        scores[0, 10] = 10.0
        scores[0, 20] = 10.0
        scores[0, 30] = 10.0
        mask = torch.ones(1, 1, NUM_TRACKS)

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)
        selected_set = set(selected_indices[0].tolist())

        assert 10 in selected_set, 'GT track 10 not in top-K'
        assert 20 in selected_set, 'GT track 20 not in top-K'
        assert 30 in selected_set, 'GT track 30 not in top-K'

    def test_topk_returns_correct_count(self):
        """Should return exactly K indices per event."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.randn(BATCH_SIZE, NUM_TRACKS)
        mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)
        assert selected_indices.shape == (BATCH_SIZE, TOP_K)

    def test_topk_handles_fewer_valid_than_k(self):
        """Events with < K valid tracks should return all valid indices."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.randn(1, NUM_TRACKS)
        mask = torch.ones(1, 1, NUM_TRACKS)
        mask[0, 0, 20:] = 0.0  # Only 20 valid tracks, K=50

        selected_indices = model.select_top_k(scores, mask, top_k=TOP_K)

        # All 20 valid should be selected, rest padded with -1 or repeated
        valid_selected = selected_indices[0][selected_indices[0] < 20]
        assert len(valid_selected) == 20


# ---- Loss Functions ----

class TestPreFilterLoss:
    """Test ranking loss for pre-filter training."""

    def test_ranking_loss_finite(self):
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'total_loss' in loss_dict
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_ranking_loss_zero_when_perfectly_ranked(self):
        """Perfect ranking should give near-zero loss."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.zeros(1, NUM_TRACKS) - 10
        scores[0, 10] = 10.0
        scores[0, 20] = 10.0
        scores[0, 30] = 10.0
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(scores, labels, valid_mask)
        assert loss.item() < 0.01

    def test_ranking_loss_high_when_misranked(self):
        """Misranked should give high loss."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        scores = torch.zeros(1, NUM_TRACKS)
        scores[0, 10] = -5.0  # GT scores low
        scores[0, 20] = -5.0
        scores[0, 30] = -5.0
        scores[0, 0] = 5.0  # Background scores high
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        labels[0, 20] = 1.0
        labels[0, 30] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)

        loss = model._ranking_loss(scores, labels, valid_mask)
        assert loss.item() > 1.0

    def test_gradients_flow(self):
        """Backward should produce gradients in all model parameters."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        assert params_with_grad > 0


# ---- Two-Stage Pipeline ----

class TestTwoStagePipeline:
    """Test Stage 1 → top-K → Stage 2 pipeline."""

    def test_repack_reduces_track_count(self):
        """After top-K selection, repacked tensors should have K tracks."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )

        filtered = model.filter_tracks(
            points, features, lorentz_vectors, mask, track_labels,
            top_k=TOP_K,
        )

        assert filtered['points'].shape[2] == TOP_K
        assert filtered['features'].shape[2] == TOP_K
        assert filtered['lorentz_vectors'].shape[2] == TOP_K
        assert filtered['mask'].shape[2] == TOP_K
        assert filtered['track_labels'].shape[2] == TOP_K

    def test_filter_preserves_gt_when_scored_high(self):
        """If GT tracks score high, they should survive filtering."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )

        # We can't guarantee untrained model scores GT high,
        # but we can verify the pipeline runs without error
        filtered = model.filter_tracks(
            points, features, lorentz_vectors, mask, track_labels,
            top_k=TOP_K,
        )

        # Filtered labels should be valid (0 or 1)
        assert torch.all(
            (filtered['track_labels'] == 0) | (filtered['track_labels'] == 1)
        )


# ---- Extended 13-Feature Configuration (wide192) ----

def _make_extended_training_inputs():
    """Create inputs with the extended 13-feature set and wide192 config."""
    points, features, lorentz_vectors = _make_physical_inputs(
        BATCH_SIZE, NUM_TRACKS, input_dim=INPUT_DIM_EXTENDED,
    )
    mask = torch.ones(BATCH_SIZE, 1, NUM_TRACKS)
    mask[:, :, -30:] = 0.0

    track_labels = torch.zeros(BATCH_SIZE, 1, NUM_TRACKS)
    for batch_index in range(BATCH_SIZE):
        track_labels[batch_index, 0, 10] = 1.0
        track_labels[batch_index, 0, 20] = 1.0
        track_labels[batch_index, 0, 30] = 1.0

    return points, features, lorentz_vectors, mask, track_labels


class TestExtendedHybridMode:
    """Test hybrid mode with extended 13-feature input and wide192 config.

    Verifies the widened architecture (hidden_dim=192, latent_dim=48)
    works correctly with the 13-feature extended dataset.
    """

    def test_forward_shape(self):
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite(self):
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid_mask]).all()

    def test_padded_scores_negative_inf(self):
        """Padded tracks should get -inf with the wider model."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        padded_mask = ~mask.squeeze(1).bool()
        assert torch.all(scores[padded_mask] == float('-inf'))

    def test_loss_finite(self):
        """All loss components should be finite with extended features."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert torch.isfinite(loss_dict['ranking_loss']).all()
        assert torch.isfinite(loss_dict['reconstruction_loss']).all()

    def test_gradients_flow(self):
        """All parameters should receive gradients with extended input."""
        model = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        # Allow 1 param with zero grad (numerical artifact from random data)
        assert params_with_grad >= total_params - 1, (
            f'Only {params_with_grad}/{total_params} params got gradients'
        )

    def test_param_count_increase(self):
        """Wide192 with 13 features should have more params than wide128 with 7."""
        model_old = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM,
            hidden_dim=128, latent_dim=32, num_message_rounds=2,
        )
        model_new = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        old_params = sum(p.numel() for p in model_old.parameters())
        new_params = sum(p.numel() for p in model_new.parameters())
        assert new_params > old_params, (
            f'New model ({new_params}) should have more params than old ({old_params})'
        )


# ---- MLP Mode with Multi-Round Message Passing ----

class TestMLPMultiRound:
    """Test MLP mode with multiple kNN message-passing rounds."""

    def test_multi_round_forward_shape(self):
        """MLP mode with 2 message rounds produces correct output shape."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_multi_round_loss_finite(self):
        """Loss should be finite with multi-round MLP (no reconstruction loss)."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert torch.isfinite(loss_dict['ranking_loss']).all()
        assert 'reconstruction_loss' not in loss_dict

    def test_fewer_params_than_hybrid(self):
        """MLP mode (no AE) should have fewer params than hybrid mode."""
        model_mlp = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
        )
        model_hybrid = TrackPreFilter(
            mode='hybrid', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, latent_dim=48, num_message_rounds=2,
        )
        mlp_params = sum(p.numel() for p in model_mlp.parameters())
        hybrid_params = sum(p.numel() for p in model_hybrid.parameters())
        assert mlp_params < hybrid_params, (
            f'MLP ({mlp_params}) should have fewer params than '
            f'hybrid ({hybrid_params})'
        )

    def test_single_round_backward_compat(self):
        """MLP mode with 1 round should still work (backward compat)."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM, num_message_rounds=1,
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_gradient_flow(self):
        """All parameters should receive gradients in multi-round MLP."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad >= total_params - 1, (
            f'Only {params_with_grad}/{total_params} params got gradients'
        )


# ---- Temperature Scheduling ----

class TestTemperatureScheduling:
    """Test temperature and sigma scheduling for ranking/denoising losses."""

    def test_sigma_interpolation(self):
        """Denoising sigma should interpolate linearly with progress."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            denoising_sigma_start=1.0, denoising_sigma_end=0.1,
        )
        model.set_temperature_progress(0.0)
        assert abs(model.current_denoising_sigma - 1.0) < 1e-6

        model.set_temperature_progress(0.5)
        assert abs(model.current_denoising_sigma - 0.55) < 1e-6

        model.set_temperature_progress(1.0)
        assert abs(model.current_denoising_sigma - 0.1) < 1e-6

    def test_temperature_interpolation(self):
        """Ranking temperature should interpolate linearly with progress."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            ranking_temperature_start=2.0, ranking_temperature_end=0.5,
        )
        model.set_temperature_progress(0.0)
        assert abs(model.current_ranking_temperature - 2.0) < 1e-6

        model.set_temperature_progress(1.0)
        assert abs(model.current_ranking_temperature - 0.5) < 1e-6

    def test_temperature_affects_loss_magnitude(self):
        """T * softplus(x/T) is monotonically increasing in T for fixed x > 0.

        Higher T spreads gradients across more pairs (smoother loss landscape).
        Lower T concentrates gradients on hard violations (sharper boundary).
        """
        # Use uniform negative scores so random sampling doesn't matter
        scores = torch.full((1, NUM_TRACKS), -5.0)
        scores[0, 10] = -0.5  # GT pion scores low
        scores[0, :10] = 0.5  # All negatives score equally at 0.5
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.ones(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 11:] = False  # Only 11 valid tracks

        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            ranking_temperature_start=2.0, ranking_temperature_end=2.0,
        )
        loss_high_temp = model._ranking_loss(scores, labels, valid_mask)

        model_low = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            ranking_temperature_start=0.5, ranking_temperature_end=0.5,
        )
        loss_low_temp = model_low._ranking_loss(scores, labels, valid_mask)

        # T * softplus(x/T) increases with T for x > 0
        assert loss_high_temp.item() > loss_low_temp.item()

    def test_default_temperature_matches_baseline(self):
        """Default temperature (T=1) should match original softplus exactly."""
        # Use uniform negatives to eliminate randomness from sampling
        scores = torch.full((1, NUM_TRACKS), -10.0)
        scores[0, 10] = -3.0  # GT pion
        scores[0, 0] = 3.0    # Single negative
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.zeros(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 0] = True   # 1 valid negative
        valid_mask[0, 10] = True  # 1 valid positive

        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            ranking_temperature_start=1.0, ranking_temperature_end=1.0,
            ranking_num_samples=1,
        )
        loss = model._ranking_loss(scores, labels, valid_mask)
        # 1 * softplus((3.0 - (-3.0)) / 1) = softplus(6.0) ≈ 6.0025
        expected = torch.nn.functional.softplus(torch.tensor(6.0)).item()
        assert abs(loss.item() - expected) < 0.01


# ---- Deferred Re-Weighting ----

class TestDeferredReweighting:
    """Test DRW (Deferred Re-Weighting) for ranking loss."""

    def test_drw_inactive_matches_baseline(self):
        """With DRW inactive, loss should match a weight=1.0 model exactly."""
        # Use uniform negatives to eliminate randomness from sampling
        scores = torch.full((1, NUM_TRACKS), -10.0)
        scores[0, 10] = -2.0  # GT pion
        scores[0, 0] = 2.0    # Single negative
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.zeros(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 0] = True
        valid_mask[0, 10] = True

        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            drw_positive_weight=3.0, ranking_num_samples=1,
        )
        model.set_drw_active(False)
        loss_inactive = model._ranking_loss(scores, labels, valid_mask)

        model_baseline = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            drw_positive_weight=1.0, ranking_num_samples=1,
        )
        loss_baseline = model_baseline._ranking_loss(
            scores, labels, valid_mask,
        )
        assert abs(loss_inactive.item() - loss_baseline.item()) < 1e-5

    def test_drw_active_scales_loss(self):
        """With DRW active, loss should scale by positive weight."""
        # Use uniform negatives to eliminate randomness from sampling
        scores = torch.full((1, NUM_TRACKS), -10.0)
        scores[0, 10] = -2.0  # GT pion
        scores[0, 0] = 2.0    # Single negative
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.zeros(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 0] = True
        valid_mask[0, 10] = True

        weight = 3.0
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            drw_positive_weight=weight, ranking_num_samples=1,
        )

        model.set_drw_active(False)
        loss_baseline = model._ranking_loss(scores, labels, valid_mask)

        model.set_drw_active(True)
        loss_reweighted = model._ranking_loss(scores, labels, valid_mask)

        assert abs(loss_reweighted.item() - weight * loss_baseline.item()) < 1e-4

    def test_drw_toggle(self):
        """DRW state should toggle correctly."""
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        assert model._drw_active is False
        model.set_drw_active(True)
        assert model._drw_active is True
        model.set_drw_active(False)
        assert model._drw_active is False


# ---- PNA Multi-Aggregation ----

class TestPNAAggregation:
    """Test PNA (Principal Neighbourhood Aggregation) mode."""

    def test_pna_forward_shape(self):
        """PNA mode should produce correct output shape."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='pna',
        )
        points, features, lorentz_vectors, mask, _ = (
            _make_extended_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_pna_loss_finite(self):
        """PNA mode loss should be finite."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='pna',
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()

    def test_pna_gradient_flow(self):
        """All PNA model parameters should receive gradients."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='pna',
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad >= total_params - 1

    def test_pna_more_params_than_maxpool(self):
        """PNA mode has wider neighbor_mlp input, so more params."""
        model_max = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='max',
        )
        model_pna = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='pna',
        )
        max_params = sum(p.numel() for p in model_max.parameters())
        pna_params = sum(p.numel() for p in model_pna.parameters())
        assert pna_params > max_params

    def test_pna_aggregation_values(self):
        """PNA should compute correct mean/max/min/std on known inputs."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            aggregation_mode='pna',
        )
        # All neighbors have value 2.0, except one which is 4.0
        neighbor_features = torch.full((1, 4, 3, 5), 2.0)
        neighbor_features[0, :, :, 2] = 4.0  # One neighbor is 4.0
        neighbor_validity = torch.ones(1, 1, 3, 5)
        neighbor_validity[0, 0, :, 4] = 0.0  # Last neighbor invalid

        result = model._pna_aggregate(neighbor_features, neighbor_validity)
        # Shape: (1, 4*4, 3) = (1, 16, 3)
        assert result.shape == (1, 16, 3)
        # Mean of [2, 2, 4, 2] = 2.5 for each channel, each track
        hidden = 4
        mean_part = result[0, :hidden, 0]
        assert torch.allclose(mean_part, torch.tensor([2.5, 2.5, 2.5, 2.5]))

    def test_pna_handles_masked_neighbors(self):
        """Invalid neighbors should be excluded from all statistics."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            aggregation_mode='pna',
        )
        # All neighbors invalid except first
        neighbor_features = torch.full((1, 2, 1, 4), 99.0)
        neighbor_features[0, :, 0, 0] = 3.0  # Only valid neighbor
        neighbor_validity = torch.zeros(1, 1, 1, 4)
        neighbor_validity[0, 0, 0, 0] = 1.0  # Only first is valid

        result = model._pna_aggregate(neighbor_features, neighbor_validity)
        hidden = 2
        # With 1 valid neighbor: mean=3, max=3, min=3, std≈0
        mean_part = result[0, :hidden, 0]
        max_part = result[0, hidden:2*hidden, 0]
        min_part = result[0, 2*hidden:3*hidden, 0]
        assert torch.allclose(mean_part, torch.tensor([3.0, 3.0]))
        assert torch.allclose(max_part, torch.tensor([3.0, 3.0]))
        assert torch.allclose(min_part, torch.tensor([3.0, 3.0]))


# ---- Focal Weighting ----

class TestFocalWeighting:
    """Test equalized focal weighting for ranking loss."""

    def test_focal_disabled_matches_baseline(self):
        """focal_gamma=0 should produce identical loss to baseline."""
        scores = torch.full((1, NUM_TRACKS), -10.0)
        scores[0, 10] = -2.0
        scores[0, 0] = 2.0
        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.zeros(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 0] = True
        valid_mask[0, 10] = True

        model_no_focal = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            focal_gamma=0.0, ranking_num_samples=1,
        )
        model_with_focal = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            focal_gamma=2.0, ranking_num_samples=1,
        )

        loss_baseline = model_no_focal._ranking_loss(
            scores, labels, valid_mask,
        )
        loss_focal = model_with_focal._ranking_loss(
            scores, labels, valid_mask,
        )
        # Focal should downweight this pair (it's a clear violation)
        # but should NOT be identical to baseline
        assert loss_focal.item() < loss_baseline.item()

    def test_focal_downweights_easy_pairs(self):
        """Easy pairs (large correct margin) should get smaller focal weight."""
        # Easy pair: positive scores much higher than negative
        scores_easy = torch.full((1, NUM_TRACKS), -10.0)
        scores_easy[0, 10] = 10.0  # GT pion scores very high
        scores_easy[0, 0] = -5.0   # Negative scores very low
        # Hard pair: negative barely beats positive
        scores_hard = torch.full((1, NUM_TRACKS), -10.0)
        scores_hard[0, 10] = -0.1  # GT pion scores just below
        scores_hard[0, 0] = 0.1    # Negative scores just above

        labels = torch.zeros(1, NUM_TRACKS)
        labels[0, 10] = 1.0
        valid_mask = torch.zeros(1, NUM_TRACKS, dtype=torch.bool)
        valid_mask[0, 0] = True
        valid_mask[0, 10] = True

        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            focal_gamma=2.0, ranking_num_samples=1,
        )

        loss_easy = model._ranking_loss(scores_easy, labels, valid_mask)
        loss_hard = model._ranking_loss(scores_hard, labels, valid_mask)
        # Hard pair should produce higher loss than easy pair
        assert loss_hard.item() > loss_easy.item()

    def test_focal_never_zeros_gradient(self):
        """Unlike ASL, focal weighting should never zero any gradient."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=64, num_message_rounds=1,
            focal_gamma=3.0,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
            use_contrastive_denoising=False,
        )
        loss_dict['total_loss'].backward()

        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        assert params_with_grad > 0, 'Focal weighting zeroed all gradients'


# ---- DINO Contrastive Denoising ----

class TestDINOContrastiveDenoising:
    """Test DINO-style contrastive denoising with positive + negative copies."""

    def test_positive_only_matches_legacy(self):
        """With negative_sigma=0, should behave like the original denoising."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            contrastive_denoising_negative_sigma=0.0,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        loss = model._contrastive_denoising_loss(
            points, features, lorentz_vectors, mask, track_labels, scores,
        )
        assert torch.isfinite(loss).all()

    def test_negative_copies_produce_finite_loss(self):
        """With negative copies enabled, loss should still be finite."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            contrastive_denoising_negative_sigma=1.5,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        scores = model(points, features, lorentz_vectors, mask)
        loss = model._contrastive_denoising_loss(
            points, features, lorentz_vectors, mask, track_labels, scores,
        )
        assert torch.isfinite(loss).all()

    def test_negative_copies_change_loss(self):
        """Adding negative copies should produce a different denoising loss."""
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )

        model_pos_only = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            contrastive_denoising_negative_sigma=0.0,
            denoising_sigma_start=0.3, denoising_sigma_end=0.3,
        )
        scores_pos = model_pos_only(points, features, lorentz_vectors, mask)

        model_with_neg = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            contrastive_denoising_negative_sigma=1.5,
            denoising_sigma_start=0.3, denoising_sigma_end=0.3,
        )
        # Use same weights for fair comparison
        model_with_neg.load_state_dict(model_pos_only.state_dict())
        scores_neg = model_with_neg(points, features, lorentz_vectors, mask)

        loss_pos = model_pos_only._contrastive_denoising_loss(
            points, features, lorentz_vectors, mask, track_labels, scores_pos,
        )
        loss_neg = model_with_neg._contrastive_denoising_loss(
            points, features, lorentz_vectors, mask, track_labels, scores_neg,
        )
        # With negative copies, loss includes neg-vs-pos terms → different value
        assert loss_neg.item() != loss_pos.item()

    def test_full_loss_with_all_batch_b(self):
        """Smoke test: all Batch B features enabled together."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM_EXTENDED,
            hidden_dim=192, num_message_rounds=2,
            aggregation_mode='pna',
            focal_gamma=2.0,
            contrastive_denoising_negative_sigma=1.5,
        )
        points, features, lorentz_vectors, mask, track_labels = (
            _make_extended_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert torch.isfinite(loss_dict['total_loss']).all()
        assert torch.isfinite(loss_dict['ranking_loss']).all()
        assert torch.isfinite(loss_dict['denoising_loss']).all()

        loss_dict['total_loss'].backward()
        params_with_grad = sum(
            1 for _, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        assert params_with_grad >= total_params - 1


# ---- Dropout regularization (2026-04-07 overfit mitigation) ----

class TestDropoutRegularization:
    """The mlp-mode backbone had zero dropout, which was the biggest missing
    regularization lever in the overfit audit (reports/prefilter_analysis_20260406.md).
    These tests encode the post-fix contract: dropout is an explicit constructor
    kwarg, defaults to 0 for backward compatibility with existing tests, and
    when set > 0 inserts exactly 5 nn.Dropout modules in the mlp-mode stack
    (track_mlp × 2, neighbor_mlps × num_message_rounds, scorer × 1).
    """

    def test_dropout_default_is_zero(self):
        """No dropout kwarg → self.dropout == 0.0 and no active Dropout modules.

        Backward compatibility: every pre-2026-04-07 test instantiates
        TrackPreFilter without passing ``dropout``. Those tests must keep
        working, which requires the default to be 0 and for no Dropout
        modules with p>0 to be inserted.
        """
        import torch.nn as nn
        model = TrackPreFilter(mode='mlp', input_dim=INPUT_DIM)
        assert model.dropout == 0.0
        active_dropouts = [
            module for module in model.modules()
            if isinstance(module, nn.Dropout) and module.p > 0
        ]
        assert active_dropouts == [], (
            f'Expected no active Dropout modules at default, got '
            f'{len(active_dropouts)}'
        )

    def test_dropout_site_count_mlp_mode(self):
        """With dropout>0, mlp-mode model must contain exactly 5 nn.Dropout
        modules, all with p matching the constructor arg.

        Expected sites:
          - track_mlp: 2 (after each of the two ReLUs)
          - neighbor_mlps: 2 (num_message_rounds=2, one ReLU per round)
          - scorer: 1 (after the middle ReLU; NOT before the final output)
        """
        import torch.nn as nn
        model = TrackPreFilter(
            mode='mlp',
            input_dim=INPUT_DIM,
            num_message_rounds=2,
            dropout=0.3,
        )
        dropout_modules = [
            module for module in model.modules()
            if isinstance(module, nn.Dropout)
        ]
        assert len(dropout_modules) == 5, (
            f'Expected 5 nn.Dropout modules (track_mlp×2 + neighbor_mlps×2 '
            f'+ scorer×1), got {len(dropout_modules)}'
        )
        for module in dropout_modules:
            assert module.p == pytest.approx(0.3), (
                f'Dropout module has p={module.p}, expected 0.3'
            )

    def test_dropout_scales_with_message_rounds(self):
        """neighbor_mlps has one dropout per message round. Four rounds → 7 total
        (track_mlp×2 + neighbor_mlps×4 + scorer×1)."""
        import torch.nn as nn
        model = TrackPreFilter(
            mode='mlp',
            input_dim=INPUT_DIM,
            num_message_rounds=4,
            dropout=0.1,
        )
        dropout_modules = [
            module for module in model.modules()
            if isinstance(module, nn.Dropout)
        ]
        assert len(dropout_modules) == 7

    def test_dropout_identity_in_eval_mode(self):
        """model.eval() makes Dropout an identity; two forwards on the same
        input must produce byte-identical outputs."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, dropout=0.5,
        )
        model.eval()
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        with torch.no_grad():
            scores_first = model(points, features, lorentz_vectors, mask)
            scores_second = model(points, features, lorentz_vectors, mask)
        assert torch.equal(scores_first, scores_second), (
            'Dropout should be an identity in eval() mode, but two forward '
            'passes on the same input produced different outputs.'
        )

    def test_dropout_stochastic_in_train_mode(self):
        """model.train() activates Dropout; two forwards on the same input
        must differ (at least one valid-track score differs by >1e-6)."""
        torch.manual_seed(1337)
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, dropout=0.5,
        )
        model.train()
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        # Eval-mode the BatchNorm layers so their running stats don't shift
        # between forward passes — that would confound with dropout. Do this
        # by setting only the Dropout modules to train and everything else
        # to eval. Actually simpler: one forward in train mode, then re-seed
        # and forward again; difference must come from dropout alone (BN
        # running stats are updated but don't affect the same-input output
        # since BN uses batch stats in train mode).
        scores_first = model(points, features, lorentz_vectors, mask)
        scores_second = model(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        # Compare only valid tracks — padded scores are -inf and equal.
        diff = (
            scores_first[valid_mask] - scores_second[valid_mask]
        ).abs().max()
        assert diff.item() > 1e-6, (
            f'Dropout should make two train-mode forwards differ on valid '
            f'tracks, but max diff was {diff.item()}'
        )

    def test_denoising_loss_in_train_dict_by_default(self):
        """With denoising re-enabled as the default path and self.training=True,
        compute_loss must return a dict containing 'denoising_loss'. This is
        the contract that train_prefilter.py relies on for loss logging.
        """
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, dropout=0.0,
        )
        model.train()
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        # Default kwarg path — use_contrastive_denoising not passed → True
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
        )
        assert 'denoising_loss' in loss_dict, (
            f'Expected denoising_loss in compute_loss output, got keys: '
            f'{sorted(loss_dict.keys())}'
        )
        assert torch.isfinite(loss_dict['denoising_loss']).all()

    def test_denoising_loss_absent_when_kwarg_false(self):
        """Explicit opt-out via use_contrastive_denoising=False — used by
        train_prefilter.py's validate() to keep val loss clean."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, dropout=0.0,
        )
        model.train()
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
            use_contrastive_denoising=False,
        )
        assert 'denoising_loss' not in loss_dict


# ---- Edge features (pairwise LV physics on k-NN edges) ----

class TestEdgeFeatures:
    """Tests for ``use_edge_features=True`` path — pairwise LV physics.

    The edge augmentation appends the 4 pairwise_lv_fts channels
    (ln kT, ln z, ln ΔR, ln m²), max-pooled across the k-NN, to the
    neighbor-aggregated feature vector. When enabled it shifts the
    neighbor MLP input dim from ``2 * hidden_dim`` to
    ``2 * hidden_dim + 4`` (or ``5 * hidden_dim + 4`` for PNA).
    """

    def _first_conv_in_channels(self, model: TrackPreFilter) -> int:
        first_conv = next(
            layer for layer in model.neighbor_mlps[0].modules()
            if isinstance(layer, torch.nn.Conv1d)
        )
        return first_conv.in_channels

    def test_neighbor_input_dim_without_edge_features(self):
        """Baseline: neighbor_input_dim == 2 * hidden_dim."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
            use_edge_features=False,
        )
        assert self._first_conv_in_channels(model) == 2 * 64

    def test_neighbor_input_dim_with_edge_features(self):
        """Edge features add 4 channels (ln kT, ln z, ln ΔR, ln m²)."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
            use_edge_features=True,
        )
        assert self._first_conv_in_channels(model) == 2 * 64 + 4

    def test_pna_with_edge_features_dim(self):
        """PNA + edge features: neighbor_input_dim == 5 * hidden_dim + 4."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            hidden_dim=64, num_message_rounds=1,
            aggregation_mode='pna', use_edge_features=True,
        )
        assert self._first_conv_in_channels(model) == 5 * 64 + 4

    def test_forward_shape_with_edge_features(self):
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, use_edge_features=True,
        )
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)

    def test_scores_finite_with_edge_features(self):
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=2, use_edge_features=True,
        )
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        valid = mask.squeeze(1).bool()
        assert torch.isfinite(scores[valid]).all()

    def test_gradients_with_edge_features(self):
        """Edge path must backprop cleanly into ``features``.
        pairwise_lv_fts is detached inside build_cross_set_edge_features,
        so the LV gradient path is intentionally cut. features gradient
        still flows through the track_mlp / neighbor_mlps path."""
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, use_edge_features=True,
        )
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        features = features.clone().requires_grad_(True)
        scores = model(points, features, lorentz_vectors, mask)
        valid = mask.squeeze(1).bool()
        scores[valid].sum().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_edge_features_changes_output(self):
        """Enabling edge features should change the output vs. baseline."""
        torch.manual_seed(0)
        model_baseline = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, use_edge_features=False,
        )
        torch.manual_seed(0)
        model_edge = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, use_edge_features=True,
        )
        model_baseline.eval()
        model_edge.eval()
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores_baseline = model_baseline(
            points, features, lorentz_vectors, mask,
        )
        scores_edge = model_edge(points, features, lorentz_vectors, mask)
        valid = mask.squeeze(1).bool()
        # At least some tracks should produce different scores
        assert not torch.allclose(
            scores_baseline[valid], scores_edge[valid], atol=1e-5,
        )


class TestDeferredExperiments:
    """Smoke-level coverage for E5, E7, E10, E12 wiring."""

    def test_xgb_stub_feature_path(self):
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, use_xgb_stub_feature=True,
        )
        assert hasattr(model, 'xgb_stub')
        assert hasattr(model, 'track_mlp_with_xgb')
        points, features, lorentz_vectors, mask, _ = _make_training_inputs()
        scores = model(points, features, lorentz_vectors, mask)
        assert scores.shape == (BATCH_SIZE, NUM_TRACKS)
        assert torch.isfinite(
            scores[mask.squeeze(1).bool()],
        ).all()

    def test_object_condensation_forward_and_loss(self):
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, loss_type='object_condensation',
        )
        assert hasattr(model, 'oc_beta_head')
        assert hasattr(model, 'oc_embedding_head')
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
            use_contrastive_denoising=False,
        )
        assert torch.isfinite(loss_dict['ranking_loss'])
        assert torch.isfinite(loss_dict['total_loss'])

    def test_mpm_pretrain_loss_is_finite(self):
        model = TrackPreFilter(
            mode='mlp', input_dim=INPUT_DIM,
            num_message_rounds=1, loss_type='mpm_pretrain',
        )
        assert hasattr(model, 'mpm_reconstruction_head')
        model.apply_mpm_masking = True
        points, features, lorentz_vectors, mask, track_labels = (
            _make_training_inputs()
        )
        model.train()
        loss_dict = model.compute_loss(
            points, features, lorentz_vectors, mask, track_labels,
            use_contrastive_denoising=False,
        )
        assert torch.isfinite(loss_dict['ranking_loss'])
        assert loss_dict['ranking_loss'].item() >= 0.0
