"""Network wrapper for TrackPreFilter (Stage 1 of two-stage pipeline).

Default configuration: MLP mode with hidden_dim=256, k=16,
num_message_rounds=3, ranking_num_samples=50, edge features ON, and
P1 per-feature embedding ON (feature_embed_mode='per_feature',
feature_embed_dim=32) — the winning configuration from the 2026-04-23
expressiveness sweep (val P@256 = 0.8976 vs 0.8904 for the pre-P1 E2a
anchor). Pass ``--feature-embed-mode none`` to reproduce the anchor.

Input features (16): px, py, pz, eta, phi, charge, dxy_significance,
log_dz_significance, normalized_chi2, log_pt_error, n_valid_pixel_hits,
dca_significance, log_covariance_phi_phi, log_covariance_lambda_lambda,
log_pt, relative_pt_error.

Key fixes over 13-feature version:
- dz_significance log-transformed (was 99.9% clipped by auto-standardization)
- pt_error, covariance_phi_phi, covariance_lambda_lambda log-transformed
- Added normalized_chi2 (#1 CMS track quality feature)
- Added log(pT) and relative pT error (standard in CMS/ATLAS taggers)
"""

from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    """Build TrackPreFilter with default wide192+2rounds config."""
    # Pop unused args from other heads
    for unused_arg in [
        'pretrained_backbone_path', 'backbone_mode',
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'no_object_weight', 'num_decoder_layers', 'num_queries',
        'focal_bce_weight', 'potential_loss_weight',
        'beta_loss_weight', 'clustering_dim',
        'per_track_loss_weight', 'refinement_loss_weight',
        'num_enrichment_layers',
        # P6 two-tier-only kwargs: only consumed by
        # networks/lowpt_tau_TwoTierPreFilter.py; silently dropped here
        # so the single-tier path can be invoked from the same training
        # CLI without argparse divergence.
        'two_tier_top_n', 'two_tier_coarse_hidden_dim',
        'two_tier_refine_hidden_dim', 'two_tier_coarse_neighbors',
        'two_tier_refine_neighbors', 'two_tier_coarse_rounds',
        'two_tier_refine_rounds', 'two_tier_composite_offset',
    ]:
        kwargs.pop(unused_arg, None)

    input_dim = len(data_config.input_dicts['pf_features'])

    configuration = dict(
        mode='mlp',
        input_dim=input_dim,
        hidden_dim=kwargs.pop('hidden_dim', 256),
        num_neighbors=kwargs.pop('num_neighbors', 16),
        # E2a winning config (prefilter_campaign_20260419): k=16, r=3,
        # edge features ON. Baseline R@200 bump 0.9166 → 0.9227.
        num_message_rounds=kwargs.pop('num_message_rounds', 3),
        aggregation_mode=kwargs.pop('aggregation_mode', 'max'),
        use_edge_features=kwargs.pop('use_edge_features', True),
        loss_type=kwargs.pop('loss_type', 'pairwise'),
        logit_adjust_tau=kwargs.pop('logit_adjust_tau', 1.0),
        listwise_temperature=kwargs.pop('listwise_temperature', 1.0),
        use_xgb_stub_feature=kwargs.pop('use_xgb_stub_feature', False),
        clustering_dim=kwargs.pop('clustering_dim', 8),
        # Expressiveness plug-ins (prefilter P@256 sweep). P1 is the new
        # baseline as of the 2026-04-23 sweep (P@256 0.8976 vs 0.8904 E2a);
        # the remaining heads (P2/P3/P4) stayed below P1 and are still off
        # by default. A checkpoint trained with P1 on diverges from the
        # pre-P1 E2a state_dict, so a stock E2a wrapper cannot load it.
        feature_embed_mode=kwargs.pop('feature_embed_mode', 'per_feature'),
        feature_embed_dim=kwargs.pop('feature_embed_dim', 32),
        use_feature_gate=kwargs.pop('use_feature_gate', False),
        feature_gate_bottleneck=kwargs.pop('feature_gate_bottleneck', 16),
        use_film_head=kwargs.pop('use_film_head', False),
        film_context_dim=kwargs.pop('film_context_dim', 32),
        use_soft_attention_aggregation=kwargs.pop(
            'use_soft_attention_aggregation', False,
        ),
        soft_attention_bottleneck=kwargs.pop(
            'soft_attention_bottleneck', 64,
        ),
        ranking_num_samples=50,
        # Dropout rate for the mlp-mode MLP hidden layers (track_mlp,
        # neighbor_mlps, scorer). 0.1 is the 2026-04-07 overfit-mitigation
        # default. Pass --dropout on the CLI to override (0.0 disables).
        dropout=kwargs.pop('dropout', 0.1),
        # -------------------------------------------------------------
        # Loss-schedule configuration (post-2026-04-06 ablation state)
        # -------------------------------------------------------------
        # Clean baseline: plain softplus pairwise ranking with no schedules.
        # Contrastive denoising is re-enabled as of 2026-04-07 (it's a
        # GT-invariance regularizer; disabling it was a side effect of the
        # DRW ablation). DRW and ranking-temperature annealing stay OFF —
        # the prefilter analysis report (reports/prefilter_analysis_20260406.md)
        # identified DRW's epoch-31 activation as the loss discontinuity
        # and the correlated val-R@200 inflection; we do not want them back.
        #
        # Original values (re-enable to restore the Kukleva/DRW recipe):
        #   ranking_temperature_start=2.0, ranking_temperature_end=0.5
        #   drw_warmup_fraction=0.3,       drw_positive_weight=2.0
        #
        # Temperature annealing OFF: T held at 1.0 throughout training, so
        # the loss reduces to plain `softplus(s_neg - s_pos)`. Removes the
        # boundary-sharpening pressure that correlates with late-epoch overfit.
        ranking_temperature_start=1.0,
        ranking_temperature_end=1.0,
        # Denoising sigma schedule — LIVE values. The contrastive denoising
        # loss is enabled in train_prefilter.py's train path (and disabled
        # in its validate path so val loss remains clean). Large → small
        # gives a curriculum: easy positives first, then hard ones near the
        # real GT manifold.
        denoising_sigma_start=1.0,
        denoising_sigma_end=0.1,
        # DRW OFF: warmup=1.0 means the DRW activation epoch is 1.0 * epochs,
        # i.e. never for a 100-epoch run. drw_positive_weight=1.0 is a second
        # safeguard: even if DRW somehow activates, the scalar multiplier
        # is a no-op.
        drw_warmup_fraction=1.0,
        drw_positive_weight=1.0,
    )
    configuration.update(**kwargs)
    _logger.info('TrackPreFilter config: %s' % str(configuration))

    model = TrackPreFilter(**configuration)

    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'TrackPreFilter: {total_params:,} params (all trainable)')

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {
            key: ((1,) + shape[1:])
            for key, shape in data_config.input_shapes.items()
        },
        'output_names': ['loss'],
        'dynamic_axes': {
            **{
                key: {0: 'N', 2: 'n_' + key.split('_')[0]}
                for key in data_config.input_names
            },
            **{'loss': {0: 'N'}},
        },
    }

    return model, model_info
