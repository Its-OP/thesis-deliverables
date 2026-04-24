from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):
    input_dim = len(data_config.input_dicts['pf_features'])

    configuration = dict(
        mode='mlp',
        input_dim=input_dim,
        hidden_dim=kwargs.pop('hidden_dim', 256),
        num_neighbors=kwargs.pop('num_neighbors', 16),
        num_message_rounds=kwargs.pop('num_message_rounds', 3),
        aggregation_mode=kwargs.pop('aggregation_mode', 'max'),
        use_edge_features=kwargs.pop('use_edge_features', True),
        loss_type=kwargs.pop('loss_type', 'pairwise'),
        logit_adjust_tau=kwargs.pop('logit_adjust_tau', 1.0),
        listwise_temperature=kwargs.pop('listwise_temperature', 1.0),
        clustering_dim=kwargs.pop('clustering_dim', 8),
        feature_embed_mode=kwargs.pop('feature_embed_mode', 'per_feature'),
        feature_embed_dim=kwargs.pop('feature_embed_dim', 32),
        ranking_num_samples=50,
        dropout=kwargs.pop('dropout', 0.1),
        # No LR-schedule curriculum: softplus pairwise with T=1 throughout.
        ranking_temperature_start=1.0,
        ranking_temperature_end=1.0,
        # Denoising sigma: curriculum easy → hard positives near GT manifold.
        denoising_sigma_start=1.0,
        denoising_sigma_end=0.1,
        # DRW OFF: warmup_fraction=1.0 defers activation past any run length.
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
