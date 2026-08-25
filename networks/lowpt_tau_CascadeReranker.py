import torch

from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def infer_stage1_kwargs(stage1_state, stage1_num_neighbors=16):
    """Recover TrackPreFilter __init__ kwargs from state_dict shapes.
    num_neighbors is runtime-only (not in state dict); pass it in.
    """
    first_layer_key = 'track_mlp.0.weight'
    if first_layer_key not in stage1_state:
        raise ValueError(
            f'Cannot infer Stage 1 dimensions: expected key '
            f'"{first_layer_key}" not found in state dict',
        )
    first_layer_weight = stage1_state[first_layer_key]
    inferred_hidden_dim = first_layer_weight.shape[0]

    # P1 per-feature embedding inflates track_mlp input from F to F*E;
    # detect by module-key prefix and recover E from LayerNorm shape.
    has_feature_embedder = any(
        key.startswith('feature_embedder.') for key in stage1_state
    )
    if has_feature_embedder:
        feature_embed_mode = 'per_feature'
        feature_embed_dim = stage1_state[
            'feature_embedder.layer_norm.weight'
        ].shape[0]
        inferred_input_dim = first_layer_weight.shape[1] // feature_embed_dim
    else:
        feature_embed_mode = 'none'
        feature_embed_dim = 32
        inferred_input_dim = first_layer_weight.shape[1]

    stage1_round_indices = {
        int(key.split('.')[1]) for key in stage1_state
        if key.startswith('neighbor_mlps.')
    }
    inferred_num_message_rounds = (
        max(stage1_round_indices) + 1 if stage1_round_indices else 0
    )
    first_neighbor_weight = stage1_state.get('neighbor_mlps.0.0.weight')
    if first_neighbor_weight is not None:
        neighbor_in_dim = first_neighbor_weight.shape[1]
        expected_no_edges = 2 * inferred_hidden_dim
        if neighbor_in_dim == expected_no_edges:
            inferred_use_edge = False
        elif neighbor_in_dim == expected_no_edges + 4:
            inferred_use_edge = True
        else:
            raise ValueError(
                f'Cannot infer use_edge_features: neighbor_mlps.0.0 '
                f'in_dim={neighbor_in_dim}, expected '
                f'{expected_no_edges} (no edges) or '
                f'{expected_no_edges + 4} (edges)',
            )
    else:
        inferred_use_edge = False

    # Dropout>0 shifts track_mlp Sequential indices by 1: with dropout
    # track_mlp.4.weight is a Conv1d (3D); without it's BN (1D).
    second_conv_weight = stage1_state.get('track_mlp.4.weight')
    inferred_dropout = (
        0.1 if second_conv_weight is not None and second_conv_weight.dim() == 3
        else 0.0
    )

    return dict(
        mode='mlp',
        input_dim=inferred_input_dim,
        hidden_dim=inferred_hidden_dim,
        num_message_rounds=inferred_num_message_rounds,
        num_neighbors=stage1_num_neighbors,
        use_edge_features=inferred_use_edge,
        dropout=inferred_dropout,
        feature_embed_mode=feature_embed_mode,
        feature_embed_dim=feature_embed_dim,
    )


def load_stage2_init(stage2, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    stage2.load_state_dict(checkpoint['model_state_dict'])
    _logger.info(f'Stage 2 warm-initialized from: {checkpoint_path}')


def get_model(data_config, **kwargs):
    stage1_checkpoint = kwargs.pop('stage1_checkpoint', None)
    stage2_init_checkpoint = kwargs.pop('stage2_init_checkpoint', None)
    top_k1 = kwargs.pop('top_k1', 600)

    stage2_embed_dim = kwargs.pop('stage2_embed_dim', 128)
    stage2_num_heads = kwargs.pop('stage2_num_heads', 4)
    stage2_num_layers = kwargs.pop('stage2_num_layers', 3)
    stage2_pair_embed_dims = kwargs.pop('stage2_pair_embed_dims', [64, 64])
    stage2_pair_extra_dim = kwargs.pop('stage2_pair_extra_dim', 6)
    stage2_pair_embed_mode = kwargs.pop('stage2_pair_embed_mode', 'concat')
    stage2_ffn_ratio = kwargs.pop('stage2_ffn_ratio', 4)
    stage2_dropout = kwargs.pop('stage2_dropout', 0.1)
    stage2_loss_mode = kwargs.pop('stage2_loss_mode', 'pairwise')
    stage2_rs_at_k_target = kwargs.pop('stage2_rs_at_k_target', 200)

    input_dim = len(data_config.input_dicts['pf_features'])

    if stage1_checkpoint is None:
        raise ValueError(
            'stage1_checkpoint is required. '
            'Pass --stage1-checkpoint to train_cascade.py'
        )

    _logger.info(f'Loading Stage 1 from: {stage1_checkpoint}')
    checkpoint = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)
    stage1_state = checkpoint.get('model_state_dict', checkpoint)

    stage1_num_neighbors = kwargs.pop('stage1_num_neighbors', 16)
    stage1_kwargs = infer_stage1_kwargs(stage1_state, stage1_num_neighbors)
    if stage1_kwargs['input_dim'] != input_dim:
        raise ValueError(
            f'Stage 1 checkpoint input_dim={stage1_kwargs["input_dim"]} '
            f'does not match data config input_dim={input_dim}. '
            f'Retrain Stage 1 on the current feature set.',
        )
    _logger.info(f'Stage 1 config from checkpoint: {stage1_kwargs}')

    stage1 = TrackPreFilter(**stage1_kwargs)
    stage1.load_state_dict(stage1_state, strict=False)
    _logger.info('Stage 1 loaded successfully')

    stage1_params = sum(p.numel() for p in stage1.parameters())
    _logger.info(f'Stage 1: {stage1_params:,} params (frozen)')

    stage2 = CascadeReranker(
        input_dim=input_dim,
        embed_dim=stage2_embed_dim,
        num_heads=stage2_num_heads,
        num_layers=stage2_num_layers,
        pair_input_dim=4,
        pair_extra_dim=stage2_pair_extra_dim,
        pair_embed_dims=stage2_pair_embed_dims,
        pair_embed_mode=stage2_pair_embed_mode,
        ffn_ratio=stage2_ffn_ratio,
        dropout=stage2_dropout,
        ranking_num_samples=50,
        ranking_temperature=1.0,
        loss_mode=stage2_loss_mode,
        rs_at_k_target=stage2_rs_at_k_target,
    )
    if stage2_init_checkpoint is not None:
        load_stage2_init(stage2, stage2_init_checkpoint)
    stage2_params = sum(p.numel() for p in stage2.parameters())
    _logger.info(f'Stage 2 (CascadeReranker): {stage2_params:,} params (trainable)')

    model = CascadeModel(stage1=stage1, stage2=stage2, top_k1=top_k1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'Cascade total: {total_params:,} params | Trainable: {trainable_params:,}')

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
