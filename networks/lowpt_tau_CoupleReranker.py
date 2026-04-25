import torch

from networks.lowpt_tau_CascadeReranker import infer_stage1_kwargs
from utils.couple_features import COUPLE_FEATURE_DIM, PAIR_PHYSICS_V3_EXTRA_DIM
from weaver.nn.model.CascadeModel import CascadeModel
from weaver.nn.model.CascadeReranker import CascadeReranker
from weaver.nn.model.CoupleCascadeModel import CoupleCascadeModel
from weaver.nn.model.CoupleReranker import CoupleReranker
from weaver.nn.model.TrackPreFilter import TrackPreFilter
from weaver.utils.logger import _logger


def _build_frozen_cascade(
    cascade_checkpoint_path: str,
    input_dim: int,
) -> CascadeModel:
    """Rebuild Stage 1 + Stage 2 from a single bundled checkpoint."""
    _logger.info(f'Loading cascade checkpoint: {cascade_checkpoint_path}')
    checkpoint = torch.load(
        cascade_checkpoint_path, map_location='cpu', weights_only=False,
    )
    cascade_state_dict = checkpoint['model_state_dict']
    saved_args = checkpoint.get('args', {})

    # Strip the "stage1." prefix so infer_stage1_kwargs (P1-aware) can read it.
    stage1_state = {
        key[len('stage1.'):]: value
        for key, value in cascade_state_dict.items()
        if key.startswith('stage1.')
    }
    if not stage1_state:
        raise ValueError(
            f'Cannot infer Stage 1: no stage1.* keys in {cascade_checkpoint_path}'
        )
    stage1_kwargs = infer_stage1_kwargs(stage1_state, stage1_num_neighbors=16)
    if stage1_kwargs['input_dim'] != input_dim:
        raise ValueError(
            f'Cascade checkpoint Stage 1 input_dim={stage1_kwargs["input_dim"]} '
            f'does not match data config input_dim={input_dim}.'
        )
    _logger.info(f'Stage 1 config from checkpoint: {stage1_kwargs}')
    stage1 = TrackPreFilter(**stage1_kwargs)

    pair_embed_dims_raw = saved_args.get('stage2_pair_embed_dims', '64,64,64')
    if isinstance(pair_embed_dims_raw, str):
        pair_embed_dims = [int(x) for x in pair_embed_dims_raw.split(',')]
    else:
        pair_embed_dims = pair_embed_dims_raw
    stage2 = CascadeReranker(
        input_dim=input_dim,
        embed_dim=saved_args.get('stage2_embed_dim', 512),
        num_heads=saved_args.get('stage2_num_heads', 8),
        num_layers=saved_args.get('stage2_num_layers', 2),
        pair_input_dim=4,
        pair_extra_dim=saved_args.get('stage2_pair_extra_dim', 6),
        pair_embed_dims=pair_embed_dims,
        pair_embed_mode=saved_args.get('stage2_pair_embed_mode', 'concat'),
        ffn_ratio=saved_args.get('stage2_ffn_ratio', 4),
        dropout=saved_args.get('stage2_dropout', 0.1),
        loss_mode=saved_args.get('stage2_loss_mode', 'pairwise'),
        rs_at_k_target=saved_args.get('stage2_rs_at_k_target', 200),
    )

    top_k1 = saved_args.get('top_k1', 256)
    cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=top_k1)
    cascade.load_state_dict(cascade_state_dict)
    _logger.info(f'Cascade loaded (top_k1={top_k1})')
    return cascade


def get_model(data_config, **kwargs):
    cascade_checkpoint = kwargs.pop('cascade_checkpoint', None)
    top_k2 = kwargs.pop('top_k2', 50)
    k_values_tracks_raw = kwargs.pop('k_values_tracks', '30,50,75,100,200')
    if isinstance(k_values_tracks_raw, str):
        k_values_tracks = tuple(
            int(x) for x in k_values_tracks_raw.split(',')
        )
    else:
        k_values_tracks = tuple(k_values_tracks_raw)

    couple_hidden_dim = kwargs.pop('couple_hidden_dim', 256)
    couple_num_residual_blocks = kwargs.pop('couple_num_residual_blocks', 4)
    couple_dropout = kwargs.pop('couple_dropout', 0.1)
    couple_ranking_num_samples = kwargs.pop('couple_ranking_num_samples', 50)
    couple_ranking_temperature = kwargs.pop('couple_ranking_temperature', 1.0)
    couple_label_smoothing = kwargs.pop('couple_label_smoothing', 0.10)
    couple_projector_dim = kwargs.pop('couple_projector_dim', 32)

    if cascade_checkpoint is None:
        raise ValueError(
            'cascade_checkpoint is required. Pass --cascade-checkpoint to '
            'train_couple_reranker.py'
        )

    input_dim = len(data_config.input_dicts['pf_features'])

    cascade = _build_frozen_cascade(cascade_checkpoint, input_dim=input_dim)
    cascade_params = sum(p.numel() for p in cascade.parameters())
    _logger.info(f'Frozen cascade: {cascade_params:,} params')

    rest_dim = COUPLE_FEATURE_DIM + PAIR_PHYSICS_V3_EXTRA_DIM - 32  # = 24
    couple_reranker = CoupleReranker(
        hidden_dim=couple_hidden_dim,
        num_residual_blocks=couple_num_residual_blocks,
        dropout=couple_dropout,
        ranking_num_samples=couple_ranking_num_samples,
        ranking_temperature=couple_ranking_temperature,
        label_smoothing=couple_label_smoothing,
        couple_projector_dim=couple_projector_dim,
        rest_dim=rest_dim,
    )
    couple_params = sum(p.numel() for p in couple_reranker.parameters())
    _logger.info(
        f'CoupleReranker: {couple_params:,} params (trainable, '
        f'hidden_dim={couple_hidden_dim}, '
        f'num_residual_blocks={couple_num_residual_blocks}, '
        f'projector_dim={couple_projector_dim}, '
        f'input_dim={couple_reranker.input_dim})'
    )

    model = CoupleCascadeModel(
        cascade=cascade,
        couple_reranker=couple_reranker,
        top_k2=top_k2,
        k_values_tracks=k_values_tracks,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    _logger.info(
        f'CoupleCascadeModel total: {total_params:,} params | '
        f'Trainable: {trainable_params:,}'
    )

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
