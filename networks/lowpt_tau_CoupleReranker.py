"""Network wrapper for CoupleCascadeModel (frozen 2-stage cascade + per-couple Stage 3 head).

Builds the post-ParT direction-A reranker from
``reports/triplet_reranking/triplet_research_plan_20260408.md``:

    Stage 1 (TrackPreFilter)            ← frozen, loaded from cascade checkpoint
    Stage 2 (CascadeReranker / ParT)    ← frozen, loaded from cascade checkpoint
    Stage 3 (CoupleReranker)            ← trainable, freshly initialized

Unlike ``lowpt_tau_CascadeReranker.py`` (which loads only Stage 1 from a
checkpoint and trains a fresh Stage 2), this wrapper loads BOTH Stage 1 and
Stage 2 from a single trained cascade checkpoint and freezes them. Stage 1
hidden_dim is auto-inferred from the saved state dict; Stage 2 architecture
kwargs come from the cascade checkpoint's saved ``args`` dict.

Usage:
    python train_couple_reranker.py \\
        --network networks/lowpt_tau_CoupleReranker.py \\
        --cascade-checkpoint models/debug_checkpoints/cascade_soap_*/checkpoints/best_model.pt \\
        --top-k2 50
"""
import torch

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
    """Reconstruct the frozen 2-stage cascade from a single checkpoint.

    Stage 1 is rebuilt from the inferred hidden_dim (the same auto-detection
    pattern used by ``lowpt_tau_CascadeReranker.py``). Stage 2 is rebuilt
    from the saved CLI args dict that ``train_cascade.py`` writes into the
    checkpoint. The full cascade state dict is then loaded over the rebuilt
    object so both stages get the trained weights.
    """
    _logger.info(f'Loading cascade checkpoint: {cascade_checkpoint_path}')
    checkpoint = torch.load(
        cascade_checkpoint_path, map_location='cpu', weights_only=False,
    )
    cascade_state_dict = checkpoint['model_state_dict']
    saved_args = checkpoint.get('args', {})

    # ---- Stage 1: infer hidden_dim from state dict shape ----
    stage1_first_key = 'stage1.track_mlp.0.weight'
    if stage1_first_key not in cascade_state_dict:
        raise ValueError(
            f'Cannot infer Stage 1 dimensions: expected key '
            f'"{stage1_first_key}" not found in {cascade_checkpoint_path}'
        )
    stage1_first_weight = cascade_state_dict[stage1_first_key]
    stage1_hidden_dim = stage1_first_weight.shape[0]
    stage1_input_dim = stage1_first_weight.shape[1]
    if stage1_input_dim != input_dim:
        raise ValueError(
            f'Cascade checkpoint Stage 1 input_dim={stage1_input_dim} does '
            f'not match data config input_dim={input_dim}.'
        )
    _logger.info(
        f'Stage 1 config: hidden_dim={stage1_hidden_dim}, '
        f'input_dim={stage1_input_dim}'
    )
    stage1 = TrackPreFilter(
        mode='mlp',
        input_dim=stage1_input_dim,
        hidden_dim=stage1_hidden_dim,
        num_message_rounds=2,
        num_neighbors=16,
    )

    # ---- Stage 2: rebuild from the saved args dict ----
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
    _logger.info(
        f'Stage 2 config: embed_dim={saved_args.get("stage2_embed_dim", 512)}, '
        f'num_layers={saved_args.get("stage2_num_layers", 2)}, '
        f'num_heads={saved_args.get("stage2_num_heads", 8)}'
    )

    # ---- Wrap in CascadeModel and load the full state dict ----
    top_k1 = saved_args.get('top_k1', 256)
    cascade = CascadeModel(stage1=stage1, stage2=stage2, top_k1=top_k1)
    cascade.load_state_dict(cascade_state_dict)
    _logger.info(f'Cascade loaded (top_k1={top_k1})')
    return cascade


def get_model(data_config, **kwargs):
    """Build CoupleCascadeModel: frozen cascade + trainable CoupleReranker.

    Required kwargs (passed via train_couple_reranker.py):
        cascade_checkpoint: Path to a trained cascade checkpoint (Stage 1+2).
        top_k2: Number of top tracks per event from which couples are
            enumerated (default 50).
    """
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
    couple_loss = kwargs.pop('couple_loss', 'pairwise')
    couple_label_smoothing = kwargs.pop('couple_label_smoothing', 0.0)
    couple_hardneg_fraction = kwargs.pop('couple_hardneg_fraction', 0.0)
    couple_hardneg_margin = kwargs.pop('couple_hardneg_margin', 0.1)
    pair_kinematics_v2 = kwargs.pop('pair_kinematics_v2', False)
    pair_physics_v3 = kwargs.pop('pair_physics_v3', False)
    pair_physics_signif = kwargs.pop('pair_physics_signif', False)
    couple_embed_mode = kwargs.pop('couple_embed_mode', 'concat')
    couple_projector_dim = kwargs.pop('couple_projector_dim', 0)
    tokenize_d = kwargs.pop('tokenize_d', 16)
    tokenize_blocks = kwargs.pop('tokenize_blocks', 2)
    tokenize_heads = kwargs.pop('tokenize_heads', 4)
    ndcg_K = kwargs.pop('ndcg_k', 100)
    lambda_sigma = kwargs.pop('lambda_sigma', 1.0)
    ndcg_alpha = kwargs.pop('ndcg_alpha', 5.0)
    couple_multi_positive = kwargs.pop('couple_multi_positive', 'none')
    couple_use_full_negative_list = kwargs.pop(
        'couple_use_full_negative_list', False,
    )
    aux_vertex_weight = kwargs.pop('aux_vertex_weight', 0.0)
    event_context = kwargs.pop('event_context', 'none')
    context_dim = kwargs.pop('context_dim', 32)

    # Pop unused args from other heads
    for unused_arg in [
        'pretrained_backbone_path', 'backbone_mode',
        'mask_ce_loss_weight', 'confidence_loss_weight',
        'no_object_weight', 'num_decoder_layers', 'num_queries',
        'focal_bce_weight', 'potential_loss_weight',
        'beta_loss_weight', 'clustering_dim',
        'per_track_loss_weight', 'refinement_loss_weight',
        'num_enrichment_layers',
        'stage1_checkpoint', 'top_k1',
        'stage2_embed_dim', 'stage2_num_heads', 'stage2_num_layers',
        'stage2_pair_embed_dims', 'stage2_pair_extra_dim',
        'stage2_pair_embed_mode', 'stage2_ffn_ratio', 'stage2_dropout',
        'stage2_loss_mode', 'stage2_rs_at_k_target',
        'stage2_use_contrastive_denoising', 'stage2_denoising_sigma_start',
        'stage2_denoising_sigma_end', 'stage2_denoising_loss_weight',
    ]:
        kwargs.pop(unused_arg, None)

    if cascade_checkpoint is None:
        raise ValueError(
            'cascade_checkpoint is required. Pass --cascade-checkpoint to '
            'train_couple_reranker.py'
        )

    input_dim = len(data_config.input_dicts['pf_features'])

    # ---- Stage 1 + Stage 2 (frozen) ----
    cascade = _build_frozen_cascade(cascade_checkpoint, input_dim=input_dim)
    cascade_params = sum(p.numel() for p in cascade.parameters())
    _logger.info(f'Frozen cascade: {cascade_params:,} params')

    # ---- Stage 3: CoupleReranker (trainable) ----
    # CoupleReranker derives input_dim internally from couple_embed_mode,
    # couple_projector_dim, and rest_dim. rest_dim = 19 by default
    # (blocks 2+3+4) or 23 with pair_kinematics_v2 (4 extra physics dims).
    from utils.couple_features import (
        COUPLE_FEATURE_DIM,
        COUPLE_FEATURE_DIM_V2,
        COUPLE_FEATURE_DIM_V3,
        PAIR_PHYSICS_V3_EXTRA_DIM,
    )
    from utils.couple_features import PAIR_PHYSICS_SIGNIF_EXTRA_DIM
    feature_dim = COUPLE_FEATURE_DIM
    if pair_kinematics_v2:
        feature_dim = COUPLE_FEATURE_DIM_V2
    if pair_physics_v3:
        feature_dim += PAIR_PHYSICS_V3_EXTRA_DIM
    if pair_physics_signif:
        feature_dim += PAIR_PHYSICS_SIGNIF_EXTRA_DIM
    rest_dim = feature_dim - 32
    couple_reranker = CoupleReranker(
        hidden_dim=couple_hidden_dim,
        num_residual_blocks=couple_num_residual_blocks,
        dropout=couple_dropout,
        ranking_num_samples=couple_ranking_num_samples,
        ranking_temperature=couple_ranking_temperature,
        couple_loss=couple_loss,
        label_smoothing=couple_label_smoothing,
        hardneg_fraction=couple_hardneg_fraction,
        hardneg_margin=couple_hardneg_margin,
        ndcg_K=ndcg_K,
        lambda_sigma=lambda_sigma,
        ndcg_alpha=ndcg_alpha,
        multi_positive=couple_multi_positive,
        use_full_negative_list=couple_use_full_negative_list,
        aux_vertex_weight=aux_vertex_weight,
        event_context=event_context,
        context_dim=context_dim,
        couple_embed_mode=couple_embed_mode,
        couple_projector_dim=couple_projector_dim,
        rest_dim=rest_dim,
        tokenize_d=tokenize_d,
        tokenize_blocks=tokenize_blocks,
        tokenize_heads=tokenize_heads,
    )
    couple_params = sum(p.numel() for p in couple_reranker.parameters())
    _logger.info(
        f'CoupleReranker: {couple_params:,} params (trainable, '
        f'hidden_dim={couple_hidden_dim}, '
        f'num_residual_blocks={couple_num_residual_blocks}, '
        f'embed_mode={couple_embed_mode}, '
        f'projector_dim={couple_projector_dim}, '
        f'input_dim={couple_reranker.input_dim})'
    )

    # ---- CoupleCascadeModel: glue ----
    model = CoupleCascadeModel(
        cascade=cascade,
        couple_reranker=couple_reranker,
        pair_physics_v3=pair_physics_v3,
        pair_physics_signif=pair_physics_signif,
        top_k2=top_k2,
        k_values_tracks=k_values_tracks,
        pair_kinematics_v2=pair_kinematics_v2,
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
