from __future__ import annotations

import torch

from networks.lowpt_tau_CascadeReranker import load_stage2_init
from weaver.nn.model.CascadeReranker import CascadeReranker


def _small_reranker() -> CascadeReranker:
    return CascadeReranker(
        input_dim=8, embed_dim=16, num_heads=2, num_layers=1,
        pair_input_dim=4, pair_extra_dim=6, pair_embed_dims=[8, 8],
        ffn_ratio=2, dropout=0.0,
    )


def test_load_stage2_init_restores_weights(tmp_path):
    source = _small_reranker()
    target = _small_reranker()
    checkpoint_path = tmp_path / 'stage2_init.pt'
    torch.save({'model_state_dict': source.state_dict()}, checkpoint_path)

    load_stage2_init(target, str(checkpoint_path))

    for name, parameter in source.state_dict().items():
        assert torch.equal(parameter, target.state_dict()[name]), name
