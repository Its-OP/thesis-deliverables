from __future__ import annotations

import torch
import torch.nn as nn


class PerFeatureEmbedding(nn.Module):
    def __init__(self, num_features: int = 16, embed_dim: int = 32):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.per_feature_conv = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features * embed_dim,
            kernel_size=1,
            groups=num_features,
            bias=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_features, P). Returns (B, num_features * embed_dim, P)."""
        if x.shape[1] != self.num_features:
            raise ValueError(
                f'PerFeatureEmbedding expects {self.num_features} input '
                f'channels, got {x.shape[1]}',
            )
        batch_size, num_features, num_positions = x.shape
        embed_dim = self.embed_dim

        embedded_flat = self.per_feature_conv(x)
        embedded = (
            embedded_flat
            .view(batch_size, num_features, embed_dim, num_positions)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        embedded = self.layer_norm(embedded)
        embedded = self.activation(embedded)
        return (
            embedded
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(batch_size, num_features * embed_dim, num_positions)
        )
