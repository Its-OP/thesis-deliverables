from __future__ import annotations

import importlib.util

import torch


def trim_to_max_valid_tracks(
    inputs: list[torch.Tensor],
    mask_input_index: int,
) -> list[torch.Tensor]:
    """inputs[i]: (B, C_i, P). mask_input_index points at a (B, 1, P) mask."""
    mask = inputs[mask_input_index]

    max_valid_tracks = max(1, int(mask.sum(dim=2).max().item()))

    # Round up to multiples of 128 so torch.compile sees ~22 shapes, not thousands.
    bucket_size = 128
    max_valid_tracks = min(
        ((max_valid_tracks + bucket_size - 1) // bucket_size) * bucket_size,
        inputs[0].shape[2],
    )

    return [tensor[:, :, :max_valid_tracks] for tensor in inputs]


def load_network_module(network_path: str):
    spec = importlib.util.spec_from_file_location('network', network_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_label_from_inputs(
    inputs: list[torch.Tensor],
    label_input_index: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Returns (model_inputs_without_label, track_labels=(B, 1, P))."""
    track_labels = inputs[label_input_index]
    model_inputs = [
        tensor for index, tensor in enumerate(inputs)
        if index != label_input_index
    ]
    return model_inputs, track_labels
