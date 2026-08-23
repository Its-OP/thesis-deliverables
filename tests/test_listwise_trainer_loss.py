from __future__ import annotations

import torch

from train_listwise_reranker import listwise_loss


def test_listwise_loss_prefers_gt_on_top():
    pos_mask = torch.tensor([[False, True, False, False]])
    valid_mask = torch.tensor([[True, True, True, False]])
    couple_ids = torch.tensor([[7, 7, 9, -1]])
    gt_on_top = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    gt_buried = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    low = listwise_loss(gt_on_top, pos_mask, valid_mask, couple_ids,
                        label_smoothing=0.1, contrast_weight=0.2)
    high = listwise_loss(gt_buried, pos_mask, valid_mask, couple_ids,
                         label_smoothing=0.1, contrast_weight=0.2)
    assert low.item() < high.item()


def test_listwise_loss_ignores_padded_slots_and_gtless_events():
    pos_mask = torch.tensor([[False, True], [False, False]])
    valid_mask = torch.tensor([[True, True], [True, True]])
    couple_ids = torch.tensor([[3, 3], [5, 5]])
    scores = torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)
    loss = listwise_loss(scores, pos_mask, valid_mask, couple_ids,
                         label_smoothing=0.0, contrast_weight=0.0)
    loss.backward()
    # The GT-less event contributes nothing: its gradient rows are zero.
    assert torch.all(scores.grad[1] == 0.0)
    assert torch.isfinite(loss)
