from __future__ import annotations

import torch

from utils.triplet_rank_data import collate_listwise_rank


def _item(n: int, offset: int):
    return {
        'features': torch.randn(n, 4),
        'pos_mask': torch.zeros(n, dtype=torch.bool),
        'filter_logit': torch.randn(n),
        'keys': torch.arange(offset, offset + 3 * n).reshape(n, 3),
        'couple_ids': torch.arange(offset, offset + n),
    }


def test_collate_listwise_pads_keys_with_minus_one_and_ids_with_minus_one():
    batch = collate_listwise_rank([_item(3, 0), _item(5, 100)])
    assert batch['features'].shape == (2, 4, 5)
    assert batch['keys'].shape == (2, 5, 3)
    assert batch['couple_ids'].shape == (2, 5)
    assert batch['valid_mask'].tolist() == [[True] * 3 + [False] * 2,
                                            [True] * 5]
    # Padded slots must never collide with real track indices or couple ids.
    assert (batch['keys'][0, 3:] == -1).all()
    assert (batch['couple_ids'][0, 3:] == -1).all()
    assert (batch['counts'] == torch.tensor([3, 5])).all()
