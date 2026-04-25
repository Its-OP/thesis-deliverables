"""ASAM / SAM optimizer wrappers for flat-minimum training.

Sharpness-aware minimisation (Foret 2020, arXiv:2010.01412) seeks
parameters w such that loss in a neighbourhood of radius ρ around w is
low, not just at w itself. The two-step training update is:

    1. gradient g₁ at w.
    2. w̃ = w + ρ · g₁ / ||g₁||     (perturb toward steepest ascent)
    3. gradient g₂ at w̃.
    4. w ← optimizer.step() using g₂ and then restore w (undo step 2).

ASAM (Kwon 2021, arXiv:2102.11600) makes the perturbation scale-invariant
by multiplying each component by |w_i| + η, giving

    e_w_i = ρ · (|w_i| + η) · g_i / ||diag(|w|+η) · g||

This wrapper composes with any torch optimiser — pass an already-
constructed AdamW (or SGD etc.) as ``base_optimizer``. The training
loop drives two forward passes per batch via ``first_step`` /
``second_step`` explicitly. Gradient clipping is applied AFTER the
second backward and BEFORE ``second_step``.
"""
from __future__ import annotations

from collections import defaultdict

import torch


class SAM:
    """Sharpness-aware minimisation wrapper.

    Implements both vanilla SAM (``adaptive=False``) and ASAM (
    ``adaptive=True``). Not a ``torch.optim.Optimizer`` subclass — the
    caller drives the two-step update explicitly.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        eta: float = 0.01,
        adaptive: bool = True,
    ):
        if rho <= 0.0:
            raise ValueError(f'rho must be > 0, got {rho}')
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.eta = eta
        self.adaptive = adaptive
        # Store per-parameter perturbation so second_step can undo it.
        self.state: dict[torch.nn.Parameter, dict] = defaultdict(dict)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Norm of the (adaptive-scaled) gradient over all params."""
        shared_device = self.param_groups[0]['params'][0].device
        sq_norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    scale = torch.abs(p) + self.eta
                    g = scale * p.grad
                else:
                    g = p.grad
                sq_norms.append(g.norm(p=2).to(shared_device))
        if not sq_norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(sq_norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        """Perturb weights to the gradient-ascent neighbourhood boundary."""
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if self.adaptive:
                    e_w = (torch.abs(p) + self.eta) ** 2 * p.grad * scale
                else:
                    e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        """Undo the perturbation, then call the base optimizer step."""
        for group in self.param_groups:
            for p in group['params']:
                entry = self.state[p]
                if 'e_w' in entry:
                    p.sub_(entry['e_w'])
                    entry.pop('e_w')
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)
