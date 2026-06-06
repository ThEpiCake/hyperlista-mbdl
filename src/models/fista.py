"""
Fast ISTA (FISTA) — ISTA with Nesterov momentum.

Update rule (Beck & Teboulle 2009):
  y^{k+1} = x^k + t_k (x^k - x^{k-1})
  x^{k+1} = S_{λ/L}( y^{k+1} + (1/L) A^T (b - A y^{k+1}) )
  t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2

where the momentum coefficient is (t_{k-1} - 1) / t_k.
"""

import torch
import torch.nn as nn
from typing import cast
from .ista import soft_threshold


class FISTA(nn.Module):
    """
    FISTA solver (non-trainable, purely algorithmic).

    Args:
        A:         Sensing matrix (m, n)
        lam:       Regularisation coefficient λ
        n_iter:    Number of iterations K
        compute_L: If True, auto-compute L = o_max(A)^2
        L:         Lipschitz constant override
    """

    def __init__(
        self,
        A: torch.Tensor,
        lam: float = 0.1,
        n_iter: int = 16,
        compute_L: bool = True,
        L: float | None = None,
    ):
        super().__init__()
        A = A.detach().clone()
        self.register_buffer("A", A)
        self.lam    = lam
        self.n_iter = n_iter

        if compute_L:
            with torch.no_grad():
                sv = torch.linalg.svdvals(A)
                self.L = float(sv[0] ** 2)
        else:
            assert L is not None
            self.L = L

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run K iterations of FISTA.

        Args:
            b:          Measurements (N, m)
            return_all: If True, return list of iterates [x^1, ..., x^K]

        Returns:
            x^K  (N, n)  or  list of (N, n) tensors
        """
        # narrow `self.A` to `torch.Tensor` for the type-checker (no runtime change)
        A = cast(torch.Tensor, self.A)
        lam, L = self.lam, self.L
        step  = 1.0 / L
        theta = lam / L

        n = A.shape[1]
        x      = torch.zeros(b.shape[0], n, device=b.device, dtype=b.dtype)
        x_prev = torch.zeros_like(x)
        t      = 1.0
        iterates = []

        for _ in range(self.n_iter):
            # Nesterov extrapolation
            t_next = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            momentum = (t - 1.0) / t_next
            y = x + momentum * (x - x_prev)

            # Proximal gradient step
            residual = b - y @ A.T
            gradient = residual @ A
            x_new = soft_threshold(y + step * gradient, theta)

            x_prev = x
            x      = x_new
            t      = t_next

            if return_all:
                iterates.append(x)

        return iterates if return_all else x

    def extra_repr(self) -> str:
        return f"n_iter={self.n_iter}, lam={self.lam:.4f}, L={self.L:.4f}"
