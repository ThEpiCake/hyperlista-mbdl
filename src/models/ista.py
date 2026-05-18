"""
Classical Iterative Shrinkage-Thresholding Algorithm (ISTA).

Update rule:
  x^{k+1} = S_{λ/L}( x^k + (1/L) A^T (b - A x^k) )

where
  L  = largest eigenvalue of A^T A  (Lipschitz constant)
  S_θ = element-wise soft-thresholding: sign(x) * max(|x| - θ, 0)
"""

import torch
import torch.nn as nn


def soft_threshold(x: torch.Tensor, theta: float | torch.Tensor) -> torch.Tensor:
    """Element-wise soft-thresholding operator S_θ(x)."""
    return x.sign() * (x.abs() - theta).clamp(min=0.0)


class ISTA(nn.Module):
    """
    ISTA solver (non-trainable, purely algorithmic).

    Args:
        A:          Sensing matrix (m, n) — fixed, not a parameter
        lam:        Regularisation coefficient λ
        n_iter:     Number of iterations K
        compute_L:  If True, compute L = σ_max(A)^2 at construction time.
                    Otherwise L must be supplied explicitly.
        L:          Lipschitz constant override (ignored when compute_L=True)
    """

    def __init__(
        self,
        A: torch.Tensor,
        lam: float = 0.1,
        n_iter: int = 16,
        compute_L: bool = True,
        L: float = None,
    ):
        super().__init__()
        self.register_buffer("A", A)
        self.lam    = lam
        self.n_iter = n_iter

        if compute_L:
            with torch.no_grad():
                sv = torch.linalg.svdvals(A)
                self.L = float(sv[0] ** 2)
        else:
            assert L is not None, "Must provide L when compute_L=False"
            self.L = L

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run K iterations of ISTA.

        Args:
            b:          Measurements (N, m)
            return_all: If True, return list of iterates [x^1, ..., x^K]

        Returns:
            x^K  (N, n)  or  list of (N, n) tensors
        """
        A, lam, L = self.A, self.lam, self.L
        step = 1.0 / L
        theta = lam / L

        x = torch.zeros(b.shape[0], A.shape[1], device=b.device, dtype=b.dtype)
        iterates = []

        for _ in range(self.n_iter):
            residual = b - x @ A.T          # (N, m)
            gradient = residual @ A          # (N, n)  = A^T (b - Ax)
            x = soft_threshold(x + step * gradient, theta)
            if return_all:
                iterates.append(x)

        return iterates if return_all else x

    def extra_repr(self) -> str:
        return f"n_iter={self.n_iter}, lam={self.lam:.4f}, L={self.L:.4f}"
