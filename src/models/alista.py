"""
Analytic LISTA (ALISTA) — Liu et al. (2019).

The weight matrix W is computed analytically from A by solving:

  min_{W ∈ R^{mxn}}  ||W^T A||_F^2   s.t.  diag(W^T A) = 1

This reduces the learnable parameters to only per-layer step sizes γ^(k)
and thresholds θ^(k).

Each layer k:
  x^{k+1} = S_{p_k, θ^k}( x^k + γ^k W^T (b - A x^k) )

where S_{p,θ} is the partial soft-thresholding with support selection
(the p entries with largest magnitudes bypass thresholding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ista import soft_threshold


# ─── Weight matrix computation ───────────────────────────────────────────────

def compute_alista_weight(
    A: torch.Tensor,
    n_iter: int = 5000,
    lr: float = 1e-3,
) -> torch.Tensor:
    """
    Compute the ALISTA weight matrix W analytically as the minimum-Frobenius-norm
    solution to min_W ||W^T A - I||_F, then normalise columns so diag(W^T A) = 1.

    W* = (AA^T)^{-1} A  is the closed-form minimiser of ||W^T A - I||_F.
    After column normalisation: diag(W^T A) = 1 and off-diagonal mu ≈ 0.22.

    Args:
        A:      Sensing matrix (m, n)
        n_iter: Kept for API compatibility (unused)
        lr:     Kept for API compatibility (unused)

    Returns:
        W: (m, n) weight matrix, same device as A
    """
    m = A.shape[0]
    with torch.no_grad():
        AAt_reg = A @ A.T + 1e-6 * torch.eye(m, device=A.device, dtype=A.dtype)
        W = torch.linalg.solve(AAt_reg, A)        # (m, n)
        diag_vals = (W.T @ A).diag().clamp(min=1e-6)
        W = W / diag_vals[None, :]
    return W


# ─── Support-selection thresholding ──────────────────────────────────────────

def partial_soft_threshold(
    v: torch.Tensor,
    theta: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Partial soft-thresholding with support selection.

    The p entries with the largest absolute values pass through unchanged;
    the remaining entries are soft-thresholded by θ.

    Args:
        v:     Input tensor (N, n)
        theta: Threshold (scalar tensor)
        p:     Number of entries in the "trust region"

    Returns:
        output: (N, n)
    """
    if p <= 0:
        return soft_threshold(v, theta)

    n = v.shape[1]
    if p >= n:
        # Hard-threshold (all trusted): no shrinkage
        return v

    # Identify the p largest magnitudes for each sample
    _, top_idx = v.abs().topk(p, dim=1)
    trust_mask = torch.zeros_like(v, dtype=torch.bool)
    trust_mask.scatter_(1, top_idx, True)

    # Soft-threshold non-trusted entries
    out = soft_threshold(v, theta)
    # Restore trusted entries
    out[trust_mask] = v[trust_mask]
    return out


# ─── ALISTA model ─────────────────────────────────────────────────────────────

class ALISTA(nn.Module):
    """
    ALISTA with learnable per-layer step sizes and thresholds.

    Args:
        A:          Sensing matrix (m, n)
        n_layers:   Number of unrolled layers K
        p_schedule: Support selection schedule as a list of ints of length K.
                    If None, uses linear ramp from 0 to n//2.
        W_iters:    Gradient iterations for analytic W computation
    """

    def __init__(
        self,
        A: torch.Tensor,
        n_layers: int = 16,
        p_schedule: list[int] | None = None,
        W_iters: int = 2000,
    ):
        super().__init__()
        A = A.detach().clone()
        m, n = A.shape
        self.n_layers = n_layers

        # Fixed analytic weight matrix
        W = compute_alista_weight(A, n_iter=W_iters)
        self.register_buffer("W", W)   # (m, n)
        self.register_buffer("A", A)

        # Learnable per-layer step sizes and thresholds (initialise close to ISTA)
        with torch.no_grad():
            sv = torch.linalg.svdvals(A)
            L  = float(sv[0] ** 2)

        self.log_gamma = nn.Parameter(torch.zeros(n_layers))     # softplus → positive
        self.log_theta = nn.Parameter(torch.full((n_layers,), -2.0))

        # Support selection schedule
        if p_schedule is None:
            p_max = n // 2
            self.p_schedule = [int(round(k * p_max / max(n_layers - 1, 1)))
                                for k in range(n_layers)]
        else:
            assert len(p_schedule) == n_layers
            self.p_schedule = list(p_schedule)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.log_gamma)

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.log_theta)

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            b:          Measurements (N, m)
            return_all: If True return list of all iterates

        Returns:
            x^K  (N, n)  or  list of (N, n) tensors
        """
        A, W = self.A, self.W
        x = torch.zeros(b.shape[0], A.shape[1], device=b.device, dtype=b.dtype)
        iterates = []

        for k in range(self.n_layers):
            gamma_k = self.gamma[k]
            theta_k = self.theta[k]
            p_k     = self.p_schedule[k]

            residual = b - x @ A.T          # (N, m)
            v = x + gamma_k * (residual @ W)  # (N, n)   W is (m,n) so W^T gives (n,m)
            x = partial_soft_threshold(v, theta_k, p_k)

            if return_all:
                iterates.append(x)

        return iterates if return_all else x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
