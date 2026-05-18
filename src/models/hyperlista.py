"""
HyperLISTA — Chen, Liu, Wang, Yin (NeurIPS 2021).

Ultra-lightweight deep-unrolled network for sparse recovery with
only **three** instance-and-layer-invariant hyperparameters (c1, c2, c3).

Full algorithm (Algorithm 1 from the paper):
─────────────────────────────────────────────
Inputs : b, A, c1, c2, c3
Init   : x^(0) = 0

1. Compute D and G via Eq. (10): min_{G,D} ||D^T D - I||_F^2 + (1/α)||D - GA||_F^2
   Set W = (G^T G) A   (symmetric Jacobian parameterisation)
2. Compute μ = max_{i≠j} |(D^T D)_{ij}|   (mutual coherence of D)

3. For k = 0, 1, 2, ... until convergence:
   γ^(k) = 1
   θ^(k) = c1 · μ · γ^(k) · ||A^+ (A x^(k) - b)||_1
   β^(k) = c2 · μ · ||x^(k)||_0
   p^(k) = c3 · min( log( ||A^+ b||_1 / ||A^+ (A x^(k) - b)||_1 ), n )

   x^(k+1) = S_{p^(k), θ^(k)}( x^(k)
              + γ^(k) W^T (b - A x^(k))
              + β^(k) (x^(k) - x^(k-1)) )   [Polyak heavy-ball momentum]

   if p^(k) is large enough: break

4. (Optional CG tail — not implemented here for deep-unrolling comparison)
─────────────────────────────────────────────

The three hyperparameters are found by **grid search** (not backprop),
implemented in src/training/tuner.py.

This module provides the forward pass for a fixed (c1, c2, c3) triple.
"""

import torch
import torch.nn as nn
import numpy as np
from .alista import compute_alista_weight, partial_soft_threshold


# ─── Symmetric Jacobian weight computation ───────────────────────────────────

def compute_hyperlista_weight(
    A: torch.Tensor,
    alpha_reg: float = 10.0,
    n_iter: int = 3000,
    lr: float = 5e-3,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute W = (G^T G) A using the symmetric Jacobian formulation (Eq. 10):

      min_{G, D}  ||D^T D - I||_F^2 + (1/alpha) ||D - GA||_F^2
      s.t.        diag(D^T D) = 1

    For simplicity we relax the diagonal constraint and re-normalise after.

    Returns:
        W:   (m, n) weight matrix such that W^T A is symmetric
        D:   (m, n) auxiliary matrix
        mu:  mutual coherence of D
    """
    m, n = A.shape
    device = A.device

    # Initialise G ≈ I (if square) or pseudo-inverse, D = G A
    G = torch.eye(m, device=device) if m == n else torch.randn(m, m, device=device) * 0.01
    G = G.requires_grad_(True)
    D_param = (G.detach() @ A.detach()).requires_grad_(True)

    opt = torch.optim.Adam([G, D_param], lr=lr)
    A_fixed = A.detach()

    for _ in range(n_iter):
        opt.zero_grad()
        DTD = D_param.T @ D_param                     # (n, n)
        loss_orth = ((DTD - torch.eye(n, device=device)) ** 2).sum()
        loss_fit  = ((D_param - G @ A_fixed) ** 2).sum() / alpha_reg
        (loss_orth + loss_fit).backward()
        opt.step()

        # Re-normalise diagonal of D^T D to 1
        with torch.no_grad():
            diag_norm = (D_param.T @ D_param).diag().clamp(min=1e-8).sqrt()
            D_param.data /= diag_norm[None, :]

    D = D_param.detach()
    G_final = G.detach()
    W = (G_final.T @ G_final) @ A_fixed     # (m, n)

    # Mutual coherence of D: max_{i≠j} |(D^T D)_{ij}|
    with torch.no_grad():
        DTD = D.T @ D                        # (n, n)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        mu = float(DTD[mask].abs().max().item())

    return W, D, mu


# ─── Utility: l1 norm of A^+ residual ────────────────────────────────────────

def _pinv_l1(A_pinv: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Compute ||A^+ r||_1 for each sample in the batch.

    Args:
        A_pinv: (n, m) pseudo-inverse of A
        r:      (N, m) residuals

    Returns:
        norms: (N,)
    """
    return (r @ A_pinv.T).abs().sum(dim=1)   # (N,)


# ─── HyperLISTA model ─────────────────────────────────────────────────────────

class HyperLISTA(nn.Module):
    """
    HyperLISTA with adaptive parameters driven by only (c1, c2, c3).

    The hyperparameters are **not** nn.Parameters — they are ordinary floats
    set by the grid-search tuner and stored as plain attributes.

    Args:
        A:             Sensing matrix (m, n)
        c1:            Threshold scaling hyperparameter
        c2:            Momentum scaling hyperparameter
        c3:            Support-selection scaling hyperparameter
        n_layers:      Number of unrolled layers K
        p_threshold:   Break when p^(k) >= n * p_threshold (fraction in [0,1])
        W_iters:       Iterations for analytic W computation
        alpha_reg:     Regularisation weight for D-G optimisation
    """

    def __init__(
        self,
        A: torch.Tensor,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
        n_layers: int = 16,
        p_threshold: float = 0.9,
        W_iters: int = 2000,
        alpha_reg: float = 10.0,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.n_layers = n_layers
        self.p_threshold = p_threshold

        m, n = A.shape
        self.n = n

        # Compute fixed weights (no grad)
        W, D, mu = compute_hyperlista_weight(A, alpha_reg=alpha_reg, n_iter=W_iters)
        self.register_buffer("A",     A)
        self.register_buffer("W",     W)      # (m, n)
        self.register_buffer("D",     D)      # (m, n)
        self.mu = mu

        # Precompute A^+ = A^T (A A^T)^{-1}  (Moore-Penrose, m < n case)
        with torch.no_grad():
            A_pinv = torch.linalg.pinv(A)     # (n, m)
        self.register_buffer("A_pinv", A_pinv)

        # ||A^+ b||_1 is input-dependent; stored per forward call
        self._Apinv_b_l1: torch.Tensor | None = None

    def set_hyperparams(self, c1: float, c2: float, c3: float):
        """Update (c1, c2, c3) without re-computing W."""
        self.c1, self.c2, self.c3 = c1, c2, c3

    def _compute_layer_params(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Compute adaptive layer parameters from current iterate.

        Returns:
            theta:  (N,) per-sample threshold
            beta:   (N,) per-sample momentum coefficient
            p:      scalar support selection count (int)
            p_val:  raw float p value
        """
        A, A_pinv = self.A, self.A_pinv
        c1, c2, c3 = self.c1, self.c2, self.c3
        mu = self.mu
        n  = self.n

        # Residual r = A x^(k) - b
        residual = x @ A.T - b                              # (N, m)
        Apinv_r_l1 = _pinv_l1(A_pinv, residual)            # (N,)

        # Adaptive threshold  θ^(k) = c1 μ ||A^+ r||_1
        theta = (c1 * mu * Apinv_r_l1).clamp(min=1e-12)    # (N,)

        # Adaptive momentum  β^(k) = c2 μ ||x^(k)||_0  (ℓ0 proxy via count)
        sparsity = (x.abs() > 1e-6).float().sum(dim=1)     # (N,)
        beta = (c2 * mu * sparsity).clamp(min=0.0)          # (N,)

        # Support selection  p^(k) = c3 log( ||A^+ b||_1 / ||A^+ r||_1 )
        Apinv_b_l1 = self._Apinv_b_l1                       # (N,) precomputed
        ratio = (Apinv_b_l1 / Apinv_r_l1.clamp(min=1e-12)).clamp(min=1.0)
        p_float = c3 * torch.log(ratio)                      # (N,)
        p = int(p_float.median().item())                     # scalar for mask
        p = max(0, min(p, n))

        return theta, beta, p

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run HyperLISTA forward pass.

        Args:
            b:          Measurements (N, m)
            return_all: If True, return list of all iterates

        Returns:
            x^K  (N, n)  or  list of (N, n) tensors
        """
        A, W, A_pinv = self.A, self.W, self.A_pinv
        n = self.n
        p_break = int(self.p_threshold * n)

        # Precompute ||A^+ b||_1 once per forward call
        self._Apinv_b_l1 = _pinv_l1(A_pinv, b).clamp(min=1e-12)  # (N,)

        x      = torch.zeros(b.shape[0], n, device=b.device, dtype=b.dtype)
        x_prev = torch.zeros_like(x)
        iterates = []

        for k in range(self.n_layers):
            theta, beta, p = self._compute_layer_params(x, x_prev, b)

            # Polyak heavy-ball with adaptive momentum β
            momentum = beta[:, None] * (x - x_prev)         # (N, n)

            # Gradient step with W (symmetric Jacobian)
            residual = b - x @ A.T                           # (N, m)
            grad_step = residual @ W                          # (N, n)  = W^T r

            v = x + grad_step + momentum                     # (N, n)

            # Per-sample threshold — broadcast (N,) → partial soft-threshold
            # We use the median threshold across the batch for simplicity
            theta_scalar = theta.median()
            x_new = partial_soft_threshold(v, theta_scalar, p)

            x_prev = x
            x      = x_new

            if return_all:
                iterates.append(x)

            # Early stopping when support selection saturates
            if p >= p_break and k >= 1:
                # Pad remaining slots with the final iterate for fair comparison
                if return_all:
                    for _ in range(self.n_layers - k - 1):
                        iterates.append(x)
                break

        return iterates if return_all else x

    def count_parameters(self) -> int:
        """HyperLISTA has no learnable parameters; report count as 3 (c1,c2,c3)."""
        return 3   # the three hyperparameters tuned by grid search
