"""
Learned ISTA (LISTA) — Gregor & LeCun (2010).

Each layer k executes:
  x^{k+1} = S_{θk}( W_y^k b + W_x^k x^k )

W_y^k ∈ R^{n×m}, W_x^k ∈ R^{n×n}, θ_k > 0 are all learnable parameters.

Trained end-to-end via backpropagation through all K layers (BPTT).
"""

import torch
import torch.nn as nn
from .ista import soft_threshold


class LISTALayer(nn.Module):
    """Single LISTA layer."""

    def __init__(self, m: int, n: int):
        super().__init__()
        # Initialise close to ISTA parameterisation for stable training
        self.W_y = nn.Linear(m, n, bias=False)
        self.W_x = nn.Linear(n, n, bias=False)
        self.log_theta = nn.Parameter(torch.zeros(1))  # softplus → positive

    @property
    def theta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_theta)

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            b: measurements (N, m)
            x: current iterate (N, n)
        Returns:
            x_next: (N, n)
        """
        return soft_threshold(self.W_y(b) + self.W_x(x), self.theta)


class LISTA(nn.Module):
    """
    LISTA network with K unrolled layers.

    Args:
        A:          Sensing matrix (m, n) — used for weight initialisation
        n_layers:   Number of unrolled layers K
        tied:       If True, share W_y and W_x across all layers (fewer params)
    """

    def __init__(
        self,
        A: torch.Tensor,
        n_layers: int = 16,
        tied: bool = False,
    ):
        super().__init__()
        m, n = A.shape
        self.n_layers = n_layers
        self.tied     = tied

        if tied:
            single = LISTALayer(m, n)
            self._init_layer(single, A)
            self.layers = nn.ModuleList([single] * n_layers)
        else:
            layers = [LISTALayer(m, n) for _ in range(n_layers)]
            for layer in layers:
                self._init_layer(layer, A)
            self.layers = nn.ModuleList(layers)

    @staticmethod
    def _init_layer(layer: LISTALayer, A: torch.Tensor):
        """Initialise weights close to the ISTA parameterisation."""
        with torch.no_grad():
            sv = torch.linalg.svdvals(A)
            L  = float(sv[0] ** 2)
            step = 1.0 / L
            # W_y ≈ (1/L) A^T,  W_x ≈ I - (1/L) A^T A
            layer.W_y.weight.copy_((step * A).T)     # (n, m)
            layer.W_x.weight.copy_(
                torch.eye(A.shape[1], device=A.device) - step * (A.T @ A)
            )
            nn.init.constant_(layer.log_theta, -2.0)   # softplus(-2) ≈ 0.127

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
        x = torch.zeros(b.shape[0], self.layers[0].W_x.in_features,
                        device=b.device, dtype=b.dtype)
        iterates = []

        for layer in self.layers:
            x = layer(b, x)
            if return_all:
                iterates.append(x)

        return iterates if return_all else x

    def count_parameters(self) -> int:
        if self.tied:
            return sum(p.numel() for p in self.layers[0].parameters())
        return sum(p.numel() for p in self.parameters())
