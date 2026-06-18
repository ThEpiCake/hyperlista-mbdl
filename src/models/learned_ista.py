"""
ISTA-derived learnable scalar variants.

These models preserve the ISTA update structure and learn only selected
scalar components per unfolded layer.

They are intended as ablation models for studying which ISTA components
are useful to learn in model-based deep learning.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from .ista import soft_threshold


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Return y such that softplus(y) = x."""
    x = torch.clamp(x, min=torch.finfo(x.dtype).eps)
    return torch.log(torch.expm1(x))


def spectral_lipschitz(A: torch.Tensor) -> float:
    """
    Compute L = ||A||_2^2.

    This is the Lipschitz constant of the gradient of
        0.5 * ||b - A x||_2^2.
    """
    with torch.no_grad():
        sv = torch.linalg.svdvals(A)
        return float(sv[0] ** 2)


class _BaseLearnedISTA(nn.Module):
    """Shared utilities for ISTA-derived unfolded models."""

    def __init__(self, A: torch.Tensor, n_layers: int = 16, lam: float = 0.1):
        super().__init__()

        if A.ndim != 2:
            raise ValueError(f"A must be a 2D tensor, got shape {tuple(A.shape)}")

        m, n = A.shape
        self.m = m
        self.n = n
        self.n_layers = n_layers
        self.lam = lam

        self.register_buffer("A", A.detach().clone())

        L = spectral_lipschitz(A)
        self.L = L
        self.step0 = 1.0 / L
        self.theta0 = lam / L

    @property
    def A_mat(self) -> torch.Tensor:
        """Return the sensing matrix buffer as a tensor."""
        A = self._buffers.get("A")
        if A is None:
            raise RuntimeError("Sensing matrix buffer A was not initialized.")
        return cast(torch.Tensor, A)

    def _initial_x(self, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(b.shape[0], self.n, device=b.device, dtype=b.dtype)

    def _gradient_step(self, b: torch.Tensor, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
        Compute x + step * A^T (b - A x).

        Args:
            b:    measurements, shape (batch, m)
            x:    current estimate, shape (batch, n)
            step: scalar tensor or broadcastable tensor
        """
        A = self.A_mat

        residual = b - x @ A.T
        grad = residual @ A
        return x + step * grad

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ThresholdISTA(_BaseLearnedISTA):
    """
    ISTA with fixed step size and learnable per-layer thresholds.

    Update:
        x^{k+1} = S_{theta_k}(x^k + (1/L) A^T (b - A x^k))

    Learned parameters:
        theta_1, ..., theta_K

    Number of learned parameters:
        K
    """

    def __init__(self, A: torch.Tensor, n_layers: int = 16, lam: float = 0.1):
        super().__init__(A=A, n_layers=n_layers, lam=lam)

        theta0 = torch.full(
            (n_layers,),
            fill_value=self.theta0,
            device=A.device,
            dtype=A.dtype,
        )
        self.log_theta = nn.Parameter(inverse_softplus(theta0))

    @property
    def theta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_theta)

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        x = self._initial_x(b)
        iterates = []

        step = torch.tensor(self.step0, device=b.device, dtype=b.dtype)

        for k in range(self.n_layers):
            z = self._gradient_step(b, x, step)
            x = soft_threshold(z, self.theta[k])

            if return_all:
                iterates.append(x)

        return iterates if return_all else x


class StepISTA(_BaseLearnedISTA):
    """
    ISTA with learnable per-layer step sizes.

    The threshold is coupled to the step size, as in ISTA:
        threshold_k = lambda * gamma_k

    Update:
        x^{k+1} = S_{lambda * gamma_k}(x^k + gamma_k A^T (b - A x^k))

    Learned parameters:
        gamma_1, ..., gamma_K

    Number of learned parameters:
        K
    """

    def __init__(self, A: torch.Tensor, n_layers: int = 16, lam: float = 0.1):
        super().__init__(A=A, n_layers=n_layers, lam=lam)

        step0 = torch.full(
            (n_layers,),
            fill_value=self.step0,
            device=A.device,
            dtype=A.dtype,
        )
        self.log_step = nn.Parameter(inverse_softplus(step0))

    @property
    def step(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_step)

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        x = self._initial_x(b)
        iterates = []

        for k in range(self.n_layers):
            step_k = self.step[k]
            theta_k = self.lam * step_k

            z = self._gradient_step(b, x, step_k)
            x = soft_threshold(z, theta_k)

            if return_all:
                iterates.append(x)

        return iterates if return_all else x


class StepThresholdISTA(_BaseLearnedISTA):
    """
    ISTA with learnable per-layer step sizes and thresholds.

    Update:
        x^{k+1} = S_{theta_k}(x^k + gamma_k A^T (b - A x^k))

    Learned parameters:
        gamma_1, ..., gamma_K
        theta_1, ..., theta_K

    Number of learned parameters:
        2K
    """

    def __init__(self, A: torch.Tensor, n_layers: int = 16, lam: float = 0.1):
        super().__init__(A=A, n_layers=n_layers, lam=lam)

        step0 = torch.full(
            (n_layers,),
            fill_value=self.step0,
            device=A.device,
            dtype=A.dtype,
        )
        theta0 = torch.full(
            (n_layers,),
            fill_value=self.theta0,
            device=A.device,
            dtype=A.dtype,
        )

        self.log_step = nn.Parameter(inverse_softplus(step0))
        self.log_theta = nn.Parameter(inverse_softplus(theta0))

    @property
    def step(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_step)

    @property
    def theta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_theta)

    def forward(
        self,
        b: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        x = self._initial_x(b)
        iterates = []

        for k in range(self.n_layers):
            z = self._gradient_step(b, x, self.step[k])
            x = soft_threshold(z, self.theta[k])

            if return_all:
                iterates.append(x)

        return iterates if return_all else x