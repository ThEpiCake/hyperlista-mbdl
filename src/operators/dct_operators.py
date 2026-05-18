"""
2D-DCT / 2D-IDCT operators compatible with PyTorch tensors.

Two implementations are provided:
  - Matrix-multiply (exact, GPU-friendly for d ≤ 784)
  - torch.fft-based (memory-efficient for larger signals)

The public API used throughout the project is:
  dct2_flat(x_flat, H, W)   -> alpha_flat
  idct2_flat(alpha, H, W)   -> x_flat
"""

from __future__ import annotations
import torch
import numpy as np
from functools import lru_cache


# ─── Basis construction ──────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _dct1d_matrix(n: int) -> torch.Tensor:
    """
    Build the (n, n) orthonormal 1D-DCT-II basis matrix on CPU.

    Ψ[k, i] = w_k * cos(π (2i+1) k / (2n))
    w_0 = 1/sqrt(n),  w_k = sqrt(2/n) for k >= 1
    """
    k = torch.arange(n, dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    angles = torch.pi * (2.0 * i[None, :] + 1.0) * k[:, None] / (2.0 * n)
    Psi = torch.cos(angles)
    Psi[0, :]  *= 1.0 / np.sqrt(n)
    Psi[1:, :] *= np.sqrt(2.0 / n)
    return Psi.float()


def get_dct_basis(H: int = 28, W: int = 28, device: torch.device = None) -> torch.Tensor:
    """
    Return the full 2D-DCT basis Ψ of shape (H*W, H*W) as a matrix.

    Computed as the Kronecker product of two 1D bases: Ψ = Ψ_h ⊗ Ψ_w.
    For 28×28 this yields a 784×784 matrix.

    Args:
        H, W:   Spatial dimensions
        device: Target device

    Returns:
        Psi: (H*W, H*W) float tensor
    """
    Psi_h = _dct1d_matrix(H)   # (H, H)
    Psi_w = _dct1d_matrix(W)   # (W, W)
    Psi_2d = torch.kron(Psi_h, Psi_w)  # (H*W, H*W)
    if device is not None:
        Psi_2d = Psi_2d.to(device)
    return Psi_2d


# ─── Separable (memory-efficient) operators ───────────────────────────────────

def dct2_flat(
    x_flat: torch.Tensor,
    H: int = 28,
    W: int = 28,
) -> torch.Tensor:
    """
    Apply 2D-DCT to a batch of flattened images via separable 1D transforms.

    Args:
        x_flat: (N, H*W) batch of flattened images
        H, W:   Spatial dimensions

    Returns:
        alpha_flat: (N, H*W) flattened DCT coefficients
    """
    N = x_flat.shape[0]
    device = x_flat.device
    Psi_h = _dct1d_matrix(H).to(device)   # (H, H)
    Psi_w = _dct1d_matrix(W).to(device)   # (W, W)

    x2d = x_flat.view(N, H, W)                            # (N, H, W)
    # DCT over rows  (axis=1): alpha[k,j] = sum_i Psi_h[k,i] * x[i,j]
    a = torch.einsum('ki,nij->nkj', Psi_h, x2d)
    # DCT over cols  (axis=2): alpha[k,l] = sum_j Psi_w[l,j] * a[k,j]
    a = torch.einsum('lj,nkj->nkl', Psi_w, a)
    return a.reshape(N, H * W)


def idct2_flat(
    alpha_flat: torch.Tensor,
    H: int = 28,
    W: int = 28,
) -> torch.Tensor:
    """
    Apply 2D-IDCT (inverse of dct2_flat) to a batch of flattened coefficients.

    Because Ψ is orthonormal, IDCT = Ψ^T applied separably.

    Args:
        alpha_flat: (N, H*W) flattened DCT coefficients
        H, W:       Spatial dimensions

    Returns:
        x_flat: (N, H*W) reconstructed images
    """
    N = alpha_flat.shape[0]
    device = alpha_flat.device
    Psi_h = _dct1d_matrix(H).to(device)
    Psi_w = _dct1d_matrix(W).to(device)

    a2d = alpha_flat.view(N, H, W)                         # (N, H, W)
    # IDCT over cols
    x = torch.einsum('lj,nkl->nkj', Psi_w, a2d)
    # IDCT over rows
    x = torch.einsum('ki,nkj->nij', Psi_h, x)
    return x.reshape(N, H * W)


# ─── Matrix-multiply version (explicit Ψ) ────────────────────────────────────

def dct2_flat_matmul(
    x_flat: torch.Tensor,
    Psi: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 2D-DCT using the explicit full basis matrix Ψ (H*W, H*W).

    alpha = x_flat @ Psi^T   (since Psi rows are basis vectors)

    Args:
        x_flat: (N, d) flattened images
        Psi:    (d, d) 2D-DCT basis (from get_dct_basis)

    Returns:
        alpha: (N, d)
    """
    return x_flat @ Psi.T


def idct2_flat_matmul(
    alpha: torch.Tensor,
    Psi: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 2D-IDCT using the explicit full basis matrix Ψ.

    x = alpha @ Psi   (since Psi^{-1} = Psi^T for orthonormal basis)

    Args:
        alpha: (N, d)
        Psi:   (d, d) 2D-DCT basis

    Returns:
        x_flat: (N, d)
    """
    return alpha @ Psi
