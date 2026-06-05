"""
Fashion-MNIST DataLoader for Part B (compressed-sensing image reconstruction).

Pipeline:
  1. Load Fashion-MNIST (grayscale 28x28, pixels in [0,1])
  2. Flatten to d = 784
  3. Compute 2D-DCT coefficients α = Ψ x  (sparse/compressible domain)
  4. Generate compressed measurements  y = A α + n  where A ∈ R^{m×d}

The DataLoader returns (y, alpha, x_flat) triplets so that all downstream
code can choose which target to use.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

from src.operators.dct_operators import dct2_flat, get_dct_basis


class CSImageDataset(Dataset):
    """
    Compressed-sensing image dataset wrapping Fashion-MNIST.

    Returns (measurements y, DCT coefficients alpha, flattened image x).
    """

    def __init__(
        self,
        images: torch.Tensor,   # (N, d) flattened, normalised to [0,1]
        alpha: torch.Tensor,    # (N, d) 2D-DCT coefficients
        y: torch.Tensor,        # (N, m) compressed measurements
        A: torch.Tensor,        # (m, d) sensing matrix (stored for reference)
    ):
        self.images = images
        self.alpha  = alpha
        self.y      = y
        self.A      = A

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.alpha[idx], self.images[idx]


def _build_dct2_matrix(n: int) -> torch.Tensor:
    """
    Build the n×n orthonormal 1D-DCT-II basis matrix Ψ_1d.

    The full 2D-DCT basis for a d = n*n image (flattened) is Ψ = Ψ_1d ⊗ Ψ_1d,
    but we compute it lazily inside apply_dct2 using separable 1D transforms
    for memory efficiency.
    """
    k = torch.arange(n, dtype=torch.float64)
    i = torch.arange(n, dtype=torch.float64)
    # DCT-II: Ψ[k,i] = cos(π(2i+1)k / (2n)) * w_k / sqrt(n)
    angles = torch.pi * (2 * i[None, :] + 1) * k[:, None] / (2 * n)
    W = torch.cos(angles)
    # Orthonormal scaling
    W[0, :] *= 1.0 / np.sqrt(n)
    W[1:, :] *= np.sqrt(2.0 / n)
    return W.float()


def apply_dct2_flat(x_flat: torch.Tensor, H: int = 28, W_img: int = 28) -> torch.Tensor:
    """
    Apply separable 2D-DCT to a batch of flattened images.

    Args:
        x_flat: (N, H*W) flattened images
        H, W_img: spatial dimensions

    Returns:
        alpha_flat: (N, H*W) flattened DCT coefficients
    """
    N = x_flat.shape[0]
    x_2d = x_flat.view(N, H, W_img)              # (N, H, W)

    Psi = _build_dct2_matrix(H).to(x_flat.device)  # assumes square images

    # Apply along columns then rows (separable)
    alpha_2d = torch.einsum('ij,njk->nik', Psi, x_2d)   # DCT over rows
    alpha_2d = torch.einsum('ij,nkj->nki', Psi, alpha_2d) # DCT over cols
    return alpha_2d.reshape(N, H * W_img)


def apply_idct2_flat(alpha_flat: torch.Tensor, H: int = 28, W_img: int = 28) -> torch.Tensor:
    """
    Apply separable 2D-IDCT (inverse of apply_dct2_flat).

    Args:
        alpha_flat: (N, H*W) DCT coefficients

    Returns:
        x_flat: (N, H*W) reconstructed images
    """
    N = alpha_flat.shape[0]
    alpha_2d = alpha_flat.view(N, H, W_img)

    Psi = _build_dct2_matrix(H).to(alpha_flat.device)
    Psi_T = Psi.T

    # IDCT = transpose of DCT (orthonormal basis)
    x_2d = torch.einsum('ij,njk->nik', Psi_T, alpha_2d)
    x_2d = torch.einsum('ij,nkj->nki', Psi_T, x_2d)
    return x_2d.reshape(N, H * W_img)


def _load_raw_fmnist(data_root: str = "./data", train: bool = True) -> torch.Tensor:
    """Load Fashion-MNIST and return flattened float tensors in [0, 1]."""
    tf = transforms.Compose([
        transforms.ToTensor(),   # scales to [0,1]
    ])
    try:
        ds = torchvision.datasets.FashionMNIST(
            root=data_root, train=train, download=True, transform=tf
        )
    except RuntimeError as exc:
        split = "train" if train else "test"
        raise RuntimeError(
            "Fashion-MNIST is not available locally and torchvision could not "
            f"download the {split} split. This environment appears to be offline. "
            f"Download Fashion-MNIST once into {data_root!r}, or run this notebook "
            "from an environment with network access."
        ) from exc
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    imgs, _ = next(iter(loader))          # (N, 1, 28, 28)
    imgs = imgs.squeeze(1)                # (N, 28, 28)
    imgs_flat = imgs.reshape(len(ds), -1) # (N, 784)
    return imgs_flat


def build_image_cs_dataloaders(
    measurement_ratio: float = 0.25,
    sigma: float = 0.0,
    batch_size: int = 128,
    data_root: str = "./data",
    device: torch.device = None,
    seed: int = 42,
):
    """
    Build train / test DataLoaders for Fashion-MNIST compressed sensing.

    Args:
        measurement_ratio: m / d, e.g. 0.125, 0.25, or 0.5
        sigma:             Additive Gaussian noise std-dev on measurements
        batch_size:        Mini-batch size
        data_root:         Where to cache the dataset
        device:            Torch device
        seed:              RNG seed for sensing matrix and noise

    Returns:
        A:            Sensing matrix (m, d)
        Psi:          Full 2D-DCT basis matrix (d, d) — for reference
        train_loader: DataLoader yielding (y, alpha, x_flat)
        test_loader:  DataLoader yielding (y, alpha, x_flat)
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)

    d = 784   # 28 * 28
    m = int(measurement_ratio * d)
    if not 0.0 < measurement_ratio <= 1.0:
        raise ValueError("measurement_ratio must be in the interval (0, 1].")

    # Sensing matrix — random Gaussian, unit-l2-norm columns
    A = torch.randn(m, d, device=device)
    A = A / A.norm(dim=0, keepdim=True)

    Psi = get_dct_basis(28, 28, device=device)  # (784, 784) full 2D basis

    def _process(imgs_flat):
        imgs_flat = imgs_flat.to(device)
        alpha = dct2_flat(imgs_flat)                # (N, d)
        y     = alpha @ A.T                          # (N, m)
        if sigma > 0:
            y = y + sigma * torch.randn_like(y)
        return imgs_flat, alpha, y

    train_flat = _load_raw_fmnist(data_root, train=True)   # (60000, 784)
    test_flat  = _load_raw_fmnist(data_root, train=False)  # (10000, 784)

    tr_x, tr_alpha, tr_y = _process(train_flat)
    te_x, te_alpha, te_y = _process(test_flat)

    train_ds = CSImageDataset(tr_x, tr_alpha, tr_y, A)
    test_ds  = CSImageDataset(te_x, te_alpha, te_y, A)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=False)

    return A, Psi, train_loader, test_loader
