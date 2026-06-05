"""
Synthetic sparse vector generation for Part A experiments.

Measurement model: b = A x* + epsilon
  x*   in R^n  -- s-sparse ground truth
  A    in R^{m x n} -- random Gaussian sensing matrix (unit-l2-norm columns)
  b    in R^m   -- noisy observations
  eps  ~ N(0, sigma^2 I)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class SparseDataset(Dataset):
    """Pre-generated dataset of (measurement, ground-truth) pairs."""

    def __init__(
        self,
        A: torch.Tensor,
        x_true: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        Args:
            A:      Sensing matrix  (m, n)
            x_true: Ground-truth    (N, n)
            b:      Measurements    (N, m)
        """
        self.A = A
        self.x_true = x_true
        self.b = b

    def __len__(self):
        return self.x_true.shape[0]

    def __getitem__(self, idx):
        return self.b[idx], self.x_true[idx]


def generate_sensing_matrix(m: int, n: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random Gaussian sensing matrix with unit-l2-norm columns.

    Args:
        m:      Number of measurements (rows)
        n:      Signal dimension (columns)
        device: Torch device

    Returns:
        A: (m, n) sensing matrix
    """
    if device is None:
        device = torch.device("cpu")
    A = torch.randn(m, n, device=device)
    # Normalise each column to unit l2 norm
    A = A / A.norm(dim=0, keepdim=True)
    return A


def generate_sparse_signals(
    N: int,
    n: int,
    s: int,
    magnitude_std: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate N s-sparse vectors of dimension n.

    Non-zero values are drawn from N(0, magnitude_std^2).

    Args:
        N:             Number of signals
        n:             Ambient dimension
        s:             Sparsity level (number of non-zeros)
        magnitude_std: Std-dev of non-zero entries
        device:        Torch device

    Returns:
        X: (N, n) sparse signal matrix
    """
    if device is None:
        device = torch.device("cpu")
    
    # Generate all-zero matrix and fill in s random positions for each signal
    # X is the ground-truth signal matrix, where each row is an s-sparse vector of dimension n
    X = torch.zeros(N, n, device=device)
    for i in range(N):
        support = torch.randperm(n, device=device)[:s]
        X[i, support] = magnitude_std * torch.randn(s, device=device)
    return X


def generate_measurements(
    A: torch.Tensor,
    X: torch.Tensor,
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Generate noisy measurements B = X @ A^T + noise.

    Args:
        A:     Sensing matrix  (m, n)
        X:     Signals         (N, n)
        sigma: Noise std-dev

    Returns:
        B: (N, m) measurement matrix
    """

    # Compute noiseless measurements
    B = X @ A.T          # (N, m)

    # Add Gaussian noise if sigma > 0
    if sigma > 0.0:
        B = B + sigma * torch.randn_like(B)
    return B


def build_sparse_dataloaders(
    m: int = 250,
    n: int = 500,
    s: int = 50,
    sigma: float = 0.0,
    n_train: int = 51200,
    n_val: int = 2048,
    n_test: int = 2048,
    batch_size: int = 256,
    magnitude_std: float = 1.0,
    device: torch.device | None = None,
    seed: int = 42,
    A: torch.Tensor = None,
):
    """
    Build train / val / test DataLoaders for the synthetic sparse-recovery task.

    Args:
        A: Optional fixed sensing matrix. If provided, generated signals and
           measurements reuse this matrix instead of sampling a new one.

    Returns:
        A:            Sensing matrix (m, n) on *device*
        train_loader: DataLoader
        val_loader:   DataLoader
        test_loader:  DataLoader
    """

    # Device and random seeds
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate sensing matrix A (or move to device if provided)
    if A is None:
        A = generate_sensing_matrix(m, n, device)
    else:
        A = A.to(device)
        if A.shape != (m, n):
            raise ValueError(f"Expected A with shape {(m, n)}, got {tuple(A.shape)}.")
        
    # Generate signals and measurements
    N_total = n_train + n_val + n_test
    X_all = generate_sparse_signals(N_total, n, s, magnitude_std, device)
    B_all = generate_measurements(A, X_all, sigma)

    # Split into train/val/test sets
    X_tr, X_va, X_te = X_all[:n_train], X_all[n_train:n_train+n_val], X_all[n_train+n_val:]
    B_tr, B_va, B_te = B_all[:n_train], B_all[n_train:n_train+n_val], B_all[n_train+n_val:]

    # Create Datasets and DataLoaders
    train_ds = SparseDataset(A, X_tr, B_tr)
    val_ds   = SparseDataset(A, X_va, B_va)
    test_ds  = SparseDataset(A, X_te, B_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return A, train_loader, val_loader, test_loader
