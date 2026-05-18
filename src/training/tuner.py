"""
Grid-search hyperparameter tuner for HyperLISTA.

HyperLISTA has only three scalar hyperparameters: c1, c2, c3.
Because there are no layer-level parameters, evaluation cost is cheap:
a single forward pass on a validation mini-batch suffices.

Strategy
--------
1. Coarse grid: search a log-spaced grid over each cᵢ in [lo, hi].
2. Zoom-in:     around the best coarse point, run a finer grid.

No backpropagation is needed — the gradient-free nature of this search
is one of HyperLISTA's key advantages over LISTA/ALISTA.
"""

from __future__ import annotations
import itertools
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def _eval_hyperlista(
    model,
    b: torch.Tensor,
    x_true: torch.Tensor,
    c1: float,
    c2: float,
    c3: float,
) -> float:
    """
    Evaluate NMSE (dB) for a given (c1, c2, c3) triple on a single batch.

    Returns:
        NMSE in dB (lower is better)
    """
    model.set_hyperparams(c1, c2, c3)
    with torch.no_grad():
        x_hat = model(b)
    num = ((x_hat - x_true) ** 2).sum(dim=-1)
    den = (x_true ** 2).sum(dim=-1).clamp(min=1e-12)
    nmse_db = 10.0 * torch.log10((num / den).mean()).item()
    return nmse_db


def coarse_grid_search(
    model,
    val_loader: DataLoader,
    device: torch.device,
    c1_range: tuple = (0.1, 5.0),
    c2_range: tuple = (0.1, 5.0),
    c3_range: tuple = (0.1, 5.0),
    n_points: int = 5,
    n_batches: int = 4,
    verbose: bool = True,
) -> tuple[float, float, float, float]:
    """
    Coarse grid search over (c1, c2, c3) on log-scale.

    Args:
        model:       HyperLISTA instance (already on device)
        val_loader:  Validation DataLoader
        device:      Torch device
        cX_range:    (lo, hi) search range for each hyperparameter
        n_points:    Number of grid points per dimension
        n_batches:   Number of validation batches to average over
        verbose:     Print progress

    Returns:
        best_c1, best_c2, best_c3, best_nmse_db
    """
    model = model.to(device)
    model.eval()

    # Gather validation batches
    val_batches = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        b, x_true = batch[0].to(device), batch[1].to(device)
        val_batches.append((b, x_true))

    c1_grid = np.logspace(np.log10(c1_range[0]), np.log10(c1_range[1]), n_points)
    c2_grid = np.logspace(np.log10(c2_range[0]), np.log10(c2_range[1]), n_points)
    c3_grid = np.logspace(np.log10(c3_range[0]), np.log10(c3_range[1]), n_points)

    combinations = list(itertools.product(c1_grid, c2_grid, c3_grid))

    best_nmse = float("inf")
    best_c1 = best_c2 = best_c3 = 1.0

    it = tqdm(combinations, desc="Coarse grid") if verbose else combinations

    for c1, c2, c3 in it:
        nmse_sum = 0.0
        for b, x_true in val_batches:
            nmse_sum += _eval_hyperlista(model, b, x_true, float(c1), float(c2), float(c3))
        avg_nmse = nmse_sum / len(val_batches)

        if avg_nmse < best_nmse:
            best_nmse = avg_nmse
            best_c1, best_c2, best_c3 = float(c1), float(c2), float(c3)

        if verbose:
            it.set_postfix(best_nmse=f"{best_nmse:.2f} dB")

    return best_c1, best_c2, best_c3, best_nmse


def fine_grid_search(
    model,
    val_loader: DataLoader,
    device: torch.device,
    c1_center: float,
    c2_center: float,
    c3_center: float,
    zoom_factor: float = 3.0,
    n_points: int = 7,
    n_batches: int = 8,
    verbose: bool = True,
) -> tuple[float, float, float, float]:
    """
    Fine grid search around the coarse best point.

    The search interval for each cᵢ is
      [cᵢ_center / zoom_factor, cᵢ_center * zoom_factor]
    on a log-scale with n_points grid points.

    Returns:
        best_c1, best_c2, best_c3, best_nmse_db
    """
    c1_range = (c1_center / zoom_factor, c1_center * zoom_factor)
    c2_range = (c2_center / zoom_factor, c2_center * zoom_factor)
    c3_range = (c3_center / zoom_factor, c3_center * zoom_factor)

    return coarse_grid_search(
        model,
        val_loader,
        device,
        c1_range=c1_range,
        c2_range=c2_range,
        c3_range=c3_range,
        n_points=n_points,
        n_batches=n_batches,
        verbose=verbose,
    )


def tune_hyperlista(
    model,
    val_loader: DataLoader,
    device: torch.device,
    coarse_points: int = 5,
    fine_points: int = 7,
    coarse_batches: int = 4,
    fine_batches: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Two-stage grid search: coarse then fine zoom-in.

    Args:
        model:          HyperLISTA instance
        val_loader:     Validation DataLoader
        device:         Torch device
        coarse_points:  Grid points per dim in coarse stage
        fine_points:    Grid points per dim in fine stage
        coarse_batches: Validation batches for coarse stage
        fine_batches:   Validation batches for fine stage
        verbose:        Print progress

    Returns:
        result dict with keys: c1, c2, c3, nmse_db
    """
    if verbose:
        print("=== Stage 1: Coarse grid search ===")
    t0 = time.time()
    c1, c2, c3, nmse_coarse = coarse_grid_search(
        model, val_loader, device,
        n_points=coarse_points,
        n_batches=coarse_batches,
        verbose=verbose,
    )
    if verbose:
        print(f"Coarse best: c1={c1:.4f}, c2={c2:.4f}, c3={c3:.4f}  "
              f"NMSE={nmse_coarse:.2f} dB  ({time.time()-t0:.1f}s)")

    if verbose:
        print("\n=== Stage 2: Fine grid search ===")
    t1 = time.time()
    c1, c2, c3, nmse_fine = fine_grid_search(
        model, val_loader, device,
        c1_center=c1, c2_center=c2, c3_center=c3,
        n_points=fine_points,
        n_batches=fine_batches,
        verbose=verbose,
    )
    if verbose:
        print(f"Fine best:   c1={c1:.4f}, c2={c2:.4f}, c3={c3:.4f}  "
              f"NMSE={nmse_fine:.2f} dB  ({time.time()-t1:.1f}s)")

    model.set_hyperparams(c1, c2, c3)
    return {"c1": c1, "c2": c2, "c3": c3, "nmse_db": nmse_fine}
