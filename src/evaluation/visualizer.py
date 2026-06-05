"""
Visualisation utilities for HyperLISTA experiments.

plot_nmse_vs_layers(results, ...)  — NMSE (dB) vs. layer index for multiple models
plot_image_comparison(...)         — Grid: original | ISTA | FISTA | LISTA | ALISTA | HyperLISTA
plot_hyperparameter_landscape(...)  — 2-D slice of the (c1, c2, c3) objective
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ─── NMSE-vs-layers plot ──────────────────────────────────────────────────────

def plot_nmse_vs_layers(
    results: dict[str, list[float]],
    title: str = "NMSE vs. Layer",
    xlabel: str = "Layer",
    ylabel: str = "NMSE (dB)",
    save_path: str | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Plot NMSE (dB) vs. unrolled layer index for multiple methods.

    Args:
        results:   {model_name: [nmse_layer_1, ..., nmse_layer_K]}
        title:     Figure title
        save_path: If given, save figure to this path

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "P", "*"]
    colors  = plt.cm.tab10.colors

    for idx, (name, nmse_list) in enumerate(results.items()):
        layers = list(range(1, len(nmse_list) + 1))
        ax.plot(
            layers,
            nmse_list,
            label=name,
            marker=markers[idx % len(markers)],
            markevery=max(1, len(layers) // 8),
            color=colors[idx % len(colors)],
            linewidth=1.8,
        )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Per-layer NMSE extraction helper ────────────────────────────────────────

@torch.no_grad()
def get_layerwise_nmse(
    model: nn.Module,
    loader,
    device: torch.device,
    n_batches: int = 8,
) -> list[float]:
    """
    Run model with return_all=True and compute NMSE at each layer.

    Returns:
        List of NMSE values (dB), one per layer
    """
    model.eval()
    model.to(device)

    layer_nmse_sums = None
    count = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        b, x_true = batch[0].to(device), batch[1].to(device)
        iterates = model(b, return_all=True)

        if layer_nmse_sums is None:
            layer_nmse_sums = [0.0] * len(iterates)

        for k, x_hat in enumerate(iterates):
            num = ((x_hat - x_true) ** 2).sum(dim=-1)
            den = (x_true ** 2).sum(dim=-1).clamp(min=1e-12)
            nmse = 10.0 * torch.log10((num / den).mean()).item()
            layer_nmse_sums[k] += nmse
        count += 1

    return [v / count for v in layer_nmse_sums]


# ─── Image comparison grid ────────────────────────────────────────────────────

@torch.no_grad()
def plot_image_comparison(
    models: dict[str, nn.Module],
    loader,
    device: torch.device,
    H: int = 28,
    W: int = 28,
    n_samples: int = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Display a grid: rows = samples, cols = [Original] + [model predictions].

    Args:
        models:    OrderedDict {name: model}.  Models predict alpha (DCT domain).
                   Reconstructed image = IDCT(alpha_hat).
        loader:    DataLoader yielding (y, alpha_true, x_flat_true)
        device:    Torch device
        H, W:      Image spatial dimensions
        n_samples: Number of samples to show (rows)
        save_path: If given, save to this path

    Returns:
        matplotlib Figure
    """
    from src.operators.dct_operators import idct2_flat

    # Fetch one batch
    batch = next(iter(loader))
    y, alpha_true, x_flat_true = [t[:n_samples].to(device) for t in batch]

    col_names = ["Original"] + list(models.keys())
    n_cols    = len(col_names)
    n_rows    = n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row in range(n_rows):
        # Original
        img_orig = x_flat_true[row].cpu().float().numpy().reshape(H, W)
        axes[row, 0].imshow(img_orig, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=10)

        for col_idx, (name, model) in enumerate(models.items()):
            model.eval()
            alpha_hat = model(y[[row]])           # (1, d)
            x_hat = idct2_flat(alpha_hat, H, W)  # (1, d)
            img_hat = x_hat[0].cpu().float().numpy().reshape(H, W).clip(0, 1)

            axes[row, col_idx + 1].imshow(img_hat, cmap="gray", vmin=0, vmax=1)
            axes[row, col_idx + 1].axis("off")
            if row == 0:
                axes[row, col_idx + 1].set_title(name, fontsize=10)

    fig.tight_layout(pad=0.2)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── Hyperparameter landscape (2-D slice) ────────────────────────────────────

def plot_hyperparameter_landscape(
    model,
    val_loader,
    device: torch.device,
    fixed: dict,
    sweep_pair: tuple[str, str],
    grid_values: list | None = None,
    n_batches: int = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot NMSE landscape over a 2-D slice of (c1, c2, c3) space.

    Args:
        model:       HyperLISTA instance
        val_loader:  Validation DataLoader
        device:      Torch device
        fixed:       Dict of fixed hyperparameter values, e.g. {'c3': 1.0}
        sweep_pair:  Names of the two hyperparams to sweep, e.g. ('c1', 'c2')
        grid_values: Values for both sweep axes (same grid applied to both)
        n_batches:   Batches to average over
        save_path:   Output path

    Returns:
        matplotlib Figure
    """
    import itertools

    if grid_values is None:
        grid_values = np.logspace(-1, 0.7, 15).tolist()

    a_name, b_name = sweep_pair
    grid = np.zeros((len(grid_values), len(grid_values)))

    # Precompute validation batches
    batches = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        batches.append((batch[0].to(device), batch[1].to(device)))

    def _eval(c1, c2, c3):
        model.set_hyperparams(c1, c2, c3)
        model.eval()
        sums = 0.0
        with torch.no_grad():
            for b, x_true in batches:
                x_hat = model(b)
                num = ((x_hat - x_true) ** 2).sum(-1)
                den = (x_true ** 2).sum(-1).clamp(1e-12)
                sums += 10.0 * torch.log10((num / den).mean()).item()
        return sums / len(batches)

    for i, av in enumerate(grid_values):
        for j, bv in enumerate(grid_values):
            params = dict(fixed)
            params[a_name] = float(av)
            params[b_name] = float(bv)
            grid[i, j] = _eval(params.get("c1", 1.0), params.get("c2", 1.0), params.get("c3", 1.0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(grid_values, grid_values, grid, levels=20, cmap="viridis_r")
    plt.colorbar(im, ax=ax, label="NMSE (dB)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(a_name, fontsize=12)
    ax.set_ylabel(b_name, fontsize=12)
    ax.set_title(f"NMSE landscape ({a_name} vs {b_name})", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
