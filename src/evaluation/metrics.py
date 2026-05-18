"""
Evaluation metrics for sparse recovery and image reconstruction.

Public API
----------
  mse(x_hat, x_true)       -> float
  nmse(x_hat, x_true)      -> float
  nmse_db(x_hat, x_true)   -> float (dB)
  psnr(x_hat, x_true, ...)  -> float (dB)
  ssim_batch(x_hat, x_true, H, W) -> float

  evaluate_model(model, loader, device, ...) -> dict with all metrics
  count_parameters(model)  -> int
  measure_runtime(model, b, n_runs) -> float (ms)
"""

from __future__ import annotations
import time
import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity


# ─── Basic signal metrics ─────────────────────────────────────────────────────

def mse(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Mean squared error averaged over batch and signal dimensions."""
    return ((x_hat - x_true) ** 2).mean().item()


def nmse(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Normalised MSE: E[ ||x_hat - x||^2 / ||x||^2 ].

    Averaged over the batch dimension.
    """
    num = ((x_hat - x_true) ** 2).sum(dim=-1)
    den = (x_true ** 2).sum(dim=-1).clamp(min=eps)
    return (num / den).mean().item()


def nmse_db(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-12) -> float:
    """NMSE in dB: 10 log10(NMSE)."""
    return 10.0 * math.log10(max(nmse(x_hat, x_true, eps), 1e-30))


def psnr(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    PSNR = 10 log10( max_val^2 / MSE )

    Args:
        x_hat:   Reconstructed signal (N, d)
        x_true:  Ground truth (N, d)
        max_val: Maximum possible pixel/signal value
    """
    mse_val = mse(x_hat, x_true)
    if mse_val < eps:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse_val)


def ssim_batch(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    H: int = 28,
    W: int = 28,
    data_range: float = 1.0,
) -> float:
    """
    Mean SSIM over a batch of flattened images.

    Args:
        x_hat:  (N, H*W) reconstructed
        x_true: (N, H*W) ground truth
        H, W:   Spatial dimensions

    Returns:
        Mean SSIM (higher is better)
    """
    x_hat_np  = x_hat.detach().cpu().float().numpy()
    x_true_np = x_true.detach().cpu().float().numpy()

    ssim_vals = []
    for i in range(x_hat_np.shape[0]):
        img_hat  = x_hat_np[i].reshape(H, W).clip(0.0, data_range)
        img_true = x_true_np[i].reshape(H, W).clip(0.0, data_range)
        s = structural_similarity(img_hat, img_true, data_range=data_range)
        ssim_vals.append(s)
    return float(np.mean(ssim_vals))


# ─── Runtime measurement ──────────────────────────────────────────────────────

def measure_runtime(
    model: nn.Module,
    b: torch.Tensor,
    n_runs: int = 50,
    warmup: int = 5,
) -> float:
    """
    Measure average inference time in milliseconds.

    Args:
        model:  Torch model in eval mode
        b:      Input batch (N, m)
        n_runs: Number of timed forward passes
        warmup: Warmup passes (not timed)

    Returns:
        Average runtime per forward pass in milliseconds
    """
    model.eval()
    device = b.device

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(b)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1000.0   # ms


def count_parameters(model: nn.Module) -> int:
    """Count learnable parameters.  For HyperLISTA, returns 3."""
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Full model evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    image_mode: bool = False,
    H: int = 28,
    W: int = 28,
    max_val: float = 1.0,
    measure_time: bool = True,
) -> dict:
    """
    Compute all metrics for a model on a DataLoader.

    Args:
        model:       Model in eval mode
        loader:      DataLoader yielding (b, x_true) or (y, alpha, x_flat)
        device:      Torch device
        image_mode:  If True, also compute PSNR and SSIM on pixel-space images
                     (loader yields (y, alpha, x_flat) and model predicts alpha)
        H, W:        Spatial dims for SSIM (only used when image_mode=True)
        max_val:     Max pixel value for PSNR
        measure_time:Measure inference speed on first batch

    Returns:
        dict with keys: mse, nmse, nmse_db, [psnr, ssim,] runtime_ms, n_params
    """
    model.eval()
    model.to(device)

    all_mse   = []
    all_nmse  = []
    all_psnr  = []
    all_ssim  = []
    runtime   = None

    from src.operators.dct_operators import idct2_flat  # lazy import

    for i, batch in enumerate(loader):
        if image_mode:
            y, alpha_true, x_flat_true = [t.to(device) for t in batch]
            # Model predicts DCT coefficients alpha
            alpha_hat = model(y)
            x_hat = idct2_flat(alpha_hat, H, W)
            x_true = x_flat_true
            target_hat  = alpha_hat
            target_true = alpha_true
        else:
            b, x_true = batch[0].to(device), batch[1].to(device)
            x_hat = model(b)
            target_hat  = x_hat
            target_true = x_true

        if measure_time and i == 0:
            input_b = batch[0].to(device)
            runtime = measure_runtime(model, input_b)

        all_mse.append(mse(target_hat, target_true))
        all_nmse.append(nmse(target_hat, target_true))

        if image_mode:
            all_psnr.append(psnr(x_hat, x_true, max_val=max_val))
            all_ssim.append(ssim_batch(x_hat, x_true, H=H, W=W, data_range=max_val))

    avg_mse  = float(np.mean(all_mse))
    avg_nmse = float(np.mean(all_nmse))
    avg_nmse_db = 10.0 * math.log10(max(avg_nmse, 1e-30))

    result = {
        "mse":        avg_mse,
        "nmse":       avg_nmse,
        "nmse_db":    avg_nmse_db,
        "runtime_ms": runtime,
        "n_params":   count_parameters(model),
    }
    if image_mode:
        result["psnr"] = float(np.mean(all_psnr))
        result["ssim"] = float(np.mean(all_ssim))

    return result
