"""
Standard training loop for learnable models (LISTA, ALISTA) via BPTT.

All models receive (b, x_true) pairs and minimise MSE at the final layer.
An optional intermediate-supervision variant adds weighted losses at each layer.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def nmse_db(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Normalised MSE in dB: 10 log10( ||x_hat - x_true||^2 / ||x_true||^2 )."""
    num = ((x_hat - x_true) ** 2).sum(dim=-1)
    den = (x_true ** 2).sum(dim=-1).clamp(min=1e-12)
    return 10.0 * torch.log10((num / den).mean()).item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    intermediate_weight: float = 0.0,
) -> float:
    """
    One epoch of supervised training.

    Args:
        model:                Learnable model (LISTA / ALISTA)
        loader:               DataLoader yielding (b, x_true)
        optimiser:            Torch optimiser
        device:               Compute device
        intermediate_weight:  If > 0, add weighted sum of intermediate losses

    Returns:
        Average training MSE loss for this epoch
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        b, x_true = batch[0].to(device), batch[1].to(device)
        optimiser.zero_grad()

        # Intermediate-layer supervision:
        if intermediate_weight > 0.0:
            iterates = model(b, return_all=True)
            K = len(iterates)

            final_loss = nn.functional.mse_loss(iterates[-1], x_true)

            weights = torch.tensor(
                [torch.log(torch.tensor(float(k + 2), device=device)) for k in range(K)],
                device=device,
            )
            weights = weights / weights.sum()

            intermediate_loss = sum(
                weights[k] * nn.functional.mse_loss(iterates[k], x_true)
                for k in range(K)
            )

            loss = final_loss + intermediate_weight * intermediate_loss
        else:
            x_hat = model(b)
            loss = nn.functional.mse_loss(x_hat, x_true)


        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate NMSE (dB) on a DataLoader.

    Returns:
        NMSE in dB averaged over the dataset
    """
    model.eval()
    all_nmse = []
    for batch in loader:
        b, x_true = batch[0].to(device), batch[1].to(device)
        x_hat = model(b)
        nmse  = nmse_db(x_hat, x_true)
        all_nmse.append(nmse)
    return sum(all_nmse) / len(all_nmse)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = None,
    patience: int = 20,
    intermediate_weight: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Full training loop with early stopping.

    Args:
        model:                Learnable model
        train_loader:         Training DataLoader
        val_loader:           Validation DataLoader
        n_epochs:             Maximum epochs
        lr:                   Adam learning rate
        weight_decay:         L2 regularisation
        device:               Compute device
        patience:             Early-stop patience (epochs without improvement)
        intermediate_weight:  Weight for intermediate-layer supervision
        verbose:              Print progress bar

    Returns:
        history: dict with 'train_loss', 'val_nmse_db', 'best_epoch'
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_nmse_db": [], "best_epoch": 0}
    best_val  = float("inf")
    no_improve = 0
    best_state = None

    epoch_iter = range(n_epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimiser, device, intermediate_weight
        )
        val_nmse = evaluate(model, val_loader, device)
        scheduler.step(val_nmse)

        history["train_loss"].append(train_loss)
        history["val_nmse_db"].append(val_nmse)

        if val_nmse < best_val:
            best_val  = val_nmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_nmse=f"{val_nmse:.2f} dB",
                lr=f"{optimiser.param_groups[0]['lr']:.2e}",
            )

        if no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (best val NMSE = {best_val:.2f} dB)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


def train_sequential(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs_per_layer: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = None,
    patience: int = 15,
    verbose: bool = True,
) -> dict:
    """
    L3 sequential (greedy) training: layer k is trained while layers 0..k-1 are frozen.

    At step k the loss is evaluated at the output of layer k (not the final layer),
    making each step a shallow optimisation problem. Intended for LISTA with tied=False.

    Args:
        model:               Model with a .layers attribute and .n signal dimension
        train_loader:        DataLoader yielding (b, x_true)
        val_loader:          DataLoader yielding (b, x_true)
        n_epochs_per_layer:  Maximum epochs per layer
        lr:                  Adam learning rate
        weight_decay:        L2 regularisation
        device:              Compute device
        patience:            Early-stop patience per layer
        verbose:             Print per-layer progress

    Returns:
        history dict with 'layer_histories' (list of per-layer dicts with
        'train_loss' and 'val_nmse_db')
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    if not hasattr(model, "layers") or not hasattr(model, "n"):
        raise ValueError(
            "train_sequential requires a model with 'layers' (nn.ModuleList) "
            "and 'n' (signal dimension) attributes."
        )

    K = model.n_layers
    n_signal = model.n
    history: dict = {"layer_histories": []}

    # Freeze all parameters before greedy training starts
    for p in model.parameters():
        p.requires_grad_(False)

    for k in range(K):
        # Unfreeze only the k-th layer
        for p in model.layers[k].parameters():
            p.requires_grad_(True)

        active_params = [p for p in model.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(active_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=10
        )

        best_val = float("inf")
        no_improve = 0
        best_state: dict | None = None
        layer_hist: dict = {"train_loss": [], "val_nmse_db": []}

        epoch_iter = range(n_epochs_per_layer)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc=f"Layer {k + 1:2d}/{K}", unit="epoch")

        for _ in epoch_iter:
            # Forward through layers 0..k, loss at layer-k output
            model.train()
            total_loss, n_batches = 0.0, 0
            for batch in train_loader:
                b, x_true = batch[0].to(device), batch[1].to(device)
                optimiser.zero_grad()
                x = torch.zeros(b.shape[0], n_signal, device=device, dtype=b.dtype)
                for i in range(k + 1):
                    x = model.layers[i](b, x)
                loss = nn.functional.mse_loss(x, x_true)
                loss.backward()
                nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
                optimiser.step()
                total_loss += loss.item()
                n_batches += 1
            train_loss = total_loss / max(n_batches, 1)

            # Validate at the same k+1-layer prefix
            model.eval()
            val_nmses = []
            with torch.no_grad():
                for batch in val_loader:
                    b, x_true = batch[0].to(device), batch[1].to(device)
                    x = torch.zeros(b.shape[0], n_signal, device=device, dtype=b.dtype)
                    for i in range(k + 1):
                        x = model.layers[i](b, x)
                    val_nmses.append(nmse_db(x, x_true))
            val_nmse = sum(val_nmses) / len(val_nmses)

            scheduler.step(val_nmse)
            layer_hist["train_loss"].append(train_loss)
            layer_hist["val_nmse_db"].append(val_nmse)

            if val_nmse < best_val:
                best_val = val_nmse
                best_state = {nm: v.cpu().clone() for nm, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                epoch_iter.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_nmse=f"{val_nmse:.2f} dB",
                )

            if no_improve >= patience:
                if verbose:
                    print(f"\n  Early stop (best {best_val:.2f} dB)")
                break

        if best_state is not None:
            model.load_state_dict({nm: v.to(device) for nm, v in best_state.items()})

        history["layer_histories"].append(layer_hist)

    # Restore requires_grad for all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    return history
