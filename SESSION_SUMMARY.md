# Session Summary — HyperLISTA Project

---

## Session 2 — June 13, 2026

### What was done

Following professor feedback ("compare training methods for unfolded optimizers" and "no DCT needed for MNIST"), two new notebooks and supporting infrastructure were added.

#### New: `src/training/trainer.py` — `train_sequential()`

Implements **L3 (sequential/greedy) training** from Lecture 6:
- Layer *k* is trained to minimise MSE at its own output while layers 0,…,k−1 are frozen
- Requires the model to have a `.layers` attribute (e.g. LISTA with `tied=False`)
- Returns per-layer training history

```python
hist = train_sequential(model, train_loader, val_loader,
                        n_epochs_per_layer=20, lr=1e-3, patience=15)
```

Existing `train()` already covered L1 (`intermediate_weight=0.0`) and L2 (`intermediate_weight>0`).

#### New: `src/data/image_loader.py` — `build_pixel_cs_dataloaders()`

Pixel-domain compressed sensing for FashionMNIST (no DCT):
- Returns `(A, train_loader, test_loader)` with batches of `(b, x_flat)`
- Signal model: `b = Ax`, `x ∈ [0,1]^{784}` (naturally ~65 % sparse)

#### New: `notebooks/05_training_methods_comparison.ipynb`

Systematic comparison of L1 / L2 / L3 × {LISTA, LISTA-Tied} with ALISTA and HyperLISTA as reference:
- LISTA-Tied uses `LISTA(A, n_layers=K, tied=True)` — single (W_y, W_x, θ) shared across all K layers
- Produces NMSE-vs-layer plot and parameter-efficiency scatter

#### New: `notebooks/06_fashion_mnist_pixel_domain.ipynb`

FashionMNIST CS in pixel domain for measurement ratios {0.125, 0.25, 0.5}:
- ISTA, FISTA, LISTA, ALISTA, HyperLISTA evaluated with PSNR, SSIM, NMSE
- Includes sparsity analysis (pixel histogram) and reconstruction image grid

#### Bugfix: `.gitignore`
- `data/` → `/data/` (root-anchored): the old pattern accidentally excluded `src/data/` Python source files.

### Updated results (Notebook 02, more recent run)

| Model | NMSE @ 16 layers (noiseless) | # Parameters |
|-------|------------------------------|--------------|
| ISTA | ~-5.3 dB | 0 |
| FISTA | ~-11.0 dB | 0 |
| LISTA | ~-22.6 dB | ~6 M |
| ALISTA | ~-29.8 dB | 32 |
| HyperLISTA | **~-61.4 dB** | **3** |

*(Previous session summary listed LISTA at -12.6 dB — that was from an early run. Newer run with updated code gives -22.6 dB.)*

### File map (Session 2)

```
src/
  data/
    image_loader.py   ← added build_pixel_cs_dataloaders()
  training/
    trainer.py        ← added train_sequential() (L3 greedy)
    __init__.py       ← added exports
  models/
    __init__.py       ← added exports

notebooks/
  05_training_methods_comparison.ipynb   ← NEW
  06_fashion_mnist_pixel_domain.ipynb    ← NEW

results/
  checkpoints/
    hyperlista_*_hparams.json            ← NEW (all experiments)

.gitignore  ← fixed data/ → /data/
```

---

## Session 1 — June 5–6, 2026
**Author:** Etay Baron

---

## TL;DR

4 critical bugs were found and fixed. All learned models (LISTA, ALISTA, HyperLISTA) were producing garbage results (+4.79 dB / 0.00 dB). After fixes: ALISTA reaches **-30 dB**, HyperLISTA reaches **-60 dB**. Classical ISTA/FISTA were fine throughout.

---

## Bug 1 — W Matrix Divergence (Root Cause of Most Problems)

**Affected files:** `src/models/alista.py`, `src/models/hyperlista.py`

**What was wrong:**  
Both ALISTA and HyperLISTA compute a fixed weight matrix W using iterative Adam optimization. The normalization step had:
```python
diag_vals = (W.T @ A).diag().clamp(min=1e-8)
W = W / diag_vals[None, :]
```
When Adam pushed diagonal elements negative, clamping to 1e-8 instead of the true (negative) value multiplied those columns by ~10^8. W exploded. The mutual coherence μ went from the expected **0.22** to **1.49**.

**Effect on models:**
- With μ=1.49: threshold θ = c1·μ·‖A⁺b‖₁ ≈ 117·c1. For c1_min=0.1 → θ≈11.7, larger than any gradient step → everything zeroed every iteration.
- ALISTA val_nmse: **+4.79 dB** (worse than guessing zero)
- HyperLISTA val_nmse: **≈0.00 dB**

**Fix:** Replaced iterative optimization with the analytic closed-form solution W* = (AA^T + εI)^{-1} A, then normalize so diag(W^T A) = 1.

```python
# alista.py and hyperlista.py
def compute_alista_weight(A, n_iter=5000, lr=1e-3):
    m = A.shape[0]
    with torch.no_grad():
        AAt_reg = A @ A.T + 1e-6 * torch.eye(m, device=A.device, dtype=A.dtype)
        W = torch.linalg.solve(AAt_reg, A)
        diag_vals = (W.T @ A).diag().clamp(min=1e-6)
        W = W / diag_vals[None, :]
    return W
```

**Result:** μ = 0.22, ALISTA → **-30 dB**, HyperLISTA → **-60 dB**

---

## Bug 2 — Heavy-Ball Momentum Instability

**Affected file:** `src/models/hyperlista.py`

**What was wrong:**  
β = c2·μ·‖x‖₀ had no upper bound. Polyak heavy-ball requires β < 1, otherwise the iteration diverges.

**Fix:**
```python
# Before:
beta = (c2 * mu * sparsity).clamp(min=0.0)
# After:
beta = (c2 * mu * sparsity).clamp(min=0.0, max=0.99)
```

---

## Bug 3 — Grid Search Range Out of Bounds

**Affected file:** `src/training/tuner.py`

**What was wrong:**  
All three hyperparameters searched in (0.1, 5.0). With μ=0.22:
- Optimal c1 ≈ 0.05 → **outside the search range** (range started at 0.1)
- For heavy-ball stability: c2·μ·50 < 1 → need c2 < 0.09. Range started at 0.1 → **always unstable**

**Fix:** Narrowed to physically meaningful values:
```python
c1_range: tuple = (0.01, 0.2),   # threshold scaling
c2_range: tuple = (0.005, 0.1),  # momentum (must keep β<1)
c3_range: tuple = (0.5, 30.0),   # support selection growth
```

---

## Bug 4 — Sensing Matrix Mismatch in Notebook 03

**Affected file:** `notebooks/03_sparse_generalization_experiments.ipynb` (cell-6)

**What was wrong:**  
The `eval_on_shifted` test function called `build_sparse_dataloaders(seed=99)`, which generated a **new random sensing matrix A** (seed=99 ≠ training seed=42). All models were trained on A_42 but tested with b = A_99 @ x → complete failure (ISTA: +1.44 dB).

**Fix in notebook cell-6:** Generate test signals directly using the same A:
```python
def eval_on_shifted(A, models, s_test, sigma_test, mag_std_test, label):
    torch.manual_seed(99)   # seeds only signal generation, not A
    X_test = generate_sparse_signals(2048, N, s_test, ...)
    B_test = generate_measurements(A, X_test, sigma=sigma_test)  # same A!
    ...
```

**Also fixed in `src/data/sparse_generator.py`:** Added optional `A` parameter to `build_sparse_dataloaders()` so callers can pass a pre-built A instead of generating a new one:
```python
def build_sparse_dataloaders(..., A: torch.Tensor = None):
    if A is None:
        A = generate_sensing_matrix(m, n, device)
    else:
        A = A.to(device)
```

> **Note:** The notebook fix is written to the file but the **kernel still has the old function cached**. Must do **Kernel → Restart Kernel and Run All Cells** before results update.

---

## Additional Fixes (Smaller but Important)

### `src/training/trainer.py` — DataLoader compatibility + tensor device
- Changed `for b, x_true in loader` → `for batch in loader; b, x_true = batch[0], batch[1]`
  - **Why:** The image DataLoader (notebook 04) returns a triplet `(y, alpha, x_flat)`. Unpacking to 2 values would crash with `ValueError`.
- Fixed `torch.tensor(float(k+1))` → `torch.tensor(float(k+1), device=device)` in intermediate loss weights.

### `src/models/lista.py` — `torch.eye` missing device
- Fixed: `torch.eye(A.shape[1])` → `torch.eye(A.shape[1], device=A.device)`
- Without this, W_x initialization would fail when A is on GPU.

### All models — `A.detach().clone()` in `__init__`
- Added `A = A.detach().clone()` at the top of `__init__` in ISTA, FISTA, ALISTA, HyperLISTA.
- Prevents gradient flow back through A (which could corrupt the sensing matrix during training).

### `src/models/hyperlista.py` — NaN guards
Added `nan_to_num` at three points to prevent NaN propagation in edge cases:
```python
Apinv_r_l1 = torch.nan_to_num(Apinv_r_l1, nan=0.0, posinf=1e6)
p_float = torch.nan_to_num(p_float, nan=0.0, posinf=float(n), neginf=0.0)
v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
```

### `src/evaluation/visualizer.py` — Removed `matplotlib.use("Agg")`
- Was called at import time, forcing non-interactive backend → no inline plots in Jupyter.

### `src/operators/dct_operators.py` — New module
- New file with proper orthonormal 2D-DCT-II operators for image CS.
- `dct2_flat(x_flat)` and `idct2_flat(alpha_flat)` for image-to-frequency and back.
- `get_dct_basis(H, W, device)` returns the full (H·W × H·W) DCT basis matrix.
- Used by `image_loader.py` for notebook 04.

### `src/data/image_loader.py` — Multiple improvements
- Replaced `_build_dct2_matrix` with `get_dct_basis` from new operators module.
- Added graceful offline error message when Fashion-MNIST download fails.
- Added validation: `measurement_ratio` must be in (0, 1].

---

## Current Expected Results (after all fixes, Session 1 + 2)

| Model | NMSE @ 16 layers (noiseless) | # Parameters |
|-------|------------------------------|--------------|
| ISTA | ~-5.3 dB | 0 |
| FISTA | ~-11.0 dB | 0 |
| LISTA | ~-22.6 dB | ~6 M |
| ALISTA | ~-29.8 dB | 32 |
| HyperLISTA | **~-61.4 dB** | **3** |

---

## What Still Needs to Be Done

### Before seminar (June 28)

1. **Notebook 03** — Restart kernel and run all cells (~15 min).  
   Verify: ISTA sanity check gives ~-5 dB (not +1.44 dB).

2. **Notebook 04 (Image CS)** — Fashion-MNIST download fails due to DNS failure.  
   **Manual fix:** Download the 4 files and place in `data/FashionMNIST/raw/`:
   - `train-images-idx3-ubyte.gz`
   - `train-labels-idx1-ubyte.gz`
   - `t10k-images-idx3-ubyte.gz`
   - `t10k-labels-idx1-ubyte.gz`  
   After placing files: run notebook 04 (~90 min). Measures m/d ∈ {0.125, 0.25, 0.5}. Metrics: PSNR and SSIM.

3. **Slides** — Currently only show LISTA as proposed method. Need to add:
   - ALISTA (analytic W, 32 parameters)
   - HyperLISTA (3 hyperparameters, grid search, adaptive θ/β/p)
   - Results graphs from notebooks 03 and 04

4. **GitHub repo** — Make public for submission (`github.com/ThEpiCake/hyperlista-mbdl`).

### Before final report (July 30)
- 10-page report: Motivation, Problem, Method, Results, Discussion, References.

---

## File Map of All Changes

```
src/
  operators/
    dct_operators.py     ← NEW: orthonormal 2D-DCT-II operators for image CS
    __init__.py          ← NEW

  data/
    sparse_generator.py  ← build_sparse_dataloaders(): added optional A param
    image_loader.py      ← DCT refactor, offline error message, ratio validation

  models/
    ista.py              ← A.detach().clone() in __init__
    fista.py             ← A.detach().clone() in __init__
    lista.py             ← torch.eye device fix
    alista.py            ← compute_alista_weight(): analytic (was iterative Adam)
                            A.detach().clone() in __init__
    hyperlista.py        ← compute_hyperlista_weight(): analytic (was iterative)
                            beta clamp max=0.99
                            NaN guards (nan_to_num x3)
                            A.detach().clone() in __init__

  training/
    trainer.py           ← DataLoader batch unpacking fix, tensor device fix
    tuner.py             ← coarse grid ranges narrowed

  evaluation/
    visualizer.py        ← Removed matplotlib.use("Agg")

notebooks/
  03_sparse_generalization_experiments.ipynb
                         ← cell-6: eval_on_shifted() uses same A (was new random A)
```

---

## Key Insight for Seminar

The main story:
- **LISTA** needs ~6 million parameters to reach -12 dB — by learning W, W_x, W_y, θ per layer
- **ALISTA** needs only **32 parameters** (per-layer γ, θ) to reach -30 dB — by computing W analytically, removing the need to learn it
- **HyperLISTA** needs only **3 hyperparameters** found by grid search (zero backprop) to reach -60 dB — by making θ, β, p adaptive to the current iterate

Each model adds more inductive bias (more model knowledge), uses fewer parameters, and performs better. This is the core MBDL message: **structured models beat black-box learners**.
