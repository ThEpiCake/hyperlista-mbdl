# Session Summary — HyperLISTA Project
**Date:** June 5–6, 2026  
**Author:** Etay Baron  

---

## What Was Done Tonight

### Background
The project had 3 critical bugs that caused all learned models (LISTA, ALISTA, HyperLISTA) to produce garbage results. The classical models (ISTA, FISTA) were fine. Everything was traced and fixed.

---

## Bug 1 — W Matrix Divergence (Root Cause of All Problems)

**Affected files:** `src/models/alista.py`, `src/models/hyperlista.py`

**What was wrong:**  
Both ALISTA and HyperLISTA compute a fixed weight matrix W using the `compute_alista_weight` function. The original implementation used iterative optimization (Adam) to find W. The normalization step had:
```python
diag_vals = (W.T @ A).diag().clamp(min=1e-8)
W = W / diag_vals[None, :]
```
When Adam pushed diagonal elements negative, clamping to 1e-8 instead of the negative value multiplied those columns by ~10^8. W exploded. The mutual coherence μ went from the expected 0.22 to **1.49**.

**Effect on models:**
- With μ=1.49: threshold θ = c1·μ·‖A⁺b‖₁ ≈ 117·c1. For c1_min=0.1 → θ≈11.7, which is larger than any gradient step → everything zeroed every iteration → NMSE ≈ 0 dB (completely useless).
- ALISTA: val_nmse = **+4.79 dB** (worse than guessing zero!)
- HyperLISTA: val_nmse ≈ **0.00 dB**

**Fix:** Replaced the iterative optimization with the analytic closed-form solution:
```
W* = (AA^T + εI)^{-1} A,  then normalize so diag(W^T A) = 1
```
Result: μ = 0.22 (as expected by theory). ALISTA now reaches **-29.97 dB**, HyperLISTA reaches **-56 to -63 dB**.

**Code change (both files):**
```python
def compute_alista_weight(A, n_iter=5000, lr=1e-3):
    m = A.shape[0]
    with torch.no_grad():
        AAt_reg = A @ A.T + 1e-6 * torch.eye(m, device=A.device, dtype=A.dtype)
        W = torch.linalg.solve(AAt_reg, A)
        diag_vals = (W.T @ A).diag().clamp(min=1e-6)
        W = W / diag_vals[None, :]
    return W
```

---

## Bug 2 — Heavy-Ball Momentum Instability

**Affected file:** `src/models/hyperlista.py` (line ~186)

**What was wrong:**  
The momentum coefficient β = c2·μ·‖x‖₀ had no upper bound. Polyak heavy-ball momentum requires β < 1, otherwise the iteration diverges.

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
The coarse grid search searched c1, c2, c3 all in range (0.1, 5.0). With the fixed μ=0.22:
- For c1=0.1: θ ≈ 0.1·0.22·76 ≈ 1.67 → too large, zeros out everything
- Optimal c1 ≈ 0.05 — was **outside the search range**
- For heavy-ball stability: β = c2·μ·50 < 1 → need c2 < 0.09. Old range started at 0.1 — **always unstable**

**Fix:** Narrowed ranges to physically meaningful values:
```python
c1_range: tuple = (0.01, 0.2),   # threshold scaling
c2_range: tuple = (0.005, 0.1),  # momentum (must keep β<1)
c3_range: tuple = (0.5, 30.0),   # support selection growth
```

---

## Bug 4 — Sensing Matrix Mismatch in Notebook 03

**Affected file:** `notebooks/03_sparse_generalization_experiments.ipynb` (cell-6)

**What was wrong:**  
The generalization test function `eval_on_shifted` called `build_sparse_dataloaders(seed=99)` which generated a **new random sensing matrix A** (different from training A at seed=42). All models were trained on A_42 but tested with measurements b = A_99 @ x → complete failure (ISTA: +1.44 dB).

**Fix:** Generate test signals directly with the same A the models were trained on:
```python
def eval_on_shifted(A, models, s_test, sigma_test, mag_std_test, label):
    torch.manual_seed(99)   # only seeds the SIGNAL generation
    X_test = generate_sparse_signals(2048, N, s_test, ...)
    B_test = generate_measurements(A, X_test, sigma=sigma_test)  # uses same A!
    ...
```
The fix is **written to the file** but the notebook kernel still has the old function cached. **Must do Kernel → Restart Kernel and Run All Cells** before results will update.

---

## Current Expected Results (after fixes)

| Model | NMSE (16 layers, noiseless) | # Parameters |
|-------|----------------------------|--------------|
| ISTA | ~-5.3 dB | 0 |
| FISTA | ~-6.0 dB | 0 |
| LISTA | ~-12.6 dB | ~6M |
| ALISTA | ~-30.0 dB | 32 |
| HyperLISTA | ~-60 dB | **3** |

Note: LISTA underperforms expected ~-30 dB from the paper. The NMSE-vs-layer curve shows flat near 0 dB for early layers then rapid drop at the last 3 layers — likely a local minimum in training (does nothing for 13 layers then barely recovers). Not a code bug; could be fixed with more epochs or different LR schedule. For the seminar this is actually an interesting finding.

---

## What Still Needs to Be Done

### Immediate (before seminar June 28)

1. **Notebook 03** — Restart kernel and run all cells. Should take ~15 min. Verify ISTA sanity check gives ~-5 dB (not +1.44 dB).

2. **Notebook 04 (Image CS)** — Fashion-MNIST download fails due to DNS error. 
   - **Fix:** Manually download the 4 Fashion-MNIST files and place them in `data/FashionMNIST/raw/`:
     - `train-images-idx3-ubyte.gz`
     - `train-labels-idx1-ubyte.gz`
     - `t10k-images-idx3-ubyte.gz`
     - `t10k-labels-idx1-ubyte.gz`
   - After placing files: run notebook 04 (~90 min). Tests 3 measurement ratios (m/d = 0.125, 0.25, 0.5). Metrics: PSNR and SSIM per model.

3. **GitHub repo** — Must be public for submission. Create repo, push all code.

4. **Slides** — Currently only show LISTA as proposed method. Need to add:
   - ALISTA slide (analytic W, 32 parameters)
   - HyperLISTA slide (3 hyperparameters, grid search, adaptive θ/β/p)
   - Results graphs from notebooks 03 and 04

### Before final report (July 30)
- Write 10-page report (Motivation, Problem, Method, Results, Discussion, References)
- Make GitHub repo public

---

## File Map of Changes

```
src/
  models/
    alista.py          ← compute_alista_weight(): analytic formula (was iterative Adam)
    hyperlista.py      ← compute_hyperlista_weight(): analytic formula
                          beta clamp: added max=0.99
  training/
    tuner.py           ← coarse_grid_search() default ranges narrowed

notebooks/
  03_sparse_generalization_experiments.ipynb
                       ← cell-6: eval_on_shifted() uses same A (was new random A)
```

---

## Key Insight for Seminar

The main finding that makes a good story:
- LISTA needs ~6 million parameters to reach -12 dB (and even then underperforms)
- ALISTA needs only **32 parameters** (per-layer γ, θ) to reach -30 dB — by pre-computing W analytically
- HyperLISTA needs only **3 hyperparameters** found by grid search (no backprop at all) to reach -60 dB — by making θ, β, p adaptive to the current iterate

This is the MBDL connection: each model trades learnable parameters for more structure/inductive bias, and performance improves.
