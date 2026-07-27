# Sparse Recovery and Compressed Sensing — A Model-Based Deep Learning Approach

Final project for **Model-Based Deep Learning (361.2.2320)**, Ben-Gurion University, Spring 2026.
**Team:** Etay Baron & Etai Wigman

> 📦 **Final deliverables (report + seminar presentation) are in [`submission/`](submission/).**

> **Reference:** Chen, Liu, Wang, Yin. *Hyperparameter Tuning is All You Need for LISTA*. NeurIPS 2021.

---

## Project Overview

We implement and evaluate a simplified fixed-depth **HyperLISTA** variant—an
ultra-lightweight deep-unrolled method for sparse linear inverse problems—and
compare it with classical, learned, and analytic alternatives. We also design
and evaluate three original ISTA-derived scalar models that explore the MBDL
design space.

Methods compared:

| Method | Type | Learnable params |
|--------|------|-----------------|
| ISTA | Classical iterative | 0 |
| FISTA | Classical + Nesterov momentum | 0 |
| **ThresholdISTA** | **Our design — ISTA + learned $\theta_k$** | **K = 16** |
| **StepISTA** | **Our design — ISTA + learned $\gamma_k$** | **K = 16** |
| **StepThresholdISTA** | **Our design — ISTA + learned $\gamma_k, \theta_k$** | **2K = 32** |
| LISTA | Deep-unrolled (Gregor & LeCun) | O(K · m · n) |
| ALISTA | Analytic weights + learned scalars | O(K) |
| **HyperLISTA** | **Analytic + 3 global hyperparams** | **3** |

Evaluation covers two tasks:

- **Part A:** Synthetic sparse vector recovery: $b = Ax^* + \varepsilon$
- **Part B:** Compressed-sensing image reconstruction from MNIST/Fashion-MNIST, first in the **pixel domain** (notebook 04; Fashion-MNIST is ~50% near-zero in pixel space) and then revisited in the **DCT transform domain** (notebook 05; same measurements, recovery via the effective dictionary $D = A\Psi^\top$)

We also systematically compare three **training strategies for unfolded optimizers** (Lecture 6):

| Strategy | Description |
|----------|-------------|
| **L1** | Final-layer loss only (standard end-to-end BPTT) |
| **L2** | Deep supervision — weighted loss at every intermediate layer |
| **L3** | Sequential/greedy — layer *k* trained while layers 1,…,k−1 are frozen |

---

## Repository Structure

```
hyperlista_mbdl_project/
├── submission/                                  # ★ Final deliverables for grading
│   ├── MBDL_Final_Report.pdf                    # Public report (student IDs omitted)
│   └── Seminar/
│       ├── MBDL_Seminar_Presentation.pptx       # Seminar slides (16 slides, EN, speaker notes)
│       └── MBDL_Seminar_Presentation.pdf        # PDF export of the slides
│
├── notebooks/
│   ├── 01_sparse_recovery_baselines.ipynb      # ISTA & FISTA sweeps (Part A baselines)
│   ├── 02_lista_alista_hyperlista.ipynb         # LISTA / ALISTA / HyperLISTA, full comparison
│   ├── 03_ista_unfolding_design_space.ipynb     # Our three scalar models + L1/L2/L3 comparison
│   ├── 04_real_image_pixel_domain.ipynb         # MNIST/Fashion-MNIST pixel-domain CS (Part B)
│   ├── 05_transform_domain.ipynb                # DCT-domain CS via D = A·Psi^T (Part B follow-up)
│   └── 06_multiseed_robustness.ipynb            # Multi-seed error bars for the headline comparisons
│
├── src/
│   ├── data/
│   │   ├── sparse_generator.py   # Synthetic sparse dataset
│   │   └── image_loader.py       # MNIST/Fashion-MNIST CS loaders (pixel + DCT domain)
│   ├── operators/
│   │   └── dct_operators.py      # Separable 2D-DCT / IDCT
│   ├── models/
│   │   ├── ista.py               # Classical ISTA
│   │   ├── fista.py              # FISTA with Nesterov momentum
│   │   ├── lista.py              # LISTA — independent or tied weights (tied=True)
│   │   ├── alista.py             # ALISTA (analytic W, learned γ, θ)
│   │   ├── hyperlista.py         # HyperLISTA (3 hyperparams only)
│   │   └── learned_ista.py       # Our models: ThresholdISTA, StepISTA, StepThresholdISTA
│   ├── training/
│   │   ├── trainer.py            # BPTT (L1/L2) + sequential greedy (L3) training
│   │   └── tuner.py              # Gradient-free grid search for HyperLISTA
│   └── evaluation/
│       ├── metrics.py            # MSE, NMSE, PSNR, SSIM, runtime
│       └── visualizer.py         # NMSE-vs-layers, image grids, landscapes
│
├── results/
│   ├── checkpoints/              # Tracked sensing-matrix snapshots
│   └── notebook_01 ... 06/       # Saved tables and figures from executed notebooks
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

GPU is used automatically when available (`cuda`). All experiments also run on CPU (slower).

---

## Data & Setup

### Why is `data/` not in Git?

The `data/` directory is listed in `.gitignore` because:
- **Large files:** Data files (CSV, NPY, NPZ, PT) can be hundreds of MB or GB
- **Auto-generated:** Synthetic data and Fashion-MNIST are generated/downloaded on demand
- **Version control best practice:** Only source code and configuration belong in Git, not artifacts

### Data for Part A (Synthetic Sparse Recovery)

**Generated automatically.** When you run notebooks 01–03, the code creates:

```python
A, train_loader, val_loader, test_loader = build_sparse_dataloaders(
    m=250, n=500, s=50, sigma=0.0,
    n_train=51200, n_val=2048, n_test=2048,
    batch_size=256, device=DEVICE,
)
```

- **Sensing matrix** $A \in \mathbb{R}^{250 \times 500}$ with unit-norm columns
- **Sparse vectors** $x^*$ with $s=50$ non-zero entries per sample
- **Measurements** $b = Ax^* + \varepsilon$ (noiseless by default)
- **Batched DataLoaders** for efficient training

No download needed — generated on first run.

### Data for Part B (MNIST / Fashion-MNIST Compressed Sensing)

**Downloaded automatically via PyTorch.**

Notebook 04 (pixel domain) uses:

```python
A, train_loader, val_loader, test_loader = build_pixel_cs_dataloaders(
    dataset_name='fashionmnist',   # or 'mnist'
    measurement_ratio=0.25,        # m/d ∈ {0.25, 0.5}
    sigma=0.0, batch_size=256, n_val=5000,
    device=DEVICE, data_root='./data',
)
```

Notebook 05 (DCT domain) uses the same physical measurements `y = A x` but
returns the effective dictionary `D = A @ Psi.T` and `(y, alpha, x)` triplets:

```python
A, D, Psi, train_loader, val_loader, test_loader = build_transform_cs_dataloaders(
    dataset_name='fashionmnist',   # or 'mnist'
    measurement_ratio=0.25,
    sigma=0.0, batch_size=256, n_val=5000,
    device=DEVICE, data_root='./data',
)
```

- **Downloaded to:** `./data/` (auto-created on first run; MNIST and Fashion-MNIST are each ~12–35 MB)
- **Format:** Grayscale 28×28 images in [0,1], flattened to d = 784

If you run the project without internet access, download the datasets once in
an online environment and copy the resulting `data/MNIST/` and/or
`data/FashionMNIST/` directories into the project. The loader intentionally
fails with an explicit offline-data message instead of substituting synthetic
images, because Part B is meant to evaluate the real reconstruction task.

### Expected Directory Structure (After First Run)

```
hyperlista_mbdl_project/
├── data/
│   ├── FashionMNIST/
│   │   ├── raw/
│   │   │   ├── train-images-idx3-ubyte.gz
│   │   │   ├── train-labels-idx1-ubyte.gz
│   │   │   ├── t10k-images-idx3-ubyte.gz
│   │   │   └── t10k-labels-idx1-ubyte.gz
│   │   └── processed/
│   │       ├── training.pt
│   │       └── test.pt
│   └── .gitkeep
│
├── results/
│   ├── sparse/
│   │   ├── nmse_vs_layers_partA.pdf
│   │   ├── adaptivity_experiments.pdf
│   │   └── .gitkeep
│   ├── images/
│   │   ├── image_comparison_0.25.pdf
│   │   ├── psnr_vs_ratio.pdf
│   │   └── .gitkeep
│   └── figures/
│       └── .gitkeep
│
├── src/
├── notebooks/
└── ...
```

### Troubleshooting

**Q: "FileNotFoundError: data not found"**
- ✅ Run the notebooks in order (they auto-generate/download)
- ✅ Ensure write permissions in the project directory
- ✅ Check internet connection for Fashion-MNIST download

**Q: "Fashion-MNIST is not available locally..."**
- The environment is offline or DNS is blocked
- Run notebook 04 once with network access, or copy a prepared
  `data/FashionMNIST/` cache into this project
- Do not replace Fashion-MNIST with random images if you need report-quality
  Part B results

**Q: "CUDA out of memory"**
- Reduce `batch_size` in notebooks (e.g., 128 → 64)
- Reduce `n_epochs` or use CPU: `DEVICE = torch.device('cpu')`

**Q: Large disk usage after running experiments**
- `data/FashionMNIST/` ~35 MB (needed for Part B)
- `results/` directory grows with saved figures (can delete safely)

---

## Quick Start

### Part A — Sparse recovery baselines

```bash
cd notebooks
jupyter notebook 01_sparse_recovery_baselines.ipynb   # ISTA & FISTA
jupyter notebook 02_lista_alista_hyperlista.ipynb      # LISTA, ALISTA, HyperLISTA
```

### Design space & training methods (Notebook 03)

```bash
jupyter notebook 03_ista_unfolding_design_space.ipynb
```

Contains two experiments:
1. **ISTA component comparison** — ThresholdISTA, StepISTA, StepThresholdISTA
2. **Training method comparison** — L1 / L2 / L3 × {LISTA, LISTA-Tied}

### Part B — Image CS (Pixel domain)

```bash
jupyter notebook 04_real_image_pixel_domain.ipynb
```

Fashion-MNIST data is downloaded automatically to `./data/`.

### Part B follow-up — Image CS (DCT transform domain)

```bash
jupyter notebook 05_transform_domain.ipynb
```

Same measurements as notebook 04, recovery in the DCT domain. Set
`DATASET_NAME` / `MEASUREMENT_RATIO` in the config cell to reproduce each of
the four {mnist, fashionmnist} × {0.25, 0.5} cells.

---

## Reproducibility Note — CPU vs GPU Sensing Matrix

**TL;DR: Run the notebooks on GPU (CUDA). Loading checkpoints on CPU will give garbage results for LISTA and ALISTA.**

PyTorch uses different random-number generators for CPU (Mersenne Twister) and GPU (Philox-4×32). Even with `torch.manual_seed(42)`, `torch.randn(..., device='cuda')` produces a **different matrix** than `torch.randn(..., device='cpu')`. LISTA and ALISTA learn weights that are specific to the training sensing matrix A, so they fail completely if evaluated with a different A.

HyperLISTA does not store a learned matrix tied to a particular sensing
matrix: its analytic `W` is recomputed from the supplied `A`. Its measured
performance can still vary with the sensing matrix, as evaluated in notebook
06.

**Fix applied:**
- Notebooks 02 and 04 save `A_partA.npy` / `A_pixel_<ratio>.npy` (tracked by Git, gitignore has `!results/checkpoints/A_*.npy`).
- To reload a LISTA/ALISTA checkpoint on CPU:

```python
import numpy as np, torch
A = torch.from_numpy(np.load('results/checkpoints/A_partA.npy'))
lista.load_state_dict(torch.load('results/checkpoints/lista_L1.pt', map_location='cpu'))
```

---

## Key Findings

### Part A — ISTA Component Comparison (Notebook 03)

| Method | NMSE @ K=16 | # Params | Observation |
|--------|------------|---------|-------------|
| ISTA | -5.3 dB | 0 | Baseline |
| FISTA | -11.0 dB | 0 | Momentum helps |
| ThresholdISTA | -7.1 dB | 16 | Learning θ alone barely helps |
| StepISTA | -20.0 dB | 16 | Learning γ alone ≈ full LISTA |
| **StepThresholdISTA** | **-23.6 dB** | **32** | **Beats LISTA with 187,000× fewer params** |
| LISTA-L1 | -20.4 dB | 6,000,016 | Reference |

**Key insight:** The step size schedule is the most valuable component to learn. `StepISTA` (16 scalars) matches `LISTA` (6M parameters). Decoupling the threshold from the step size adds another 3.5 dB at negligible cost (16 extra scalars). This is the MBDL principle: structured scalar models beat unstructured matrix learners.

### Part A — Training Methods (Notebook 03)

| Method | NMSE @ K=16 | # Params | Observation |
|--------|------------|---------|-------------|
| LISTA-L1 | -20.4 dB | 6,000,016 | Baseline (end-to-end BPTT) |
| **LISTA-L2** | **-22.2 dB** | 6,000,016 | Deep supervision — best LISTA variant |
| LISTA-L3 | -16.6 dB | 6,000,016 | Greedy training plateaus (~layer 8) |
| LISTA-Tied-L1 | -19.3 dB | 375,001 | Near-untied accuracy, 16× fewer params |
| ALISTA | -30.0 dB | 32 | Reference |
| HyperLISTA | -62.9 dB | 3 | Reference |

**Notable: LISTA-Tied nearly matches LISTA-Independent** (−19.3 vs −20.4 dB) despite 16× fewer parameters.

*Why?* LISTA with 6M independent parameters has a complex, poorly-conditioned loss landscape. The RNN-style weight tying acts as implicit regularization: all layers must share a single (W_y, W_x, θ), which makes gradient flow smoother and convergence more reliable. This is a concrete example of the MBDL principle: adding structure (even the mild constraint of weight sharing) preserves accuracy at a fraction of the parameter cost.

**L3 observation:** NMSE-vs-layer flattens from ~layer 8 onward. Greedy training optimizes each step independently, so later layers have no incentive to improve over what earlier layers already solved.

### Part B — Fashion-MNIST Pixel Domain (Notebook 04)

| Method | NMSE (ratio=0.25) | PSNR | SSIM |
|--------|------------------|------|------|
| ISTA | -1.1 dB | 8.8 | 0.134 |
| FISTA | -1.1 dB | 8.7 | 0.135 |
| StepThresholdISTA | -1.1 dB | 8.8 | 0.137 |
| **LISTA-L2** | **-14.1 dB** | **22.5** | **0.828** |
| ALISTA | -0.8 dB | 8.4 | 0.128 |
| HyperLISTA | -0.3 dB | 7.7 | 0.091 |

**ALISTA and HyperLISTA perform poorly in this setting because their structure
is mismatched to the image distribution.**

The reason is a signal-model mismatch:

- ALISTA and HyperLISTA were **designed for i.i.d. Gaussian sparse signals**: `x*` has exactly `s` random non-zero entries drawn from `N(0,1)`.
- Fashion-MNIST pixels are bounded `[0,1]`, spatially smooth, and only "soft-sparse" (~50% exact zeros). The remaining ~50% are *continuous, structured* non-zeros — not Gaussian.
- HyperLISTA's threshold `θ = c1·μ·‖A⁺(Ax−b)‖₁` calibrates to the residual under a Gaussian sparse model. On Fashion-MNIST the residual reflects spatial image structure, not sparse noise → threshold zeros out valid signal.
**Interpretation:** LISTA wins here because its learned update matrices can
adapt to the image distribution from 55K training images, while the structured
methods retain correction directions determined by the sensing matrix. This
motivates testing a representation that better matches their sparsity prior.

### Part B follow-up — DCT Transform Domain (Notebook 05)

Same measurements `y = A x` as notebook 04 (bit-identical A and splits), but recovery
happens in the DCT domain via the effective dictionary `D = A @ Psi.T`; the image is
reconstructed as `x_hat = Psi.T @ alpha_hat`. SSIM across all four cells (pixel → DCT):

| Cell | ISTA | FISTA | StepThr | LISTA-L2 | ALISTA | HyperLISTA |
|------|------|-------|---------|----------|--------|------------|
| FMNIST @ 0.25 | 0.13 → **0.26** | 0.13 → **0.35** | 0.14 → **0.52** | 0.83 → 0.81 | 0.13 → **0.40** | 0.09 → **0.52** |
| FMNIST @ 0.5 | 0.29 → **0.45** | 0.31 → **0.57** | 0.32 → **0.68** | 0.89 → 0.82 | 0.30 → **0.69** | 0.32 → **0.71** |
| MNIST @ 0.25 | 0.17 → **0.22** | 0.20 → **0.31** | 0.21 → **0.43** | 0.97 → 0.88 | 0.22 → **0.37** | 0.22 → **0.43** |
| MNIST @ 0.5 | 0.42 → 0.45 | 0.58 → 0.57 | 0.73 → 0.65 | 0.99 → 0.85 | 0.81 → 0.66 | 0.82 → 0.66 |

**Two-sided conclusion:**

1. **The Fashion-MNIST collapse was a domain problem, not an algorithm problem.**
   On FMNIST @ 0.25 HyperLISTA jumps from dead last (−0.28 dB, SSIM 0.09) to
   −7.7 dB / SSIM 0.52 with only three tuned constants, and its tuned `c3` returns to a
   healthy support-selection range (2.5–15.8 vs 0.17 in pixel space). At ratio 0.5
   HyperLISTA is the **best** structured method (−11.4 dB, SSIM 0.71).
2. **The transform is not a free lunch.** On MNIST @ 0.5 — where pixels already form
   a good sparse domain (~81% exact zeros) — the DCT actually *hurts* the structured
   learners (ALISTA 0.81 → 0.66). Sparse recovery succeeds in whichever domain the
   signal is genuinely sparsest; the domain choice is part of the model.
3. **LISTA-L2 stays the robust ceiling** (SSIM 0.81–0.88 everywhere), but on
   FMNIST @ 0.25 its NMSE lead over the best structured method shrinks from 13.0 dB
   to 7.1 dB once the prior matches the data.

---

## Mathematical Background

### Measurement model
$$b = Ax^* + \varepsilon, \quad x^* \in \mathbb{R}^n \text{ (s-sparse)}, \quad A \in \mathbb{R}^{m \times n}, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### HyperLISTA update (layer k)
$$x^{(k+1)} = \mathcal{S}_{p^{(k)}, \theta^{(k)}}\!\left(x^{(k)} + W^T(b - Ax^{(k)}) + \beta^{(k)}(x^{(k)} - x^{(k-1)})\right)$$

with adaptive parameters driven by only **three** scalars $(c_1, c_2, c_3)$:

$$\theta^{(k)} = c_1 \mu \|A^+(Ax^{(k)}-b)\|_1, \quad \beta^{(k)} = c_2 \mu \|x^{(k)}\|_0, \quad p^{(k)} = c_3 \log\!\frac{\|A^+b\|_1}{\|A^+(Ax^{(k)}-b)\|_1}$$

The weight matrix $W = (AA^T)^{-1}A$ is computed analytically (minimum-Frobenius-norm solution of $\|W^T A - I\|_F$), column-normalized so that $\mathrm{diag}(W^T A) = 1$.

### Pixel-domain CS (Notebook 04)
$$b = Ax + \varepsilon, \quad x \in [0,1]^{784} \text{ (≈50\% exact zeros in pixel space)}$$

Fashion-MNIST images have many zero-valued background pixels, but the
pixel-domain representation proved insufficiently sparse and structured for
the lightweight recovery methods. Notebook 05 therefore evaluates the same
measurements in the DCT domain.

### Training strategies (Notebook 03)

| Loss | Definition |
|------|-----------|
| L1 | $\mathcal{L} = \|x^{(K)} - x^*\|^2$ |
| L2 | $\mathcal{L} = \|x^{(K)} - x^*\|^2 + \lambda \sum_k w_k \|x^{(k)} - x^*\|^2$ |
| L3 | Train $\theta_k$ to minimize $\|x^{(k)} - x^*\|^2$ with $\theta_1,\ldots,\theta_{k-1}$ frozen |

---

## Project Deadlines

| Milestone | Date |
|-----------|------|
| Progress report | 24-May-2026 |
| Recorded seminar | 28-June-2026 |
| Final report | 30-July-2026 |
