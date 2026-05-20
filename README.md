# HyperLISTA — Model-Based Deep Learning Project

Final project for **Model-Based Deep Learning (361.2.2320)**, Ben-Gurion University, Spring 2025–2026.

> **Reference:** Chen, Liu, Wang, Yin. *Hyperparameter Tuning is All You Need for LISTA*. NeurIPS 2021.

---

## Project Overview

We implement and evaluate **HyperLISTA** — an ultra-lightweight deep-unrolled network for sparse linear inverse problems — against four competing methods:

| Method | Type | Learnable params |
|--------|------|-----------------|
| ISTA | Classical iterative | 0 |
| FISTA | Classical + Nesterov momentum | 0 |
| LISTA | Deep-unrolled (Gregor & LeCun) | O(K · m · n) |
| ALISTA | Analytic weights + learned scalars | O(K) |
| **HyperLISTA** | **Analytic + 3 global hyperparams** | **3** |

Evaluation covers two tasks:

- **Part A:** Synthetic sparse vector recovery: $b = Ax^* + \varepsilon$
- **Part B:** Compressed-sensing image reconstruction from Fashion-MNIST using 2D-DCT domain sparsity

---

## Repository Structure

```
hyperlista_mbdl_project/
├── notebooks/
│   ├── 01_sparse_recovery_baselines.ipynb      # ISTA & FISTA sweeps
│   ├── 02_lista_alista_hyperlista.ipynb         # Trained models, full comparison
│   ├── 03_sparse_generalization_experiments.ipynb  # OOD tests (Fig. 5 of paper)
│   └── 04_image_cs_experiments.ipynb           # Fashion-MNIST CS (Part B)
│
├── src/
│   ├── data/
│   │   ├── sparse_generator.py   # Synthetic sparse dataset
│   │   └── image_loader.py       # Fashion-MNIST + CS measurements
│   ├── operators/
│   │   └── dct_operators.py      # Separable 2D-DCT / IDCT
│   ├── models/
│   │   ├── ista.py               # Classical ISTA
│   │   ├── fista.py              # FISTA with Nesterov momentum
│   │   ├── lista.py              # LISTA (end-to-end trainable)
│   │   ├── alista.py             # ALISTA (analytic W, learned γ, θ)
│   │   └── hyperlista.py         # HyperLISTA (3 hyperparams only)
│   ├── training/
│   │   ├── trainer.py            # BPTT training loop with early stopping
│   │   └── tuner.py              # Gradient-free grid search for HyperLISTA
│   └── evaluation/
│       ├── metrics.py            # MSE, NMSE, PSNR, SSIM, runtime
│       └── visualizer.py         # NMSE-vs-layers, image grids, landscapes
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

### Data for Part B (Fashion-MNIST Compressed Sensing)

**Downloaded automatically via PyTorch.**

When you run notebook 04, the code fetches Fashion-MNIST:

```python
A_r, Psi, tr_loader, te_loader = build_image_cs_dataloaders(
    measurement_ratio=0.25,  # m/d ∈ {0.125, 0.25, 0.5}
    sigma=0.0,
    batch_size=128,
    device=DEVICE,
    data_root='./data',
)
```

- **Downloaded to:** `./data/` (auto-created on first run)
- **Storage:** ~30 MB (train) + ~5 MB (test) per ratio
- **Format:** Grayscale 28×28 images, automatically converted to 2D-DCT coefficients

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

**Q: "CUDA out of memory"**
- Reduce `batch_size` in notebooks (e.g., 128 → 64)
- Reduce `n_epochs` or use CPU: `DEVICE = torch.device('cpu')`

**Q: Large disk usage after running experiments**
- `data/FashionMNIST/` ~35 MB (needed for Part B)
- `results/` directory grows with saved figures (can delete safely)

---

## Quick Start

### Part A — Sparse recovery

```bash
cd notebooks
jupyter notebook 01_sparse_recovery_baselines.ipynb   # classical baselines
jupyter notebook 02_lista_alista_hyperlista.ipynb      # all five methods
jupyter notebook 03_sparse_generalization_experiments.ipynb  # OOD tests
```

### Part B — Image CS

```bash
jupyter notebook 04_image_cs_experiments.ipynb
```

Fashion-MNIST data is downloaded automatically to `./data/`.

---

## Mathematical Background

### Measurement model
$$b = Ax^* + \varepsilon, \quad x^* \in \mathbb{R}^n \text{ (s-sparse)}, \quad A \in \mathbb{R}^{m \times n}, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

### HyperLISTA update (layer k)
$$x^{(k+1)} = \mathcal{S}_{p^{(k)}, \theta^{(k)}}\!\left(x^{(k)} + W^T(b - Ax^{(k)}) + \beta^{(k)}(x^{(k)} - x^{(k-1)})\right)$$

with adaptive parameters driven by only **three** scalars $(c_1, c_2, c_3)$:

$$\theta^{(k)} = c_1 \mu \|A^+(Ax^{(k)}-b)\|_1, \quad \beta^{(k)} = c_2 \mu \|x^{(k)}\|_0, \quad p^{(k)} = c_3 \log\!\frac{\|A^+b\|_1}{\|A^+(Ax^{(k)}-b)\|_1}$$

The weight matrix $W = (G^TG)A$ is computed analytically (symmetric Jacobian parameterisation).

### DCT-domain CS (Part B)
$$y = A\alpha + n = A\Psi x + n, \quad \hat{x} = \Psi^T \hat{\alpha}$$

---

## Project Deadlines

| Milestone | Date |
|-----------|------|
| Progress report | 24-May-2026 |
| Recorded seminar | 28-June-2026 |
| Final report | 30-July-2026 |
