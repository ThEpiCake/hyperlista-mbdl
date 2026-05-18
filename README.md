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
