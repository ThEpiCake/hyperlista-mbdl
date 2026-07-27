# Sparse Recovery and Compressed Sensing with Model-Based Deep Learning

Final project for **Model-Based Deep Learning (361.2.2320)**, Ben-Gurion
University, Spring 2026.

**Team:** Etay Baron and Etai Wigman

> The [final report](submission/MBDL_Final_Report.pdf) and
> [seminar presentation](submission/Seminar/) are available under
> [`submission/`](submission/).

## Overview

This project studies deep-unrolled methods for the sparse linear inverse
problem

$$
b = Ax^* + \varepsilon.
$$

We compare classical optimization, learned unfolding, analytic unfolding, and
a simplified fixed-depth HyperLISTA variant. We also introduce three
ISTA-derived models to test which scalar components are most valuable to
learn:

- **ThresholdISTA:** learns one threshold per layer.
- **StepISTA:** learns one step size per layer.
- **StepThresholdISTA:** learns both scalars independently per layer.

All unfolded models use \(K=16\) layers.

| Method | Main idea | Learned or tuned quantities |
|---|---|---:|
| ISTA | Classical proximal-gradient method | 0 |
| FISTA | ISTA with Nesterov momentum | 0 |
| ThresholdISTA | ISTA with learned thresholds | 16 |
| StepISTA | ISTA with learned step sizes | 16 |
| StepThresholdISTA | Learned step sizes and thresholds | 32 |
| LISTA | Learned layer-wise update matrices | \(O(Kmn)\) |
| ALISTA | Analytic matrix with learned scalars | 32 |
| HyperLISTA | Adaptive analytic updates | 3 tuned constants |

Our HyperLISTA implementation retains the adaptive threshold, support
selection, and momentum mechanisms, but uses the same analytic matrix
construction as ALISTA. It also uses batch-median aggregation and omits the
original paper's final conjugate-gradient stage. In this repository,
**HyperLISTA refers to this simplified variant**.

## Experiments

The project contains three connected experimental studies:

1. **Synthetic sparse recovery:** noiseless Gaussian-sparse signals with
   \(m=250\), \(n=500\), and sparsity \(s=50\).
2. **ISTA design and training:** comparison of our scalar models and three
   unfolded-training strategies:
   - **L1:** final-layer loss;
   - **L2:** weighted losses at intermediate layers;
   - **L3:** sequential layer-wise training.
3. **Image compressed sensing:** reconstruction of flattened \(28\times28\)
   MNIST and Fashion-MNIST images at measurement ratios 0.25 and 0.50. We
   compare direct pixel-domain recovery with recovery in the 2D-DCT domain
   using the effective dictionary \(D=A\Psi^\top\).

The final robustness experiment repeats the main comparisons with independently
drawn sensing matrices to check whether the conclusions depend on one
favorable random seed.

## Main Findings

- On synthetic data, learning the **step-size schedule** accounts for most of
  LISTA's improvement. StepISTA reaches \(-20.0\) dB NMSE, close to LISTA-L1
  at \(-20.4\) dB, using only 16 learned scalars.
- StepThresholdISTA reaches \(-23.6\) dB with 32 parameters, showing that
  decoupling the threshold from the step size provides an additional gain.
- Intermediate supervision improves LISTA: LISTA-L2 reaches \(-22.2\) dB,
  compared with \(-20.4\) dB for final-layer training.
- ALISTA and our simplified HyperLISTA are particularly strong on the
  synthetic sparse model, reaching \(-30.0\) dB and \(-62.9\) dB,
  respectively.
- In Fashion-MNIST pixel-domain recovery at ratio 0.25, LISTA-L2 performs well
  (\(-14.1\) dB NMSE and 0.828 SSIM), while the more structured methods remain
  close to the zero estimate. This indicates a mismatch between their sparse
  prior and the pixel representation.
- Moving to the DCT domain substantially improves the structured methods on
  Fashion-MNIST. For example, HyperLISTA's SSIM at ratio 0.25 increases from
  0.09 to 0.52. The transform is not universally beneficial: it can reduce
  performance on MNIST when the pixels are already sparse.
- The multi-seed experiment preserves the main method ordering, suggesting
  that the headline conclusions are not caused by one lucky sensing matrix.

These results highlight a central model-based deep-learning trade-off:
stronger learned models can adapt to complex data distributions, while compact
structured models can perform extremely well when their assumed signal model
matches the representation.

## Repository Structure

```text
hyperlista-mbdl/
├── notebooks/                 # Six experiments, saved with their outputs
├── src/
│   ├── data/                  # Synthetic and image data loaders
│   ├── evaluation/            # NMSE, PSNR, SSIM, runtime, and plots
│   ├── models/                # ISTA, FISTA, LISTA, ALISTA, HyperLISTA, our models
│   ├── operators/             # 2D-DCT and inverse-DCT operators
│   └── training/              # L1/L2/L3 training and HyperLISTA tuning
├── results/                   # Saved tables, figures, and sensing matrices
├── submission/                # Final report and seminar presentation
├── requirements.txt
└── README.md
```

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

MNIST and Fashion-MNIST are downloaded automatically through PyTorch when
needed. Synthetic data are generated by the notebooks. A CUDA GPU is
recommended for training, but the experiments can also run on CPU.

## Running the Project

Run the notebooks in numerical order:

| Notebook | Purpose |
|---|---|
| `01_sparse_recovery_baselines.ipynb` | ISTA and FISTA baselines |
| `02_lista_alista_hyperlista.ipynb` | LISTA, ALISTA, and HyperLISTA |
| `03_ista_unfolding_design_space.ipynb` | Our scalar models and L1/L2/L3 training |
| `04_real_image_pixel_domain.ipynb` | Pixel-domain image compressed sensing |
| `05_transform_domain.ipynb` | DCT-domain image compressed sensing |
| `06_multiseed_robustness.ipynb` | Multi-seed robustness analysis |

From the repository root:

```bash
jupyter notebook
```

The notebooks contain their experiment configurations and save outputs under
`results/`.

## Reproducibility Note

Learned LISTA and ALISTA checkpoints are tied to the sensing matrix used during
training. PyTorch's CPU and CUDA random-number generators can produce different
matrices even when initialized with the same seed, so regenerating \(A\) on a
different device can severely degrade checkpoint performance.

The relevant sensing matrices are therefore saved under
`results/checkpoints/`. When loading a trained checkpoint, load its associated
matrix instead of regenerating it.

## Reference

X. Chen, J. Liu, Z. Wang, and W. Yin, “Hyperparameter Tuning Is All You Need
for LISTA,” *Advances in Neural Information Processing Systems*, 2021.
