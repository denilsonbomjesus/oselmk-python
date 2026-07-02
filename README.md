# pyoselmk

**Online Sequential Extreme Learning Machine with Kernels (OS-ELMK)**  
Python implementation of the algorithm proposed by Huang et al. (2014) for nonstationary time series prediction.

---

## Table of Contents

- [Algorithm Overview](#algorithm-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)

---

## Algorithm Overview

The OS-ELMK extends the standard kernel ELM (ELMK) with an **online sequential learning** mechanism that updates the model incrementally as new data arrives — without retraining from scratch.

### Why OS-ELMK?

Classical batch ELMK trains on a fixed dataset. In practice, time series are often **nonstationary**: the statistical properties (mean, variance, spectral content) drift over time. Retraining from scratch at every step is prohibitively expensive. OS-ELMK solves this via a Schur-complement block-matrix update that runs in **O(n·bs + bs³)** instead of O((n+bs)³), where `n` is the current support size and `bs` is the new block size.

### Mathematical Background

The kernel ELM decision function is:

```
f(x) = K(x, X_train) · w
     = K(x, X_train) · R⁻¹ · y_train
```

where:
- `K(·, ·)` is the kernel matrix
- `R = K_train + I/C` is the regularised kernel matrix
- `w = R⁻¹ · y_train` are the output weights (Lagrange multipliers θ)

The OS-ELMK online update (Section 2.3 of the paper) keeps `R⁻¹` and `θ` in memory and updates them block-wise using the following equations:

| Symbol    | Equation                                  | Paper Eq. | Description                  |
|-----------|-------------------------------------------|-----------|------------------------------|
| `G`       | `G = −R⁻¹ · K_cross`                     | Eq. 17    | Sensitivity matrix           |
| `γ`       | `γ = K_new + I_bs/C + K_crossᵀ · G`      | Eq. 26    | Schur complement (SPD)       |
| `θ_new`   | `θ_new = γ⁻¹ · E_bs`                     | Eq. 27    | New-block multipliers        |
| `θ*`      | `θ* = [θ + G·θ_new ; θ_new]`             | Eq. 20    | Stacked multipliers          |
| `R⁻¹_new` | Block matrix from `R11, R12, R21, R22`    | Schur     | Updated inverse              |

### Update Modes

**Sequential (incremental):** the support set **grows** by `bs` samples at each update. Suitable when memory is not a concern and the series is slowly drifting.

**Decremental (sliding window):** after the full incremental expansion to size `n + bs`, the **oldest** `bs` support vectors are pruned, keeping `n_train_` fixed at `window_size`. This allows the model to *forget* stale patterns and adapt to regime changes.

```
Decremental pruning (applied after incremental expansion):

    R_inv_   = Schur-corrected sub-block R_inv_expanded[bs:, bs:]
    theta_   = R_inv_ @ y_train_
    K_elm_   = K_expanded[bs:, bs:]
    X_train_ = X_expanded[bs:]
    y_train_ = y_expanded[bs:]
```

### Available Kernels

| Kernel     | `kernel=`   | `kernel_params`           | Formula                                         |
|------------|-------------|---------------------------|-------------------------------------------------|
| RBF        | `'rbf'`     | `σ` (float, default 1.0)  | `exp(−‖xᵢ − xⱼ‖² / σ)`                        |
| Linear     | `'linear'`  | ignored                   | `xᵢᵀ xⱼ`                                       |
| Polynomial | `'poly'`    | `[c, d]` (default [1, 2]) | `(xᵢᵀ xⱼ + c)^d`                               |
| Wavelet    | `'wavelet'` | `[b, a, ω₀]` (default 1s) | `cos(ω₀ Δx / a) · exp(−‖Δx‖² / b)`            |

---

## Architecture

```
oselmk-python/
├── src/
│   └── oselmk/
│       ├── __init__.py          # Public exports
│       ├── model.py             # OSELMK class (fit / update / predict)
│       └── utils/
│           ├── kernels.py       # kernel_matrix() — all 4 kernels
│           ├── metrics.py       # compute_all() — RMSE, NRMSE, MAPE, SMAPE
│           ├── normalization.py # ZScoreNormalizer (fit / transform / inverse)
│           ├── results.py       # save_run() — JSON + CSV persistence
│           └── windowing.py     # make_lag_features() — lag embedding
├── tests/                       # pytest unit tests (one file per module)
├── examples/                    # End-to-end runnable scripts
├── datasets/                    # Benchmark time series (see §Datasets)
├── results/                     # Auto-generated experiment outputs
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

### Data flow

```
Raw time series
      │
      ▼
 ZScoreNormalizer.fit_transform(y)
      │
      ▼
 make_lag_features(y_norm, n_lags)
      │
      ├────────────────────────────────────────┐
      ▼                                        │
 X_train, y_train                              │
      │                                        │
      ▼                                        │
 OSELMK.fit()  ◄── offline batch phase         │
      │                                        │
      │  stores: R_inv_, theta_,               │
      │          X_train_, K_elm_              │
      │                                        │
      ├── OSELMK.update(mode='sequential')     │
      │       support set grows by bs          │
      │                                        │
      ├── OSELMK.update(mode='decremental')    │
      │       support set stays fixed          │
      │       (oldest bs samples pruned)       │
      │                                        │
      ▼                                        │
 OSELMK.predict(X_test)                        │
      │                                        │
      ▼                                        │
 y_pred_norm                                   │
      │                                        │
      ▼                                        │
 ZScoreNormalizer.inverse_transform()          │
      │                                        │
      ▼                                        │
 y_pred  (original scale)                      │
      │                                        │
      ▼                                        │
 save_run()  ────────────────────────────────►─┘
      │
      ▼
 results/<timestamp>/
   ├── metrics.json
   ├── config.json
   └── predictions.csv
```

---

## Installation

### Prerequisites

- Python >= 3.10
- pip

### From source (recommended during development)

```bash
git clone https://github.com/denilsonbomjesus/oselmk-python.git
cd oselmk-python

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install the package in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt
```

### Dependencies

Runtime (`requirements.txt`):

```
numpy>=1.24
scipy>=1.10
```

Dev (`requirements-dev.txt`):

```
pytest>=7.4
pytest-cov>=4.1
ruff==0.11.13
```

---

## Quickstart

### 1 — Minimal example (sinc regression)

```python
import numpy as np
from oselmk.model import OSELMK
from oselmk.utils.normalization import ZScoreNormalizer
from oselmk.utils.windowing import make_lag_features

# Generate a synthetic series
t = np.linspace(-10, 10, 500)
y = np.sinc(t)

# Normalise
norm = ZScoreNormalizer()
y_norm = norm.fit_transform(y)

# Lag embedding (4 past values → predict next)
X, target = make_lag_features(y_norm, n_lags=4)

n_train = 300
X_train, y_train = X[:n_train], target[:n_train]
X_test,  y_test  = X[n_train:], target[n_train:]

# Batch training
model = OSELMK(C=1.0, kernel="rbf", kernel_params=1.0)
model.fit(X_train, y_train)

# Predict
y_pred_norm = model.predict(X_test)
y_pred = norm.inverse_transform(y_pred_norm)
```

### 2 — Online update (sequential)

```python
block_size = 10

for start in range(0, len(X_test) - block_size, block_size):
    X_block = X_test[start : start + block_size]
    y_block = y_test[start : start + block_size]
    model.update(X_block, y_block, mode="sequential")

y_pred_final = norm.inverse_transform(model.predict(X_test))
```

### 3 — Sliding window (decremental)

```python
# Support set stays fixed at n_train=300; oldest block is pruned each step
for start in range(0, len(X_test) - block_size, block_size):
    X_block = X_test[start : start + block_size]
    y_block = y_test[start : start + block_size]
    model.update(X_block, y_block, mode="decremental")
```

### 4 — Saving results

```python
import time
from oselmk.utils.results import save_run

t0 = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - t0

t1 = time.perf_counter()
y_pred_norm = model.predict(X_test)
predict_time = time.perf_counter() - t1

run_dir = save_run(
    y_true=norm.inverse_transform(y_test),
    y_pred=norm.inverse_transform(y_pred_norm),
    train_time_s=train_time,
    predict_time_s=predict_time,
    config={
        "kernel": "rbf",
        "C": 1.0,
        "n_lags": 4,
        "update_mode": "decremental",
        "dataset": "sinc",
    },
)
print(f"Results saved to {run_dir}")
```

---

## API Reference

### `OSELMK`

```
oselmk.model.OSELMK(C=1.0, kernel='rbf', kernel_params=None)
```

| Parameter       | Type                           | Default  | Description                                         |
|-----------------|--------------------------------|----------|-----------------------------------------------------|
| `C`             | `float`                        | `1.0`    | Regularisation constant. Must be > 0.               |
| `kernel`        | `str`                          | `'rbf'`  | One of `'rbf'`, `'linear'`, `'poly'`, `'wavelet'`.  |
| `kernel_params` | `float \| list[float] \| None` | `None`   | Kernel-specific parameters (see kernel table).      |

#### Methods

| Method                                                      | Returns   | Description                       |
|-------------------------------------------------------------|-----------|-----------------------------------|
| `fit(X, y)`                                                 | `self`    | Offline batch training.           |
| `update(X_new, y_new, mode='sequential', window_size=None)` | `self`    | Online update with a new block.   |
| `predict(X)`                                                | `ndarray` | Predict on new inputs.            |

#### Key attributes after `fit()`

| Attribute  | Shape                     | Description                        |
|------------|---------------------------|------------------------------------|
| `R_inv_`   | `(n_support, n_support)`  | Inverse regularised kernel matrix. |
| `theta_`   | `(n_support, n_outputs)`  | Lagrange multipliers.              |
| `X_train_` | `(n_support, n_features)` | Current support inputs.            |
| `K_elm_`   | `(n_support, n_support)`  | Kernel matrix (no regularisation). |
| `n_train_` | `int`                     | Current support size.              |

---

### `ZScoreNormalizer`

```
oselmk.utils.normalization.ZScoreNormalizer()
```

| Method                      | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `fit_transform(y)`          | Compute `μ`, `σ` and return `(y − μ) / σ`.            |
| `transform(y)`              | Apply stored `μ` and `σ` to new data.                 |
| `inverse_transform(y_norm)` | Recover original scale: `y_norm · σ + μ`.             |

---

### `make_lag_features`

```
oselmk.utils.windowing.make_lag_features(y, n_lags) -> (X, y_target)
```

Converts a 1-D normalised series into a supervised learning dataset.  
Each row of `X` contains the `n_lags` previous values; `y_target[i]` is the value to predict.

---

### `compute_all`

```
oselmk.utils.metrics.compute_all(y_true, y_pred) -> dict
```

Returns a dictionary with keys `rmse`, `nrmse`, `mape`, `smape`.  
MAPE is set to `nan` when any `y_true` value is zero. NRMSE is set to `nan` when `range(y_true) == 0`.

---

### `save_run`

```
oselmk.utils.results.save_run(
    y_true, y_pred,
    train_time_s, predict_time_s,
    config=None, run_dir=None, base_dir='results'
) -> Path
```

Writes three artefacts under a timestamped directory:

| File              | Content                                           |
|-------------------|---------------------------------------------------|
| `metrics.json`    | RMSE, NRMSE, MAPE, SMAPE + wall-clock timing.     |
| `config.json`     | Experiment hyperparameters (kernel, C, n_lags, …).|
| `predictions.csv` | Columns: `index`, `y_true`, `y_pred`.             |

`NaN` and `Inf` values in metrics are serialised as JSON `null`.

---

## Datasets

The `datasets/` directory is not tracked by git. The benchmark series used in the original paper are all publicly available.

```
datasets/
├── mackeyglass/
│   └── mackeyglass.csv            # columns: t, y  (chaotic, τ=17)
├── sunspot/
│   └── SN_y_tot_V1700_2_2003.txt  # SIDC yearly sunspot numbers
├── djia/
│   └── djia.csv                   # Dow Jones Industrial Average
├── sp500/
│   └── SP500.csv                  # S&P 500 daily closes
├── santafe/
│   └── santafe.csv                # Santa Fe competition (laser, set A)
└── sinc/
    └── sinc.csv                   # Synthetic sinc (regression baseline)
```

All files should be plain CSV with at minimum a `y` column containing the univariate series values.  
The original Octave datasets are available at [phmferreira/project_ST_OS_ELMK](https://github.com/phmferreira/project_ST_OS_ELMK/tree/master/datasets).

### Dataset characteristics

| Dataset      | Type      | Length  | Key property                          |
|--------------|-----------|---------|---------------------------------------|
| Mackey-Glass | Synthetic | ~10 000 | Chaotic, standard ELM benchmark       |
| Sunspot      | Real      | ~300    | Quasi-periodic, slowly nonstationary  |
| DJIA         | Real      | varies  | Financial, highly nonstationary       |
| S&P 500      | Real      | varies  | Financial, regime changes             |
| Santa Fe A   | Real      | 1 000   | Laser intensity, competition dataset  |
| Sinc         | Synthetic | 500     | Smooth, regression baseline           |

---

## Results

Every call to `save_run()` writes a timestamped directory under `results/`:

```
results/
└── 2026-07-02_17-00-00/
    ├── metrics.json        ← prediction quality + timing
    ├── config.json         ← kernel, C, n_lags, dataset, update_mode
    └── predictions.csv     ← index, y_true, y_pred (original scale)
```

Example `metrics.json`:

```json
{
  "metrics": {
    "rmse": 0.0312,
    "nrmse": 0.0289,
    "mape": 4.71,
    "smape": 4.68
  },
  "timing": {
    "train_time_s": 0.043,
    "predict_time_s": 0.007
  }
}
```

The `results/` directory is listed in `.gitignore`. Committed experiment snapshots, if any, live in `results/published/`.

---

## Running Tests

```bash
pytest                           # run all tests
pytest tests/test_model.py -v    # specific file, verbose
pytest --cov=src/oselmk          # with coverage report
```

Linting:

```bash
ruff check .           # lint all files
ruff check . --fix     # auto-fix safe issues
```

---

## Contributing

1. Fork the repository and create a feature branch from `main`.
2. Run the full test suite before opening a PR: `pytest`.
3. Lint with `ruff check .` — all checks must pass.
4. Follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`.

---

## References

### Primary reference

> Huang, G., Huang, G.-B., Song, S., & You, K. (2014).  
> **Online sequential extreme learning machine with kernels for nonstationary time series prediction.**  
> *Neurocomputing*, 145, 90–97.  
> <https://doi.org/10.1016/j.neucom.2014.05.068>

### Octave reference implementation

The algorithm structure and dataset selection were validated against:

> phmferreira. *project_ST_OS_ELMK* (GNU Octave).  
> <https://github.com/phmferreira/project_ST_OS_ELMK>

Differences from the Octave reference:

- Restricted to regression (the paper's validated scope); the classification branch in `os_elmk_model.m` referenced undefined variables (`TV.T`, `NumberofTestingData`) and was removed.
- `scipy.linalg.solve(..., assume_a='pos')` replaces explicit `inv()` for numerical stability.
- Output weights are cached and recomputed only on state change (the original recomputed on every prediction call).
- Decremental pruning uses a Schur-complement correction to the `R_inv` sub-block instead of a raw slice, improving approximation quality.

### Related reading

- Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: Theory and applications. *Neurocomputing*, 70(1–3), 489–501. <https://doi.org/10.1016/j.neucom.2005.12.126>
- Liang, N.-Y., Huang, G.-B., Saratchandran, P., & Sundararajan, N. (2006). A fast and accurate online sequential learning algorithm for feedforward networks. *IEEE Transactions on Neural Networks*, 17(6), 1411–1423. <https://doi.org/10.1109/TNN.2006.880583>
