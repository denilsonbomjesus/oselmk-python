#!/usr/bin/env python3
"""Batch-only ELMK baseline (no online update).

This script implements the simplest OS-ELMK usage mode: a single offline
train on an initial data window, followed by evaluation on a held-out test
set.  No online update is performed.  Use this as a baseline to compare
against the sequential and decremental modes.

Pipeline
--------
1. Load a univariate time series from datasets/  (or generate a synthetic
   Mackey-Glass-like signal if the file is absent).
2. Normalise the raw series with ZScoreNormalizer (fit on train only).
3. Build lag features with make_lag_features.
4. Split into train / test sets (no shuffling — time order is preserved).
5. Fit OSELMK on the training set.
6. Predict on the test set and denormalise predictions.
7. Compute and print RMSE, NRMSE, MAPE, SMAPE.
8. Persist metrics, config and predictions to results/.

Usage
-----
    python examples/run_batch_only.py
    python examples/run_batch_only.py --dataset datasets/mackeyglass.csv
    python examples/run_batch_only.py --dataset datasets/mackeyglass.csv \\
        --kernel rbf --C 100 --n-lags 6 --test-ratio 0.2

Arguments
---------
--dataset PATH      Path to a single-column CSV or whitespace-delimited
                    text file.  If omitted, a 1 000-point synthetic signal
                    is generated automatically.
--kernel NAME       Kernel to use: rbf | linear | poly | wavelet
                    (default: rbf).
--C FLOAT           Regularisation constant (default: 100.0).
--n-lags INT        Number of lag features (default: 4).
--test-ratio FLOAT  Fraction of windowed samples reserved for testing
                    (default: 0.2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make sure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oselmk.model import OSELMK
from oselmk.utils.metrics import compute_all
from oselmk.utils.normalization import ZScoreNormalizer
from oselmk.utils.results import save_run
from oselmk.utils.windowing import make_lag_features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-only ELMK baseline (no online update).")
    p.add_argument("--dataset", default=None, help="Path to a single-column CSV / text file.")
    p.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly", "wavelet"])
    p.add_argument(
        "--C", type=float, default=100.0, help="Regularisation constant (default: 100.0)."
    )
    p.add_argument(
        "--n-lags", type=int, default=4, dest="n_lags", help="Number of lag features (default: 4)."
    )
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        dest="test_ratio",
        help="Fraction of windowed samples for testing (default: 0.2).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_series(path: str | None) -> np.ndarray:
    """Return a 1-D float array from *path*, or a synthetic signal."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            print(f"[warn] Dataset file not found: {p}")
            print("[warn] Falling back to synthetic Mackey-Glass signal.")
        else:
            raw = np.loadtxt(p, delimiter=",", comments="#")
            series = raw.ravel().astype(float)
            print(f"[info] Loaded {len(series)} samples from {p.name}")
            return series
    return _mackey_glass_synthetic(n=1000)


def _mackey_glass_synthetic(
    n: int = 1000,
    tau: int = 17,
    n0: float = 1.2,
    a: float = 0.2,
    b: float = 0.1,
    dt: float = 1.0,
) -> np.ndarray:
    """Integrate the Mackey-Glass delay-differential equation numerically.

    x'(t) = a*x(t-tau)/(1 + x(t-tau)^10) - b*x(t)

    Parameters match the benchmark configuration used in the paper.
    """
    x = [n0] * (tau + 1)
    for _ in range(n):
        x_now = x[-1]
        x_tau = x[-tau - 1] if len(x) > tau else n0
        dx = a * x_tau / (1.0 + x_tau**10) - b * x_now
        x.append(x_now + dt * dx)
    series = np.array(x[tau + 1 :], dtype=float)  # discard warm-up
    print(f"[info] Generated synthetic Mackey-Glass signal ({len(series)} samples)")
    return series


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # --- 1. Load data -------------------------------------------------------
    series = _load_series(args.dataset)

    # --- 2. Normalise (fit stats on the full series before windowing) --------
    #        We fit the normaliser on the training portion only to avoid
    #        data leakage.  The split index is determined after windowing,
    #        so we first window then split.
    X_raw, y_raw = make_lag_features(series, n_lags=args.n_lags)

    n_total = len(y_raw)
    n_test = max(1, int(n_total * args.test_ratio))
    n_train = n_total - n_test

    if n_train < 2:
        sys.exit("[error] Not enough data for the chosen n_lags / test_ratio.")

    X_train_raw, X_test_raw = X_raw[:n_train], X_raw[n_train:]
    y_train_raw, y_test_raw = y_raw[:n_train], y_raw[n_train:]

    # Normalise features (fit on train only)
    scaler_X = ZScoreNormalizer()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_test = scaler_X.transform(X_test_raw)

    # Normalise target (fit on train only)
    scaler_y = ZScoreNormalizer()
    y_train = scaler_y.fit_transform(y_train_raw)
    y_test = scaler_y.transform(y_test_raw)

    print(f"[info] Train samples : {n_train}")
    print(f"[info] Test  samples : {n_test}")
    print(f"[info] n_lags        : {args.n_lags}")
    print(f"[info] Kernel        : {args.kernel} | C = {args.C}")

    # --- 3. Fit (offline batch) ---------------------------------------------
    model = OSELMK(C=args.C, kernel=args.kernel)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[info] Training time : {train_time:.4f} s  (support size = {model.n_train_})")

    # --- 4. Predict on test set and denormalise ------------------------------
    t1 = time.perf_counter()
    y_pred_norm = model.predict(X_test)
    predict_time = time.perf_counter() - t1

    y_pred = scaler_y.inverse_transform(y_pred_norm)
    y_true = scaler_y.inverse_transform(y_test)

    # --- 5. Metrics ---------------------------------------------------------
    metrics = compute_all(y_true, y_pred)
    print()
    print("=== Metrics (original scale) ===")
    for name, value in metrics.items():
        print(f"  {name.upper():6s} : {value:.6f}")
    print(f"  Predict time : {predict_time:.4f} s")

    # --- 6. Save results ----------------------------------------------------
    dataset_name = Path(args.dataset).stem if args.dataset else "mackeyglass_synthetic"
    run_dir = save_run(
        y_true=y_true,
        y_pred=y_pred,
        train_time_s=train_time,
        predict_time_s=predict_time,
        config={
            "kernel": args.kernel,
            "C": args.C,
            "n_lags": args.n_lags,
            "update_mode": "none (batch only)",
            "dataset": dataset_name,
            "n_train": n_train,
            "n_test": n_test,
        },
    )
    print(f"\n[info] Results saved to {run_dir}")


if __name__ == "__main__":
    main()
