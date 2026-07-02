#!/usr/bin/env python3
"""OS-ELMK with sequential (incremental) online update.

This script demonstrates the incremental learning mode of OS-ELMK:
the model starts with an initial batch fit, then receives new data
block-by-block.  The support set grows with each update (n_train_ += bs),
so memory usage increases over time but the model accumulates all past
information.

This mode is suitable for **stationary or slowly drifting** series where
retaining all past data is beneficial.

Pipeline
--------
1. Load and normalise the series (train scaler on the initial batch only).
2. Build lag features and split into: initial batch | online blocks | test.
3. Fit OSELMK on the initial batch.
4. Stream new blocks one at a time via model.update(mode='sequential').
5. After all updates, predict on the held-out test set.
6. Denormalise, compute metrics, save results.

Usage
-----
    python examples/run_sequential.py
    python examples/run_sequential.py --dataset datasets/mackeyglass.csv
    python examples/run_sequential.py --dataset datasets/mackeyglass.csv \\
        --kernel rbf --C 100 --n-lags 6 --block-size 10 \\
        --init-ratio 0.5 --test-ratio 0.2

Arguments
---------
--dataset PATH      Path to a single-column CSV or text file.
--kernel NAME       rbf | linear | poly | wavelet  (default: rbf).
--C FLOAT           Regularisation constant (default: 100.0).
--n-lags INT        Number of lag features (default: 4).
--block-size INT    Number of new samples per online update (default: 10).
--init-ratio FLOAT  Fraction of windowed samples used for initial batch
                    (default: 0.5).
--test-ratio FLOAT  Fraction of windowed samples held out for final test
                    (default: 0.2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

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
    p = argparse.ArgumentParser(
        description="OS-ELMK sequential (incremental) online update."
    )
    p.add_argument("--dataset", default=None)
    p.add_argument("--kernel", default="rbf",
                   choices=["rbf", "linear", "poly", "wavelet"])
    p.add_argument("--C", type=float, default=100.0)
    p.add_argument("--n-lags", type=int, default=4, dest="n_lags")
    p.add_argument("--block-size", type=int, default=10, dest="block_size")
    p.add_argument("--init-ratio", type=float, default=0.5, dest="init_ratio")
    p.add_argument("--test-ratio", type=float, default=0.2, dest="test_ratio")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic dataset (shared helper)
# ---------------------------------------------------------------------------

def _mackey_glass_synthetic(n: int = 1200) -> np.ndarray:
    """Mackey-Glass delay-differential equation (tau=17)."""
    tau, n0, a, b, dt = 17, 1.2, 0.2, 0.1, 1.0
    x = [n0] * (tau + 1)
    for _ in range(n):
        x_now = x[-1]
        x_tau = x[-tau - 1] if len(x) > tau else n0
        x.append(x_now + dt * (a * x_tau / (1.0 + x_tau ** 10) - b * x_now))
    series = np.array(x[tau + 1:], dtype=float)
    print(f"[info] Generated synthetic Mackey-Glass signal ({len(series)} samples)")
    return series


def _load_series(path: str | None) -> np.ndarray:
    if path is not None:
        p = Path(path)
        if p.exists():
            series = np.loadtxt(p, delimiter=",", comments="#").ravel().astype(float)
            print(f"[info] Loaded {len(series)} samples from {p.name}")
            return series
        print(f"[warn] File not found: {path} — using synthetic signal.")
    return _mackey_glass_synthetic()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # --- 1. Load and window data --------------------------------------------
    series = _load_series(args.dataset)
    X_raw, y_raw = make_lag_features(series, n_lags=args.n_lags)
    n_total = len(y_raw)

    n_test   = max(1, int(n_total * args.test_ratio))
    n_online = n_total - n_test          # samples available for train + stream
    n_init   = max(2, int(n_online * args.init_ratio))

    if n_init >= n_online:
        sys.exit("[error] init_ratio too large; no samples left for online updates.")

    # Splits (chronological order preserved)
    X_init,   y_init   = X_raw[:n_init],         y_raw[:n_init]
    X_stream, y_stream = X_raw[n_init:n_online],  y_raw[n_init:n_online]
    X_test,   y_test   = X_raw[n_online:],        y_raw[n_online:]

    # --- 2. Normalise (fit on initial batch only — no leakage) --------------
    scaler_X = ZScoreNormalizer()
    X_init_n = scaler_X.fit_transform(X_init)
    X_stream_n = scaler_X.transform(X_stream)
    X_test_n   = scaler_X.transform(X_test)

    scaler_y = ZScoreNormalizer()
    y_init_n   = scaler_y.fit_transform(y_init)
    y_stream_n = scaler_y.transform(y_stream)
    y_test_n   = scaler_y.transform(y_test)

    print(f"[info] Initial batch : {n_init} samples")
    print(f"[info] Online stream : {len(y_stream)} samples  "
          f"(block_size = {args.block_size})")
    print(f"[info] Test set      : {n_test} samples")
    print(f"[info] Kernel: {args.kernel} | C: {args.C} | n_lags: {args.n_lags}")

    # --- 3. Initial batch fit -----------------------------------------------
    model = OSELMK(C=args.C, kernel=args.kernel)

    t_fit_start = time.perf_counter()
    model.fit(X_init_n, y_init_n)
    t_fit_end = time.perf_counter()
    train_time = t_fit_end - t_fit_start

    print(f"\n[info] Batch fit done  — support size = {model.n_train_}  "
          f"({train_time:.4f} s)")

    # --- 4. Sequential online updates ---------------------------------------
    bs = args.block_size
    n_updates = 0
    t_update_total = 0.0

    n_stream = len(y_stream_n)
    for start in range(0, n_stream, bs):
        end = min(start + bs, n_stream)
        X_blk = X_stream_n[start:end]
        y_blk = y_stream_n[start:end]

        t0 = time.perf_counter()
        model.update(X_blk, y_blk, mode="sequential")
        t_update_total += time.perf_counter() - t0
        n_updates += 1

    print(f"[info] Online updates  — {n_updates} block(s)  "
          f"(total {t_update_total:.4f} s)")
    print(f"[info] Final support size = {model.n_train_}")

    # --- 5. Predict and denormalise -----------------------------------------
    t1 = time.perf_counter()
    y_pred_n = model.predict(X_test_n)
    predict_time = time.perf_counter() - t1

    y_pred = scaler_y.inverse_transform(y_pred_n)
    y_true = scaler_y.inverse_transform(y_test_n)

    # --- 6. Metrics ---------------------------------------------------------
    metrics = compute_all(y_true, y_pred)
    print()
    print("=== Metrics (original scale) ===")
    for name, value in metrics.items():
        print(f"  {name.upper():6s} : {value:.6f}")
    print(f"  Predict time : {predict_time:.4f} s")

    # --- 7. Save results ----------------------------------------------------
    dataset_name = (
        Path(args.dataset).stem if args.dataset else "mackeyglass_synthetic"
    )
    run_dir = save_run(
        y_true=y_true,
        y_pred=y_pred,
        train_time_s=train_time + t_update_total,
        predict_time_s=predict_time,
        config={
            "kernel": args.kernel,
            "C": args.C,
            "n_lags": args.n_lags,
            "update_mode": "sequential",
            "dataset": dataset_name,
            "block_size": bs,
            "n_init": n_init,
            "n_online_blocks": n_updates,
            "final_support_size": model.n_train_,
            "n_test": n_test,
        },
    )
    print(f"\n[info] Results saved to {run_dir}")


if __name__ == "__main__":
    main()
