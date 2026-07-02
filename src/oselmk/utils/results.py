"""Results persistence for OS-ELMK experiments.

Every call to :func:`save_run` writes three artefacts into a timestamped
sub-directory of ``results/``:

``metrics.json``
    Prediction quality metrics (RMSE, NRMSE, MAPE, SMAPE) and wall-clock
    timing for the train and predict phases.

``config.json``
    Hyper-parameters and experiment metadata (kernel, C, n_lags,
    update_mode, dataset, plus any extra keys the caller supplies).

``predictions.csv``
    One row per test sample: ``index,y_true,y_pred``.

Directory layout
----------------
::

    results/
    ├── 2026-07-02_14-30-00/
    │   ├── metrics.json
    │   ├── config.json
    │   └── predictions.csv
    └── 2026-07-02_15-05-42/
        ├── metrics.json
        ...

Timestamps are UTC-based (``datetime.utcnow``) so runs started from
different time-zones are still sortable on disk.  Two runs started within
the same second get distinct directories via a ``_{n}`` numeric suffix.

NaN serialisation
-----------------
Some metrics return ``nan`` for degenerate series (e.g. NRMSE on a
constant series, MAPE when all true values are zero).  Standard
``json.dump`` raises :class:`ValueError` for ``float('nan')``.  This
module uses a custom :class:`_NaNEncoder` that maps ``nan`` / ``inf`` to
JSON ``null`` so the file always parses correctly.

Usage example
-------------
::

    from oselmk.utils.results import save_run
    import time

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t1

    run_dir = save_run(
        y_true=y_test,
        y_pred=y_pred,
        train_time_s=train_time,
        predict_time_s=predict_time,
        config={
            "kernel": "rbf",
            "C": 1.0,
            "n_lags": 4,
            "update_mode": "decremental",
            "dataset": "mackeyglass",
        },
    )
    print(f"Results saved to {run_dir}")
"""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oselmk.utils.metrics import compute_all

# ---------------------------------------------------------------------------
# JSON encoder that maps NaN / Inf -> null
# ---------------------------------------------------------------------------


class _NaNEncoder(json.JSONEncoder):
    """JSON encoder that serialises ``nan`` and ``inf`` as ``null``.

    Standard :mod:`json` raises ``ValueError`` for non-finite floats.
    OS-ELMK metrics can legitimately return ``nan`` (e.g. NRMSE on a
    constant series), so we map them to JSON ``null`` instead.
    """

    def iterencode(self, o: Any, _one_shot: bool = False):
        """Walk the object tree and replace non-finite floats with None."""
        return super().iterencode(self._sanitise(o), _one_shot)

    @classmethod
    def _sanitise(cls, obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: cls._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._sanitise(v) for v in obj]
        return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_run(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    train_time_s: float,
    predict_time_s: float,
    config: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
    base_dir: str | Path = "results",
) -> Path:
    """Persist metrics, config, and predictions for a single experiment run.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth test values.
    y_pred : array-like, shape (n,)
        Model predictions on the test set.
    train_time_s : float
        Wall-clock seconds spent in the training / fit phase.
    predict_time_s : float
        Wall-clock seconds spent in the prediction phase.
    config : dict, optional
        Experiment configuration to persist.  Recognised (but not required)
        keys:

        * ``'kernel'``      — kernel name string (e.g. ``'rbf'``)
        * ``'C'``           — regularisation constant
        * ``'n_lags'``      — lag window used for feature construction
        * ``'update_mode'`` — ``'sequential'`` or ``'decremental'``
        * ``'dataset'``     — dataset identifier string

        Any additional keys are preserved verbatim in ``config.json``.
        If ``None``, an empty dict is written.

    run_dir : str or Path, optional
        Explicit directory to write artefacts into.  If supplied, it
        overrides the automatic ``<base_dir>/<timestamp>/`` path.  Useful
        for tests and reproducible pipelines.
    base_dir : str or Path, default ``'results'``
        Root directory under which timestamped sub-directories are created.
        Resolved relative to the current working directory when not absolute.
        Ignored when ``run_dir`` is supplied.

    Returns
    -------
    Path
        Absolute path to the directory where artefacts were written.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    # --- Resolve output directory -----------------------------------------
    target_dir = _resolve_run_dir(run_dir, base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics.json -------------------------------------------------------
    metric_values = compute_all(y_true, y_pred)
    metrics_payload: dict[str, Any] = {
        "metrics": metric_values,
        "timing": {
            "train_time_s": float(train_time_s),
            "predict_time_s": float(predict_time_s),
        },
    }
    _write_json(target_dir / "metrics.json", metrics_payload)

    # --- config.json -------------------------------------------------------
    config_payload: dict[str, Any] = dict(config) if config else {}
    _write_json(target_dir / "config.json", config_payload)

    # --- predictions.csv ---------------------------------------------------
    _write_predictions_csv(target_dir / "predictions.csv", y_true, y_pred)

    return target_dir.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_run_dir(
    run_dir: str | Path | None,
    base_dir: str | Path,
) -> Path:
    """Return the Path where artefacts should be written.

    If ``run_dir`` is given, return it directly.
    Otherwise, build ``<base_dir>/<timestamp>`` and add a numeric suffix
    (``_1``, ``_2``, ...) if the directory already exists so that two runs
    starting within the same second never collide.
    """
    if run_dir is not None:
        return Path(run_dir)

    # UTC timestamp — sortable and timezone-independent
    stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    candidate = Path(base_dir) / stamp

    # Collision avoidance: append _1, _2, ... if the directory already exists
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        suffixed = Path(base_dir) / f"{stamp}_{counter}"
        if not suffixed.exists():
            return suffixed
        counter += 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as indented JSON, mapping NaN to null."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, cls=_NaNEncoder, indent=2, ensure_ascii=False)
        fh.write("\n")  # POSIX-friendly trailing newline


def _write_predictions_csv(
    path: Path,
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> None:
    """Write index/y_true/y_pred rows to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "y_true", "y_pred"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred, strict=True)):
            writer.writerow([i, float(yt), float(yp)])
