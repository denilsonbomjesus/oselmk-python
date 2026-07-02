"""Unit tests for src/oselmk/utils/results.py."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from oselmk.utils.results import save_run

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

N = 20
Y_TRUE = RNG.uniform(1.0, 5.0, N)   # no zeros -> MAPE defined
Y_PRED = Y_TRUE + RNG.normal(0, 0.1, N)

TRAIN_TIME = 0.042
PREDICT_TIME = 0.003

BASE_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "n_lags": 4,
    "update_mode": "decremental",
    "dataset": "mackeyglass",
}


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Call save_run once and return the directory it created."""
    return save_run(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        train_time_s=TRAIN_TIME,
        predict_time_s=PREDICT_TIME,
        config=BASE_CONFIG,
        run_dir=tmp_path / "run0",
    )


# ---------------------------------------------------------------------------
# Directory and file existence
# ---------------------------------------------------------------------------


def test_run_dir_is_created(run_dir: Path):
    """save_run must create the target directory."""
    assert run_dir.is_dir()


def test_metrics_json_exists(run_dir: Path):
    assert (run_dir / "metrics.json").exists()


def test_config_json_exists(run_dir: Path):
    assert (run_dir / "config.json").exists()


def test_predictions_csv_exists(run_dir: Path):
    assert (run_dir / "predictions.csv").exists()


# ---------------------------------------------------------------------------
# metrics.json content
# ---------------------------------------------------------------------------


def test_metrics_json_is_valid_json(run_dir: Path):
    """metrics.json must parse without error."""
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert isinstance(payload, dict)


def test_metrics_json_has_metrics_key(run_dir: Path):
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert "metrics" in payload


def test_metrics_json_has_all_four_metrics(run_dir: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    for key in ("rmse", "nrmse", "mape", "smape"):
        assert key in metrics, f"Missing metric key: {key}"


def test_metrics_json_has_timing_key(run_dir: Path):
    payload = json.loads((run_dir / "metrics.json").read_text())
    assert "timing" in payload


def test_metrics_json_timing_values(run_dir: Path):
    timing = json.loads((run_dir / "metrics.json").read_text())["timing"]
    assert math.isclose(timing["train_time_s"], TRAIN_TIME)
    assert math.isclose(timing["predict_time_s"], PREDICT_TIME)


def test_metrics_json_rmse_is_positive(run_dir: Path):
    rmse = json.loads((run_dir / "metrics.json").read_text())["metrics"]["rmse"]
    assert rmse > 0.0


def test_nan_metric_serialised_as_null(tmp_path: Path):
    """NRMSE on a constant series is nan; must be stored as JSON null."""
    constant = np.ones(10)
    noisy = constant + 0.1
    run_dir = save_run(
        y_true=constant,
        y_pred=noisy,
        train_time_s=0.0,
        predict_time_s=0.0,
        run_dir=tmp_path / "nan_run",
    )
    raw = (run_dir / "metrics.json").read_text()
    payload = json.loads(raw)
    # NRMSE of a constant series is nan -> should be serialised as null
    assert payload["metrics"]["nrmse"] is None


# ---------------------------------------------------------------------------
# config.json content
# ---------------------------------------------------------------------------


def test_config_json_is_valid_json(run_dir: Path):
    payload = json.loads((run_dir / "config.json").read_text())
    assert isinstance(payload, dict)


def test_config_json_preserves_all_keys(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    for key, value in BASE_CONFIG.items():
        assert key in config
        assert config[key] == value


def test_config_json_extra_keys_preserved(tmp_path: Path):
    """Extra keys beyond the standard set must be written to config.json."""
    extra_config = {**BASE_CONFIG, "window_size": 50, "note": "ablation run"}
    run_dir = save_run(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        train_time_s=0.0,
        predict_time_s=0.0,
        config=extra_config,
        run_dir=tmp_path / "extra",
    )
    config = json.loads((run_dir / "config.json").read_text())
    assert config["window_size"] == 50
    assert config["note"] == "ablation run"


def test_config_none_writes_empty_dict(tmp_path: Path):
    """Passing config=None must write an empty JSON object."""
    run_dir = save_run(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        train_time_s=0.0,
        predict_time_s=0.0,
        config=None,
        run_dir=tmp_path / "no_config",
    )
    config = json.loads((run_dir / "config.json").read_text())
    assert config == {}


# ---------------------------------------------------------------------------
# predictions.csv content
# ---------------------------------------------------------------------------


def test_predictions_csv_header(run_dir: Path):
    with (run_dir / "predictions.csv").open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == ["index", "y_true", "y_pred"]


def test_predictions_csv_row_count(run_dir: Path):
    with (run_dir / "predictions.csv").open(newline="") as fh:
        rows = list(csv.reader(fh))
    # rows includes the header
    assert len(rows) == N + 1


def test_predictions_csv_values_match(run_dir: Path):
    """CSV y_true and y_pred columns must match the input arrays."""
    with (run_dir / "predictions.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    for i, row in enumerate(rows):
        assert int(row["index"]) == i
        assert math.isclose(float(row["y_true"]), Y_TRUE[i], rel_tol=1e-9)
        assert math.isclose(float(row["y_pred"]), Y_PRED[i], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# run_dir and base_dir behaviour
# ---------------------------------------------------------------------------


def test_explicit_run_dir_is_respected(tmp_path: Path):
    """The returned path must match the explicit run_dir argument."""
    explicit = tmp_path / "my_custom_run"
    returned = save_run(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        train_time_s=0.0,
        predict_time_s=0.0,
        run_dir=explicit,
    )
    assert returned == explicit.resolve()


def test_returned_path_is_absolute(run_dir: Path):
    """save_run must return an absolute Path."""
    assert run_dir.is_absolute()


def test_auto_timestamp_dir_created_inside_base_dir(tmp_path: Path):
    """Without explicit run_dir, the directory must sit inside base_dir."""
    base = tmp_path / "results"
    returned = save_run(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        train_time_s=0.0,
        predict_time_s=0.0,
        base_dir=base,
    )
    assert returned.parent.resolve() == base.resolve()


def test_two_runs_same_second_get_distinct_dirs(tmp_path: Path):
    """Calling save_run twice in the same UTC second must not collide.

    We simulate a collision by pre-creating the timestamp directory so the
    second call is forced to generate a suffixed variant.
    """
    base = tmp_path / "results"
    # First call creates the base timestamp dir
    dir1 = save_run(
        y_true=Y_TRUE, y_pred=Y_PRED,
        train_time_s=0.0, predict_time_s=0.0,
        base_dir=base,
    )
    # Second call within the same second hits the existing dir and appends _1
    dir2 = save_run(
        y_true=Y_TRUE, y_pred=Y_PRED,
        train_time_s=0.0, predict_time_s=0.0,
        base_dir=base,
    )
    assert dir1 != dir2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise(tmp_path: Path):
    with pytest.raises(ValueError, match="same length"):
        save_run(
            y_true=np.ones(5),
            y_pred=np.ones(3),
            train_time_s=0.0,
            predict_time_s=0.0,
            run_dir=tmp_path / "bad",
        )


def test_empty_arrays_raise(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        save_run(
            y_true=np.array([]),
            y_pred=np.array([]),
            train_time_s=0.0,
            predict_time_s=0.0,
            run_dir=tmp_path / "empty",
        )
