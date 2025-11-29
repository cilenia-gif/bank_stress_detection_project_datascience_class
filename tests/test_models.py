#!/usr/bin/env python3
"""
Tests for src/bank_stress/models.py.

These tests use small synthetic CSV files written to a temporary
directory (pytest's tmp_path fixture), so they do not depend on
the real project data.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from bank_stress.models import (
    DatasetPaths,
    build_ml_dataset,
    evaluate_model,
    train_baseline_logit,
    train_test_split_by_date,
)


def _write_synthetic_processed(tmp_path: Path) -> Tuple[Path, Path]:
    """
    Helper to create tiny synthetic stress_index.csv and stress_labels.csv.

    Returns
    -------
    stress_index_path, labels_path : Path, Path
    """
    dates = pd.date_range("2000-01-01", periods=10, freq="D")

    stress_index_df = pd.DataFrame(
        {
            "avg_vol": np.linspace(0.01, 0.05, len(dates)),
            "avg_corr": np.linspace(0.2, 0.8, len(dates)),
            # base metric
            "stress_index": np.linspace(-1.0, 2.0, len(dates)),
            # smoothed metric (slightly different values)
            "stress_index_smooth": np.linspace(-0.8, 2.2, len(dates)),
        },
        index=dates,
    )

    # A few stress days towards the end
    labels_df = pd.DataFrame(
        {
            "stress_label": [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        },
        index=dates,
    )

    stress_index_path = tmp_path / "stress_index.csv"
    labels_path = tmp_path / "stress_labels.csv"

    stress_index_df.to_csv(stress_index_path)
    labels_df.to_csv(labels_path)

    return stress_index_path, labels_path


def test_build_ml_dataset_basic(tmp_path: Path) -> None:
    """
    build_ml_dataset should return non-empty (X, y) with lagged features
    and aligned indices.
    """
    stress_index_path, labels_path = _write_synthetic_processed(tmp_path)
    paths = DatasetPaths(
        stress_index_path=stress_index_path,
        labels_path=labels_path,
    )

    X, y = build_ml_dataset(paths=paths, lags=2)

    # Basic shape checks
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

    # Stress index smooth should be the core metric (present in features)
    assert "stress_index_smooth" in X.columns

    # Lagged features should exist
    assert "stress_index_smooth_lag1" in X.columns
    assert "stress_index_smooth_lag2" in X.columns

    # After dropping NaNs from lags, we should have fewer rows than original
    assert len(X) < 10


def test_build_ml_dataset_missing_label_column(tmp_path: Path) -> None:
    """
    If the labels CSV does not contain 'stress_label', the function
    should raise a KeyError with a clear message.
    """
    dates = pd.date_range("2000-01-01", periods=5, freq="D")

    stress_index_df = pd.DataFrame(
        {
            "avg_vol": np.linspace(0.01, 0.05, len(dates)),
            "avg_corr": np.linspace(0.2, 0.8, len(dates)),
            "stress_index": np.linspace(-1.0, 2.0, len(dates)),
        },
        index=dates,
    )
    labels_df = pd.DataFrame(
        {
            "not_stress_label": [0, 0, 1, 0, 1],
        },
        index=dates,
    )

    stress_index_path = tmp_path / "stress_index.csv"
    labels_path = tmp_path / "stress_labels.csv"

    stress_index_df.to_csv(stress_index_path)
    labels_df.to_csv(labels_path)

    paths = DatasetPaths(
        stress_index_path=stress_index_path,
        labels_path=labels_path,
    )

    with pytest.raises(KeyError):
        build_ml_dataset(paths=paths, lags=1)


def test_train_test_split_by_date_respects_boundary(tmp_path: Path) -> None:
    """
    train_test_split_by_date should split deterministically on the calendar
    date and produce non-empty train and test sets.
    """
    stress_index_path, labels_path = _write_synthetic_processed(tmp_path)
    paths = DatasetPaths(
        stress_index_path=stress_index_path,
        labels_path=labels_path,
    )

    X, y = build_ml_dataset(paths=paths, lags=1)

    split_date = "2000-01-06"
    X_train, X_test, y_train, y_test = train_test_split_by_date(
        X,
        y,
        split_date=split_date,
    )

    assert not X_train.empty
    assert not X_test.empty
    assert len(X_train) + len(X_test) == len(X)

    boundary = pd.to_datetime(split_date)
    assert X_train.index.max() < boundary
    assert X_test.index.min() >= boundary
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_train_and_evaluate_baseline_logit() -> None:
    """
    End-to-end smoke test:
    - create a tiny synthetic time series dataset in memory
    - split by date
    - train baseline logistic regression
    - evaluate and check that metrics are finite
    """
    # Synthetic dates
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    # Two simple feature trends
    x0 = np.linspace(0.0, 1.0, len(dates))
    x1 = np.linspace(1.0, 0.0, len(dates))

    X = pd.DataFrame(
        {
            "avg_vol": x0,
            "avg_corr": x1,
            "stress_index_smooth": x0,
        },
        index=dates,
    )

    # Binary target with both classes present (every 5th day is stress = 1)
    y = pd.Series(0, index=dates, name="stress_label")
    y.iloc[::5] = 1

    # Split deterministically by date
    X_train, X_test, y_train, y_test = train_test_split_by_date(
        X,
        y,
        split_date="2020-01-21",
    )

    # Train baseline model
    model = train_baseline_logit(
        X_train,
        y_train,
    )

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Basic sanity checks: metrics exist and are finite numbers
    expected_keys = {
        "roc_auc",
        "avg_precision",
        "accuracy",
        "precision_1",
        "recall_1",
    }
    assert set(metrics.keys()) == expected_keys
    assert not np.isnan(metrics["accuracy"])
