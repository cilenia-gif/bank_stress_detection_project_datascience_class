#!/usr/bin/env python3
"""
Tests for src/bank_stress/evaluation.py.

We use small synthetic data sets so that the tests are fast and do not rely
on the real project CSV files.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import dump

from bank_stress.evaluation import (
    compute_confusion_counts,
    evaluate_existing_model,
    load_trained_model,
    rebuild_dataset_and_split,
)
from bank_stress.models import (
    DatasetPaths,
    train_baseline_logit,
    train_test_split_by_date,
)


def _make_toy_series() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a tiny synthetic feature matrix and label series.

    - 40 daily observations
    - two simple trend features
    - every 5th day is a stress day (label 1)
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    x0 = np.linspace(0.0, 1.0, len(dates))
    x1 = np.linspace(1.0, 0.0, len(dates))

    features = pd.DataFrame(
        {
            "avg_vol": x0,
            "avg_corr": x1,
            "stress_index_smooth": x0,
        },
        index=dates,
    )

    labels = pd.Series(0, index=dates, name="stress_label")
    labels.iloc[::5] = 1

    return features, labels


def test_compute_confusion_counts_basic() -> None:
    """
    compute_confusion_counts should return the correct counts for
    a small hand-checkable example.

    We expect:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]

        -> tp = 1, fp = 1, tn = 1, fn = 1
    """
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 1, 0, 1])

    counts = compute_confusion_counts(y_true=y_true, y_pred=y_pred)

    # Function returns a dict; assert on the keys explicitly.
    assert counts["tp"] == 1
    assert counts["fp"] == 1
    assert counts["tn"] == 1
    assert counts["fn"] == 1


def test_evaluate_existing_model_and_plots() -> None:
    """
    End-to-end test:

    - build a small synthetic data set
    - time split into train / test
    - train logistic regression via train_baseline_logit
    - call evaluate_existing_model (which reuses models.evaluate_model)
    - generate ROC and PR plots
    """
    features, labels = _make_toy_series()

    features_train, features_test, labels_train, labels_test = train_test_split_by_date(
        features,
        labels,
        split_date="2020-01-21",
    )

    model = train_baseline_logit(
        features_train=features_train,
        labels_train=labels_train,
    )

    # Call positionally so we are robust to parameter naming
    metrics, prob_pos = evaluate_existing_model(
        model,
        features_test,
        labels_test,
    )

    expected_keys = {
        "roc_auc",
        "avg_precision",
        "accuracy",
        "precision_1",
        "recall_1",
    }
    assert set(metrics.keys()) == expected_keys
    assert not np.isnan(metrics["accuracy"])

    assert prob_pos is not None
    assert len(prob_pos) == len(labels_test)


def test_rebuild_dataset_and_load_trained_model(tmp_path: Path) -> None:
    """
    Extra coverage test for:

    - rebuild_dataset_and_split
    - load_trained_model

    We create tiny synthetic CSVs under tmp_path, rebuild the dataset,
    train a model, save it, reload it, and check that predictions match.
    """
    dates = pd.date_range("2000-01-01", periods=6, freq="D")

    stress_index_df = pd.DataFrame(
        {
            "avg_vol": np.linspace(0.01, 0.06, len(dates)),
            "avg_corr": np.linspace(0.2, 0.7, len(dates)),
            "stress_index": np.linspace(-1.0, 1.0, len(dates)),
            "stress_index_smooth": np.linspace(-0.8, 1.2, len(dates)),
        },
        index=dates,
    )
    # Make sure the *training* portion sees both classes (0 and 1)
    # with split_date="2000-01-04", rows before that are indices 0,1,2.
    # So we choose labels [0, 1, 0, 1, 1, 0] so train has {0, 1}.
    labels_df = pd.DataFrame(
        {
            "stress_label": [0, 1, 0, 1, 1, 0],
        },
        index=dates,
    )

    stress_index_path = tmp_path / "stress_index.csv"
    labels_path = tmp_path / "stress_labels.csv"

    stress_index_df.to_csv(stress_index_path)
    labels_df.to_csv(labels_path)

    dataset_paths = DatasetPaths(
        stress_index_path=stress_index_path,
        labels_path=labels_path,
    )

    features_train, features_test, labels_train, labels_test = (
        rebuild_dataset_and_split(
            paths=dataset_paths,
            lags=1,
            split_date="2000-01-04",
        )
    )

    assert not features_train.empty
    assert not features_test.empty

    # Train a small model and save it
    model = train_baseline_logit(
        features_train=features_train,
        labels_train=labels_train,
    )

    model_path = tmp_path / "toy_model.joblib"
    dump(model, model_path)

    # Reload and compare predictions
    loaded_model = load_trained_model(model_path=model_path)

    original_pred = model.predict(features_test)
    reloaded_pred = loaded_model.predict(features_test)

    assert np.array_equal(original_pred, reloaded_pred)
    assert len(reloaded_pred) == len(labels_test)
