#!/usr/bin/env python3
"""
src/bank_stress/models.py

Machine learning utilities for the bank stress project.

Provides:
- DatasetPaths: small dataclass collecting important CSV paths.
- build_ml_dataset: build feature matrix and label vector from CSV files.
- train_test_split_by_date: deterministic time-based train/test split.
- train_baseline_logit: simple logistic-regression pipeline.
- evaluate_model: evaluation metrics + printed classification report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetPaths:
    """
    Container for processed CSV paths used to build the ML dataset.

    Attributes
    ----------
    stress_index_path : Path
        Path to `data/processed/stress_index.csv`.

    labels_path : Path
        Path to `data/processed/stress_labels.csv` (main spec).
    """

    stress_index_path: Path
    labels_path: Path


def build_ml_dataset(
    paths: DatasetPaths,
    lags: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a supervised-learning dataset from processed stress index data.

    The function:
      1. Loads the stress index and labels CSVs.
      2. Ensures both have *timezone-naive* DatetimeIndex indices.
      3. Chooses the stress metric:
         - 'stress_index_smooth' if available, else 'stress_index'.
      4. Creates features from:
         - 'avg_vol', 'avg_corr' if present, and
         - the chosen stress metric plus its lagged values.
      5. Aligns with the binary 'stress_label' from the labels file.
    """
    if lags < 0:
        raise ValueError("lags must be >= 0")

    # ------------------------------------------------------------------
    # 1) Load CSVs
    # ------------------------------------------------------------------
    stress_df = pd.read_csv(
        paths.stress_index_path,
        index_col=0,
        parse_dates=True,
    )

    labels_df = pd.read_csv(
        paths.labels_path,
        index_col=0,
        parse_dates=True,
    )

    # ------------------------------------------------------------------
    # 2) Ensure both indices are timezone-naive DatetimeIndex
    #    (this fixes the 'tz-naive vs tz-aware' join error)
    # ------------------------------------------------------------------
    def _make_naive_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with a tz-naive DatetimeIndex."""
        idx = df.index

        # If not already datetime, try to parse.
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, errors="coerce")

        # Drop rows where the index could not be parsed.
        valid_mask = ~idx.isna()
        df = df.loc[valid_mask].copy()
        idx = idx[valid_mask]

        # If timezone-aware, drop the timezone information.
        if idx.tz is not None:
            idx = idx.tz_convert(None)

        df.index = idx
        return df.sort_index()

    stress_df = _make_naive_datetime_index(stress_df)
    labels_df = _make_naive_datetime_index(labels_df)

    # ------------------------------------------------------------------
    # 3) Basic checks and metric choice
    # ------------------------------------------------------------------
    if "stress_label" not in labels_df.columns:
        raise KeyError(
            "labels_path must contain a 'stress_label' column; "
            "did you run Notebook 03?"
        )

    # Choose which stress metric to use as the main signal.
    if "stress_index_smooth" in stress_df.columns:
        metric_column = "stress_index_smooth"
    elif "stress_index" in stress_df.columns:
        metric_column = "stress_index"
    else:
        raise KeyError(
            "No 'stress_index_smooth' or 'stress_index' column found in "
            f"{paths.stress_index_path}"
        )

    # ------------------------------------------------------------------
    # 4) Align stress index and labels on overlapping dates
    # ------------------------------------------------------------------
    combined = stress_df.join(labels_df[["stress_label"]], how="inner")
    if combined.empty:
        raise ValueError("No overlapping dates between stress index and labels.")

    # ------------------------------------------------------------------
    # 5) Build feature matrix
    # ------------------------------------------------------------------
    feature_columns: list[str] = []
    for column in ("avg_vol", "avg_corr"):
        if column in combined.columns:
            feature_columns.append(column)

    feature_columns.append(metric_column)

    features = combined[feature_columns].copy()
    metric_series = combined[metric_column]

    # Add lagged versions of the main stress metric.
    for lag in range(1, lags + 1):
        features[f"{metric_column}_lag{lag}"] = metric_series.shift(lag)

    # Target vector.
    labels = combined["stress_label"].astype(int)

    # Drop rows with NaNs in any feature (e.g. early rows due to lags).
    valid_mask = ~features.isna().any(axis=1)
    features = features.loc[valid_mask].copy()
    labels = labels.loc[valid_mask].copy()

    if features.empty:
        raise ValueError(
            "All rows were dropped after creating lagged features. "
            "Try using fewer lags or check for NaNs in the inputs."
        )

    return features, labels


def train_test_split_by_date(
    features: pd.DataFrame,
    labels: pd.Series,
    split_date: str = "2018-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Deterministic train/test split using a calendar date.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix indexed by date.

    labels : pd.Series
        Binary labels indexed by the same dates as `features`.

    split_date : str, default "2018-01-01"
        Date string. All observations strictly before this date
        go to the train set; all on or after go to the test set.

    Returns
    -------
    features_train, features_test, labels_train, labels_test
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features.index must be a DatetimeIndex")

    if not features.index.equals(labels.index):
        raise ValueError("features and labels must have the same index")

    boundary = pd.to_datetime(split_date)

    train_mask = features.index < boundary
    test_mask = features.index >= boundary

    features_train = features.loc[train_mask]
    labels_train = labels.loc[train_mask]
    features_test = features.loc[test_mask]
    labels_test = labels.loc[test_mask]

    if features_train.empty or features_test.empty:
        raise ValueError(
            "Train or test split is empty; adjust split_date to fall "
            "within the data range."
        )

    return features_train, features_test, labels_train, labels_test


def train_baseline_logit(
    features_train: pd.DataFrame,
    labels_train: pd.Series,
    c_param: float = 1.0,
    solver: str = "lbfgs",
    random_state: int | None = 0,
) -> Pipeline:
    """
    Train a simple logistic-regression baseline pipeline.

    The pipeline has two steps:
      - StandardScaler
      - LogisticRegression(class_weight='balanced')

    Parameters
    ----------
    features_train : pd.DataFrame
        Training feature matrix.

    labels_train : pd.Series
        Training labels (0/1).

    c_param : float, default 1.0
        Inverse regularization strength for LogisticRegression.

    solver : str, default "lbfgs"
        Solver passed to LogisticRegression.

    random_state : int or None, default 0
        Random seed for reproducibility.

    Returns
    -------
    model : Pipeline
        Fitted sklearn pipeline.
    """
    unique_labels = set(np.unique(labels_train))
    if unique_labels - {0, 1}:
        raise ValueError("labels_train must be binary (0/1)")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    class_weight="balanced",
                    C=c_param,
                    solver=solver,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(features_train, labels_train)
    return model


def evaluate_model(
    model: Any,
    features_test: pd.DataFrame,
    labels_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a fitted classifier on a test set.

    Metrics:
      - ROC AUC (if scores available)
      - Average precision (PR AUC, if scores available)
      - Accuracy
      - Precision for class 1
      - Recall for class 1

    A short classification report is printed.

    The function tries `predict_proba` first. If unavailable, it falls
    back to `decision_function` and rescales scores to [0, 1]. If that
    also fails, probability-based metrics are returned as NaN.
    """
    if features_test.empty:
        raise ValueError("features_test is empty")
    if labels_test.empty:
        raise ValueError("labels_test is empty")

    # Try to obtain "scores" for the positive class.
    prob_pos: np.ndarray | None
    try:
        prob_pos = model.predict_proba(features_test)[:, 1]
    except (AttributeError, IndexError, TypeError, ValueError):
        # Fall back to decision_function if available.
        try:
            raw_scores = model.decision_function(features_test)
            raw_scores = np.asarray(raw_scores).ravel()
            min_score = raw_scores.min()
            max_score = raw_scores.max()
            if max_score > min_score:
                prob_pos = (raw_scores - min_score) / (max_score - min_score)
            else:
                prob_pos = np.zeros_like(raw_scores)
        except (AttributeError, TypeError, ValueError):
            prob_pos = None

    predicted_labels = model.predict(features_test)

    # Metrics that need probabilities.
    if prob_pos is not None:
        try:
            roc_value = roc_auc_score(labels_test, prob_pos)
        except ValueError:
            roc_value = float("nan")
        try:
            avg_precision_value = average_precision_score(labels_test, prob_pos)
        except ValueError:
            avg_precision_value = float("nan")
    else:
        roc_value = float("nan")
        avg_precision_value = float("nan")

    accuracy_value = accuracy_score(labels_test, predicted_labels)
    precision_pos = float(
        precision_score(
            labels_test,
            predicted_labels,
            pos_label=1,
            zero_division=0,
        )
    )
    recall_pos = float(
        recall_score(
            labels_test,
            predicted_labels,
            pos_label=1,
            zero_division=0,
        )
    )

    metrics: Dict[str, float] = {
        "roc_auc": float(roc_value),
        "avg_precision": float(avg_precision_value),
        "accuracy": float(accuracy_value),
        "precision_1": precision_pos,
        "recall_1": recall_pos,
    }

    print("\n=== Test set metrics ===")
    for name, value in metrics.items():
        print(f"{name:>12}: {value:.3f}")

    print("\n=== Classification report ===")
    print(
        classification_report(
            labels_test,
            predicted_labels,
            zero_division=0,
        )
    )

    return metrics
