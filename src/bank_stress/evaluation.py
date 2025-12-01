#!/usr/bin/env python3
"""
src/bank_stress/evaluation.py

Evaluation helpers for the bank stress project.

Provides
--------
- rebuild_dataset_and_split:
    convenience wrapper to build the ML dataset and perform the
    time-based train/test split.

- load_trained_model:
    load a serialized model (e.g. baseline logistic regression).

- evaluate_existing_model:
    run models.evaluate_model() and additionally return a vector
    of probability-like scores for the positive class.

- plot_roc_and_pr_curves:
    create and save ROC and Precision–Recall plots.

- compute_confusion_counts:
    compute TP, FP, TN, FN counts from true and predicted labels.

This module is designed to be small and notebook-friendly and is
used mainly in Notebook 05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from .models import (
    DatasetPaths,
    build_ml_dataset,
    evaluate_model,
    train_test_split_by_date,
)

# ---------------------------------------------------------------------------
# Core dataset + split helpers
# ---------------------------------------------------------------------------


def rebuild_dataset_and_split(
    paths: DatasetPaths,
    lags: int = 5,
    split_date: str = "2018-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience wrapper around build_ml_dataset + train_test_split_by_date.

    Parameters
    ----------
    paths : DatasetPaths
        Paths to stress_index.csv and a labels CSV.

    lags : int, default 5
        Number of lags for the stress metric in build_ml_dataset.

    split_date : str, default "2018-01-01"
        Calendar date used for the time-based train/test split.

    Returns
    -------
    features_train, features_test, labels_train, labels_test
    """
    features, labels = build_ml_dataset(paths=paths, lags=lags)

    # train_test_split_by_date expects keyword names: features, labels
    features_train, features_test, labels_train, labels_test = train_test_split_by_date(
        features=features,
        labels=labels,
        split_date=split_date,
    )
    return features_train, features_test, labels_train, labels_test


# ---------------------------------------------------------------------------
# Model loading and evaluation
# ---------------------------------------------------------------------------


def load_trained_model(model_path: Path) -> Any:
    """
    Load a previously trained model (e.g. baseline logistic regression).

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _get_probabilities_or_scores(
    model: Any,
    features_test: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Internal helper to compute a continuous score in [0, 1] for class 1.

    Order of attempts
    -----------------
    1. model.predict_proba(features_test)[:, 1]
    2. model.decision_function(features_test) → rescaled to [0, 1]
    3. return None if neither is available.
    """
    # Try predict_proba first
    try:
        prob_matrix = model.predict_proba(features_test)
        prob_pos = np.asarray(prob_matrix)[:, 1].ravel()
        return prob_pos
    except (AttributeError, TypeError, ValueError, IndexError):
        # Model doesn't implement predict_proba or returned an
        # unexpected shape; fall back to decision_function.
        pass

    # Fall back to decision_function
    try:
        raw_scores = model.decision_function(features_test)
        raw_scores = np.asarray(raw_scores).ravel()
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))
        if max_score > min_score:
            prob_pos = (raw_scores - min_score) / (max_score - min_score)
        else:
            prob_pos = np.zeros_like(raw_scores, dtype=float)
        return prob_pos
    except (AttributeError, TypeError, ValueError):
        # No usable decision_function either → give up on scores.
        return None


def evaluate_existing_model(
    model: Any,
    features_test: pd.DataFrame,
    labels_test: pd.Series,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Evaluate an already-trained model on a test set.

    Parameters
    ----------
    model : Any
        Fitted classifier (e.g. pipeline from train_baseline_logit).

    features_test : pd.DataFrame
        Test feature matrix.

    labels_test : pd.Series
        Test labels.

    Returns
    -------
    metrics : dict
        Dictionary of performance metrics as returned by models.evaluate_model.

    prob_pos : np.ndarray or None
        Probability-like scores for the positive class (y = 1) or
        None if they could not be computed.
    """
    metrics = evaluate_model(
        model=model,
        features_test=features_test,
        labels_test=labels_test,
    )

    prob_pos = _get_probabilities_or_scores(model, features_test)
    return metrics, prob_pos


# ---------------------------------------------------------------------------
# Confusion counts
# ---------------------------------------------------------------------------


def compute_confusion_counts(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, int]:
    """
    Compute TP, FP, TN, FN counts from true and predicted binary labels.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels (0/1).

    y_pred : pd.Series
        Predicted labels (0/1).

    Returns
    -------
    counts : dict
        Dictionary with keys: "tp", "fp", "tn", "fn".
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    counts = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print("\n=== Confusion counts (rows=true, cols=pred) ===")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return counts


# ---------------------------------------------------------------------------
# Plotting helpers (ROC and Precision–Recall curves)
# ---------------------------------------------------------------------------


def plot_roc_and_pr_curves(
    y_test: pd.Series,
    prob_pos: np.ndarray,
    output_dir: Path,
    prefix: str = "baseline_logit",
    show_plots: bool = True,
) -> Tuple[Path, Path]:
    """
    Plot ROC and Precision–Recall curves and save them to disk.

    Parameters
    ----------
    y_test : pd.Series
        Ground-truth test labels.

    prob_pos : np.ndarray
        Probability-like scores for the positive class.

    output_dir : Path
        Directory where the figures will be saved.

    prefix : str, default "baseline_logit"
        Filename prefix for the saved images.

    show_plots : bool, default True
        Whether to display the plots inline (useful in notebooks).

    Returns
    -------
    roc_path : Path
        Path to the saved ROC curve PNG.

    pr_path : Path
        Path to the saved Precision–Recall curve PNG.
    """
    if prob_pos is None:
        raise ValueError("prob_pos is None — cannot plot ROC/PR curves")

    output_dir.mkdir(parents=True, exist_ok=True)

    y_test_array = np.asarray(y_test).ravel()
    prob_pos_array = np.asarray(prob_pos).ravel()

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    pr_path = output_dir / f"{prefix}_pr_curve.png"

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_true=y_test_array,
        y_score=prob_pos_array,
        ax=ax_roc,
    )
    ax_roc.set_title("ROC Curve")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig_roc)

    # Precision–Recall
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(
        y_true=y_test_array,
        y_pred=prob_pos_array,
        ax=ax_pr,
    )
    ax_pr.set_title("Precision–Recall Curve")
    fig_pr.tight_layout()
    fig_pr.savefig(pr_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig_pr)

    return roc_path, pr_path
