#!/usr/bin/env python3
"""
Main entry point for the bank stress detection project.

This script glues together the functionality from src/bank_stress:

- models.py       → dataset building, time-based split, baseline model
- evaluation.py   → evaluation helpers and plots

It:
  1. Loads processed data and label files.
  2. Builds a supervised dataset with lagged features.
  3. Performs a time-based train/test split.
  4. Trains a logistic-regression baseline.
  5. Evaluates performance and prints metrics.
  6. (Optionally) repeats the evaluation for robustness label specs.
  7. Prints a short conclusion that answers the research question:
     "Can a simple logistic-regression model based on the stress index
      detect bank stress days in out-of-sample data?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths and sys.path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Make src/ importable when running `python main.py` from the project root.
# We modify sys.path first, then import from bank_stress below.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bank_stress.models import (  # noqa: E402
    DatasetPaths,
    build_ml_dataset,
    train_baseline_logit,
    train_test_split_by_date,
)
from bank_stress.evaluation import (  # noqa: E402
    compute_confusion_counts,
    evaluate_existing_model,
    plot_roc_and_pr_curves,
)

# ---------------------------------------------------------------------------
# Project paths and global parameters
# ---------------------------------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"

for directory in (RESULTS_DIR, FIGURES_DIR, MODEL_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

STRESS_INDEX_PATH = PROCESSED_DIR / "stress_index.csv"
LABELS_MAIN_PATH = PROCESSED_DIR / "stress_labels.csv"
LABELS_Q90_PATH = PROCESSED_DIR / "stress_labels_q90.csv"
LABELS_COMBINED_PATH = PROCESSED_DIR / "stress_labels_combined.csv"

BASELINE_MODEL_PATH = MODEL_DIR / "baseline_logit_main.joblib"
BASELINE_METRICS_PATH = METRICS_DIR / "baseline_logit_main_metrics.json"

# ML hyperparameters
LAGS = 5
SPLIT_DATE = "2018-01-01"


# ---------------------------------------------------------------------------
# Helper for one label specification
# ---------------------------------------------------------------------------


def run_spec(label_path: Path, spec_name: str) -> Dict[str, float]:
    """
    Train and evaluate the baseline model for a given label CSV.

    Parameters
    ----------
    label_path:
        Path to the label CSV to use in this specification.
    spec_name:
        Human-readable name for this label specification
        (e.g. "q=0.95 (main)").

    Returns
    -------
    dict
        Dictionary with main performance metrics and confusion counts.
    """
    print(f"\n=== Evaluating spec: {spec_name} ===")
    print(f"Labels file: {label_path}")

    if not STRESS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {STRESS_INDEX_PATH}. "
            "Did you run the feature / index notebooks?"
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing {label_path}. " "Did you run the labeling notebook for this spec?"
        )

    paths = DatasetPaths(
        stress_index_path=STRESS_INDEX_PATH,
        labels_path=label_path,
    )

    # 1) Build supervised dataset
    features, labels = build_ml_dataset(paths=paths, lags=LAGS)
    print("  Dataset built.")
    print(f"  Features shape: {features.shape}")
    print(f"  Label share (stress days): {float(labels.mean()):.3f}")

    # 2) Time-based split
    features_train, features_test, labels_train, labels_test = train_test_split_by_date(
        features,
        labels,
        split_date=SPLIT_DATE,
    )
    print("  Time-based split done.")
    print(f"  Train rows: {len(features_train)}, test rows: {len(features_test)}")

    # 3) Train baseline logistic regression
    model = train_baseline_logit(
        features_train=features_train,
        labels_train=labels_train,
        c_param=1.0,
        solver="lbfgs",
        random_state=0,
    )
    print("  Baseline logistic model trained.")

    # 4) Evaluate (reuses models.evaluate_model internally)
    metrics, prob_pos = evaluate_existing_model(
        model,
        features_test,
        labels_test,
    )

    # Compute confusion counts using predicted labels
    y_pred = pd.Series(
        model.predict(features_test),
        index=labels_test.index,
        name="y_pred",
    )
    confusion = compute_confusion_counts(
        y_true=labels_test,
        y_pred=y_pred,
    )

    # 5) Plots (ROC + PR) saved to results/figures
    if prob_pos is not None:
        prefix = spec_name.replace(" ", "_").replace("=", "")
        roc_path, pr_path = plot_roc_and_pr_curves(
            y_test=labels_test,
            prob_pos=prob_pos,
            output_dir=FIGURES_DIR,
            prefix=f"baseline_{prefix}",
            show_plots=False,
        )
        print(f"  Saved ROC curve to: {roc_path}")
        print(f"  Saved PR  curve to: {pr_path}")
    else:
        print("  Warning: prob_pos is None → ROC/PR plots skipped.")

    # Return combined row for summary table
    row: Dict[str, float] = {
        "spec": spec_name,
        "roc_auc": float(metrics["roc_auc"]),
        "avg_precision": float(metrics["avg_precision"]),
        "accuracy": float(metrics["accuracy"]),
        "precision_1": float(metrics["precision_1"]),
        "recall_1": float(metrics["recall_1"]),
        "tp": float(confusion["tp"]),
        "fp": float(confusion["fp"]),
        "tn": float(confusion["tn"]),
        "fn": float(confusion["fn"]),
    }

    # For the main spec, also save the model + metrics to disk
    if label_path == LABELS_MAIN_PATH:
        joblib.dump(model, BASELINE_MODEL_PATH)
        with BASELINE_METRICS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print(f"  Saved main baseline model to: {BASELINE_MODEL_PATH}")
        print(f"  Saved main metrics JSON to: {BASELINE_METRICS_PATH}")

    return row


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run the full bank-stress detection pipeline.

    This function:
      * builds the dataset for each label specification,
      * trains and evaluates the baseline logistic model,
      * prints a summary table across specifications, and
      * prints a short final conclusion.
    """
    print("=" * 70)
    print("Bank Stress Detection: Baseline Logistic Regression")
    print("=" * 70)

    print("\nProject root:", PROJECT_ROOT)
    print("Processed data dir:", PROCESSED_DIR)
    print("Results dir:", RESULTS_DIR)

    # Always run the main label spec (q=0.95)
    rows: List[Dict[str, float]] = []
    rows.append(run_spec(LABELS_MAIN_PATH, "q=0.95 (main)"))

    # Optional robustness specs (only if the CSVs exist)
    if LABELS_Q90_PATH.exists():
        rows.append(run_spec(LABELS_Q90_PATH, "q=0.90 (robustness)"))
    else:
        print("\n[Info] q=0.90 label file not found → skipping robustness spec.")

    if LABELS_COMBINED_PATH.exists():
        rows.append(run_spec(LABELS_COMBINED_PATH, "combined labels"))
    else:
        print("[Info] combined-labels file not found → skipping that spec.")

    # ------------------------------------------------------------------
    # Summary table across specs
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(rows)
    print("\n=== Summary across label specifications ===")
    print(
        summary_df[["spec", "roc_auc", "avg_precision", "recall_1"]].to_string(
            index=False,
        )
    )

    # Pick the "best" spec by ROC AUC (ties broken by recall_1)
    summary_sorted = summary_df.sort_values(
        ["roc_auc", "recall_1"],
        ascending=[False, False],
    )
    best = summary_sorted.iloc[0]

    # ------------------------------------------------------------------
    # Final high-level conclusion (for your presentation / report)
    # ------------------------------------------------------------------
    print("\n=== Conclusion ===")
    spec = str(best["spec"])
    roc_auc = float(best["roc_auc"])
    avg_precision = float(best["avg_precision"])
    recall_1 = float(best["recall_1"])
    precision_1 = float(best["precision_1"])

    print(
        "Using the stress-index-based features and a logistic-regression "
        f"baseline, the best-performing label specification is '{spec}'."
    )
    print(
        f"On the held-out test set this model achieves "
        f"ROC AUC = {roc_auc:.3f}, average precision = {avg_precision:.3f}, "
        f"recall for stress days (class 1) = {recall_1:.3f}, "
        f"and precision for stress days = {precision_1:.3f}."
    )

    if recall_1 >= 0.8:
        print(
            "This suggests that the stress index provides a strong early-warning "
            "signal for bank stress days: most stress periods are detected, "
            "though some false alarms remain."
        )
    else:
        print(
            "This suggests that, in its current form, the stress index alone is "
            "not sufficient as a reliable early-warning signal for bank stress "
            "days; additional features or models may be needed."
        )

    print("\nDone. All outputs saved under 'results/'.")


if __name__ == "__main__":
    main()
