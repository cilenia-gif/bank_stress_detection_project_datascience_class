#!/usr/bin/env python3
"""
tests/test_labels_minimal.py

Minimal tests for src/bank_stress/labels.py.

These tests focus on:
- threshold_labels: selecting high-stress days via a quantile threshold
- event_window_labels: labeling windows around specific event dates
- apply_crisis_overrides: forcing crisis windows to be stress
- construct_labels: combining threshold, event, and crisis logic
"""

import numpy as np
import pandas as pd

from bank_stress.labels import (
    apply_crisis_overrides,
    construct_labels,
    event_window_labels,
    threshold_labels,
)


def _series_0_to_9() -> pd.Series:
    """
    Create a simple time series [0, 1, 2, ..., 9] with daily dates.

    This makes it easy to reason about which values should be labeled
    as "high" when using the 0.95 quantile threshold.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.Series(np.arange(10.0), index=dates)


def test_threshold_event_and_overrides_minimal() -> None:
    """
    Minimal integrated test for threshold-, event-, and crisis-based labeling.

    Steps:
    1. threshold_labels:
       - With quantile=0.95 on the series [0..9], only the largest value
         (9.0) should be labeled as stress.
    2. event_window_labels:
       - Label a small window around a central event date.
       - Check that events outside the index fall back to the nearest date.
    3. construct_labels with method="combined":
       - Use threshold + event windows + crisis overrides together.
    4. apply_crisis_overrides:
       - Directly ensure labels are set to 1 inside specified windows.
    """
    stress_series = _series_0_to_9()
    dates = stress_series.index

    # ------------------------------------------------------------------
    # 1) Threshold-based labels: with quantile=0.95 on values 0..9,
    #    only the last value (9.0) should be labeled as stress.
    # ------------------------------------------------------------------
    labels_thresh, threshold_value = threshold_labels(
        stress_series,
        quantile=0.95,
    )
    assert isinstance(labels_thresh, pd.Series)
    assert labels_thresh.sum() == 1
    assert labels_thresh.iloc[-1] == 1

    assert isinstance(threshold_value, float)
    assert np.isfinite(threshold_value)

    # ------------------------------------------------------------------
    # 2) Event-window labels: mark a window around a central event date.
    # ------------------------------------------------------------------
    event_date = pd.Timestamp("2020-01-04")
    event_window_series = event_window_labels(
        dates,
        event_date,
        pre=1,
        post=1,
    )
    assert isinstance(event_window_series, pd.Series)
    # With pre=1, post=1, we expect 3 days labeled as 1.
    assert event_window_series.sum() == 3

    # Event before the index: should fall back to the nearest index date.
    early_event_date = pd.Timestamp("2019-12-31")
    early_event_labels = event_window_labels(
        dates,
        early_event_date,
        pre=1,
        post=0,
    )
    # Nearest date is the first date in the index, which should be labeled.
    assert early_event_labels.iloc[0] == 1

    # ------------------------------------------------------------------
    # 3) construct_labels with method="combined":
    #    Combine threshold-based labels, event windows, and crisis windows.
    # ------------------------------------------------------------------
    labels_df, threshold_value_2 = construct_labels(
        stress_series,
        method="combined",
        quantile=0.95,
        event_dates=[pd.Timestamp("2020-01-05")],
        event_window=(1, 1),
        crisis_windows=[("2020-01-02", "2020-01-03")],
    )
    assert "stress_label" in labels_df.columns
    assert "reason" in labels_df.columns
    assert "threshold" in labels_df.columns

    # In the crisis window [2020-01-02, 2020-01-03], all dates should
    # be labeled as stress (1), regardless of threshold or event logic.
    crisis_mask = (labels_df.index >= pd.Timestamp("2020-01-02")) & (
        labels_df.index <= pd.Timestamp("2020-01-03")
    )
    assert labels_df.loc[crisis_mask, "stress_label"].sum() == crisis_mask.sum()

    # ------------------------------------------------------------------
    # 4) apply_crisis_overrides directly:
    #    Starting from an all-zero label series, we expect the specified
    #    crisis window to be fully set to 1.
    # ------------------------------------------------------------------
    base_labels = pd.Series(0, index=dates, dtype=int)
    overridden_labels = apply_crisis_overrides(
        base_labels,
        crisis_windows=[("2020-01-02", "2020-01-03")],
    )
    assert overridden_labels.loc["2020-01-02"] == 1
    assert overridden_labels.loc["2020-01-03"] == 1
