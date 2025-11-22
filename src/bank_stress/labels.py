#!/usr/bin/env python3
"""
src/bank_stress/labels.py

Simple label construction utilities that convert a stress index into
binary stress/non-stress labels and optionally enforce crisis windows.
"""

from typing import Iterable, Optional, Tuple

import pandas as pd

DateRange = Tuple[str, str]


def threshold_labels(
    stress_index: pd.Series,
    quantile: float = 0.95,
) -> Tuple[pd.Series, float]:
    """
    Create binary labels based on a quantile threshold of the stress index.

    A date is labeled as a stress day (1) if:
        stress_index_t >= threshold
    where 'threshold' is the chosen quantile of the series.

    Any NaN values in the stress_index are mapped to label 0.

    Parameters
    ----------
    stress_index : pd.Series
        Continuous stress index (one value per date).

    quantile : float, default 0.95
        Quantile in (0, 1) used to define the high-stress threshold.

    Returns
    -------
    labels : pd.Series
        Binary series (0/1) named 'stress_label', indexed like stress_index.

    threshold_value : float
        The numeric threshold corresponding to the chosen quantile.
    """
    if not isinstance(stress_index, pd.Series):
        raise TypeError("stress_index must be a pandas Series")
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0,1)")

    threshold_value = float(stress_index.quantile(quantile))
    labels = (stress_index >= threshold_value).fillna(0).astype(int)
    labels.name = "stress_label"
    return labels, threshold_value


def event_window_labels(
    index: Iterable[pd.Timestamp],
    event_date: pd.Timestamp,
    pre: int = 0,
    post: int = 0,
) -> pd.Series:
    """
    Create labels for a contiguous window around a specific event date.

    The function marks dates in [event_date - pre, event_date + post]
    as 1 and all other dates as 0, using the provided index.

    If event_date is not exactly in the index, the nearest date
    in the index is used as the center of the window.

    Parameters
    ----------
    index : iterable of pd.Timestamp
        Sequence of dates (typically the index of the stress index series).

    event_date : pd.Timestamp
        Central event date around which the window is defined.

    pre : int, default 0
        Number of days before event_date to include.

    post : int, default 0
        Number of days after event_date to include.

    Returns
    -------
    pd.Series
        Binary series indexed by 'index' with 1 for dates in the
        event window and 0 otherwise.
    """
    idx = pd.DatetimeIndex(index)
    if idx.empty:
        return pd.Series(dtype=int)

    # Locate the position of the event date (or the nearest date)
    try:
        event_pos = idx.get_loc(pd.Timestamp(event_date))
    except KeyError:
        event_pos = idx.get_indexer(
            [pd.Timestamp(event_date)],
            method="nearest",
        )[0]

    start_pos = max(0, int(event_pos) - int(pre))
    end_pos = min(len(idx) - 1, int(event_pos) + int(post))

    window_labels = pd.Series(0, index=idx, dtype=int)
    window_labels.iloc[start_pos : end_pos + 1] = 1
    return window_labels


def apply_crisis_overrides(
    labels: pd.Series,
    crisis_windows: Optional[Iterable[DateRange]] = None,
) -> pd.Series:
    """
    Force labels to 1 inside specified crisis windows.

    Each crisis window is a (start_str, end_str) pair that is
    converted to timestamps, and all dates between start and end
    (inclusive) are set to 1.

    Parameters
    ----------
    labels : pd.Series
        Initial label series (0/1) with a DatetimeIndex.

    crisis_windows : iterable of (start_str, end_str), optional
        Date ranges specified as strings parsable by pandas,
        e.g. ("2008-09-01", "2009-03-31").

    Returns
    -------
    pd.Series
        Modified labels series where dates inside crisis windows
        are forced to 1.
    """
    overridden_labels = labels.copy()

    if crisis_windows is None:
        return overridden_labels

    idx = pd.DatetimeIndex(overridden_labels.index)
    for start_str, end_str in crisis_windows:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        mask = (idx >= start) & (idx <= end)
        overridden_labels.loc[mask] = 1

    return overridden_labels


def construct_labels(
    stress_index: pd.Series,
    method: str = "threshold",
    quantile: float = 0.95,
    event_dates: Optional[Iterable[pd.Timestamp]] = None,
    event_window: Tuple[int, int] = (0, 0),
    crisis_windows: Optional[Iterable[DateRange]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Construct a labeled DataFrame from a stress index using different schemes.

    The function can combine:
    - threshold-based labeling (high-quantile stress index)
    - event-window labeling around specific dates
    - crisis-window overrides

    Parameters
    ----------
    stress_index : pd.Series
        Continuous stress index time series.

    method : {"threshold", "event", "combined"}, default "threshold"
        Labeling scheme:
        - "threshold": label days above the quantile threshold.
        - "event":    label only days in event windows.
        - "combined": label days that are either high-threshold
                      or lie in event windows (union of both).

    quantile : float, default 0.95
        Quantile used for threshold-based labeling.

    event_dates : iterable of pd.Timestamp, optional
        Central dates for event windows (used if method involves "event").

    event_window : (int, int), default (0, 0)
        Tuple (pre, post) specifying how many days before and after
        each event_date to include in the event window.

    crisis_windows : iterable of (start_str, end_str), optional
        Calendar windows used to force labels to 1 regardless of
        threshold or event logic.

    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame indexed by date with columns:
            - "metric_value" : the original stress index values
            - "threshold"    : scalar threshold (NaN if not applicable)
            - "stress_label" : final binary label (0/1)
            - "reason"       : text description of label source
                               ("threshold", "event", "override", "none")

    threshold_value : float
        The threshold used for threshold-based labeling
        (NaN if no threshold was computed).
    """
    if not isinstance(stress_index, pd.Series):
        raise TypeError("stress_index must be a pandas Series")

    labels_df = pd.DataFrame({"metric_value": stress_index.copy()})
    labels_df.index.name = "date"

    threshold_value = float("nan")
    combined_labels = pd.Series(0, index=labels_df.index, dtype=int)

    # Threshold-based labels (if used)
    if method in ("threshold", "combined"):
        threshold_labels_series, threshold_value = threshold_labels(
            stress_index,
            quantile=quantile,
        )
        combined_labels = combined_labels | threshold_labels_series

    # Event-based labels (if used)
    if method in ("event", "combined") and event_dates is not None:
        event_mask = pd.Series(0, index=labels_df.index, dtype=int)
        pre, post = event_window
        for event_date in event_dates:
            event_mask = event_mask | event_window_labels(
                labels_df.index,
                pd.Timestamp(event_date),
                pre=pre,
                post=post,
            )
        combined_labels = combined_labels | event_mask

    # Crisis windows override (if provided)
    if crisis_windows is not None:
        combined_labels = apply_crisis_overrides(
            combined_labels,
            crisis_windows,
        )

    labels_df["threshold"] = threshold_value
    labels_df["stress_label"] = combined_labels.astype(int)

    # Initialize reasons as "none"
    labels_df["reason"] = "none"

    # Mark threshold-driven labels
    if method in ("threshold", "combined"):
        labels_df.loc[
            (labels_df["stress_label"] == 1)
            & (labels_df["metric_value"] >= threshold_value),
            "reason",
        ] = "threshold"

    # Mark event-driven labels (only on exact event dates, if still "none")
    if method in ("event", "combined") and event_dates is not None:
        event_dates_index = pd.DatetimeIndex(
            [pd.Timestamp(d) for d in (event_dates or [])],
        )
        mask_event_date = labels_df.index.isin(event_dates_index)
        labels_df.loc[
            (labels_df["stress_label"] == 1)
            & (labels_df["reason"] == "none")
            & mask_event_date,
            "reason",
        ] = "event"

    # Mark crisis-window overrides
    if crisis_windows:
        for start_str, end_str in crisis_windows:
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            crisis_mask = (labels_df.index >= start) & (labels_df.index <= end)
            labels_df.loc[crisis_mask, "reason"] = "override"

    return labels_df, threshold_value
