#!/usr/bin/env python3
"""
src/bank_stress/index.py

Compute a simple market-based stress index from bank equity returns.

Overview
--------
This module constructs two system-level indicators from log-returns:

1. avg_vol  : the average rolling volatility across all banks
2. avg_corr : the average pairwise correlation across all banks

These two series capture different aspects of market stress:
- volatility describes individual uncertainty
- correlation captures systemic co-movement

Both series are transformed into z-scores so that they have
comparable scales. They are then combined into a single stress
index using user-defined weights. An optional smoothing step can
apply an exponential moving average to the final index.

The module relies only on functions from src/bank_stress/features.py.
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .features import (
    average_crossbank_stat,
    rolling_pairwise_correlation,
    rolling_volatility,
)


def _zscore(
    series: pd.Series,
    baseline_idx: Optional[Sequence[pd.Timestamp]] = None,
) -> pd.Series:
    """
    Compute a z-score for each value in a time series.

    Parameters
    ----------
    series : pd.Series
        Input time series (typically avg_vol or avg_corr).

    baseline_idx : optional sequence of timestamps
        If provided, the mean and standard deviation are computed
        only over the values belonging to these baseline dates.
        If None, the entire non-NaN series is used.

    Notes
    -----
    - NaN values are preserved.
    - If the baseline has zero variance (std = 0), the function
      returns 0 for all non-NaN entries (to avoid division by zero).

    Returns
    -------
    pd.Series
        Z-scored time series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if baseline_idx is None:
        baseline_values = series.dropna().values
    else:
        baseline_mask = series.index.isin(baseline_idx)
        baseline_values = series.loc[baseline_mask].dropna().values

    if baseline_values.size == 0:
        return pd.Series(np.nan, index=series.index)

    mean_value = float(np.mean(baseline_values))
    std_value = float(np.std(baseline_values, ddof=0))

    # If no variation in baseline, return zeros (no scaled differences)
    if std_value == 0 or np.isnan(std_value):
        return (series - mean_value) * 0.0

    return (series - mean_value) / std_value


def build_stress_components(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute the two core components of the stress index.

    Components
    ----------
    avg_vol : pd.Series
        Cross-sectional average of per-bank rolling volatility
        computed over the specified window.

    avg_corr : pd.Series
        Rolling average of all off-diagonal pairwise correlations
        among banks over the same window.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of log returns (index = dates, columns = banks).
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        A two-column DataFrame with:
            - "avg_vol"
            - "avg_corr"
        indexed by date.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if window < 1:
        raise ValueError("window must be >= 1")

    # Rolling volatility per bank → average across banks
    rolling_vol_df = rolling_volatility(returns, window=window)
    avg_vol = average_crossbank_stat(rolling_vol_df)

    # Rolling average pairwise correlation
    avg_corr = rolling_pairwise_correlation(returns, window=window)

    components_df = pd.concat(
        [avg_vol.rename("avg_vol"), avg_corr.rename("avg_corr")],
        axis=1,
    )
    return components_df


def build_stress_index(
    returns: pd.DataFrame,
    window: int = 20,
    weight_vol: float = 0.5,
    weight_corr: float = 0.5,
    baseline_idx: Optional[Sequence[pd.Timestamp]] = None,
    smooth_span: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build the full stress index and return all intermediate components.

    Output columns
    --------------
    - avg_vol              : average rolling volatility
    - avg_corr             : average rolling pairwise correlation
    - z_vol                : z-scored volatility component
    - z_corr               : z-scored correlation component
    - stress_index         : weighted combination of z_vol and z_corr
    - stress_index_smooth  : (optional) EWMA-smoothed version

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns (index = dates, columns = tickers).
    window : int, default 20
        Rolling window for vol/corr calculation.
    weight_vol : float, default 0.5
        Weight applied to the volatility component.
    weight_corr : float, default 0.5
        Weight applied to the correlation component.
    baseline_idx : sequence of timestamps, optional
        If provided, z-scoring uses only these dates as baseline.
    smooth_span : int, optional
        If provided and > 1, compute an exponentially weighted
        moving average smoothing of the final stress index.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all components and the final stress index,
        aligned to the full returns index.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not (0.0 <= weight_vol <= 1.0 and 0.0 <= weight_corr <= 1.0):
        raise ValueError("weights must be between 0 and 1")
    if window < 1:
        raise ValueError("window must be >= 1")

    # Step 1: compute avg_vol and avg_corr
    components_df = build_stress_components(returns, window=window)

    # Step 2: z-score each component
    components_df["z_vol"] = _zscore(
        components_df["avg_vol"],
        baseline_idx=baseline_idx,
    )
    components_df["z_corr"] = _zscore(
        components_df["avg_corr"],
        baseline_idx=baseline_idx,
    )

    # Step 3: combine components linearly
    components_df["stress_index"] = (
        weight_vol * components_df["z_vol"] + weight_corr * components_df["z_corr"]
    )

    # Step 4 (optional): smooth the stress index
    if smooth_span is not None and smooth_span > 1:
        components_df["stress_index_smooth"] = (
            components_df["stress_index"]
            .ewm(span=int(smooth_span), adjust=False)
            .mean()
        )

    # Re-align to full returns index (ensures consistency downstream)
    components_df = components_df.reindex(returns.index)
    return components_df
