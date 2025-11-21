#!/usr/bin/env python3
"""
src/bank_stress/features.py

Feature engineering utilities for the bank stress project.

Provides:
- compute_log_returns:      compute daily log returns from price data
- rolling_volatility:       per-ticker rolling volatility (standard deviation)
- rolling_pairwise_correlation: average off-diagonal rolling correlation
- average_crossbank_stat:   cross-sectional average of a per-ticker statistic
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from an adjusted-close price DataFrame.

    Intuition
    ---------
    For each ticker and each day t, we compute:

        log_return_t = ln(price_t) - ln(price_{t-1})

    This is equivalent to ln(price_t / price_{t-1}) and is commonly used
    in finance because log returns add nicely over time.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with:
        - index   = dates (DatetimeIndex)
        - columns = ticker symbols
        Values are assumed to be (adjusted) prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns with the same columns.
        The first row (and any row with all-NaN values) is dropped.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    # Take natural log of prices, then first difference over time.
    # np.errstate is used to silence warnings if any price is zero/negative.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.log(prices).diff()

    # Replace any infinities that may appear with NaN.
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)

    # Drop rows where *all* tickers are NaN (typically the first row).
    return log_returns.dropna(how="all")


def rolling_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility (sample standard deviation) per ticker.

    Intuition
    ---------
    Volatility is often estimated as the standard deviation of returns
    over a recent window (e.g. last 20 trading days).

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of log returns (output of compute_log_returns).
        - index   = dates
        - columns = ticker symbols

    window : int, default 20
        Length of the rolling window in periods (days).

    Returns
    -------
    pd.DataFrame
        DataFrame of rolling standard deviations for each ticker.
        NaN values appear at the beginning where there is not enough data.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if window < 1:
        raise ValueError("window must be >= 1")

    # pandas uses ddof=1 by default for std() → sample standard deviation.
    # min_periods=2 ensures at least 2 data points in each window.
    return returns.rolling(window=window, min_periods=2).std()


def rolling_pairwise_correlation(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute a time series of average off-diagonal pairwise correlations
    using a trailing window.

    Intuition
    ---------
    For each date t:
      1. Look back over the last `window` days of returns.
      2. Compute the correlation matrix between tickers in that window.
      3. Take the average of all *off-diagonal* entries in that matrix
         (i.e. all pairwise correlations between different banks).

    This gives a single "system-level" number per day that captures how
    tightly the banks tend to move together.

    If fewer than two tickers have data in the window, the result is NaN.

    Parameters
    ----------
    returns : pd.DataFrame
        Log returns with datetime index and columns = tickers.

    window : int
        Rolling window size in days.

    Returns
    -------
    pd.Series
        Series indexed by date, where each value is the average
        off-diagonal correlation for that date.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if window < 1:
        raise ValueError("window must be >= 1")

    dates = returns.index
    # This will hold the resulting average correlation values.
    avg_corr_series = pd.Series(index=dates, dtype=float)

    # Iterate through all dates and compute correlation on the trailing window.
    for i, date in enumerate(dates):
        # Determine start index of the window [i - window + 1, ..., i].
        start_idx = max(0, i - window + 1)
        window_returns = returns.iloc[start_idx : i + 1]

        # Drop tickers that have only NaN values in this window.
        window_returns = window_returns.dropna(axis=1, how="all")
        if window_returns.shape[1] < 2:
            # Not enough valid tickers to compute correlations.
            avg_corr_series.iloc[i] = np.nan
            continue

        # Correlation matrix between tickers over this window.
        corr_matrix = window_returns.corr()

        # Create a mask that selects only off-diagonal entries (i != j).
        off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag_values = corr_matrix.values[off_diag_mask]

        # If no valid off-diagonal values, set NaN; otherwise store the mean.
        if off_diag_values.size == 0 or np.all(pd.isna(off_diag_values)):
            avg_corr_series.iloc[i] = np.nan
        else:
            avg_corr_series.iloc[i] = float(np.nanmean(off_diag_values))

    return avg_corr_series


def average_crossbank_stat(stat_df: pd.DataFrame) -> pd.Series:
    """
    Compute the cross-sectional mean of a per-ticker statistic.

    Intuition
    ---------
    Given a DataFrame where each column is a bank and each row is a date,
    this function computes, for each date, the average across all banks.
    This can be used to obtain, for example:
      - average volatility across banks
      - average return across banks
      - average any-other-metric across banks

    Parameters
    ----------
    stat_df : pd.DataFrame
        DataFrame indexed by date, columns = tickers,
        values = some per-ticker statistic (e.g. volatility).

    Returns
    -------
    pd.Series
        One value per date: the mean across tickers (ignoring NaNs).
    """
    if not isinstance(stat_df, pd.DataFrame):
        raise TypeError("stat_df must be a pandas DataFrame")

    # Mean across columns (tickers) for each row (date). NaNs are skipped.
    return stat_df.mean(axis=1, skipna=True)
