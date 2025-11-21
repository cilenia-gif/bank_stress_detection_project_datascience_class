#!/usr/bin/env python3
"""
tests/test_features.py

Unit tests for src/bank_stress/features.py.

These tests use a small synthetic price DataFrame so that:
- expected values are easy to compute by hand
- the behaviour of each feature function is clear.
"""

import numpy as np
import pandas as pd

from bank_stress.features import (
    average_crossbank_stat,
    compute_log_returns,
    rolling_pairwise_correlation,
    rolling_volatility,
)


def _make_prices() -> pd.DataFrame:
    """
    Create a small synthetic adjusted-close price DataFrame for two banks.

    - Both series are identical, so their returns should be perfectly
      correlated. This makes it easy to check that the correlation
      calculations behave as expected.
    """
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    prices_a = np.array([100.0, 101.0, 100.5, 101.5, 102.0, 101.0])
    prices_b = prices_a * 1.0  # identical series -> perfect correlation

    prices = pd.DataFrame({"A": prices_a, "B": prices_b}, index=dates)
    return prices


def test_compute_log_returns_shape_and_values() -> None:
    """
    compute_log_returns should:
    - return one fewer row than the price DataFrame
    - produce log returns consistent with the formula ln(P_t / P_{t-1}).
    """
    prices = _make_prices()
    returns = compute_log_returns(prices)

    # Number of rows should decrease by 1 because diff() drops the first row.
    assert returns.shape[0] == prices.shape[0] - 1

    # For the first return, check that value = log(101 / 100).
    expected_first = float(np.log(101.0 / 100.0))
    assert np.isclose(returns.iloc[0]["A"], expected_first)


def test_rolling_volatility_simple() -> None:
    """
    rolling_volatility should:
    - produce finite values once enough data is available
    - give identical volatility series for identical return series.
    """
    prices = _make_prices()
    returns = compute_log_returns(prices)

    # Use a small window so volatility is defined quickly.
    vol = rolling_volatility(returns, window=2)

    # At the second row, volatility should already be defined (not NaN).
    assert not np.isnan(vol.iloc[1]["A"])

    # Because A and B are identical, their volatility paths should match.
    vol_a = vol["A"].dropna().values
    vol_b = vol["B"].dropna().values
    assert np.allclose(vol_a, vol_b)


def test_rolling_pairwise_correlation_perfect_corr() -> None:
    """
    rolling_pairwise_correlation should return an average correlation
    of 1.0 when the two return series are identical.

    Because there are only two banks, the average pairwise correlation
    is just the correlation between A and B.
    """
    prices = _make_prices()
    returns = compute_log_returns(prices)

    # Use window=3 so that after a few rows we have enough data.
    avg_corr = rolling_pairwise_correlation(returns, window=3)

    # Drop initial NaNs (where window is not yet full).
    defined = avg_corr.dropna()

    # There should be at least one defined value...
    assert defined.shape[0] > 0
    # ...and all defined values should be very close to 1.0.
    assert np.allclose(defined.values, 1.0)


def test_average_crossbank_stat() -> None:
    """
    average_crossbank_stat should compute, for each date, the mean
    across banks of a per-ticker statistic (here: volatility).
    """
    prices = _make_prices()
    returns = compute_log_returns(prices)
    vol = rolling_volatility(returns, window=2)

    avg_vol = average_crossbank_stat(vol)

    # There should be some non-NaN average values.
    assert avg_vol.dropna().shape[0] > 0

    # For two banks A and B, the cross-bank mean is simply (A + B) / 2.
    manual_avg = (vol["A"] + vol["B"]) / 2.0
    mask = ~manual_avg.isna()  # only compare where both sides are defined

    assert np.allclose(avg_vol[mask], manual_avg[mask])
