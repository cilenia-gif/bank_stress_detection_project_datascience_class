#!/usr/bin/env python3
"""
tests/test_stress_simple.py

Minimal tests for src/bank_stress/stress.py.

Focus:
- Check that build_stress_components returns the expected columns
  (avg_vol, avg_corr) with sensible values.
- Check that build_stress_index returns a DataFrame with the expected
  columns, aligned index, and a stress_index that is the linear
  combination of z_vol and z_corr.
"""

import numpy as np
import pandas as pd

from bank_stress.index import build_stress_components, build_stress_index


def _sample_returns() -> pd.DataFrame:
    """
    Create a small synthetic returns DataFrame for two banks.

    The two series are positively correlated but not identical, so that
    both volatility and correlation are non-trivial and can be used to
    test the stress index construction.
    """
    dates = pd.date_range("2020-01-01", periods=8, freq="D")

    # Simple pattern with small positive and negative returns
    returns_a = np.array([0.00, 0.01, -0.01, 0.02, 0.00, 0.01, -0.02, 0.00])
    # Correlated series with slightly different scale and offset
    returns_b = returns_a * 0.8 + 0.002

    return pd.DataFrame({"A": returns_a, "B": returns_b}, index=dates)


def test_components_have_expected_columns_and_some_values() -> None:
    """
    build_stress_components should return a DataFrame with:

    - columns "avg_vol" and "avg_corr"
    - a DatetimeIndex
    - some non-NaN values once the rolling window has enough data
    """
    returns = _sample_returns()
    components_df = build_stress_components(returns, window=3)

    # Check that the core columns are present
    assert "avg_vol" in components_df.columns
    assert "avg_corr" in components_df.columns

    # Index should be a DatetimeIndex (matching the time-series nature)
    assert isinstance(components_df.index, pd.DatetimeIndex)

    # After a small rolling window, we expect at least a few valid values
    assert components_df["avg_vol"].dropna().size > 0
    assert components_df["avg_corr"].dropna().size > 0


def test_build_stress_index_structure_and_linear_combination() -> None:
    """
    build_stress_index should:

    - return a DataFrame with the expected columns,
    - preserve the same index as the input returns,
    - produce at least some non-NaN stress_index values,
    - define stress_index as the linear combination:
          stress_index = weight_vol * z_vol + weight_corr * z_corr.
    """
    returns = _sample_returns()
    stress_df = build_stress_index(
        returns,
        window=3,
        weight_vol=0.5,
        weight_corr=0.5,
        smooth_span=3,
    )

    # Check presence of core columns produced by build_stress_index
    for col in ("avg_vol", "avg_corr", "z_vol", "z_corr", "stress_index"):
        assert col in stress_df.columns

    # If smoothing is requested, the smoothed column should be present
    assert "stress_index_smooth" in stress_df.columns

    # The index of the output should match the index of the input returns
    assert stress_df.index.equals(returns.index)

    # There should be at least some non-NaN stress_index values
    assert stress_df["stress_index"].dropna().size > 0

    # On at least one row where everything is defined, check the linear combo:
    #   stress_index_t = 0.5 * z_vol_t + 0.5 * z_corr_t
    row = stress_df.dropna(subset=["z_vol", "z_corr", "stress_index"]).iloc[0]
    expected = 0.5 * row["z_vol"] + 0.5 * row["z_corr"]

    # Use a small tolerance for floating-point comparison
    assert abs(float(row["stress_index"]) - float(expected)) <= 1e-6
