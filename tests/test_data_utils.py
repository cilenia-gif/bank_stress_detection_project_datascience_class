"""
Unit tests for src.bank_stress.data_utils (uses monkeypatch to avoid network).
"""

import numpy as np
import pandas as pd
import yfinance as yf

from bank_stress.data_utils import download_prices


def _fake_yf_download_single(
    tickers,
    start=None,
    end=None,
    progress=False,
    auto_adjust=True,
    **kwargs,
):
    """
    Fake yfinance.download replacement for tests.

    Accepts **kwargs to avoid errors if callers pass extra keyword
    arguments (e.g., progress, auto_adjust). Produces a DataFrame
    with a MultiIndex column structure: (ticker, "Adj Close").
    """
    dates = pd.date_range(start, end, freq="B")

    col_index = pd.MultiIndex.from_product([[tickers[0]], ["Adj Close"]])

    df = pd.DataFrame(
        np.linspace(100.0, 110.0, len(dates)),
        index=dates,
        columns=col_index,
    )
    return df


def test_download_prices_single_ticker(monkeypatch):
    """
    Test extraction of 'Adj Close' when yfinance returns MultiIndex columns.
    """
    # Patch yfinance.download so no real network call happens.
    monkeypatch.setattr(yf, "download", _fake_yf_download_single)

    df = download_prices(["JPM"], start="2020-01-01", end="2020-01-31")

    assert not df.empty
    assert "JPM" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df["JPM"].min() >= 100.0
    assert df["JPM"].max() <= 110.0
