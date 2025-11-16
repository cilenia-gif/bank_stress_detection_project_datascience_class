#!/usr/bin/env python3
"""
src/bank_stress/data_utils.py

Purpose:
- Download adjusted-close prices for a list of tickers (yfinance).
- Normalize the various shapes that yfinance.download can return so callers
  always get a pandas.DataFrame with columns = tickers and a DatetimeIndex.
- Save and load raw CSVs under data/raw/.

This file contains inline comments explaining what each function and key block
does.
"""
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf

# Central place to store raw downloaded CSVs.
# Using Path makes directory handling easy.
RAW_DIR = Path("data/raw")


def _to_list(tickers: Iterable[str]) -> List[str]:
    """
    Helper to accept either:
    - a string (e.g. "JPM" or "JPM,BAC")
    - an iterable/list of tickers (["JPM", "BAC"])
    and always return a Python list of ticker symbols.

    This makes the public API flexible for callers.
    """
    # If a single string is passed, split on commas to allow "JPM,BAC" form.
    if isinstance(tickers, str):
        return [t.strip() for t in tickers.split(",") if t.strip()]
    # Otherwise cast the iterable to a list.
    return list(tickers)


def download_prices(
    tickers,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted-close prices for a list of tickers using yfinance.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - index = DatetimeIndex (business/trading days)
        - columns = requested ticker symbols (if available)

    Why the defensive logic
    -----------------------
    yfinance may return different shapes:

    * pd.Series (single series)
    * pd.DataFrame with single-level columns
    * pd.DataFrame with MultiIndex columns (ticker, field),
      e.g. ('JPM', 'Adj Close')

    This function inspects the returned object and extracts the adjusted-close
    prices in multiple ways so downstream code always receives a consistent
    DataFrame.
    """
    tickers_list = _to_list(tickers)
    if not tickers_list:
        raise ValueError("tickers list is empty")

    # Request auto_adjust so prices are adjusted for splits/dividends when
    # possible.
    data = yf.download(
        tickers_list,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    # If yfinance returned a Series (single column), convert to DataFrame and
    # name the column.
    if isinstance(data, pd.Series):
        # Name the single column with the requested ticker symbol.
        return pd.DataFrame({tickers_list[0]: data})

    # If columns are a MultiIndex (most common when multiple tickers are
    # requested), the second level often contains fields like 'Adj Close',
    # 'Close', 'Open', etc.
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer 'Adj Close' (adjusted close) when present.
        if "Adj Close" in data.columns.get_level_values(1):
            # xs extracts the sub-frame for the 'Adj Close' level across
            # tickers.
            adj = data.xs("Adj Close", axis=1, level=1)
            # Reindex columns to the requested tickers and preserve order.
            adj = adj.reindex(columns=[t for t in tickers_list if t in adj.columns])
            return adj

        # If 'Adj Close' not present, fall back to 'Close'.
        if "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1)
            close = close.reindex(
                columns=[t for t in tickers_list if t in close.columns]
            )
            return close

        # Final MultiIndex fallback: for each ticker take the first numeric
        # subcolumn available.
        out = pd.DataFrame(index=data.index)
        for ticker in tickers_list:
            if ticker in data.columns:
                part = data[ticker]
                if isinstance(part, pd.Series):
                    out[ticker] = part
                else:
                    # choose first numeric column if present
                    numcols = part.select_dtypes(include="number").columns
                    if len(numcols) > 0:
                        out[ticker] = part[numcols[0]]
        return out

    # If we get here, data.columns is a simple Index (single-level columns)
    cols = list(data.columns)

    # Case: columns already contain the requested tickers -> select and return
    # them.
    if set(tickers_list).issubset(set(cols)):
        return data.loc[:, tickers_list]

    # If there is only one column, assume it's the requested ticker's price and
    # name it.
    if len(cols) == 1:
        return pd.DataFrame({tickers_list[0]: data.iloc[:, 0]})

    # If 'Adj Close' or 'Close' exists as a column name in a single-level
    # DataFrame, use it.
    if "Adj Close" in cols:
        return data[["Adj Close"]].rename(columns={"Adj Close": tickers_list[0]})
    if "Close" in cols:
        return data[["Close"]].rename(columns={"Close": tickers_list[0]})

    # Lastly, try to align any available columns that match requested tickers.
    available = [c for c in cols if c in tickers_list]
    if available:
        return data.loc[:, available]

    # If nothing matches, return the DataFrame as-is (caller will need to
    # handle it).
    return data


def save_raw_prices(df: pd.DataFrame, filename: str = "sample_prices.csv") -> None:
    """
    Save a DataFrame of raw prices to data/raw/.

    - Ensures the directory exists.
    - Writes CSV with the DataFrame index as first column (dates).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / filename)


def load_raw_prices(filename: str = "sample_prices.csv") -> pd.DataFrame:
    """
    Load raw price CSV from data/raw/ and return a DataFrame with a
    DatetimeIndex.
    """
    return pd.read_csv(RAW_DIR / filename, index_col=0, parse_dates=True)
