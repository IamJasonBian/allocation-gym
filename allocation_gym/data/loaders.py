"""
Shared OHLCV data loaders for Alpaca and Yahoo Finance.

Used by both the backtester (runner.py) and the simulation module
(simulation/data.py) to avoid duplicating data-fetch logic.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Crypto symbol set (Alpaca uses the crypto endpoint for these)
# ──────────────────────────────────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD", "AVAX/USD", "SOL/USD"}


def is_crypto(symbol: str) -> bool:
    """Return True if *symbol* should be fetched via the Alpaca crypto endpoint."""
    return symbol.upper() in CRYPTO_SYMBOLS or "/USD" in symbol.upper()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers — return a normalised DataFrame (Open/High/Low/Close/Volume)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_alpaca_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten a multi-index Alpaca DataFrame and normalise column names."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _load_alpaca_crypto_df(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """Fetch daily crypto OHLCV from Alpaca and return a normalised DataFrame."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = CryptoHistoricalDataClient(api_key, secret_key)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=TimeFrame.Day,
    )
    bars = client.get_crypto_bars(request)
    return _normalise_alpaca_df(bars.df, symbol)


def _load_alpaca_stock_df(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """Fetch daily stock OHLCV from Alpaca and return a normalised DataFrame."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=TimeFrame.Day,
    )
    bars = client.get_stock_bars(request)
    return _normalise_alpaca_df(bars.df, symbol)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_alpaca_data(
    symbol: str,
    start: str,
    end: str,
    api_key: str,
    secret_key: str,
    as_feed: bool = True,
):
    """
    Fetch daily OHLCV bars from Alpaca for *symbol* between *start* and *end*.

    Parameters
    ----------
    symbol:
        Ticker symbol.  Crypto symbols (e.g. ``"BTC/USD"``) are fetched via
        the crypto endpoint; everything else via the stock endpoint.
    start, end:
        Date strings in ``"YYYY-MM-DD"`` format.
    api_key, secret_key:
        Alpaca API credentials.
    as_feed:
        If ``True`` (default) return a ``backtrader.feeds.PandasData`` feed
        ready to be added to a Cerebro instance.
        If ``False`` return the raw ``pd.DataFrame``.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if is_crypto(symbol):
        df = _load_alpaca_crypto_df(symbol, start_dt, end_dt, api_key, secret_key)
    else:
        df = _load_alpaca_stock_df(symbol, start_dt, end_dt, api_key, secret_key)

    if as_feed:
        import backtrader as bt
        return bt.feeds.PandasData(dataname=df)
    return df


def load_alpaca_ohlcv(
    symbol: str = "BTC/USD",
    calibration_days: int = 90,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch recent daily OHLCV bars from Alpaca for calibration purposes.

    Parameters
    ----------
    symbol:
        Ticker symbol (crypto or stock).
    calibration_days:
        Number of trading days to return.  The request window is padded by
        50 % to ensure enough bars survive any weekend/holiday gaps.
    api_key, secret_key:
        Alpaca API credentials.  If not supplied, resolved via
        :func:`allocation_gym.credentials.get_alpaca_keys`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Open, High, Low, Close, Volume`` and a
        tz-naive DatetimeIndex.  Exactly *calibration_days* rows are returned
        (tail of the fetched window).
    """
    if api_key is None or secret_key is None:
        from allocation_gym.credentials import get_alpaca_keys
        api_key, secret_key = get_alpaca_keys()

    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca credentials required. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY environment variables."
        )

    end = datetime.utcnow()
    start = end - timedelta(days=int(calibration_days * 1.5))

    df = _load_alpaca_crypto_df(symbol, start, end, api_key, secret_key) \
        if is_crypto(symbol) \
        else _load_alpaca_stock_df(symbol, start, end, api_key, secret_key)

    return df.tail(calibration_days)


def load_yfinance_data(symbol: str, start: str, end: str, as_feed: bool = True):
    """
    Fetch daily OHLCV data from Yahoo Finance for *symbol*.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker symbol.
    start, end:
        Date strings in ``"YYYY-MM-DD"`` format.
    as_feed:
        If ``True`` (default) return a ``backtrader.feeds.PandasData`` feed.
        If ``False`` return the raw ``pd.DataFrame``.
    """
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} from Yahoo Finance")
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)

    if as_feed:
        import backtrader as bt
        return bt.feeds.PandasData(dataname=df)
    return df


def load_ohlcv(
    symbol: str,
    start: str,
    end: str,
    source: str = "yfinance",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    as_feed: bool = True,
):
    """
    Unified data loader — dispatches to Alpaca or Yahoo Finance based on
    *source*.

    Parameters
    ----------
    symbol:
        Ticker symbol.
    start, end:
        Date strings in ``"YYYY-MM-DD"`` format.
    source:
        ``"yfinance"`` (default) or ``"alpaca"``.
    api_key, secret_key:
        Alpaca credentials (only required when *source* is ``"alpaca"``).
    as_feed:
        If ``True`` return a Backtrader feed; if ``False`` return a DataFrame.
    """
    if source == "alpaca":
        if api_key is None or secret_key is None:
            from allocation_gym.credentials import get_alpaca_keys
            api_key, secret_key = get_alpaca_keys()
        return load_alpaca_data(symbol, start, end, api_key, secret_key, as_feed=as_feed)
    return load_yfinance_data(symbol, start, end, as_feed=as_feed)
