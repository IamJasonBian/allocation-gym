"""Load recent BTC OHLCV data for Monte Carlo calibration."""

import pandas as pd
from datetime import datetime, timedelta

from allocation_gym.credentials import get_alpaca_keys


def load_btc_ohlcv(
    symbol: str = "BTC/USD",
    calibration_days: int = 90,
) -> pd.DataFrame:
    """
    Fetch recent daily OHLCV bars from Alpaca.

    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Index: DatetimeIndex (tz-naive).
    """
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = get_alpaca_keys()
    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca credentials required. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY environment variables."
        )

    end = datetime.utcnow()
    start = end - timedelta(days=int(calibration_days * 1.5))

    client = CryptoHistoricalDataClient(api_key, secret_key)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=TimeFrame.Day,
    )
    bars = client.get_crypto_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df.tail(calibration_days)
