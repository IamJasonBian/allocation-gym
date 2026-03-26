"""Load recent OHLCV data for Monte Carlo calibration."""

from __future__ import annotations

from allocation_gym.data.loaders import load_alpaca_ohlcv


def load_btc_ohlcv(
    symbol: str = "BTC/USD",
    calibration_days: int = 90,
) -> "pd.DataFrame":
    """
    Fetch recent daily OHLCV bars from Alpaca.

    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Index: DatetimeIndex (tz-naive).

    .. deprecated::
        Import :func:`allocation_gym.data.loaders.load_alpaca_ohlcv` directly.
        This wrapper is kept for backwards compatibility.
    """
    return load_alpaca_ohlcv(symbol=symbol, calibration_days=calibration_days)
