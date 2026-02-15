import numpy as np
import pandas as pd
import pytest


def generate_synthetic_ohlc(days=365, base_price=100.0, seed=42):
    """Generate synthetic OHLC DataFrame for testing."""
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.02, days)
    closes = base_price * np.cumprod(1 + returns)
    opens = closes * (1 + np.random.normal(0, 0.005, days))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    volume = np.random.randint(100_000, 1_000_000, days)

    dates = pd.bdate_range(start="2023-01-01", periods=days)
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def synthetic_ohlc():
    return generate_synthetic_ohlc()


@pytest.fixture
def synthetic_arrays():
    df = generate_synthetic_ohlc()
    return {
        "opens": df["Open"].values,
        "highs": df["High"].values,
        "lows": df["Low"].values,
        "closes": df["Close"].values,
    }
