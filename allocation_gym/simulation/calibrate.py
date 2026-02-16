"""
Calibrate GBM parameters (mu, sigma) from historical OHLCV data.
Uses Yang-Zhang volatility from VarianceMetrics for sigma.
"""

from dataclasses import dataclass

import numpy as np

from allocation_gym.metrics.variance_metrics import VarianceMetrics, VarianceResult


@dataclass
class CalibrationResult:
    mu: float                        # annualized log drift
    sigma: float                     # annualized Yang-Zhang volatility
    initial_price: float             # latest close
    variance_result: VarianceResult  # full VarianceMetrics output
    n_days_used: int


def calibrate_gbm(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    trading_days: int = 365,
    yz_period: int | None = None,
) -> CalibrationResult:
    n = len(closes)
    if n < 5:
        raise ValueError(f"Need at least 5 bars for calibration, got {n}")

    period = yz_period or (n - 1)

    vr = VarianceMetrics.compute(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        period=period,
        trading_days=trading_days,
    )

    sigma = vr.yang_zhang_vol_ann
    log_returns = np.log(closes[1:] / closes[:-1])
    mu = float(np.mean(log_returns)) * trading_days

    return CalibrationResult(
        mu=mu,
        sigma=sigma,
        initial_price=float(closes[-1]),
        variance_result=vr,
        n_days_used=n,
    )
