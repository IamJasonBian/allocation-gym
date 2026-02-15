"""
Backtrader Indicator wrapping VarianceMetrics.compute().
"""

import backtrader as bt
import numpy as np

from allocation_gym.metrics.variance_metrics import VarianceMetrics


class VarianceIndicator(bt.Indicator):
    """
    Wraps VarianceMetrics.compute() as a Backtrader indicator.
    Exposes all 9 variance metrics + regime as line objects.
    """

    lines = (
        "yz_var",
        "yz_vol",
        "yz_vol_ann",
        "variance_ratio",
        "efficiency_ratio",
        "vol_of_vol",
        "downside_semivol",
        "upside_semivol",
        "vol_skew",
        "regime",
    )

    params = (
        ("period", 14),
        ("vr_k", 3),
        ("trading_days", 252),
    )

    REGIME_MAP = {
        "RANDOM_WALK": 0.0,
        "STRONG_TREND": 1.0,
        "NOISY_TREND": 2.0,
        "CHOP": 3.0,
        "MEAN_REVERT": 4.0,
    }

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        n = self.p.period + 1
        opens = np.array(self.data.open.get(size=n))
        highs = np.array(self.data.high.get(size=n))
        lows = np.array(self.data.low.get(size=n))
        closes = np.array(self.data.close.get(size=n))

        result = VarianceMetrics.compute(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            period=self.p.period,
            vr_k=self.p.vr_k,
            trading_days=self.p.trading_days,
        )

        self.lines.yz_var[0] = result.yang_zhang_var
        self.lines.yz_vol[0] = result.yang_zhang_vol
        self.lines.yz_vol_ann[0] = result.yang_zhang_vol_ann
        self.lines.variance_ratio[0] = result.variance_ratio
        self.lines.efficiency_ratio[0] = result.efficiency_ratio
        self.lines.vol_of_vol[0] = result.vol_of_vol
        self.lines.downside_semivol[0] = result.downside_semivol
        self.lines.upside_semivol[0] = result.upside_semivol
        self.lines.vol_skew[0] = result.vol_skew
        self.lines.regime[0] = self.REGIME_MAP.get(result.regime, 0.0)
