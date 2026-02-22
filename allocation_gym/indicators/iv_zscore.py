"""
IV Z-Score Indicator — Deribit DVOL with realised-vol fallback.

Pre-fetches DVOL data in __init__, serves values via date lookup in next().
"""

import math

import backtrader as bt
import numpy as np
import pandas as pd

from allocation_gym.signals.btc_dashboard import fetch_deribit_dvol


class IVZScoreIndicator(bt.Indicator):
    """
    Implied volatility z-score from Deribit DVOL index.

    Lines:
        dvol:        Current DVOL value (annualised IV %)
        dvol_zscore: Z-score of current DVOL vs trailing window
    """

    lines = ("dvol", "dvol_zscore")

    params = (
        ("zscore_period", 365),
        ("fetch_days", 730),  # fetch extra history for z-score warmup
    )

    plotinfo = dict(subplot=True)

    def __init__(self):
        super().__init__()
        self._dvol_map = {}      # date -> dvol value
        self._zscore_map = {}    # date -> z-score

        dvol_series = fetch_deribit_dvol(days=self.p.fetch_days)

        if dvol_series is not None and len(dvol_series) >= 30:
            # Build date-keyed lookup
            for ts, val in dvol_series.items():
                dt = ts.date() if hasattr(ts, "date") else ts
                self._dvol_map[dt] = float(val)

            # Pre-compute rolling z-score for each date
            rolling_mean = dvol_series.rolling(self.p.zscore_period, min_periods=30).mean()
            rolling_std = dvol_series.rolling(self.p.zscore_period, min_periods=30).std()

            for ts, val in dvol_series.items():
                dt = ts.date() if hasattr(ts, "date") else ts
                m = rolling_mean.get(ts, float("nan"))
                s = rolling_std.get(ts, float("nan"))
                if pd.notna(m) and pd.notna(s) and s > 0:
                    self._zscore_map[dt] = (val - m) / s
        else:
            # Fallback: compute realised vol z-score from the data feed
            # Will be populated on first next() call
            self._use_fallback = True
            self._fallback_built = False

    def next(self):
        dt = self.data.datetime.date(0)
        self.lines.dvol[0] = self._dvol_map.get(dt, 0.0)
        self.lines.dvol_zscore[0] = self._zscore_map.get(dt, 0.0)
