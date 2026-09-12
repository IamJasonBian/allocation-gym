"""
Momentum Strategy — Trend-following using VR + ER + SMA.
Ported from trading_engine.py MomentumStrategy.
"""

import backtrader as bt
import numpy as np

from allocation_gym.indicators.variance import VarianceIndicator


class MomentumStrategy(bt.Strategy):
    """
    Entry: VR >= min_vr AND ER >= min_efficiency AND price > SMA
    Exit:  price < SMA OR regime == CHOP
    """

    params = (
        ("sma_period", 50),
        ("min_efficiency", 0.4),
        ("min_variance_ratio", 1.1),
        ("variance_lookback", 14),
        ("vr_k", 3),
        ("trading_days", 252),
        ("signals", False),
    )

    def __init__(self):
        self.variance_indicators = {}
        self.sma = {}
        self.order_refs = {}
        self._current_expected_return = 0.10
        self.signal_iv = None
        self.signal_flow = None

        for data in self.datas:
            name = data._name
            self.variance_indicators[name] = VarianceIndicator(
                data,
                period=self.p.variance_lookback,
                vr_k=self.p.vr_k,
                trading_days=self.p.trading_days,
            )
            self.sma[name] = bt.indicators.SMA(data.close, period=self.p.sma_period)

        if self.p.signals:
            from allocation_gym.indicators.iv_zscore import IVZScoreIndicator
            from allocation_gym.indicators.etf_flow import ETFFlowIndicator
            self.signal_iv = IVZScoreIndicator(self.datas[0])
            self.signal_flow = ETFFlowIndicator(self.datas[0])

    def next(self):
        for data in self.datas:
            name = data._name
            var = self.variance_indicators[name]
            sma = self.sma[name]

            price = data.close[0]
            pos = self.getposition(data)

            if name in self.order_refs and self.order_refs[name] is not None:
                continue

            trending = (
                var.variance_ratio[0] >= self.p.min_variance_ratio
                and var.efficiency_ratio[0] >= self.p.min_efficiency
            )
            above_sma = price > sma[0]
            is_chop = var.regime[0] == VarianceIndicator.REGIME_MAP["CHOP"]

            if trending and above_sma and pos.size == 0:
                closes_20 = np.array(data.close.get(size=21))
                if len(closes_20) >= 21:
                    ret_20d = (closes_20[-1] / closes_20[0]) - 1
                    self._current_expected_return = max(
                        ret_20d * (self.p.trading_days / 20), 0.05
                    )
                else:
                    self._current_expected_return = 0.05
                self.order_refs[name] = self.buy(data=data)

            elif pos.size > 0 and (not above_sma or is_chop):
                self.order_refs[name] = self.close(data=data)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            name = order.data._name
            self.order_refs[name] = None
