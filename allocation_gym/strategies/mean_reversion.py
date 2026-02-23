"""
Mean Reversion Strategy — RSI oversold + VR < threshold.
Ported from trading_engine.py MeanReversionStrategy.
"""

import backtrader as bt

from allocation_gym.indicators.variance import VarianceIndicator


class MeanReversionStrategy(bt.Strategy):
    """
    Entry: VR < max_vr AND RSI < oversold
    Exit:  RSI > overbought
    """

    params = (
        ("rsi_period", 14),
        ("oversold", 30),
        ("overbought", 70),
        ("max_vr", 0.9),
        ("variance_lookback", 14),
        ("vr_k", 3),
        ("trading_days", 252),
        ("signals", False),
    )

    def __init__(self):
        self.variance_indicators = {}
        self.rsi = {}
        self.sma20 = {}
        self.order_refs = {}
        self._current_expected_return = 0.05
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
            self.rsi[name] = bt.indicators.RSI(data.close, period=self.p.rsi_period)
            self.sma20[name] = bt.indicators.SMA(data.close, period=20)

        if self.p.signals:
            from allocation_gym.indicators.iv_zscore import IVZScoreIndicator
            from allocation_gym.indicators.etf_flow import ETFFlowIndicator
            self.signal_iv = IVZScoreIndicator(self.datas[0])
            self.signal_flow = ETFFlowIndicator(self.datas[0])

    def next(self):
        for data in self.datas:
            name = data._name
            var = self.variance_indicators[name]
            rsi = self.rsi[name]
            pos = self.getposition(data)

            if name in self.order_refs and self.order_refs[name] is not None:
                continue

            if var.variance_ratio[0] > self.p.max_vr:
                continue

            price = data.close[0]

            if rsi[0] < self.p.oversold and pos.size == 0:
                sma20_val = self.sma20[name][0]
                expected_move = (sma20_val - price) / price if price > 0 else 0
                self._current_expected_return = max(
                    expected_move * (self.p.trading_days / 10), 0.03
                )
                self.order_refs[name] = self.buy(data=data)

            elif pos.size > 0 and rsi[0] > self.p.overbought:
                self.order_refs[name] = self.close(data=data)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order_refs[order.data._name] = None
