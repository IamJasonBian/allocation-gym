"""
Variance Kelly Strategy — Multi-asset Kelly rebalancing.
Ported from trading_engine.py VarianceKellyStrategy.
"""

import backtrader as bt
import numpy as np

from allocation_gym.indicators.variance import VarianceIndicator


class VarianceKellyStrategy(bt.Strategy):
    """
    Computes diagonal-covariance Kelly weights across multiple assets.
    Rebalances when any position drifts beyond rebalance_threshold.
    """

    params = (
        ("expected_returns", {}),
        ("rebalance_threshold", 0.05),
        ("kelly_fraction", 0.25),
        ("risk_free_rate", 0.045),
        ("variance_lookback", 14),
        ("vr_k", 3),
        ("trading_days", 252),
        ("rebalance_days", 5),
    )

    def __init__(self):
        self.variance_indicators = {}
        self.bar_count = 0

        for data in self.datas:
            name = data._name
            self.variance_indicators[name] = VarianceIndicator(
                data,
                period=self.p.variance_lookback,
                vr_k=self.p.vr_k,
                trading_days=self.p.trading_days,
            )

    def next(self):
        self.bar_count += 1
        if self.bar_count % self.p.rebalance_days != 0:
            return

        active = []
        vols = []
        excess = []

        for data in self.datas:
            name = data._name
            var = self.variance_indicators[name]
            vol = var.yz_vol_ann[0]
            if vol < 0.01:
                continue
            active.append(data)
            vols.append(vol)
            er = self.p.expected_returns.get(name, 0.05)
            excess.append(er - self.p.risk_free_rate)

        if len(active) < 2:
            return

        vols_arr = np.array(vols)
        excess_arr = np.array(excess)

        cov = np.diag(vols_arr ** 2)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return

        full_kelly = inv_cov @ excess_arr
        frac_kelly = full_kelly * self.p.kelly_fraction
        frac_kelly = np.maximum(frac_kelly, 0)
        total = frac_kelly.sum()
        if total > 1.0:
            frac_kelly /= total

        equity = self.broker.getvalue()

        for i, data in enumerate(active):
            target_pct = frac_kelly[i]
            target_value = target_pct * equity

            pos = self.getposition(data)
            current_value = pos.size * data.close[0]

            drift = abs(target_value - current_value) / equity if equity > 0 else 0
            if drift < self.p.rebalance_threshold:
                continue

            delta_value = target_value - current_value
            delta_shares = int(delta_value / data.close[0])

            if delta_shares > 0:
                self.buy(data=data, size=abs(delta_shares))
            elif delta_shares < 0:
                self.sell(data=data, size=abs(delta_shares))
