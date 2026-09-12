"""
Variance Kelly Strategy — Multi-asset Kelly rebalancing.
Ported from trading_engine.py VarianceKellyStrategy.

Supports per-asset minimum weight constraints enforced at month-end.
"""

import backtrader as bt
import numpy as np

from allocation_gym.indicators.variance import VarianceIndicator


class VarianceKellyStrategy(bt.Strategy):
    """
    Computes diagonal-covariance Kelly weights across multiple assets.
    Rebalances when any position drifts beyond rebalance_threshold.

    min_weights: {symbol: float} — floor constraints enforced every rebalance.
        At month-end, positions are forced to meet the floor regardless of drift.
    """

    params = (
        ("expected_returns", {}),
        ("min_weights", {}),             # e.g. {"BTC-USD": 0.50}
        ("rebalance_threshold", 0.05),
        ("kelly_fraction", 0.25),
        ("risk_free_rate", 0.045),
        ("variance_lookback", 14),
        ("vr_k", 3),
        ("trading_days", 252),
        ("rebalance_days", 5),
        ("signals", False),
    )

    def __init__(self):
        self.variance_indicators = {}
        self.bar_count = 0
        self._prev_month = None
        self.signal_iv = None
        self.signal_flow = None

        self._initial_alloc_done = False
        self._initial_order_refs = set()

        for data in self.datas:
            name = data._name
            td = 365 if "BTC" in name.upper() else self.p.trading_days
            self.variance_indicators[name] = VarianceIndicator(
                data,
                period=self.p.variance_lookback,
                vr_k=self.p.vr_k,
                trading_days=td,
            )

        if self.p.signals:
            from allocation_gym.indicators.iv_zscore import IVZScoreIndicator
            from allocation_gym.indicators.etf_flow import ETFFlowIndicator
            self.signal_iv = IVZScoreIndicator(self.datas[0])
            self.signal_flow = ETFFlowIndicator(self.datas[0])

    def _is_month_end(self):
        """True on the last trading day of a month (next bar is a new month)."""
        dt = self.data.datetime.date(0)
        cur_month = (dt.year, dt.month)
        if self._prev_month is not None and cur_month != self._prev_month:
            # We just crossed into a new month — previous bar was month-end
            return False
        # Check if next bar exists and is a different month
        try:
            next_dt = self.data.datetime.date(1)
            is_end = (next_dt.year, next_dt.month) != cur_month
        except IndexError:
            is_end = True  # last bar in dataset
        self._prev_month = cur_month
        return is_end

    def next(self):
        self.bar_count += 1
        month_end = self._is_month_end()

        # On first bar, buy into min_weight positions immediately
        if not self._initial_alloc_done and self.p.min_weights:
            self._initial_alloc_done = True
            equity = self.broker.getvalue()
            for data in self.datas:
                name = data._name
                min_w = self.p.min_weights.get(name, 0)
                if min_w > 0:
                    target_value = min_w * equity
                    shares = int(target_value / data.close[0])
                    if shares > 0:
                        o = self.buy(data=data, size=shares)
                        self._initial_order_refs.add(o.ref)
            return

        # Regular rebalance cadence, or force on month-end
        if not month_end and self.bar_count % self.p.rebalance_days != 0:
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

        # Apply minimum weight constraints
        active_names = [d._name for d in active]
        for name, min_w in self.p.min_weights.items():
            if name in active_names:
                idx = active_names.index(name)
                frac_kelly[idx] = max(frac_kelly[idx], min_w)

        # Normalize if sum > 1
        total = frac_kelly.sum()
        if total > 1.0:
            # Scale non-floor assets down while preserving floors
            floor_indices = set()
            for name, min_w in self.p.min_weights.items():
                if name in active_names:
                    idx = active_names.index(name)
                    if frac_kelly[idx] <= min_w + 1e-9:
                        floor_indices.add(idx)

            floor_total = sum(frac_kelly[i] for i in floor_indices)
            non_floor_total = sum(frac_kelly[i] for i in range(len(frac_kelly)) if i not in floor_indices)
            remaining = 1.0 - floor_total

            if non_floor_total > 0 and remaining > 0:
                for i in range(len(frac_kelly)):
                    if i not in floor_indices:
                        frac_kelly[i] = frac_kelly[i] / non_floor_total * remaining

        equity = self.broker.getvalue()
        # Use tighter threshold at month-end to enforce floors
        threshold = 0.01 if month_end else self.p.rebalance_threshold

        for i, data in enumerate(active):
            target_pct = frac_kelly[i]
            target_value = target_pct * equity

            pos = self.getposition(data)
            current_value = pos.size * data.close[0]

            drift = abs(target_value - current_value) / equity if equity > 0 else 0
            if drift < threshold:
                continue

            delta_value = target_value - current_value
            delta_shares = int(delta_value / data.close[0])

            if delta_shares > 0:
                self.buy(data=data, size=abs(delta_shares))
            elif delta_shares < 0:
                self.sell(data=data, size=abs(delta_shares))
