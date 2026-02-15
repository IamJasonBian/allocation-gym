"""
Kelly position sizer for Backtrader.
Ports PositionSizer logic from trading_engine.py, drops PDT/Margin guards.
"""

import math
import backtrader as bt


class KellySizer(bt.Sizer):
    """Quarter-Kelly sizing with downside semivariance stop distance."""

    params = (
        ("kelly_fraction", 0.25),
        ("risk_free_rate", 0.045),
        ("risk_per_trade_pct", 0.01),
        ("max_portfolio_heat_pct", 0.06),
        ("max_single_position_pct", 0.30),
        ("default_atr_multiplier", 2),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        strategy = self.strategy
        var_ind = getattr(strategy, "variance_indicators", {}).get(data._name)
        if var_ind is None:
            return 0

        price = data.close[0]
        if price <= 0:
            return 0
        equity = strategy.broker.getvalue()
        if equity <= 0:
            return 0

        # Kelly-optimal fraction
        sigma = var_ind.yz_vol_ann[0]
        if sigma < 0.01:
            sigma = 0.01

        expected_return = getattr(strategy, "_current_expected_return", 0.10)
        full_kelly = (expected_return - self.p.risk_free_rate) / (sigma ** 2)
        frac_kelly = full_kelly * self.p.kelly_fraction
        kelly_pct = max(0.0, min(frac_kelly, self.p.max_single_position_pct))
        kelly_dollars = kelly_pct * equity

        # Stop distance from downside semivariance
        n = self.p.default_atr_multiplier
        if var_ind.vol_of_vol[0] > 1.0:
            n = 3

        skew_adj = max(1.0, math.sqrt(max(var_ind.vol_skew[0], 0)))
        stop_distance = n * var_ind.downside_semivol[0] * price * skew_adj
        stop_distance = max(stop_distance, price * 0.005)

        # Risk-budget sizing
        risk_budget = self.p.risk_per_trade_pct * equity
        risk_sized_shares = int(risk_budget / stop_distance) if stop_distance > 0 else 0
        kelly_shares = int(kelly_dollars / price) if price > 0 else 0

        shares = min(risk_sized_shares, kelly_shares)
        return max(shares, 0)
