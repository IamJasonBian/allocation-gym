"""
Performance analyzer — Sharpe, Sortino, max drawdown, CAGR, Calmar.
"""

import math

import numpy as np
import backtrader as bt


class PerformanceAnalyzer(bt.Analyzer):
    """Computes key performance metrics in a single pass."""

    params = (
        ("risk_free_rate", 0.045),
        ("trading_days", 252),
    )

    def start(self):
        self.daily_values = []

    def next(self):
        self.daily_values.append(self.strategy.broker.getvalue())

    def stop(self):
        values = np.array(self.daily_values)
        if len(values) < 2:
            self.rets = {}
            return

        returns = np.diff(values) / values[:-1]
        daily_rf = self.p.risk_free_rate / self.p.trading_days
        excess = returns - daily_rf

        # Sharpe
        std_excess = np.std(excess)
        sharpe = (
            np.mean(excess) / std_excess * math.sqrt(self.p.trading_days)
            if std_excess > 0
            else 0.0
        )

        # Sortino
        downside = excess[excess < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-12
        sortino = np.mean(excess) / downside_std * math.sqrt(self.p.trading_days)

        # Max Drawdown
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        max_dd = abs(np.min(dd))

        # CAGR
        n_years = len(values) / self.p.trading_days
        cagr = (values[-1] / values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Calmar
        calmar = cagr / max_dd if max_dd > 0 else 0

        self.rets = {
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "calmar": round(calmar, 3),
            "total_return_pct": round((values[-1] / values[0] - 1) * 100, 2),
            "final_value": round(values[-1], 2),
        }

    def get_analysis(self):
        return self.rets
