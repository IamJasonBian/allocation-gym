"""
Performance analyzer — Sharpe, Sortino, max drawdown, CAGR, Calmar.
Also captures daily equity curve, order fills, and closed trades for plotting.
"""

import math

import numpy as np
import backtrader as bt


class PerformanceAnalyzer(bt.Analyzer):
    """Computes key performance metrics and captures data for plotting."""

    params = (
        ("risk_free_rate", 0.045),
        ("trading_days", 252),
    )

    def start(self):
        self.daily_values = []
        self.daily_dates = []
        self.trades = []
        self.orders = []      # all filled orders (for trade markers)
        self.daily_positions = []  # per-day position snapshots
        self.initial_cash = self.strategy.broker.getvalue()

    def next(self):
        portfolio_value = self.strategy.broker.getvalue()
        self.daily_values.append(portfolio_value)
        self.daily_dates.append(self.strategy.data.datetime.date(0))

        # Snapshot per-symbol positions
        snap = {"cash": self.strategy.broker.getcash()}
        for data in self.strategy.datas:
            pos = self.strategy.getposition(data)
            mkt_val = pos.size * data.close[0]
            snap[data._name] = {
                "size": pos.size,
                "price": data.close[0],
                "value": round(mkt_val, 2),
                "weight": round(mkt_val / portfolio_value, 4) if portfolio_value > 0 else 0,
            }
        self.daily_positions.append(snap)

    def notify_order(self, order):
        if order.status == order.Completed:
            self.orders.append({
                "symbol": order.data._name,
                "side": "buy" if order.isbuy() else "sell",
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "dt": bt.num2date(order.executed.dt),
            })

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "symbol": trade.data._name,
                "pnl": trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "size": trade.size,
                "price": trade.price,
                "dtopen": bt.num2date(trade.dtopen),
                "dtclose": bt.num2date(trade.dtclose),
            })

    def stop(self):
        values = np.array(self.daily_values)
        if len(values) < 2:
            self.rets = {}
            return

        returns = np.diff(values) / values[:-1]
        daily_rf = self.p.risk_free_rate / self.p.trading_days
        excess = returns - daily_rf

        std_excess = np.std(excess)
        sharpe = (
            np.mean(excess) / std_excess * math.sqrt(self.p.trading_days)
            if std_excess > 0
            else 0.0
        )

        downside = excess[excess < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-12
        sortino = np.mean(excess) / downside_std * math.sqrt(self.p.trading_days)

        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        max_dd = abs(np.min(dd))

        n_years = len(values) / self.p.trading_days
        cagr = (values[-1] / values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        calmar = cagr / max_dd if max_dd > 0 else 0

        # P&L from daily equity changes
        daily_pnl = np.diff(values)

        self.rets = {
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "cagr_pct": round(cagr * 100, 2),
            "calmar": round(calmar, 3),
            "total_return_pct": round((values[-1] / values[0] - 1) * 100, 2),
            "final_value": round(values[-1], 2),
            "total_orders": len(self.orders),
            "buy_orders": len([o for o in self.orders if o["side"] == "buy"]),
            "sell_orders": len([o for o in self.orders if o["side"] == "sell"]),
            "closed_trades": len(self.trades),
        }

    def get_analysis(self):
        return self.rets

    def get_equity_curve(self):
        return self.daily_dates, self.daily_values

    def get_orders(self):
        return self.orders

    def get_trades(self):
        return self.trades

    def get_daily_positions(self):
        return self.daily_dates, self.daily_positions
