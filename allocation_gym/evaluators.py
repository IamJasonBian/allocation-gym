"""Band strategy evaluators — score backtests using unified TradeEvent records."""

from __future__ import annotations

from typing import TypedDict

from allocation_gym.enums import (
    AssetClass, BandType, EvalMetric, SignalType, TradeState,
)


class TradeEvent(TypedDict, total=False):
    """Normalised trade record produced by any band strategy backtest."""
    trade_id: int
    symbol: str
    asset_class: AssetClass
    band_type: BandType
    signal: SignalType
    state: TradeState
    entry_price: float
    exit_price: float | None
    entry_date: str
    exit_date: str | None
    quantity: float
    pnl_pct: float | None
    # OU-specific params (absent for non-OU strategies)
    kappa: float | None          # mean-reversion speed
    sigma: float | None          # volatility
    theta: float | None          # long-run mean
    band_width_pct: float | None


def score(trades: list[TradeEvent]) -> dict[str, float]:
    """Compute summary statistics from a list of closed trades.

    Returns a dict keyed by EvalMetric values — mirrors the summary
    table in the IWN optimal bands backtest PDF.
    """
    closed = [t for t in trades if t.get("state") == TradeState.CLOSED]
    if not closed:
        return {m: 0.0 for m in EvalMetric}

    pnls = [t.get("pnl_pct", 0.0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(closed)
    long_count = sum(1 for t in closed if t.get("signal") == SignalType.ENTRY_LONG)

    return {
        EvalMetric.WIN_RATE: len(wins) / n * 100 if n else 0.0,
        EvalMetric.TOTAL_PNL: sum(pnls),
        EvalMetric.AVG_PNL: sum(pnls) / n,
        EvalMetric.BEST_TRADE: max(pnls) if pnls else 0.0,
        EvalMetric.WORST_TRADE: min(pnls) if pnls else 0.0,
        EvalMetric.AVG_WIN: sum(wins) / len(wins) if wins else 0.0,
        EvalMetric.AVG_LOSS: sum(losses) / len(losses) if losses else 0.0,
        EvalMetric.SHARPE: _sharpe(pnls),
        "total_trades": float(n),
        "long_count": float(long_count),
        "short_count": float(n - long_count),
    }


def _sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
    """Annualised Sharpe from per-trade PnL percentages."""
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls) - risk_free
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    std = var ** 0.5
    if std == 0:
        return 0.0
    return mean / std
