"""Shared enums for band strategies, signals, and evaluation metrics.

Follows the StrEnum pattern from allocation-engine-2.0/app/enums.py,
adapted for backtest evaluation of mean-reversion band strategies.
"""

from enum import StrEnum


class SignalType(StrEnum):
    ENTRY_LONG = "entry_long"
    EXIT_LONG = "exit_long"
    ENTRY_SHORT = "entry_short"
    EXIT_SHORT = "exit_short"


class BandType(StrEnum):
    OU = "ou"
    BOLLINGER = "bollinger"
    FIXED = "fixed"


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"


class TradeState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"


class EvalMetric(StrEnum):
    WIN_RATE = "win_rate"
    TOTAL_PNL = "total_pnl"
    AVG_PNL = "avg_pnl"
    BEST_TRADE = "best_trade"
    WORST_TRADE = "worst_trade"
    AVG_WIN = "avg_win"
    AVG_LOSS = "avg_loss"
    SHARPE = "sharpe"
