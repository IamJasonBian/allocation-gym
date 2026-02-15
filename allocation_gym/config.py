from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Backtesting configuration — stripped of live-trading concerns."""

    # Variance metrics
    variance_lookback: int = 14
    yz_period: int = 7
    vr_k: int = 3

    # Kelly sizing
    kelly_fraction: float = 0.25
    risk_free_rate: float = 0.045
    risk_per_trade_pct: float = 0.01
    max_portfolio_heat_pct: float = 0.06
    max_single_position_pct: float = 0.30

    # Risk
    default_atr_multiplier: int = 2
    trading_days: int = 252  # 252 equities, 365 crypto

    # Backtest-specific
    initial_cash: float = 100_000.0
    commission_pct: float = 0.001   # 10bps
    slippage_pct: float = 0.0005    # 5bps
