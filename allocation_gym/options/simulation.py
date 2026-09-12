"""
Options hedging simulation: protective puts and collars with monthly rolling.

Ported from allocation-engine and adapted for allocation-gym's data pipeline
(Alpaca CryptoHistoricalDataClient / yfinance).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from allocation_gym.options.black_scholes import bs_call_price, bs_put_price


class OptionsStrategyType(Enum):
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    COLLAR = "COLLAR"


@dataclass
class OptionRoll:
    roll_date: str
    expiry_bar_idx: int
    put_strike: float
    put_premium_paid: float
    put_intrinsic_at_expiry: float = 0.0
    call_strike: Optional[float] = None
    call_premium_received: Optional[float] = None
    call_intrinsic_at_expiry: float = 0.0
    shares: int = 100


@dataclass
class OptionsDailySnapshot:
    date: str
    underlying_price: float
    equity_value: float
    put_mark: float
    call_mark: float
    cumulative_premium_paid: float
    cumulative_premium_received: float
    cumulative_put_intrinsic: float
    cumulative_call_intrinsic: float
    net_portfolio_value: float


@dataclass
class OptionsSimulationResult:
    symbol: str
    strategy_type: OptionsStrategyType
    snapshots: List[OptionsDailySnapshot]
    rolls: List[OptionRoll]
    initial_shares: int
    initial_price: float
    iv: float
    otm_pct: float
    roll_period_days: int


def run_options_simulation(
    bars: List[Dict],
    symbol: str,
    strategy_type: OptionsStrategyType,
    initial_shares: int,
    initial_price: float,
    iv: float = 0.20,
    otm_pct: float = 0.05,
    roll_period_days: int = 21,
    risk_free_rate: float = 0.05,
) -> OptionsSimulationResult:
    """Run an options hedging simulation over daily OHLCV bars."""
    rolls: List[OptionRoll] = []
    snapshots: List[OptionsDailySnapshot] = []

    cum_premium_paid = 0.0
    cum_premium_received = 0.0
    cum_put_intrinsic = 0.0
    cum_call_intrinsic = 0.0

    active_roll: Optional[OptionRoll] = None
    next_roll_idx = 0
    shares = initial_shares

    for bar_idx, bar in enumerate(bars):
        date = bar["date"]
        price = bar["close"]

        # Settle expiring options
        if active_roll is not None and bar_idx >= active_roll.expiry_bar_idx:
            put_intrinsic = max(active_roll.put_strike - price, 0.0)
            active_roll.put_intrinsic_at_expiry = put_intrinsic
            cum_put_intrinsic += put_intrinsic * active_roll.shares

            if (
                strategy_type == OptionsStrategyType.COLLAR
                and active_roll.call_strike is not None
            ):
                call_intrinsic = max(price - active_roll.call_strike, 0.0)
                active_roll.call_intrinsic_at_expiry = call_intrinsic
                cum_call_intrinsic += call_intrinsic * active_roll.shares

            active_roll = None
            next_roll_idx = bar_idx

        # Open new roll
        if bar_idx >= next_roll_idx and active_roll is None:
            T = roll_period_days / 365.0  # Crypto: 365 trading days

            put_strike = round(price * (1.0 - otm_pct), 2)
            put_prem = bs_put_price(price, put_strike, T, risk_free_rate, iv)
            cum_premium_paid += put_prem * shares

            call_strike = None
            call_prem = None
            if strategy_type == OptionsStrategyType.COLLAR:
                call_strike = round(price * (1.0 + otm_pct), 2)
                call_prem = bs_call_price(price, call_strike, T, risk_free_rate, iv)
                cum_premium_received += call_prem * shares

            roll = OptionRoll(
                roll_date=date,
                expiry_bar_idx=bar_idx + roll_period_days,
                put_strike=put_strike,
                put_premium_paid=put_prem,
                call_strike=call_strike,
                call_premium_received=call_prem,
                shares=shares,
            )
            rolls.append(roll)
            active_roll = roll
            next_roll_idx = bar_idx + roll_period_days

        # Daily mark-to-market
        put_mark_total = 0.0
        call_mark_total = 0.0

        if active_roll is not None:
            remaining_bars = active_roll.expiry_bar_idx - bar_idx
            T_remaining = max(remaining_bars / 365.0, 1.0 / 365.0)

            put_mark_total = (
                bs_put_price(
                    price, active_roll.put_strike, T_remaining, risk_free_rate, iv
                )
                * active_roll.shares
            )

            if (
                strategy_type == OptionsStrategyType.COLLAR
                and active_roll.call_strike is not None
            ):
                call_mark_total = (
                    bs_call_price(
                        price, active_roll.call_strike, T_remaining, risk_free_rate, iv
                    )
                    * active_roll.shares
                )

        equity_value = shares * price
        net_value = (
            equity_value
            + put_mark_total
            - call_mark_total
            - cum_premium_paid
            + cum_premium_received
            + cum_put_intrinsic
            - cum_call_intrinsic
        )

        snapshots.append(
            OptionsDailySnapshot(
                date=date,
                underlying_price=price,
                equity_value=equity_value,
                put_mark=put_mark_total,
                call_mark=call_mark_total,
                cumulative_premium_paid=cum_premium_paid,
                cumulative_premium_received=cum_premium_received,
                cumulative_put_intrinsic=cum_put_intrinsic,
                cumulative_call_intrinsic=cum_call_intrinsic,
                net_portfolio_value=net_value,
            )
        )

    return OptionsSimulationResult(
        symbol=symbol,
        strategy_type=strategy_type,
        snapshots=snapshots,
        rolls=rolls,
        initial_shares=initial_shares,
        initial_price=initial_price,
        iv=iv,
        otm_pct=otm_pct,
        roll_period_days=roll_period_days,
    )
