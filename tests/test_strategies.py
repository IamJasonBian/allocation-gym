"""Smoke tests: each strategy runs on synthetic data without errors."""

import backtrader as bt
import pandas as pd

from tests.conftest import generate_synthetic_ohlc
from allocation_gym.strategies.momentum import MomentumStrategy
from allocation_gym.strategies.mean_reversion import MeanReversionStrategy
from allocation_gym.strategies.variance_kelly import VarianceKellyStrategy
from allocation_gym.sizers.kelly import KellySizer
from allocation_gym.analyzers.performance import PerformanceAnalyzer


def _run_strategy(strategy_cls, symbols=None, strategy_params=None, use_sizer=True):
    symbols = symbols or ["TEST"]
    strategy_params = strategy_params or {}
    cerebro = bt.Cerebro()

    for sym in symbols:
        df = generate_synthetic_ohlc(days=200, seed=hash(sym) % 2**31)
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=sym)

    cerebro.addstrategy(strategy_cls, **strategy_params)
    if use_sizer:
        cerebro.addsizer(KellySizer)
    cerebro.addanalyzer(PerformanceAnalyzer)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.001)

    results = cerebro.run()
    perf = results[0].analyzers.performanceanalyzer.get_analysis()
    return perf


def test_momentum_strategy_runs():
    perf = _run_strategy(MomentumStrategy, symbols=["SPY"])
    assert "sharpe" in perf
    assert "final_value" in perf


def test_mean_reversion_strategy_runs():
    perf = _run_strategy(MeanReversionStrategy, symbols=["SPY"])
    assert "sharpe" in perf


def test_variance_kelly_strategy_runs():
    perf = _run_strategy(
        VarianceKellyStrategy,
        symbols=["SPY", "GLD"],
        strategy_params={
            "expected_returns": {"SPY": 0.10, "GLD": 0.08},
        },
        use_sizer=False,
    )
    assert "sharpe" in perf


def test_momentum_multi_symbol():
    perf = _run_strategy(MomentumStrategy, symbols=["SPY", "QQQ", "NVDA"])
    assert "final_value" in perf
