"""
Forward stress-test: run variance_kelly on Monte Carlo simulated paths.

Generates N forward scenarios for multiple assets, feeds each scenario
into Backtrader, and aggregates strategy performance across scenarios.
"""

import numpy as np
import pandas as pd
import backtrader as bt

from allocation_gym.simulation.calibrate import calibrate_gbm
from allocation_gym.simulation.engine import MonteCarloGBM


def calibrate_symbols(symbol_dfs: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Calibrate GBM parameters for each symbol from historical data."""
    params = {}
    for sym, df in symbol_dfs.items():
        is_crypto = "/" in sym or sym.upper() in {"BTC", "BTC/USD", "ETH/USD"}
        td = 365 if is_crypto else 252
        cal = calibrate_gbm(
            opens=df["Open"].values,
            highs=df["High"].values,
            lows=df["Low"].values,
            closes=df["Close"].values,
            trading_days=td,
        )
        params[sym] = {
            "mu": cal.mu,
            "sigma": cal.sigma,
            "initial_price": cal.initial_price,
            "trading_days": td,
            "regime": cal.variance_result.regime,
        }
    return params


def generate_scenario(
    params: dict[str, dict],
    n_days: int = 365,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate one forward scenario: synthetic OHLCV DataFrames per symbol."""
    rng = np.random.default_rng(seed)
    scenario = {}

    for sym, p in params.items():
        mc = MonteCarloGBM(mu=p["mu"], sigma=p["sigma"],
                           initial_price=p["initial_price"])
        # Use a unique sub-seed per symbol for independence
        sub_seed = rng.integers(0, 2**31)
        result = mc.simulate(n_paths=1, n_days=n_days, seed=int(sub_seed))
        closes = result.paths[0]  # single path

        # Synthesize OHLCV from close path
        daily_vol = p["sigma"] / np.sqrt(p["trading_days"])
        opens = closes * (1 + rng.normal(0, daily_vol * 0.3, len(closes)))
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, daily_vol * 0.5, len(closes))))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, daily_vol * 0.5, len(closes))))
        volume = rng.integers(100_000, 1_000_000, len(closes))

        dates = pd.bdate_range(start="2026-02-16", periods=len(closes))
        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": volume.astype(float),
        }, index=dates)
        scenario[sym] = df

    return scenario


def run_scenario_backtest(
    scenario: dict[str, pd.DataFrame],
    min_weights: dict[str, float],
    initial_cash: float = 100_000,
) -> dict:
    """Run variance_kelly backtest on a single simulated scenario."""
    from allocation_gym.strategies.variance_kelly import VarianceKellyStrategy
    from allocation_gym.analyzers.performance import PerformanceAnalyzer

    cerebro = bt.Cerebro()

    for sym, df in scenario.items():
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=sym)

    exp_ret = {}
    for s in scenario:
        is_crypto = "/" in s or s.upper() in {"BTC", "BTC/USD"}
        exp_ret[s] = 0.40 if is_crypto else 0.10

    cerebro.addstrategy(
        VarianceKellyStrategy,
        expected_returns=exp_ret,
        min_weights=min_weights,
        variance_lookback=14,
        vr_k=3,
        trading_days=252,
    )
    cerebro.addanalyzer(PerformanceAnalyzer, risk_free_rate=0.045, trading_days=252)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)

    results = cerebro.run()
    return results[0].analyzers.performanceanalyzer.get_analysis()


def run_forward_test(
    symbol_dfs: dict[str, pd.DataFrame],
    min_weights: dict[str, float],
    n_scenarios: int = 50,
    n_days: int = 365,
    initial_cash: float = 100_000,
    seed: int = 42,
) -> list[dict]:
    """
    Run the full forward test pipeline:
    1. Calibrate from historical data
    2. Generate N scenarios
    3. Run backtest on each
    4. Return list of performance dicts
    """
    print("\n  Calibrating from historical data...")
    params = calibrate_symbols(symbol_dfs)
    for sym, p in params.items():
        print(f"    {sym:>8}: mu={p['mu']:+.1%}  sigma={p['sigma']:.1%}  "
              f"S0=${p['initial_price']:,.2f}  regime={p['regime']}")

    rng = np.random.default_rng(seed)
    all_results = []

    print(f"\n  Running {n_scenarios} scenarios x {n_days} days...")
    for i in range(n_scenarios):
        scenario_seed = int(rng.integers(0, 2**31))
        scenario = generate_scenario(params, n_days=n_days, seed=scenario_seed)

        try:
            perf = run_scenario_backtest(scenario, min_weights, initial_cash)
            all_results.append(perf)
        except Exception as e:
            print(f"    Scenario {i+1} failed: {e}")
            continue

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    {i+1}/{n_scenarios} done  "
                  f"(latest: {perf.get('total_return_pct', 0):+.1f}%)")

    return all_results


def print_forward_test_summary(results: list[dict], n_days: int = 365):
    """Print aggregate statistics across all scenarios."""
    if not results:
        print("\nNo results to summarize.")
        return

    returns = np.array([r.get("total_return_pct", 0) for r in results])
    sharpes = np.array([r.get("sharpe", 0) for r in results])
    max_dds = np.array([r.get("max_drawdown_pct", 0) for r in results])
    finals = np.array([r.get("final_value", 0) for r in results])

    print("\n" + "=" * 60)
    print(f"  FORWARD TEST RESULTS ({len(results)} scenarios x {n_days} days)")
    print("=" * 60)

    print(f"\n  Total Return %:")
    print(f"    Mean:     {np.mean(returns):>+8.1f}%")
    print(f"    Median:   {np.median(returns):>+8.1f}%")
    print(f"    Std Dev:  {np.std(returns):>8.1f}%")
    print(f"    P10:      {np.percentile(returns, 10):>+8.1f}%")
    print(f"    P25:      {np.percentile(returns, 25):>+8.1f}%")
    print(f"    P75:      {np.percentile(returns, 75):>+8.1f}%")
    print(f"    P90:      {np.percentile(returns, 90):>+8.1f}%")
    print(f"    Best:     {np.max(returns):>+8.1f}%")
    print(f"    Worst:    {np.min(returns):>+8.1f}%")

    print(f"\n  Sharpe Ratio:")
    print(f"    Mean:     {np.mean(sharpes):>8.3f}")
    print(f"    Median:   {np.median(sharpes):>8.3f}")

    print(f"\n  Max Drawdown %:")
    print(f"    Mean:     {np.mean(max_dds):>8.1f}%")
    print(f"    Worst:    {np.max(max_dds):>8.1f}%")

    print(f"\n  Final Portfolio Value:")
    print(f"    Mean:     ${np.mean(finals):>12,.2f}")
    print(f"    Median:   ${np.median(finals):>12,.2f}")
    print(f"    P10:      ${np.percentile(finals, 10):>12,.2f}")
    print(f"    P90:      ${np.percentile(finals, 90):>12,.2f}")

    prob_profit = np.mean(returns > 0) * 100
    prob_loss_10 = np.mean(returns < -10) * 100
    prob_gain_20 = np.mean(returns > 20) * 100
    print(f"\n  Probabilities:")
    print(f"    Profit (>0%):       {prob_profit:>6.1f}%")
    print(f"    Gain > 20%:         {prob_gain_20:>6.1f}%")
    print(f"    Loss > 10%:         {prob_loss_10:>6.1f}%")
    print("=" * 60)
