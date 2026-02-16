"""
CLI runner for forward Monte Carlo simulation.

Usage:
    python -m allocation_gym.simulation --n-paths 1000 --n-days 90

    # From backtest data (calibrate from backtest period, then simulate forward)
    python -m allocation_gym.simulation --from-backtest \
        --symbol BTC --data-source alpaca \
        --start 2025-10-15 --end 2026-02-15 --n-paths 1000 --n-days 90

    # Manual overrides (no data loading)
    python -m allocation_gym.simulation --mu 0.30 --sigma 0.65 \
        --initial-price 97000 --n-paths 5000 --n-days 180
"""

import argparse

from allocation_gym.simulation.config import SimulationConfig


def _load_backtest_data(symbol, start, end, data_source):
    """Load historical OHLCV for a symbol using the backtest data pipeline."""
    import pandas as pd
    from allocation_gym.credentials import get_alpaca_keys

    api_key, secret_key = get_alpaca_keys()

    if data_source == "alpaca" and api_key and secret_key:
        from datetime import datetime

        # Detect crypto vs stock
        is_crypto = "/" in symbol or symbol.upper() in {
            "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"
        }

        if is_crypto:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            client = CryptoHistoricalDataClient(api_key, secret_key)
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d"),
                timeframe=TimeFrame.Day,
            )
            df = client.get_crypto_bars(request).df
        else:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            client = StockHistoricalDataClient(api_key, secret_key)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=datetime.strptime(start, "%Y-%m-%d"),
                end=datetime.strptime(end, "%Y-%m-%d"),
                timeframe=TimeFrame.Day,
            )
            df = client.get_stock_bars(request).df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]]
    else:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)
        return df


def run(args=None):
    parser = argparse.ArgumentParser(
        description="Forward Monte Carlo simulation (GBM)"
    )
    parser.add_argument("--symbol", default="BTC/USD",
                        help="Symbol to simulate (default: BTC/USD)")
    parser.add_argument("--n-paths", type=int, default=1000,
                        help="Number of Monte Carlo paths (default: 1000)")
    parser.add_argument("--n-days", type=int, default=90,
                        help="Forward horizon in days (default: 90)")
    parser.add_argument("--calibration-days", type=int, default=90,
                        help="Trailing days for calibration (default: 90)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42, use -1 for random)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Override annualized drift")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Override annualized volatility")
    parser.add_argument("--initial-price", type=float, default=None,
                        help="Override initial price")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable visualization")

    # From-backtest mode
    parser.add_argument("--from-backtest", action="store_true",
                        help="Calibrate from a backtest date range")
    parser.add_argument("--data-source", choices=["yfinance", "alpaca"],
                        default="alpaca")
    parser.add_argument("--start", default=None,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="Backtest end date (YYYY-MM-DD)")

    parsed = parser.parse_args(args)

    seed = parsed.seed if parsed.seed >= 0 else None
    is_crypto = "/" in parsed.symbol or parsed.symbol.upper() in {"BTC", "ETH", "SOL"}
    trading_days = 365 if is_crypto else 252

    config = SimulationConfig(
        symbol=parsed.symbol,
        calibration_days=parsed.calibration_days,
        trading_days=trading_days,
        n_paths=parsed.n_paths,
        n_days_forward=parsed.n_days,
        seed=seed,
        mu_override=parsed.mu,
        sigma_override=parsed.sigma,
        initial_price=parsed.initial_price,
        no_plot=parsed.no_plot,
    )

    historical_df = None

    if parsed.from_backtest and parsed.start and parsed.end:
        # ── From-backtest mode: load exact date range ──
        print(f"\nLoading {parsed.symbol} from {parsed.start} to {parsed.end} ({parsed.data_source})...")
        historical_df = _load_backtest_data(
            parsed.symbol, parsed.start, parsed.end, parsed.data_source,
        )
        print(f"  Loaded {len(historical_df)} bars: {historical_df.index[0].date()} to {historical_df.index[-1].date()}")

        from allocation_gym.simulation.calibrate import calibrate_gbm
        cal = calibrate_gbm(
            opens=historical_df["Open"].values,
            highs=historical_df["High"].values,
            lows=historical_df["Low"].values,
            closes=historical_df["Close"].values,
            trading_days=trading_days,
        )

        mu = config.mu_override if config.mu_override is not None else cal.mu
        sigma = config.sigma_override if config.sigma_override is not None else cal.sigma
        initial_price = config.initial_price if config.initial_price is not None else cal.initial_price

        print(f"\n  Calibration ({cal.n_days_used} bars, {parsed.start} to {parsed.end}):")
        print(f"    Yang-Zhang Vol (ann): {cal.sigma:.1%}")
        print(f"    Log Drift (ann):      {cal.mu:+.1%}")
        print(f"    Regime:               {cal.variance_result.regime}")
        print(f"    Variance Ratio:       {cal.variance_result.variance_ratio:.3f}")
        print(f"    Efficiency Ratio:     {cal.variance_result.efficiency_ratio:.3f}")
        print(f"    Latest Price:         ${cal.initial_price:,.2f}")

    elif (config.mu_override is not None
          and config.sigma_override is not None
          and config.initial_price is not None):
        # ── Full manual mode ──
        mu = config.mu_override
        sigma = config.sigma_override
        initial_price = config.initial_price
        print(f"\nUsing manual parameters: mu={mu:.1%}, sigma={sigma:.1%}, S0=${initial_price:,.2f}")

    else:
        # ── Default: load trailing N days from Alpaca crypto ──
        print(f"\nLoading {config.calibration_days} days of {config.symbol} data from Alpaca...")
        from allocation_gym.simulation.data import load_btc_ohlcv
        df = load_btc_ohlcv(
            symbol=config.symbol,
            calibration_days=config.calibration_days,
        )
        historical_df = df
        print(f"  Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")

        from allocation_gym.simulation.calibrate import calibrate_gbm
        cal = calibrate_gbm(
            opens=df["Open"].values,
            highs=df["High"].values,
            lows=df["Low"].values,
            closes=df["Close"].values,
            trading_days=config.trading_days,
        )

        mu = config.mu_override if config.mu_override is not None else cal.mu
        sigma = config.sigma_override if config.sigma_override is not None else cal.sigma
        initial_price = config.initial_price if config.initial_price is not None else cal.initial_price

        print(f"\n  Calibration ({cal.n_days_used} bars):")
        print(f"    Yang-Zhang Vol (ann): {cal.sigma:.1%}")
        print(f"    Log Drift (ann):      {cal.mu:+.1%}")
        print(f"    Regime:               {cal.variance_result.regime}")
        print(f"    Variance Ratio:       {cal.variance_result.variance_ratio:.3f}")
        print(f"    Efficiency Ratio:     {cal.variance_result.efficiency_ratio:.3f}")
        print(f"    Latest Price:         ${cal.initial_price:,.2f}")

    # ── Run simulation ──
    from allocation_gym.simulation.engine import MonteCarloGBM

    print(f"\nSimulating {config.n_paths:,} paths x {config.n_days_forward} days...")
    mc = MonteCarloGBM(mu=mu, sigma=sigma, initial_price=initial_price)
    result = mc.simulate(
        n_paths=config.n_paths,
        n_days=config.n_days_forward,
        seed=config.seed,
    )
    stats = MonteCarloGBM.summary_stats(result, percentiles=config.percentiles)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print(f"  {config.symbol} FORWARD MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"  Model:          GBM (Geometric Brownian Motion)")
    print(f"  Paths:          {stats['n_paths']:,}")
    print(f"  Horizon:        {stats['n_days']} days")
    print(f"  Drift (mu):     {stats['mu']:+.2%} annualized")
    print(f"  Vol (sigma):    {stats['sigma']:.2%} annualized")
    print(f"  Initial Price:  ${stats['initial_price']:>12,.2f}")
    print("  " + "-" * 56)
    print(f"  Median Final:   ${stats['median_final']:>12,.2f}")
    print(f"  Mean Final:     ${stats['mean_final']:>12,.2f}")
    print(f"  Std Dev:        ${stats['std_final']:>12,.2f}")
    print(f"  Min Final:      ${stats['min_final']:>12,.2f}")
    print(f"  Max Final:      ${stats['max_final']:>12,.2f}")
    print("  " + "-" * 56)
    for p in config.percentiles:
        key = f"P{p}"
        if key in stats:
            print(f"  {key:>14}:   ${stats[key]:>12,.2f}")
    print("  " + "-" * 56)
    print(f"  Expected Return:    {stats['expected_return_pct']:>+8.1f}%  (median)")
    print(f"  Prob of Profit:     {stats['prob_above_initial'] * 100:>8.1f}%")
    print("=" * 60)

    # ── Plot ──
    if not config.no_plot:
        from allocation_gym.simulation.plotting import plot_simulation
        plot_simulation(
            stats=stats,
            result=result,
            symbol=config.symbol,
            historical_df=historical_df,
        )

    return stats


def main():
    run()


if __name__ == "__main__":
    main()
