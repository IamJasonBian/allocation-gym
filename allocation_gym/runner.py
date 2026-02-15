"""
CLI runner for allocation-gym backtests.

Usage:
    python -m allocation_gym.runner --strategy momentum --symbols SPY QQQ \
        --start 2020-01-01 --end 2024-01-01

    # BTC-heavy Kelly with 50% BTC floor
    python -m allocation_gym.runner --strategy variance_kelly \
        --symbols BTC/USD SPY GLD QQQ --min-weight BTC/USD=0.50 \
        --data-source alpaca --start 2025-02-15 --end 2026-02-15

    # With Alpaca data
    python -m allocation_gym.runner --strategy momentum --symbols SPY \
        --data-source alpaca --start 2024-02-01 --end 2025-02-01
"""

import argparse
import os
from datetime import datetime

import backtrader as bt

from allocation_gym.config import BacktestConfig
from allocation_gym.strategies.momentum import MomentumStrategy
from allocation_gym.strategies.mean_reversion import MeanReversionStrategy
from allocation_gym.strategies.variance_kelly import VarianceKellyStrategy
from allocation_gym.sizers.kelly import KellySizer
from allocation_gym.analyzers.performance import PerformanceAnalyzer


STRATEGY_MAP = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "variance_kelly": VarianceKellyStrategy,
}

CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD", "AVAX/USD", "SOL/USD"}


def _is_crypto(symbol):
    return symbol.upper() in CRYPTO_SYMBOLS or "/USD" in symbol.upper()


def _load_alpaca_crypto(symbol, start, end, api_key, secret_key):
    """Fetch historical crypto bars from Alpaca."""
    import pandas as pd
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
    bars = client.get_crypto_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return bt.feeds.PandasData(dataname=df)


def _load_alpaca_stock(symbol, start, end, api_key, secret_key):
    """Fetch historical stock bars from Alpaca."""
    import pandas as pd
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
    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return bt.feeds.PandasData(dataname=df)


def _load_alpaca_data(symbol, start, end, api_key, secret_key):
    if _is_crypto(symbol):
        return _load_alpaca_crypto(symbol, start, end, api_key, secret_key)
    return _load_alpaca_stock(symbol, start, end, api_key, secret_key)


def _load_yfinance_data(symbol, start, end):
    """Fetch data from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} from Yahoo Finance")
    if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return bt.feeds.PandasData(dataname=df)


def _parse_min_weights(raw):
    """Parse 'SYM=0.5,SYM2=0.3' into dict."""
    if not raw:
        return {}
    weights = {}
    for pair in raw:
        sym, val = pair.split("=")
        weights[sym.strip()] = float(val.strip())
    return weights


def build_cerebro(args, config: BacktestConfig) -> bt.Cerebro:
    cerebro = bt.Cerebro()

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    for symbol in args.symbols:
        if args.data_source == "alpaca" and api_key and secret_key:
            data = _load_alpaca_data(symbol, args.start, args.end, api_key, secret_key)
        else:
            data = _load_yfinance_data(symbol, args.start, args.end)
        cerebro.adddata(data, name=symbol)

    strategy_cls = STRATEGY_MAP[args.strategy]
    strategy_params = {
        "variance_lookback": config.variance_lookback,
        "vr_k": config.vr_k,
        "trading_days": config.trading_days,
    }

    if args.strategy == "variance_kelly":
        exp_ret = {}
        for s in args.symbols:
            if _is_crypto(s):
                exp_ret[s] = 0.40  # higher expected return for crypto
            else:
                exp_ret[s] = 0.10
        strategy_params["expected_returns"] = exp_ret
        strategy_params["min_weights"] = _parse_min_weights(args.min_weight)

    cerebro.addstrategy(strategy_cls, **strategy_params)

    if args.strategy != "variance_kelly":
        cerebro.addsizer(
            KellySizer,
            kelly_fraction=config.kelly_fraction,
            risk_free_rate=config.risk_free_rate,
            risk_per_trade_pct=config.risk_per_trade_pct,
        )

    cerebro.addanalyzer(
        PerformanceAnalyzer,
        risk_free_rate=config.risk_free_rate,
        trading_days=config.trading_days,
    )

    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_pct)

    return cerebro


def run(args=None):
    parser = argparse.ArgumentParser(description="allocation-gym backtester")
    parser.add_argument("--strategy", required=True, choices=STRATEGY_MAP.keys())
    parser.add_argument("--symbols", nargs="+", default=["SPY"])
    parser.add_argument("--min-weight", nargs="*", default=[],
                        help="Min weight constraints, e.g. BTC/USD=0.50")
    parser.add_argument("--data-source", choices=["yfinance", "alpaca"], default="yfinance")
    parser.add_argument("--start", default="2024-02-15")
    parser.add_argument("--end", default="2025-02-15")
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args(args)
    config = BacktestConfig(initial_cash=args.cash)

    min_weights = _parse_min_weights(args.min_weight)

    print(f"\nLoading data for {args.symbols} ({args.data_source})...")
    if min_weights:
        print(f"  Min weight constraints: {min_weights}")
    cerebro = build_cerebro(args, config)

    print(f"Running {args.strategy} backtest...")
    results = cerebro.run()

    perf = results[0].analyzers.performanceanalyzer.get_analysis()

    print("\n" + "=" * 60)
    print(f"  Strategy:   {args.strategy}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    if min_weights:
        for s, w in min_weights.items():
            print(f"  Min Weight: {s} >= {w*100:.0f}%")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Initial:    ${config.initial_cash:,.0f}")
    print("=" * 60)
    for k, v in perf.items():
        label = k.replace("_", " ").title()
        if "pct" in k.lower():
            print(f"  {label:.<35} {v}%")
        elif "value" in k.lower():
            print(f"  {label:.<35} ${v:,.2f}")
        else:
            print(f"  {label:.<35} {v}")
    print("=" * 60)

    if args.plot:
        cerebro.plot(style="candlestick")

    return perf


def main():
    run()


if __name__ == "__main__":
    main()
