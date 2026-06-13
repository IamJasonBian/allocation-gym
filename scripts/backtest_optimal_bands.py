#!/usr/bin/env python3
"""
Backtest Optimal Band Selection on BTC-USD and IWN.

Downloads historical data, calibrates OU parameters on a rolling window,
computes optimal entry/exit bands, simulates trades, and generates PDF
reports with price charts, buy/sell signals, and band overlays.
"""

import io
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocation_gym.optimal_bands import calibrate_ou, compute_optimal_bands, BandResult


# ---------------------------------------------------------------------------
# Data download — Twelve Data API (primary) + Yahoo Finance (fallback)
# ---------------------------------------------------------------------------

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# Map display symbols to Twelve Data symbols
TWELVE_DATA_SYMBOL_MAP = {
    "BTC-USD": "BTC",      # Grayscale Bitcoin Mini Trust ETF
    "IWN": "IWN",           # iShares Russell 2000 Value ETF
    "IWM": "IWM",
    "SPY": "SPY",
    "QQQ": "QQQ",
}


def download_twelve_data(symbol: str, outputsize: int = 800) -> List[Dict]:
    """Download daily OHLCV via Twelve Data time_series API."""
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY environment variable is not set. "
            "Export it before running this script."
        )
    td_symbol = TWELVE_DATA_SYMBOL_MAP.get(symbol, symbol)
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={td_symbol}&interval=1day&outputsize={outputsize}"
        f"&apikey={TWELVE_DATA_API_KEY}&format=JSON"
    )
    print(f"  Twelve Data: GET {td_symbol} ({outputsize} bars)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode("utf-8"))

    if data.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {data.get('message')}")
    if "values" not in data:
        raise RuntimeError(f"No 'values' in Twelve Data response for {td_symbol}")

    bars = []
    for point in data["values"]:
        bars.append({
            "date": point["datetime"],
            "open": float(point["open"]),
            "high": float(point["high"]),
            "low": float(point["low"]),
            "close": float(point["close"]),
            "volume": int(point.get("volume", 0)),
        })

    # Twelve Data returns newest first — reverse to chronological
    bars.reverse()
    return bars


def download_yahoo_csv(symbol: str, start: str, end: str) -> List[Dict]:
    """Download OHLCV data from Yahoo Finance via CSV API (fallback)."""
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())

    url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d"
           f"&events=history&includeAdjustedClose=true")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    text = resp.read().decode("utf-8")

    bars = []
    for i, line in enumerate(text.strip().split("\n")):
        if i == 0:
            continue  # header
        parts = line.split(",")
        if len(parts) < 5 or parts[1] == "null":
            continue
        bars.append({
            "date": parts[0],
            "open": float(parts[1]),
            "high": float(parts[2]),
            "low": float(parts[3]),
            "close": float(parts[4]),
            "volume": int(float(parts[5])) if parts[5] != "null" else 0,
        })
    return bars


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    side: str = "LONG"  # LONG or SHORT
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestState:
    position: int = 0       # +1 = long, -1 = short, 0 = flat
    entry_price: float = 0.0
    entry_date: str = ""
    trades: List[Trade] = field(default_factory=list)
    bands_history: List[Dict] = field(default_factory=list)


def run_backtest(
    bars: List[Dict],
    ou_lookback: int = 60,
    recalib_period: int = 21,
    rho: float = 0.05,
    c: float = 0.01,
    use_log_prices: bool = True,
) -> BacktestState:
    """Run optimal band selection backtest on OHLCV bars.

    Uses log-prices for OU calibration (more appropriate for price series),
    then maps bands back to price levels.
    """
    state = BacktestState()
    closes = [b["close"] for b in bars]

    bands: Optional[BandResult] = None
    entry_level = None
    exit_long_level = None
    exit_short_level = None

    for i, bar in enumerate(bars):
        price = bar["close"]
        date = bar["date"]

        # Recalibrate periodically
        if i >= ou_lookback and i % recalib_period == 0:
            window = closes[i - ou_lookback:i]
            if use_log_prices:
                window = [math.log(p) for p in window]
            window = np.array(window)

            try:
                kappa, theta, sigma = calibrate_ou(window)
                if kappa > 0 and sigma > 0:
                    bands = compute_optimal_bands(kappa, theta, sigma, rho, c)

                    if use_log_prices:
                        # Clamp band values to prevent overflow in exp()
                        max_log = math.log(price) + 1.0  # ~2.7x price
                        min_log = math.log(price) - 1.0  # ~0.37x price
                        el = min(max(bands.exit_long, min_log), max_log)
                        en = min(max(bands.entry_long, min_log), max_log)
                        es = min(max(bands.exit_short, min_log), max_log)

                        exit_long_level = math.exp(el)
                        entry_level = math.exp(en)
                        exit_short_level = math.exp(es)
                        theta_price = math.exp(min(max(theta, min_log), max_log))
                    else:
                        exit_long_level = bands.exit_long
                        entry_level = bands.entry_long
                        exit_short_level = bands.exit_short
                        theta_price = theta

                    # Sanity check: entry must be below exit
                    if entry_level < exit_long_level:
                        state.bands_history.append({
                            "bar_idx": i,
                            "date": date,
                            "price": price,
                            "kappa": kappa,
                            "theta": theta_price if use_log_prices else theta,
                            "sigma": sigma,
                            "entry_long": entry_level,
                            "exit_long": exit_long_level,
                            "exit_short": exit_short_level,
                        })
            except (ValueError, RuntimeError, OverflowError):
                pass

        if bands is None or entry_level is None:
            continue

        # Trading logic
        if state.position == 0:
            # Enter long when price drops to entry level
            if price <= entry_level:
                state.position = 1
                state.entry_price = price
                state.entry_date = date

            # Enter short when price rises above exit_long (overextended)
            elif price >= exit_long_level:
                state.position = -1
                state.entry_price = price
                state.entry_date = date

        elif state.position == 1:
            # Exit long when price hits exit level
            if price >= exit_long_level:
                pnl = price - state.entry_price
                pnl_pct = pnl / state.entry_price * 100
                state.trades.append(Trade(
                    entry_date=state.entry_date,
                    entry_price=state.entry_price,
                    exit_date=date,
                    exit_price=price,
                    side="LONG",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                state.position = 0

            # Stop loss: exit if price drops below exit_short (double the entry distance)
            elif price <= exit_short_level:
                pnl = price - state.entry_price
                pnl_pct = pnl / state.entry_price * 100
                state.trades.append(Trade(
                    entry_date=state.entry_date,
                    entry_price=state.entry_price,
                    exit_date=date,
                    exit_price=price,
                    side="LONG",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                state.position = 0

        elif state.position == -1:
            # Exit short when price drops to entry level
            if price <= entry_level:
                pnl = state.entry_price - price
                pnl_pct = pnl / state.entry_price * 100
                state.trades.append(Trade(
                    entry_date=state.entry_date,
                    entry_price=state.entry_price,
                    exit_date=date,
                    exit_price=price,
                    side="SHORT",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                state.position = 0

            # Stop loss for short
            elif price >= exit_long_level * 1.02:
                pnl = state.entry_price - price
                pnl_pct = pnl / state.entry_price * 100
                state.trades.append(Trade(
                    entry_date=state.entry_date,
                    entry_price=state.entry_price,
                    exit_date=date,
                    exit_price=price,
                    side="SHORT",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                ))
                state.position = 0

    return state


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------

def generate_pdf_report(
    symbol: str,
    bars: List[Dict],
    state: BacktestState,
    output_path: str,
):
    """Generate a PDF report with price chart, signals, and bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime as dt

    dates = [dt.strptime(b["date"], "%Y-%m-%d") for b in bars]
    closes = [b["close"] for b in bars]

    with PdfPages(output_path) as pdf:
        # --- Page 1: Price chart with buy/sell signals ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        height_ratios=[3, 1],
                                        gridspec_kw={"hspace": 0.3})

        # Price line
        ax1.plot(dates, closes, color="black", linewidth=0.8, label="Close", zorder=2)

        # Band overlays (interpolated between recalibration points)
        if state.bands_history:
            band_dates = [dt.strptime(b["date"], "%Y-%m-%d") for b in state.bands_history]
            entry_levels = [b["entry_long"] for b in state.bands_history]
            exit_levels = [b["exit_long"] for b in state.bands_history]
            exit_short_levels = [b["exit_short"] for b in state.bands_history]
            theta_levels = [b["theta"] for b in state.bands_history]

            ax1.plot(band_dates, entry_levels, "g--", linewidth=0.7, alpha=0.7, label="Entry Long Band")
            ax1.plot(band_dates, exit_levels, "r--", linewidth=0.7, alpha=0.7, label="Exit Long Band")
            ax1.plot(band_dates, exit_short_levels, "b--", linewidth=0.7, alpha=0.7, label="Exit Short Band")
            ax1.plot(band_dates, theta_levels, "gray", linewidth=0.5, alpha=0.5, label="θ (mean)")

            # Shade between bands
            ax1.fill_between(band_dates, entry_levels, exit_levels,
                           alpha=0.05, color="green", label="_nolegend_")

        # Buy/sell markers
        for trade in state.trades:
            entry_dt = dt.strptime(trade.entry_date, "%Y-%m-%d")
            exit_dt = dt.strptime(trade.exit_date, "%Y-%m-%d")

            if trade.side == "LONG":
                ax1.scatter(entry_dt, trade.entry_price, marker="^", color="green",
                          s=80, zorder=5, edgecolors="black", linewidth=0.5)
                ax1.scatter(exit_dt, trade.exit_price, marker="v", color="red",
                          s=80, zorder=5, edgecolors="black", linewidth=0.5)
            else:
                ax1.scatter(entry_dt, trade.entry_price, marker="v", color="orange",
                          s=80, zorder=5, edgecolors="black", linewidth=0.5)
                ax1.scatter(exit_dt, trade.exit_price, marker="^", color="blue",
                          s=80, zorder=5, edgecolors="black", linewidth=0.5)

        ax1.set_title(f"{symbol} — Optimal Band Selection Backtest", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Volume bars
        volumes = [b["volume"] for b in bars]
        ax2.bar(dates, volumes, width=1, color="steelblue", alpha=0.4)
        ax2.set_ylabel("Volume")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Trade-level PnL and band parameters ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Trade PnL histogram
        ax = axes[0, 0]
        if state.trades:
            pnls = [t.pnl_pct for t in state.trades]
            colors = ["green" if p > 0 else "red" for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_xlabel("Trade #")
            ax.set_ylabel("PnL (%)")
            ax.set_title("Per-Trade PnL (%)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14)
            ax.set_title("Per-Trade PnL (%)")

        # Cumulative PnL
        ax = axes[0, 1]
        if state.trades:
            cum_pnl = np.cumsum([t.pnl_pct for t in state.trades])
            ax.plot(cum_pnl, color="blue", linewidth=1.5)
            ax.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.1, color="blue")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Cumulative PnL (%)")
            ax.set_title("Cumulative PnL (%)")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=14)
            ax.set_title("Cumulative PnL (%)")

        # OU parameters over time
        ax = axes[1, 0]
        if state.bands_history:
            bdates = [dt.strptime(b["date"], "%Y-%m-%d") for b in state.bands_history]
            kappas = [b["kappa"] for b in state.bands_history]
            ax.plot(bdates, kappas, color="purple", linewidth=1)
            ax.set_ylabel("κ (mean-reversion speed)")
            ax.set_title("OU κ Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            sigmas = [b["sigma"] for b in state.bands_history]
            ax2.plot(bdates, sigmas, color="orange", linewidth=1, alpha=0.7)
            ax2.set_ylabel("σ (volatility)", color="orange")

        # Summary statistics table
        ax = axes[1, 1]
        ax.axis("off")
        if state.trades:
            n_trades = len(state.trades)
            winners = sum(1 for t in state.trades if t.pnl > 0)
            losers = sum(1 for t in state.trades if t.pnl <= 0)
            win_rate = winners / n_trades * 100 if n_trades > 0 else 0
            total_pnl = sum(t.pnl_pct for t in state.trades)
            avg_pnl = total_pnl / n_trades if n_trades > 0 else 0
            max_win = max(t.pnl_pct for t in state.trades)
            max_loss = min(t.pnl_pct for t in state.trades)
            avg_win = (sum(t.pnl_pct for t in state.trades if t.pnl > 0) / winners
                      if winners > 0 else 0)
            avg_loss = (sum(t.pnl_pct for t in state.trades if t.pnl <= 0) / losers
                       if losers > 0 else 0)
            n_long = sum(1 for t in state.trades if t.side == "LONG")
            n_short = sum(1 for t in state.trades if t.side == "SHORT")

            table_data = [
                ["Total Trades", f"{n_trades}"],
                ["Long / Short", f"{n_long} / {n_short}"],
                ["Winners / Losers", f"{winners} / {losers}"],
                ["Win Rate", f"{win_rate:.1f}%"],
                ["Total PnL", f"{total_pnl:.2f}%"],
                ["Avg PnL per Trade", f"{avg_pnl:.2f}%"],
                ["Best Trade", f"{max_win:.2f}%"],
                ["Worst Trade", f"{max_loss:.2f}%"],
                ["Avg Win", f"{avg_win:.2f}%"],
                ["Avg Loss", f"{avg_loss:.2f}%"],
            ]

            table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                           loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#4472C4")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#D9E2F3")
            ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)
        else:
            ax.text(0.5, 0.5, "No trades to summarize", ha="center", va="center")

        fig.suptitle(f"{symbol} — Optimal Band Selection Report", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: Band width and trade annotations ---
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        if state.bands_history:
            bdates = [dt.strptime(b["date"], "%Y-%m-%d") for b in state.bands_history]
            bandwidths = [b["exit_long"] - b["entry_long"] for b in state.bands_history]
            bw_pct = [(b["exit_long"] - b["entry_long"]) / b["price"] * 100
                     for b in state.bands_history]

            ax.plot(bdates, bw_pct, color="teal", linewidth=1.5)
            ax.fill_between(bdates, bw_pct, alpha=0.1, color="teal")
            ax.set_ylabel("Band Width (% of price)")
            ax.set_title(f"{symbol} — Optimal Band Width Over Time")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  PDF saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent.parent / "docs" / "optimal_bands"
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = {
        "BTC-USD": {"start": "2023-01-01", "end": "2026-03-14", "name": "Bitcoin"},
        "IWN": {"start": "2023-01-01", "end": "2026-03-14", "name": "iShares Russell 2000 Value ETF"},
    }

    for symbol, config in symbols.items():
        print(f"\n{'='*60}")
        print(f"  {config['name']} ({symbol})")
        print(f"{'='*60}")

        print(f"  Downloading {symbol} data...")
        bars = None

        # Try Twelve Data API first (real market data)
        try:
            bars = download_twelve_data(symbol, outputsize=800)
            print(f"  Twelve Data: OK")
        except Exception as e:
            print(f"  Twelve Data failed: {e}")

        # Fallback to Yahoo Finance
        if not bars:
            try:
                bars = download_yahoo_csv(symbol, config["start"], config["end"])
                print(f"  Yahoo Finance: OK")
            except Exception as e:
                print(f"  Yahoo Finance failed: {e}")

        # Final fallback to synthetic data
        if not bars:
            print(f"  Using synthetic data for {symbol}...")
            bars = _generate_synthetic_bars(symbol, config["start"], config["end"])

        print(f"  Got {len(bars)} bars ({bars[0]['date']} to {bars[-1]['date']})")

        print(f"  Running backtest...")
        state = run_backtest(
            bars,
            ou_lookback=60,
            recalib_period=21,
            rho=0.05,
            c=0.01,
        )

        print(f"  Trades: {len(state.trades)}")
        if state.trades:
            total_pnl = sum(t.pnl_pct for t in state.trades)
            winners = sum(1 for t in state.trades if t.pnl > 0)
            print(f"  Total PnL: {total_pnl:.2f}%")
            print(f"  Win rate: {winners}/{len(state.trades)} "
                  f"({winners/len(state.trades)*100:.1f}%)")

        safe_name = symbol.replace("/", "_").replace("-", "_")
        pdf_path = output_dir / f"{safe_name}_optimal_bands.pdf"
        print(f"  Generating PDF...")
        generate_pdf_report(symbol, bars, state, str(pdf_path))

    print(f"\nDone! Reports saved to {output_dir}/")


def _generate_synthetic_bars(symbol: str, start: str, end: str) -> List[Dict]:
    """Generate synthetic price data when download fails."""
    import pandas as pd

    np.random.seed(hash(symbol) % 2**31)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    # Base prices
    base = {"BTC-USD": 25000.0, "IWN": 140.0}.get(symbol, 100.0)
    vol = {"BTC-USD": 0.04, "IWN": 0.015}.get(symbol, 0.02)

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Mean-reverting component + drift
    returns = np.random.normal(0.0003, vol, n)
    # Add mean reversion
    x = np.zeros(n)
    x[0] = 0
    for i in range(1, n):
        x[i] = x[i-1] + 0.05 * (0 - x[i-1]) + vol * np.random.randn()

    prices = base * np.exp(np.cumsum(returns) + x * 0.1)

    bars = []
    for i, date in enumerate(dates):
        close = prices[i]
        open_ = close * (1 + np.random.normal(0, 0.003))
        high = max(open_, close) * (1 + abs(np.random.normal(0, 0.008)))
        low = min(open_, close) * (1 - abs(np.random.normal(0, 0.008)))
        bars.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(np.random.randint(1_000_000, 10_000_000)),
        })

    return bars


if __name__ == "__main__":
    main()
