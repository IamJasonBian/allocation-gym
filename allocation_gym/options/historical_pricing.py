"""
BTC + IWM historical options pricing analysis.

Computes rolling realized vol and prices ATM/OTM calls and puts at every
historical point over the last 6 months. Shows how option premiums evolved
leading into and through the BTC drawdown, and compares BTC vs IWM vol
regimes and pricing.

Usage:
    python -m allocation_gym.options.historical_pricing
    python -m allocation_gym.options.historical_pricing --days 180
"""

import argparse
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from allocation_gym.options.black_scholes import bs_call_price, bs_put_price
from allocation_gym.credentials import get_alpaca_keys


# ── Data ─────────────────────────────────────────────────────────────────


def load_btc(days: int = 200) -> pd.DataFrame:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = get_alpaca_keys()
    end = datetime.now(tz=None)
    start = end - timedelta(days=int(days * 1.5))
    client = CryptoHistoricalDataClient(api_key, secret_key)
    request = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD", start=start, end=end,
        timeframe=TimeFrame.Day,
    )
    df = client.get_crypto_bars(request).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level="symbol")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)


def load_iwm(days: int = 200) -> pd.DataFrame:
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    df = yf.Ticker("IWM").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].tail(days)


# ── Vol ──────────────────────────────────────────────────────────────────


def rolling_yang_zhang(df: pd.DataFrame, window: int = 14,
                       trading_days: int = 252) -> pd.Series:
    """Rolling Yang-Zhang vol."""
    log_oc = np.log(df["Close"] / df["Open"])
    log_co = np.log(df["Open"] / df["Close"].shift(1))
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)

    close_var = log_co.rolling(window).var()
    open_var = log_oc.rolling(window).var()
    rs_mean = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = close_var + k * open_var + (1 - k) * rs_mean
    return np.sqrt(yz_var.clip(lower=0) * trading_days)


def rolling_close_vol(df: pd.DataFrame, window: int = 21,
                      trading_days: int = 252) -> pd.Series:
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(trading_days)


# ── Historical options pricing ───────────────────────────────────────────


def price_options_history(
    df: pd.DataFrame,
    vol_series: pd.Series,
    expiry_days: int = 30,
    r: float = 0.045,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    At each date, price ATM and OTM calls/puts using the rolling vol
    as the IV proxy. Returns a DataFrame with option prices over time.
    """
    T = expiry_days / trading_days
    records = []

    for i in range(len(df)):
        date = df.index[i]
        S = float(df["Close"].iloc[i])
        iv = float(vol_series.iloc[i]) if not np.isnan(vol_series.iloc[i]) else None

        if iv is None or iv <= 0 or S <= 0:
            continue

        rec = {"date": date, "spot": S, "iv": iv}

        for otm_label, otm_pct in [("ATM", 0.0), ("2%OTM", 0.02), ("5%OTM", 0.05), ("10%OTM", 0.10)]:
            K_call = S * (1 + otm_pct)
            K_put = S * (1 - otm_pct)

            c = bs_call_price(S, K_call, T, r, iv)
            p = bs_put_price(S, K_put, T, r, iv)

            # As percentage of spot
            rec[f"call_{otm_label}"] = c
            rec[f"call_{otm_label}_pct"] = c / S * 100
            rec[f"put_{otm_label}"] = p
            rec[f"put_{otm_label}_pct"] = p / S * 100

        records.append(rec)

    return pd.DataFrame(records).set_index("date")


# ── Main ─────────────────────────────────────────────────────────────────


def run_historical_pricing(days: int = 180):
    print("=" * 70)
    print("BTC + IWM HISTORICAL OPTIONS PRICING")
    print("=" * 70)

    print("\nLoading data...")
    btc = load_btc(days=days)
    iwm = load_iwm(days=days)
    print(f"  BTC: {len(btc)} bars, {btc.index[0].date()} to {btc.index[-1].date()}")
    print(f"  IWM: {len(iwm)} bars, {iwm.index[0].date()} to {iwm.index[-1].date()}")

    # Compute rolling vols
    btc_yz14 = rolling_yang_zhang(btc, window=14, trading_days=365)
    btc_cc21 = rolling_close_vol(btc, window=21, trading_days=365)
    iwm_yz14 = rolling_yang_zhang(iwm, window=14, trading_days=252)
    iwm_cc21 = rolling_close_vol(iwm, window=21, trading_days=252)

    # Price options history
    print("\nPricing options at each historical date...")
    btc_opts = price_options_history(btc, btc_yz14, expiry_days=30, trading_days=365)
    iwm_opts = price_options_history(iwm, iwm_yz14, expiry_days=30, trading_days=252)

    # ── Print summary table ──────────────────────────────────────────
    # Current vs 1 month ago vs 3 months ago
    print(f"\n{'=' * 70}")
    print("OPTIONS PRICING SNAPSHOTS")
    print(f"{'=' * 70}")

    for label, opts, asset_df, yz in [
        ("BTC/USD", btc_opts, btc, btc_yz14),
        ("IWM", iwm_opts, iwm, iwm_yz14),
    ]:
        print(f"\n  {label}:")
        print(f"  {'':>12s}  {'Now':>12s}  {'1mo ago':>12s}  {'3mo ago':>12s}  {'6mo ago':>12s}")
        print(f"  {'─' * 64}")

        now_idx = -1
        mo1_idx = _find_nearest_idx(opts, opts.index[-1] - pd.Timedelta(days=30))
        mo3_idx = _find_nearest_idx(opts, opts.index[-1] - pd.Timedelta(days=90))
        mo6_idx = 0

        for row_label, col in [
            ("Spot", "spot"),
            ("YZ Vol", "iv"),
            ("ATM Call $", "call_ATM"),
            ("ATM Call %", "call_ATM_pct"),
            ("5%OTM Call $", "call_5%OTM"),
            ("5%OTM Call %", "call_5%OTM_pct"),
            ("ATM Put $", "put_ATM"),
            ("ATM Put %", "put_ATM_pct"),
            ("5%OTM Put $", "put_5%OTM"),
            ("5%OTM Put %", "put_5%OTM_pct"),
        ]:
            vals = []
            for idx in [now_idx, mo1_idx, mo3_idx, mo6_idx]:
                v = opts.iloc[idx][col] if idx < len(opts) else float("nan")
                vals.append(v)

            if "pct" in col or col == "iv":
                fmt = lambda v: f"{v:.2f}%" if not np.isnan(v) else "N/A"
            elif col == "spot":
                fmt = lambda v: f"${v:,.0f}" if v > 100 else f"${v:.2f}"
            else:
                fmt = lambda v: f"${v:,.2f}" if not np.isnan(v) else "N/A"

            print(f"  {row_label:>12s}  {fmt(vals[0]):>12s}  {fmt(vals[1]):>12s}  {fmt(vals[2]):>12s}  {fmt(vals[3]):>12s}")

    # ── Vol comparison stats ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VOLATILITY REGIME COMPARISON")
    print(f"{'=' * 70}")

    for label, yz, cc in [
        ("BTC/USD", btc_yz14, btc_cc21),
        ("IWM", iwm_yz14, iwm_cc21),
    ]:
        yz_clean = yz.dropna()
        cc_clean = cc.dropna()
        print(f"\n  {label}:")
        print(f"    YZ(14d) now:    {yz_clean.iloc[-1]:.1%}")
        print(f"    YZ(14d) mean:   {yz_clean.mean():.1%}")
        print(f"    YZ(14d) min:    {yz_clean.min():.1%} ({yz_clean.idxmin().date()})")
        print(f"    YZ(14d) max:    {yz_clean.max():.1%} ({yz_clean.idxmax().date()})")
        print(f"    CC(21d) now:    {cc_clean.iloc[-1]:.1%}")
        print(f"    Percentile now: {_percentile_rank(yz_clean, yz_clean.iloc[-1]):.0f}th")

    # ── BTC vs IWM vol ratio ─────────────────────────────────────────
    # Align dates
    common = btc_yz14.dropna().index.intersection(iwm_yz14.dropna().index)
    if len(common) > 0:
        ratio = btc_yz14.loc[common] / iwm_yz14.loc[common]
        print(f"\n  BTC/IWM vol ratio:")
        print(f"    Now:    {ratio.iloc[-1]:.2f}x")
        print(f"    Mean:   {ratio.mean():.2f}x")
        print(f"    Min:    {ratio.min():.2f}x")
        print(f"    Max:    {ratio.max():.2f}x")

    # ── Key observations ─────────────────────────────────────────────
    btc_vol_now = btc_yz14.dropna().iloc[-1]
    iwm_vol_now = iwm_yz14.dropna().iloc[-1]
    btc_vol_3mo = btc_yz14.dropna().iloc[_find_nearest_idx(btc_yz14.dropna(), btc_yz14.dropna().index[-1] - pd.Timedelta(days=90))]
    iwm_vol_3mo = iwm_yz14.dropna().iloc[_find_nearest_idx(iwm_yz14.dropna(), iwm_yz14.dropna().index[-1] - pd.Timedelta(days=90))]

    print(f"\n{'=' * 70}")
    print("KEY OBSERVATIONS")
    print(f"{'=' * 70}")
    print(f"\n  BTC vol {btc_vol_3mo:.0%} -> {btc_vol_now:.0%} ({(btc_vol_now-btc_vol_3mo)*100:+.0f}pp over 3mo)")
    print(f"  IWM vol {iwm_vol_3mo:.0%} -> {iwm_vol_now:.0%} ({(iwm_vol_now-iwm_vol_3mo)*100:+.0f}pp over 3mo)")

    btc_atm_now = btc_opts["call_ATM_pct"].iloc[-1]
    btc_atm_3mo = btc_opts["call_ATM_pct"].iloc[_find_nearest_idx(btc_opts, btc_opts.index[-1] - pd.Timedelta(days=90))]
    iwm_atm_now = iwm_opts["call_ATM_pct"].iloc[-1]
    iwm_atm_3mo = iwm_opts["call_ATM_pct"].iloc[_find_nearest_idx(iwm_opts, iwm_opts.index[-1] - pd.Timedelta(days=90))]

    print(f"\n  BTC ATM 30d call: {btc_atm_3mo:.2f}% -> {btc_atm_now:.2f}% of spot")
    print(f"  IWM ATM 30d call: {iwm_atm_3mo:.2f}% -> {iwm_atm_now:.2f}% of spot")

    if btc_vol_now > btc_vol_3mo:
        print(f"\n  BTC options are significantly MORE expensive than 3 months ago")
        print(f"  -> Selling premium (covered calls / collars) captures rich vol")
    else:
        print(f"\n  BTC options have cheapened vs 3 months ago")
        print(f"  -> Better entry for buying protective puts")

    print()

    # ── Plot ─────────────────────────────────────────────────────────
    print("Generating plot...")
    _plot_historical(btc, iwm, btc_opts, iwm_opts,
                     btc_yz14, btc_cc21, iwm_yz14, iwm_cc21)


def _find_nearest_idx(series_or_df, target_date):
    """Find the index position nearest to target_date."""
    idx = series_or_df.index
    if len(idx) == 0:
        return 0
    diffs = abs(idx - target_date)
    return int(np.argmin(diffs))


def _percentile_rank(series: pd.Series, value: float) -> float:
    return float(np.sum(series <= value) / len(series) * 100)


# ── Plot ─────────────────────────────────────────────────────────────────


def _plot_historical(btc, iwm, btc_opts, iwm_opts,
                     btc_yz14, btc_cc21, iwm_yz14, iwm_cc21):
    plt.close("all")
    fig = plt.figure(figsize=(18, 20))

    # 6 panels: 3 rows x 2 columns
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    C_BTC = "#F7931A"
    C_IWM = "#4CAF50"
    C_VOL = "#9C27B0"
    C_CALL = "#2196F3"
    C_PUT = "#F44336"

    fig.suptitle(
        "BTC vs IWM: Historical Options Pricing & Volatility",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # ── Row 1: Price ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(btc.index, btc["Close"], color=C_BTC, linewidth=1.5)
    ax.set_ylabel("BTC/USD ($)", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("BTC/USD Price", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    # Annotate high/low
    hi_idx = btc["Close"].idxmax()
    lo_idx = btc["Close"].idxmin()
    ax.annotate(f"${btc['Close'].max():,.0f}", xy=(hi_idx, btc["Close"].max()),
                fontsize=8, color=C_BTC, ha="center", va="bottom")
    ax.annotate(f"${btc['Close'].min():,.0f}", xy=(lo_idx, btc["Close"].min()),
                fontsize=8, color="red", ha="center", va="top")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(iwm.index, iwm["Close"], color=C_IWM, linewidth=1.5)
    ax.set_ylabel("IWM ($)", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.set_title("IWM Price", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    hi_idx = iwm["Close"].idxmax()
    lo_idx = iwm["Close"].idxmin()
    ax.annotate(f"${iwm['Close'].max():.2f}", xy=(hi_idx, iwm["Close"].max()),
                fontsize=8, color=C_IWM, ha="center", va="bottom")
    ax.annotate(f"${iwm['Close'].min():.2f}", xy=(lo_idx, iwm["Close"].min()),
                fontsize=8, color="red", ha="center", va="top")

    # ── Row 2: Volatility ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(btc_yz14.index, btc_yz14 * 100, color=C_VOL, linewidth=1.5, label="YZ 14d")
    ax.plot(btc_cc21.index, btc_cc21 * 100, color=C_BTC, linewidth=1, alpha=0.6, label="CC 21d")
    ax.fill_between(btc_yz14.index, btc_yz14 * 100, alpha=0.1, color=C_VOL)
    ax.set_ylabel("Ann. Vol (%)", fontsize=10)
    ax.set_title("BTC Realized Volatility", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    # Current vol annotation
    if not btc_yz14.dropna().empty:
        v = btc_yz14.dropna().iloc[-1] * 100
        ax.annotate(f"Now: {v:.0f}%", xy=(btc_yz14.dropna().index[-1], v),
                    fontsize=9, fontweight="bold", color=C_VOL,
                    xytext=(-60, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=C_VOL))

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(iwm_yz14.index, iwm_yz14 * 100, color=C_VOL, linewidth=1.5, label="YZ 14d")
    ax.plot(iwm_cc21.index, iwm_cc21 * 100, color=C_IWM, linewidth=1, alpha=0.6, label="CC 21d")
    ax.fill_between(iwm_yz14.index, iwm_yz14 * 100, alpha=0.1, color=C_VOL)
    ax.set_ylabel("Ann. Vol (%)", fontsize=10)
    ax.set_title("IWM Realized Volatility", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    if not iwm_yz14.dropna().empty:
        v = iwm_yz14.dropna().iloc[-1] * 100
        ax.annotate(f"Now: {v:.0f}%", xy=(iwm_yz14.dropna().index[-1], v),
                    fontsize=9, fontweight="bold", color=C_VOL,
                    xytext=(-60, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=C_VOL))

    # ── Row 3: ATM Call/Put as % of spot ─────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(btc_opts.index, btc_opts["call_ATM_pct"], color=C_CALL, linewidth=1.5, label="ATM Call")
    ax.plot(btc_opts.index, btc_opts["call_5%OTM_pct"], color=C_CALL, linewidth=1, alpha=0.5, linestyle="--", label="5% OTM Call")
    ax.plot(btc_opts.index, btc_opts["put_ATM_pct"], color=C_PUT, linewidth=1.5, label="ATM Put")
    ax.plot(btc_opts.index, btc_opts["put_5%OTM_pct"], color=C_PUT, linewidth=1, alpha=0.5, linestyle="--", label="5% OTM Put")
    ax.set_ylabel("Premium (% of spot)", fontsize=10)
    ax.set_title("BTC 30d Options as % of Spot", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, linestyle="--")

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(iwm_opts.index, iwm_opts["call_ATM_pct"], color=C_CALL, linewidth=1.5, label="ATM Call")
    ax.plot(iwm_opts.index, iwm_opts["call_5%OTM_pct"], color=C_CALL, linewidth=1, alpha=0.5, linestyle="--", label="5% OTM Call")
    ax.plot(iwm_opts.index, iwm_opts["put_ATM_pct"], color=C_PUT, linewidth=1.5, label="ATM Put")
    ax.plot(iwm_opts.index, iwm_opts["put_5%OTM_pct"], color=C_PUT, linewidth=1, alpha=0.5, linestyle="--", label="5% OTM Put")
    ax.set_ylabel("Premium (% of spot)", fontsize=10)
    ax.set_title("IWM 30d Options as % of Spot", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── Row 4: ATM Call $ and vol ratio ──────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(btc_opts.index, btc_opts["call_ATM"], color=C_BTC, linewidth=1.5, label="BTC ATM Call ($)")
    ax.set_ylabel("BTC Call Premium ($)", fontsize=10, color=C_BTC)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(axis="y", labelcolor=C_BTC)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("ATM Call Premium (absolute $)", fontsize=11, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(iwm_opts.index, iwm_opts["call_ATM"], color=C_IWM, linewidth=1.5, label="IWM ATM Call ($)")
    ax2.set_ylabel("IWM Call Premium ($)", fontsize=10, color=C_IWM)
    ax2.tick_params(axis="y", labelcolor=C_IWM)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # BTC/IWM vol ratio
    ax = fig.add_subplot(gs[3, 1])
    common = btc_yz14.dropna().index.intersection(iwm_yz14.dropna().index)
    if len(common) > 5:
        ratio = btc_yz14.loc[common] / iwm_yz14.loc[common]
        ax.plot(common, ratio, color="#FF5722", linewidth=1.5)
        ax.fill_between(common, ratio, ratio.mean(), alpha=0.1, color="#FF5722")
        ax.axhline(ratio.mean(), color="grey", linewidth=1, linestyle="--", alpha=0.6,
                    label=f"Mean: {ratio.mean():.1f}x")
        ax.annotate(f"Now: {ratio.iloc[-1]:.1f}x",
                    xy=(common[-1], ratio.iloc[-1]),
                    fontsize=9, fontweight="bold", color="#FF5722",
                    xytext=(-60, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="#FF5722"))
    ax.set_ylabel("BTC Vol / IWM Vol", fontsize=10)
    ax.set_title("BTC / IWM Volatility Ratio", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Format all x-axes
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "btc_iwm_historical_pricing.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    plt.show(block=True)


def parse_args():
    parser = argparse.ArgumentParser(description="BTC + IWM Historical Options Pricing")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default: 180)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_historical_pricing(days=args.days)
