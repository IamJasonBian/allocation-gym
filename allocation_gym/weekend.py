"""
BTC weekend analysis — price with weekend overlays + weekend implied vol.

Usage:
    python -m allocation_gym.weekend [--days 365] [--vol-window 21]
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from allocation_gym.credentials import get_alpaca_keys


def load_btc_hourly(days: int = 60) -> pd.DataFrame:
    """Fetch hourly BTC/USD bars from Alpaca."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = get_alpaca_keys()
    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca credentials required. Set ALPACA_API_KEY / ALPACA_SECRET_KEY."
        )

    end = datetime.utcnow()
    start = end - timedelta(days=int(days * 1.5))

    client = CryptoHistoricalDataClient(api_key, secret_key)
    request = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        start=start,
        end=end,
        timeframe=TimeFrame.Hour,
    )
    bars = client.get_crypto_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level="symbol")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Keep only the last N days of hourly bars
    cutoff = end - timedelta(days=days)
    df = df[df.index >= cutoff]
    return df


def _classify_hours(idx: pd.DatetimeIndex) -> pd.Series:
    """
    Classify each hourly bar into market regime.

    Returns Series with values: 'market', 'afterhours', 'weekend'.
    Uses ET (UTC-5 approx) for market hours: Mon-Fri 9 AM - 5 PM ET.
    Data index is assumed UTC (tz-naive).

    Weekend = Fri after 5 PM ET through Mon 9 AM ET (includes Sat/Sun
    plus Friday post-market and Monday pre-market).
    Afterhours = Mon-Thu outside 9 AM - 5 PM ET.
    """
    # Approximate ET as UTC-5 (ignores DST, close enough)
    et_hour = (idx.hour - 5) % 24
    et_dow = idx.dayofweek

    # Adjust: if UTC hour < 5, the ET date is previous day
    et_date_shift = idx.hour < 5
    et_dow_adj = et_dow.where(~et_date_shift, (et_dow - 1) % 7)

    # Weekend: Sat(5), Sun(6), Fri(4) after 5PM, Mon(0) before 9AM
    is_sat_sun = et_dow_adj.isin([5, 6])
    is_fri_evening = (et_dow_adj == 4) & (et_hour >= 17)
    is_mon_premarket = (et_dow_adj == 0) & (et_hour < 9)
    is_weekend = is_sat_sun | is_fri_evening | is_mon_premarket

    is_market = (~is_weekend) & (et_hour >= 9) & (et_hour < 17)

    labels = pd.Series("afterhours", index=idx)
    labels[is_weekend] = "weekend"
    labels[is_market] = "market"
    return labels


def compute_weekend_vol(df: pd.DataFrame, window_hours: int = 168) -> pd.DataFrame:
    """
    Compute rolling annualized vol for market hours vs off-hours.

    Three regimes: market (Mon-Fri 9AM-5PM ET), afterhours, weekend.
    Vol is split: market hours vs off-hours (afterhours + weekend combined).

    window_hours: rolling lookback in hours (default 168 = 7 days).
    """
    df = df.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["regime"] = _classify_hours(df.index)
    df["is_weekend"] = df["regime"] == "weekend"
    df["is_afterhours"] = df["regime"] == "afterhours"
    df["is_offhours"] = df["regime"] != "market"

    # Annualize from hourly: sqrt(24 * 365)
    annualize = np.sqrt(24 * 365)

    weekend_vol = []
    daily_vol = []

    for i in range(len(df)):
        start_idx = max(0, i - window_hours + 1)
        chunk = df.iloc[start_idx:i + 1]

        wkend = chunk.loc[chunk["is_weekend"], "log_ret"].dropna()
        all_rets = chunk["log_ret"].dropna()

        weekend_vol.append(wkend.std() * annualize if len(wkend) >= 10 else np.nan)
        daily_vol.append(all_rets.std() * annualize if len(all_rets) >= 10 else np.nan)

    df["weekend_vol"] = weekend_vol
    df["daily_vol"] = daily_vol
    df["vol_ratio"] = df["weekend_vol"] / df["daily_vol"]

    return df


def _shade_weekends(ax, dates, regimes):
    """Add weekend shading (Fri 5PM ET → Mon 9AM ET)."""
    in_span = False
    span_start = None
    for i in range(len(dates)):
        if regimes.iloc[i] == "weekend" and not in_span:
            span_start = dates[i]
            in_span = True
        elif regimes.iloc[i] != "weekend" and in_span:
            ax.axvspan(span_start, dates[i], color="#ff9800", alpha=0.10)
            in_span = False
    if in_span:
        ax.axvspan(span_start, dates[-1], color="#ff9800", alpha=0.10)


def _plot_regime_line(ax, dates, values, regimes):
    """Plot price line: solid for market hours, dotted for off-hours."""
    # Build contiguous segments by regime
    segments = []
    seg_start = 0
    for i in range(1, len(dates)):
        cur_off = regimes.iloc[i] != "market"
        prev_off = regimes.iloc[i - 1] != "market"
        if cur_off != prev_off:
            segments.append((seg_start, i, prev_off))
            seg_start = i
    segments.append((seg_start, len(dates), regimes.iloc[-1] != "market"))

    for s, e, is_off in segments:
        # Extend by 1 on each side so segments connect
        sl = max(0, s - 1)
        ax.plot(
            dates[sl:e], values[sl:e],
            color="#e65100" if is_off else "#1f77b4",
            linewidth=0.6 if is_off else 0.9,
            linestyle=":" if is_off else "-",
            alpha=0.7 if is_off else 0.9,
        )


def plot_weekend_analysis(df: pd.DataFrame, window_hours: int, zoom_days: int = None,
                          save_path: str = None):
    """Three-panel plot: price, vol, and volume with weekend shading."""
    plt.close("all")

    fig, (ax_price, ax_vol, ax_volume) = plt.subplots(
        3, 1, figsize=(16, 13), height_ratios=[2, 1, 1], sharex=True,
    )

    dates = df.index

    # ── Summary stats for title ──
    annualize = np.sqrt(24 * 365)
    wkend_rets = df.loc[df["is_weekend"], "log_ret"].dropna()
    all_rets = df["log_ret"].dropna()
    overall_wkend_vol = wkend_rets.std() * annualize
    overall_daily_vol = all_rets.std() * annualize
    vol_premium = overall_wkend_vol / overall_daily_vol
    window_days = window_hours // 24

    fig.suptitle(
        f"BTC/USD Weekend Vol Analysis (hourly)  |  "
        f"{dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y}  |  "
        f"Weekend σ: {overall_wkend_vol:.1%}  Daily σ: {overall_daily_vol:.1%}  "
        f"Ratio: {vol_premium:.2f}x",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Price — solid market, dotted off-hours ──
    _plot_regime_line(ax_price, dates, df["Close"].values, df["regime"])
    _shade_weekends(ax_price, dates, df["regime"])

    from matplotlib.patches import Patch
    ax_price.legend(
        handles=[
            plt.Line2D([], [], color="#1f77b4", linewidth=0.9, linestyle="-",
                       label="Market hours"),
            plt.Line2D([], [], color="#e65100", linewidth=0.6, linestyle=":",
                       label="Off-hours"),
            Patch(facecolor="#ff9800", alpha=0.15,
                  label="Weekend (Fri 5PM–Mon 9AM ET)"),
        ],
        loc="upper left", fontsize=9, framealpha=0.9,
    )
    ax_price.set_ylabel("Price ($)", fontsize=10)
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax_price.grid(True, alpha=0.25, linestyle="--")
    ax_price.margins(x=0.02)

    # ── Panel 2: Weekend vol vs daily vol (hourly, no shading) ──
    ax_vol.plot(dates, df["weekend_vol"], color="#e74c3c", linewidth=0.8,
                label=f"Weekend vol ({window_days}d)", alpha=0.85)
    ax_vol.plot(dates, df["daily_vol"], color="#1f77b4", linewidth=0.8,
                label=f"Daily vol ({window_days}d)", alpha=0.85)
    ax_vol.fill_between(
        dates, df["weekend_vol"], df["daily_vol"],
        where=df["weekend_vol"] > df["daily_vol"],
        color="#e74c3c", alpha=0.08,
    )
    ax_vol.fill_between(
        dates, df["weekend_vol"], df["daily_vol"],
        where=df["weekend_vol"] <= df["daily_vol"],
        color="#1f77b4", alpha=0.08,
    )
    ax_vol.set_ylabel("Annualized Vol", fontsize=10)
    ax_vol.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_vol.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_vol.grid(True, alpha=0.25, linestyle="--")
    ax_vol.margins(x=0.02)

    # ── Panel 3: CryptoQuant Exchange Whale Ratio (manually extracted) ──
    # Source: CryptoQuant "Exchange Whale Ratio - All Exchanges" (daily)
    # = top-10 inflow txns / total inflow volume. Higher = more whale-driven.
    whale_data = {
        "2026-01-01": 0.74, "2026-01-02": 0.75, "2026-01-03": 0.73,
        "2026-01-04": 0.72, "2026-01-05": 0.75, "2026-01-06": 0.50,
        "2026-01-07": 0.51, "2026-01-08": 0.49, "2026-01-09": 0.50,
        "2026-01-10": 0.48, "2026-01-11": 0.43, "2026-01-12": 0.45,
        "2026-01-13": 0.80, "2026-01-14": 0.83, "2026-01-15": 0.60,
        "2026-01-16": 0.52, "2026-01-17": 0.50, "2026-01-18": 0.42,
        "2026-01-19": 0.50, "2026-01-20": 0.63, "2026-01-21": 0.55,
        "2026-01-22": 0.48, "2026-01-23": 0.65, "2026-01-24": 0.52,
        "2026-01-25": 0.45, "2026-01-26": 0.60, "2026-01-27": 0.52,
        "2026-01-28": 0.48, "2026-01-29": 0.46, "2026-01-30": 0.44,
        "2026-01-31": 0.35, "2026-02-01": 0.50, "2026-02-02": 0.50,
        "2026-02-03": 0.55, "2026-02-04": 0.50, "2026-02-05": 0.50,
        "2026-02-06": 0.48, "2026-02-07": 0.52, "2026-02-08": 0.68,
        "2026-02-09": 0.50, "2026-02-10": 0.55, "2026-02-11": 0.50,
        "2026-02-12": 0.68, "2026-02-13": 0.70, "2026-02-14": 0.55,
        "2026-02-15": 0.82, "2026-02-16": 0.55, "2026-02-17": 0.58,
        "2026-02-18": 0.50, "2026-02-19": 0.47, "2026-02-20": 0.42,
        "2026-02-21": 0.42, "2026-02-22": 0.43, "2026-02-23": 0.45,
    }
    whale_dates = pd.to_datetime(list(whale_data.keys()))
    whale_vals = list(whale_data.values())
    whale_sma = pd.Series(whale_vals).rolling(14, min_periods=1).mean()

    # Classify whale dates as weekend or not for coloring
    whale_regimes = _classify_hours(whale_dates)
    whale_colors = ["#ff9800" if r == "weekend" else "#5C6BC0"
                    for r in whale_regimes]

    ax_volume.scatter(whale_dates, whale_vals, c=whale_colors, s=30,
                      zorder=5, edgecolors="white", linewidths=0.4)
    ax_volume.plot(whale_dates, whale_vals, color="#5C6BC0", linewidth=1.0,
                   alpha=0.7, label="Exchange Whale Ratio")
    ax_volume.plot(whale_dates, whale_sma, color="#5C6BC0", linewidth=0.7,
                   linestyle="--", alpha=0.4, label="SMA(14)")

    _shade_weekends(ax_volume, dates, df["regime"])

    from matplotlib.patches import Patch
    ax_volume.legend(
        handles=[
            plt.Line2D([], [], color="#5C6BC0", linewidth=1.0,
                       marker="o", markersize=4, label="Whale Ratio"),
            plt.Line2D([], [], color="#5C6BC0", linewidth=0.7,
                       linestyle="--", alpha=0.5, label="SMA(14)"),
            plt.Line2D([], [], color="#ff9800", marker="o", markersize=5,
                       linestyle="none", label="Weekend point"),
            Patch(facecolor="#ff9800", alpha=0.15,
                  label="Weekend (Fri 5PM–Mon 9AM ET)"),
        ],
        loc="upper left", fontsize=8, framealpha=0.9,
    )
    ax_volume.set_ylabel("Exchange Whale Ratio", fontsize=10)
    ax_volume.set_xlabel("Date", fontsize=10)
    ax_volume.set_ylim(0.30, 0.90)
    ax_volume.grid(True, alpha=0.25, linestyle="--")
    ax_volume.margins(x=0.02)

    # Zoom to last N days if requested
    if zoom_days is not None and zoom_days < (dates[-1] - dates[0]).days:
        zoom_start = dates[-1] - timedelta(days=zoom_days)
        for ax in (ax_price, ax_vol, ax_volume):
            ax.set_xlim(zoom_start, dates[-1] + timedelta(hours=6))
        display_days = zoom_days
    else:
        display_days = (dates[-1] - dates[0]).days

    # Format x-axis
    if display_days <= 90:
        ax_volume.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax_volume.xaxis.set_minor_locator(mdates.DayLocator())
    else:
        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_volume.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description="BTC weekend vol analysis")
    parser.add_argument("--days", type=int, default=61,
                        help="Calendar days of history (default: 61)")
    parser.add_argument("--vol-window", type=int, default=7,
                        help="Rolling window in days (default: 7)")
    parser.add_argument("--zoom", type=int, default=None,
                        help="Zoom x-axis to last N days (default: show all)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to PNG file path")
    args = parser.parse_args()

    window_hours = args.vol_window * 24

    print(f"Loading {args.days} days of hourly BTC/USD data...")
    df = load_btc_hourly(args.days)
    print(f"Loaded {len(df)} hourly bars: "
          f"{df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M}")

    print(f"Computing rolling vol (window={args.vol_window}d / {window_hours}h)...")
    df = compute_weekend_vol(df, window_hours=window_hours)

    # Print summary table
    annualize = np.sqrt(24 * 365)
    all_rets = df["log_ret"].dropna()
    wkend = df.loc[df["is_weekend"], "log_ret"].dropna()
    mkt = df.loc[df["regime"] == "market", "log_ret"].dropna()

    print("\n── Weekend vs Daily Summary (hourly) ──")
    print(f"  All hours:     {len(all_rets):>5}   "
          f"Mean ret: {all_rets.mean():+.6%}   "
          f"Vol (ann): {all_rets.std() * annualize:.1%}")
    print(f"  Market hours:  {len(mkt):>5}   "
          f"Mean ret: {mkt.mean():+.6%}   "
          f"Vol (ann): {mkt.std() * annualize:.1%}")
    print(f"  Weekend*:      {len(wkend):>5}   "
          f"Mean ret: {wkend.mean():+.6%}   "
          f"Vol (ann): {wkend.std() * annualize:.1%}")
    print(f"  Wkend/Daily:   {wkend.std() / all_rets.std():.3f}x")
    print(f"  * Weekend = Fri 5PM – Mon 9AM ET")
    print()

    plot_weekend_analysis(df, window_hours, zoom_days=args.zoom,
                          save_path=args.save)


if __name__ == "__main__":
    main()
