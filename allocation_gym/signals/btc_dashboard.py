"""
BTC Market Dashboard — IV z-score, ETF flows, historical vol.

Usage:
    python -m allocation_gym.signals.btc_dashboard
    python -m allocation_gym.signals.btc_dashboard --days 365 --no-plot
"""

import argparse
import math
import json
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
import pandas as pd

from allocation_gym.metrics.variance_metrics import VarianceMetrics


# ── Major BTC spot ETFs (launched Jan 2024, GBTC converted) ──────────────
BTC_ETFS = ["IBIT", "FBTC", "GBTC", "ARKB", "BITB", "HODL", "BRRR", "EZBC", "BTCO"]


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════════════════

def fetch_btc_daily(days: int = 500) -> pd.DataFrame:
    """Fetch BTC-USD daily OHLCV from yfinance."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(
        "BTC-USD",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("No BTC-USD data returned from yfinance")
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.droplevel(1)
    return df



def fetch_etf_data(days: int = 365) -> dict[str, pd.DataFrame]:
    """Fetch daily data for major BTC spot ETFs."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=days)
    etf_data = {}
    for ticker in BTC_ETFS:
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
                    df.columns = df.columns.droplevel(1)
                etf_data[ticker] = df
        except Exception:
            pass
    return etf_data


def fetch_deribit_dvol(days: int = 365) -> pd.Series | None:
    """
    Fetch BTC DVOL (Deribit Volatility Index) from Deribit public API.

    Returns a Series of daily DVOL values (annualised IV %), or None on failure.
    """
    now = datetime.now(timezone.utc)
    end_ts = int(now.timestamp() * 1000)
    start_ts = int((now - timedelta(days=days)).timestamp() * 1000)

    url = (
        "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        f"?currency=BTC&start_timestamp={start_ts}&end_timestamp={end_ts}"
        "&resolution=86400"
    )

    try:
        req = Request(url, headers={"User-Agent": "allocation-gym/0.1"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        if "result" not in data or "data" not in data["result"]:
            return None

        records = data["result"]["data"]
        # Each record: [timestamp_ms, open, high, low, close]
        timestamps = [pd.Timestamp(r[0], unit="ms") for r in records]
        closes = [r[4] for r in records]
        series = pd.Series(closes, index=timestamps, name="DVOL")
        series.index = series.index.tz_localize(None)
        return series
    except (URLError, json.JSONDecodeError, KeyError, IndexError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Calculations
# ═══════════════════════════════════════════════════════════════════════════

def calc_iv_zscore(
    dvol: pd.Series | None,
    daily_df: pd.DataFrame,
    lookback: int = 365,
) -> dict:
    """
    Compute IV z-score.

    Primary: Deribit DVOL z-score over trailing `lookback` days.
    Fallback: 30-day realised vol z-score over trailing `lookback` days.
    """
    result = {"source": None, "current": None, "mean": None, "std": None, "zscore": None, "series": None}

    if dvol is not None and len(dvol) >= 30:
        tail = dvol.tail(lookback)
        current = float(tail.iloc[-1])
        result.update({
            "source": "Deribit DVOL",
            "current": current,
            "mean": float(tail.mean()),
            "std": float(tail.std()),
            "zscore": float((current - tail.mean()) / tail.std()) if tail.std() > 0 else 0.0,
            "series": tail,
        })
        return result

    # Fallback: rolling 30-day realised vol z-score
    closes = daily_df["Close"].values.astype(float)
    log_ret = np.log(closes[1:] / closes[:-1])

    window = 30
    if len(log_ret) < window + 10:
        return result

    rolling_vol = pd.Series(log_ret).rolling(window).std() * math.sqrt(365)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < 30:
        return result

    tail = rolling_vol.tail(lookback)
    current = float(tail.iloc[-1])
    result.update({
        "source": "30d Realised Vol",
        "current": current * 100,
        "mean": float(tail.mean()) * 100,
        "std": float(tail.std()) * 100,
        "zscore": float((current - tail.mean()) / tail.std()) if tail.std() > 0 else 0.0,
    })
    return result


def calc_etf_flows(
    etf_data: dict[str, pd.DataFrame],
    btc_daily: pd.DataFrame,
    window: int = 365,
) -> dict:
    """
    Estimate aggregate BTC ETF net flows.

    Method: For each ETF, compare its daily return to BTC's daily return.
    Premium (ETF outperforms) signals inflows; discount signals outflows.
    Flow estimate = dollar_volume × sign(etf_return - btc_return).
    """
    if not etf_data:
        return {
            "total_dollar_volume": 0,
            "net_flow_estimate": 0,
            "daily_flows": pd.Series(dtype=float),
            "cumulative_flows": pd.Series(dtype=float),
            "etf_count": 0,
        }

    btc_ret = btc_daily["Close"].pct_change()

    agg_flows = None

    for ticker, df in etf_data.items():
        if len(df) < 5:
            continue
        etf_ret = df["Close"].pct_change()
        dollar_vol = (df["Close"] * df["Volume"]).fillna(0)

        # Premium/discount: positive = ETF outperforms BTC = inflows
        # Align indices
        common = etf_ret.index.intersection(btc_ret.index)
        if len(common) < 5:
            continue

        premium = etf_ret.loc[common] - btc_ret.loc[common]
        flow_sign = np.sign(premium).fillna(0)
        daily_flow = (dollar_vol.loc[common] * flow_sign).fillna(0)

        if agg_flows is None:
            agg_flows = daily_flow.copy()
        else:
            agg_flows = agg_flows.add(daily_flow, fill_value=0)

    if agg_flows is None:
        agg_flows = pd.Series(dtype=float)

    total_dv = sum(
        float((df["Close"] * df["Volume"]).sum()) for df in etf_data.values()
    )

    return {
        "total_dollar_volume": total_dv,
        "net_flow_estimate": float(agg_flows.sum()) if len(agg_flows) > 0 else 0,
        "daily_flows": agg_flows,
        "cumulative_flows": agg_flows.cumsum() if len(agg_flows) > 0 else agg_flows,
        "etf_count": len(etf_data),
        "recent_7d": float(agg_flows.tail(7).sum()) if len(agg_flows) >= 7 else 0,
        "recent_30d": float(agg_flows.tail(30).sum()) if len(agg_flows) >= 30 else 0,
    }



def calc_historical_vol(daily_df: pd.DataFrame) -> dict:
    """
    Compute Yang-Zhang historical volatility at multiple windows.

    Returns annualised volatility for 30d, 60d, 90d, and full-period windows.
    """
    opens = daily_df["Open"].values.astype(float)
    highs = daily_df["High"].values.astype(float)
    lows = daily_df["Low"].values.astype(float)
    closes = daily_df["Close"].values.astype(float)

    full_period = len(closes) - 2
    windows = {"30d": 30, "60d": 60, "90d": 90}
    if full_period > 90:
        windows[f"{full_period}d"] = full_period
    vol_results = {}

    for label, period in windows.items():
        if len(closes) < period + 2:
            continue
        vr = VarianceMetrics.compute(
            opens=opens, highs=highs, lows=lows, closes=closes,
            period=period, trading_days=365,
        )
        vol_results[label] = {
            "yang_zhang_vol_ann": vr.yang_zhang_vol_ann,
            "variance_ratio": vr.variance_ratio,
            "efficiency_ratio": vr.efficiency_ratio,
            "vol_of_vol": vr.vol_of_vol,
            "downside_semivol": vr.downside_semivol,
            "upside_semivol": vr.upside_semivol,
            "vol_skew": vr.vol_skew,
            "regime": vr.regime,
        }

    # Rolling 30d annualised vol series for plotting
    log_ret = np.log(closes[1:] / closes[:-1])
    rolling_vol = pd.Series(
        log_ret, index=daily_df.index[1:]
    ).rolling(30).std() * math.sqrt(365)

    vol_results["rolling_30d_series"] = rolling_vol.dropna()

    return vol_results


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard output
# ═══════════════════════════════════════════════════════════════════════════

def print_dashboard(iv: dict, flows: dict, vol: dict):
    """Pretty-print the BTC market dashboard."""

    print("\n" + "=" * 72)
    print("  BTC MARKET DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # ── IV Z-Score ────────────────────────────────────────────────────────
    print("\n  IMPLIED VOLATILITY Z-SCORE")
    if iv["source"] == "Deribit DVOL":
        print("  source: Deribit DVOL index (public API, daily)")
    else:
        print("  source: yfinance BTC-USD close-to-close (30d rolling)")
    print("  " + "-" * 50)
    if iv["source"]:
        print(f"  Current IV .................. {iv['current']:>12.1f}%")
        print(f"  1Y Mean ..................... {iv['mean']:>12.1f}%")
        print(f"  1Y Std Dev .................. {iv['std']:>12.1f}%")
        zscore = iv["zscore"]
        label = _zscore_label(zscore)
        print(f"  Z-Score ..................... {zscore:>+12.2f}  ({label})")
    else:
        print("  Data unavailable — Deribit API and fallback both failed")

    # ── ETF Net Flows ─────────────────────────────────────────────────────
    print("\n  BTC ETF NET FLOWS (estimated)")
    print("  source: yfinance (IBIT FBTC GBTC ARKB BITB HODL BRRR EZBC BTCO)")
    print("  method: dollar_volume * sign(etf_return - btc_return)")
    print("  " + "-" * 50)
    if flows["etf_count"] > 0:
        print(f"  ETFs tracked ................ {flows['etf_count']}")
        print(f"  Total $ Volume (period) ..... ${flows['total_dollar_volume']:>12,.0f}")
        print(f"  Est. Net Flow (period) ...... ${flows['net_flow_estimate']:>+12,.0f}")
        if "recent_7d" in flows:
            print(f"  Est. Net Flow (7d) .......... ${flows['recent_7d']:>+12,.0f}")
        if "recent_30d" in flows:
            print(f"  Est. Net Flow (30d) ......... ${flows['recent_30d']:>+12,.0f}")
    else:
        print("  No ETF data available")

    # ── Historical Volatility ─────────────────────────────────────────────
    print("\n  HISTORICAL VOLATILITY (Yang-Zhang, annualised)")
    print("  source: yfinance BTC-USD OHLCV (daily)")
    print("  " + "-" * 50)
    vol_labels = [k for k in vol if k != "rolling_30d_series"]
    vol_labels.sort(key=lambda k: int(k.rstrip("d")))
    for label in vol_labels:
        v = vol[label]
        print(f"  {label:>5} Vol ................... {v['yang_zhang_vol_ann']*100:>11.1f}%")

    # Detailed metrics for 30d window
    if "30d" in vol:
        v = vol["30d"]
        print()
        print(f"  30d Variance Ratio .......... {v['variance_ratio']:>12.3f}")
        print(f"  30d Efficiency Ratio ........ {v['efficiency_ratio']:>12.3f}")
        print(f"  30d Vol of Vol .............. {v['vol_of_vol']:>12.3f}")
        print(f"  30d Downside Semivol ........ {v['downside_semivol']*100:>11.1f}%  (daily)")
        print(f"  30d Upside Semivol .......... {v['upside_semivol']*100:>11.1f}%  (daily)")
        print(f"  30d Vol Skew (down/up) ...... {v['vol_skew']:>12.3f}")
        print(f"  30d Regime .................. {v['regime']:>12}")

    print("\n" + "=" * 72)


def _zscore_label(z: float) -> str:
    az = abs(z)
    if az < 0.5:
        return "normal"
    elif az < 1.0:
        return "slightly elevated" if z > 0 else "slightly depressed"
    elif az < 2.0:
        return "elevated" if z > 0 else "depressed"
    else:
        return "extremely elevated" if z > 0 else "extremely depressed"


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_dashboard(
    daily_df: pd.DataFrame,
    iv: dict,
    flows: dict,
    vol: dict,
    days: int,
):
    """3-panel matplotlib dashboard."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"BTC Market Dashboard — {days}d lookback", fontsize=14, fontweight="bold")

    # ── Panel 1: IV Z-Score ───────────────────────────────────────────────
    ax2 = axes[0]
    if iv["source"] == "Deribit DVOL" and iv.get("series") is not None and len(iv["series"]) > 5:
        dvol_s = iv["series"]
        ax2.plot(dvol_s.index, dvol_s.values, label="Deribit DVOL", color="purple", linewidth=1)
        ax2.axhline(iv["mean"], color="gray", linestyle="--", linewidth=0.8, label=f"Mean {iv['mean']:.1f}%")
        ax2.axhline(iv["mean"] + iv["std"], color="red", linestyle=":", linewidth=0.8, alpha=0.6, label=f"+1\u03c3 {iv['mean']+iv['std']:.1f}%")
        ax2.axhline(iv["mean"] - iv["std"], color="green", linestyle=":", linewidth=0.8, alpha=0.6, label=f"-1\u03c3 {iv['mean']-iv['std']:.1f}%")
        ax2.set_title(f"IV Z-Score: {iv['zscore']:+.2f} (Deribit DVOL)")
        ax2.set_ylabel("IV %")
        ax2.legend(loc="upper left", fontsize=7)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3)
    elif "rolling_30d_series" in vol:
        # Plot rolling realised vol with mean/std bands
        rv = vol["rolling_30d_series"] * 100
        # Trim to requested days
        cutoff = daily_df.index[-1] - timedelta(days=days)
        rv = rv[rv.index >= cutoff]
        if len(rv) > 0:
            ax2.plot(rv.index, rv.values, label="30d Realised Vol (ann.)", color="purple", linewidth=1)
            ax2.axhline(iv["mean"], color="gray", linestyle="--", linewidth=0.8, label=f"Mean {iv['mean']:.1f}%")
            ax2.axhline(iv["mean"] + iv["std"], color="red", linestyle=":", linewidth=0.8, alpha=0.6, label=f"+1σ {iv['mean']+iv['std']:.1f}%")
            ax2.axhline(iv["mean"] - iv["std"], color="green", linestyle=":", linewidth=0.8, alpha=0.6, label=f"-1σ {iv['mean']-iv['std']:.1f}%")
            ax2.set_title(f"IV Z-Score: {iv['zscore']:+.2f} ({iv['source']})")
            ax2.set_ylabel("Annualised Vol %")
            ax2.legend(loc="upper left", fontsize=7)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax2.tick_params(axis="x", rotation=30)
            ax2.grid(True, alpha=0.3)
        else:
            _draw_zscore_gauge(ax2, iv)
    else:
        _draw_zscore_gauge(ax2, iv)

    # ── Panel 2: ETF Cumulative Flows ─────────────────────────────────────
    ax3 = axes[1]
    cum_flows = flows.get("cumulative_flows", pd.Series(dtype=float))
    if len(cum_flows) > 0:
        colors = ["green" if v >= 0 else "red" for v in cum_flows.values]
        ax3.fill_between(cum_flows.index, 0, cum_flows.values, alpha=0.3,
                         color="green", where=cum_flows.values >= 0)
        ax3.fill_between(cum_flows.index, 0, cum_flows.values, alpha=0.3,
                         color="red", where=cum_flows.values < 0)
        ax3.plot(cum_flows.index, cum_flows.values, color="steelblue", linewidth=1)
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("BTC ETF Cumulative Net Flows (est.)")
        ax3.set_ylabel("USD")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B" if abs(x) >= 1e9 else f"${x/1e6:.0f}M"))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.tick_params(axis="x", rotation=30)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No ETF flow data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("BTC ETF Cumulative Net Flows")

    # ── Panel 3: Historical Volatility ────────────────────────────────────
    ax4 = axes[2]
    if "rolling_30d_series" in vol:
        rv = vol["rolling_30d_series"] * 100
        cutoff = daily_df.index[-1] - timedelta(days=days)
        rv = rv[rv.index >= cutoff]
        if len(rv) > 0:
            ax4.plot(rv.index, rv.values, label="30d Yang-Zhang Vol (ann.)", color="darkorange", linewidth=1)
            # Add regime color bands
            ax4.axhline(50, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
            ax4.axhline(80, color="red", linestyle=":", alpha=0.4, linewidth=0.7)
            ax4.set_title("Historical Volatility (30d rolling, annualised)")
            ax4.set_ylabel("Vol %")
            ax4.legend(loc="upper left", fontsize=8)
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax4.tick_params(axis="x", rotation=30)
            ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _draw_zscore_gauge(ax, iv: dict):
    """Draw a simple z-score gauge when time series isn't available."""
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)

    if iv["zscore"] is not None:
        z = iv["zscore"]
        color = "green" if abs(z) < 1 else ("orange" if abs(z) < 2 else "red")
        ax.barh(0.5, z, height=0.3, color=color, alpha=0.7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"IV Z-Score: {z:+.2f} ({iv.get('source', 'N/A')})")
        ax.set_xlabel("Z-Score (σ)")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, "IV data unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("IV Z-Score")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run(args=None):
    parser = argparse.ArgumentParser(description="BTC Market Dashboard")
    parser.add_argument("--days", type=int, default=365, help="Lookback period in days (default: 365)")
    parser.add_argument("--no-plot", action="store_true", help="Disable chart")
    args = parser.parse_args(args)

    days = args.days

    print("Fetching BTC daily data...")
    daily_df = fetch_btc_daily(days=days + 60)
    analysis_df = daily_df.tail(days + 60)

    print("Fetching BTC ETF data...")
    etf_data = fetch_etf_data(days=days)

    print("Fetching Deribit DVOL...")
    dvol = fetch_deribit_dvol(days=days)

    print("Calculating metrics...\n")

    iv = calc_iv_zscore(dvol, analysis_df, lookback=days)
    flows = calc_etf_flows(etf_data, daily_df.tail(days + 5), window=days)
    vol = calc_historical_vol(analysis_df)

    print_dashboard(iv, flows, vol)

    if not args.no_plot:
        plot_dashboard(daily_df, iv, flows, vol, days)

    return {"iv": iv, "flows": flows, "vol": vol}


def main():
    run()


if __name__ == "__main__":
    main()
