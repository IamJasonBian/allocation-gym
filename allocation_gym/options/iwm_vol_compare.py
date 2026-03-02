"""
IWM options pricing comparison: this week vs last week.

Fetches recent IWM data, computes realized vol for both windows,
prices a grid of calls/puts at each vol level, and shows the delta.

Usage:
    python -m allocation_gym.options.iwm_vol_compare
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from allocation_gym.options.black_scholes import bs_call_price, bs_put_price


# ── Data ─────────────────────────────────────────────────────────────────


def load_iwm(days: int = 60) -> pd.DataFrame:
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.8))
    df = yf.Ticker("IWM").history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        raise RuntimeError("No IWM data from yfinance")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── Vol computations ─────────────────────────────────────────────────────


def realized_vol(closes: np.ndarray, trading_days: int = 252) -> float:
    """Annualized close-to-close realized vol."""
    log_ret = np.log(closes[1:] / closes[:-1])
    return float(np.std(log_ret, ddof=1) * np.sqrt(trading_days))


def yang_zhang_vol(df: pd.DataFrame, trading_days: int = 252) -> float:
    """Yang-Zhang OHLC volatility estimator."""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    n = len(c)
    if n < 3:
        return realized_vol(c, trading_days)

    log_oc = np.log(c / o)
    log_co = np.log(o[1:] / c[:-1])
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    rs = log_ho * (log_ho - log_oc) + log_lo * (log_lo - log_oc)

    close_var = np.var(log_co, ddof=1)
    open_var = np.var(log_oc, ddof=1)
    rs_mean = np.mean(rs)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz_var = close_var + k * open_var + (1 - k) * rs_mean
    return float(np.sqrt(max(yz_var, 0) * trading_days))


def parkinson_vol(df: pd.DataFrame, trading_days: int = 252) -> float:
    """Parkinson high-low volatility estimator."""
    h, l = df["High"].values, df["Low"].values
    log_hl = np.log(h / l)
    return float(np.sqrt(np.mean(log_hl**2) / (4 * np.log(2)) * trading_days))


# ── Main ─────────────────────────────────────────────────────────────────


def run_comparison():
    print("=" * 70)
    print("IWM OPTIONS PRICING: THIS WEEK vs LAST WEEK")
    print("=" * 70)

    # Load data
    print("\nLoading IWM data...")
    df = load_iwm(days=60)
    print(f"  {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Current price: ${df['Close'].iloc[-1]:.2f}")

    # Split into this week and last week
    today = df.index[-1]
    # This week = last 5 trading days, last week = 5 before that
    this_week = df.iloc[-5:]
    last_week = df.iloc[-10:-5]
    # Also compute over broader windows for context
    two_week = df.iloc[-10:]
    month = df.iloc[-21:]

    S_now = float(df["Close"].iloc[-1])
    S_last_week = float(last_week["Close"].iloc[-1])

    print(f"\n  This week:  {this_week.index[0].date()} to {this_week.index[-1].date()}  |  Close: ${this_week['Close'].iloc[-1]:.2f}")
    print(f"  Last week:  {last_week.index[0].date()} to {last_week.index[-1].date()}  |  Close: ${last_week['Close'].iloc[-1]:.2f}")
    price_chg = (S_now / S_last_week - 1) * 100
    print(f"  Price change: {price_chg:+.2f}%")

    # ── Volatility comparison ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REALIZED VOLATILITY COMPARISON")
    print("=" * 70)

    vol_table = {}
    for label, window_df in [
        ("This week (5d)", this_week),
        ("Last week (5d)", last_week),
        ("2 weeks (10d)", two_week),
        ("1 month (21d)", month),
    ]:
        rv = realized_vol(window_df["Close"].values)
        yz = yang_zhang_vol(window_df)
        pk = parkinson_vol(window_df)
        vol_table[label] = {"Close-Close": rv, "Yang-Zhang": yz, "Parkinson": pk}
        print(f"\n  {label}:")
        print(f"    Close-to-close: {rv:.1%}")
        print(f"    Yang-Zhang:     {yz:.1%}")
        print(f"    Parkinson:      {pk:.1%}")

    rv_this = realized_vol(this_week["Close"].values)
    rv_last = realized_vol(last_week["Close"].values)
    yz_this = yang_zhang_vol(this_week)
    yz_last = yang_zhang_vol(last_week)

    print(f"\n  Vol change (close-close): {rv_this:.1%} vs {rv_last:.1%}  ({(rv_this - rv_last)*100:+.1f}pp)")
    print(f"  Vol change (Yang-Zhang): {yz_this:.1%} vs {yz_last:.1%}  ({(yz_this - yz_last)*100:+.1f}pp)")

    # ── Options pricing grid ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTIONS PRICING GRID (30-day expiry)")
    print("=" * 70)

    T = 30 / 252  # 30 calendar days / trading days
    r = 0.045     # risk-free rate

    # Use YZ vol as proxy for IV in each period
    iv_this = yz_this
    iv_last = yz_last

    # Strike grid: ATM +/- 1%, 2%, 5%, 10%
    otm_pcts = [-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10]

    print(f"\n  Using IWM spot = ${S_now:.2f}")
    print(f"  IV this week: {iv_this:.1%}  |  IV last week: {iv_last:.1%}")
    print(f"  T = 30 days  |  r = {r:.1%}")

    # ── CALLS ────────────────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print(f"  CALL OPTIONS")
    print(f"  {'Strike':>10s} {'OTM%':>6s}  {'This Wk':>10s} {'Last Wk':>10s} {'Chg($)':>10s} {'Chg(%)':>8s}")
    print(f"  {'─' * 68}")

    call_data = []
    for pct in otm_pcts:
        K = round(S_now * (1 + pct), 2)
        c_this = bs_call_price(S_now, K, T, r, iv_this)
        c_last = bs_call_price(S_last_week, K, T, r, iv_last)
        chg = c_this - c_last
        chg_pct = (chg / c_last * 100) if c_last > 0.001 else 0
        moneyness = "ATM" if pct == 0 else f"{pct:+.0%}"
        print(f"  ${K:>9.2f} {moneyness:>6s}  ${c_this:>9.4f} ${c_last:>9.4f} ${chg:>+9.4f} {chg_pct:>+7.1f}%")
        call_data.append({"strike": K, "pct": pct, "this": c_this, "last": c_last, "chg": chg, "chg_pct": chg_pct})

    # ── PUTS ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print(f"  PUT OPTIONS")
    print(f"  {'Strike':>10s} {'OTM%':>6s}  {'This Wk':>10s} {'Last Wk':>10s} {'Chg($)':>10s} {'Chg(%)':>8s}")
    print(f"  {'─' * 68}")

    put_data = []
    for pct in otm_pcts:
        K = round(S_now * (1 - pct), 2)
        p_this = bs_put_price(S_now, K, T, r, iv_this)
        p_last = bs_put_price(S_last_week, K, T, r, iv_last)
        chg = p_this - p_last
        chg_pct = (chg / p_last * 100) if p_last > 0.001 else 0
        moneyness = "ATM" if pct == 0 else f"{pct:+.0%}"
        print(f"  ${K:>9.2f} {moneyness:>6s}  ${p_this:>9.4f} ${p_last:>9.4f} ${chg:>+9.4f} {chg_pct:>+7.1f}%")
        put_data.append({"strike": K, "pct": pct, "this": p_this, "last": p_last, "chg": chg, "chg_pct": chg_pct})

    # ── Greeks comparison (ATM) ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ATM GREEKS COMPARISON (approx)")
    print("=" * 70)

    K_atm = round(S_now, 2)
    # Vega ~ S * sqrt(T) * N'(d1) -- approximate by bumping IV 1%
    c_base = bs_call_price(S_now, K_atm, T, r, iv_this)
    c_bump = bs_call_price(S_now, K_atm, T, r, iv_this + 0.01)
    vega = (c_bump - c_base) * 100  # per 1% IV move

    # Theta ~ bump T by 1 day
    c_tminus1 = bs_call_price(S_now, K_atm, T - 1/252, r, iv_this)
    theta = c_tminus1 - c_base  # negative = decay

    # Delta ~ bump S by $1
    c_sbump = bs_call_price(S_now + 1, K_atm, T, r, iv_this)
    delta = c_sbump - c_base

    print(f"\n  ATM Call @ ${K_atm:.2f} (IV={iv_this:.1%}, T=30d):")
    print(f"    Price:  ${c_base:.4f}")
    print(f"    Delta:  {delta:.4f} ($ per $1 move)")
    print(f"    Theta:  ${theta:.4f}/day")
    print(f"    Vega:   ${vega:.4f} per 1% IV")
    print()

    # Same for last week
    c_base_lw = bs_call_price(S_last_week, K_atm, T, r, iv_last)
    c_bump_lw = bs_call_price(S_last_week, K_atm, T, r, iv_last + 0.01)
    vega_lw = (c_bump_lw - c_base_lw) * 100
    c_tminus1_lw = bs_call_price(S_last_week, K_atm, T - 1/252, r, iv_last)
    theta_lw = c_tminus1_lw - c_base_lw
    c_sbump_lw = bs_call_price(S_last_week + 1, K_atm, T, r, iv_last)
    delta_lw = c_sbump_lw - c_base_lw

    print(f"  ATM Call @ ${K_atm:.2f} (IV={iv_last:.1%}, last week S=${S_last_week:.2f}):")
    print(f"    Price:  ${c_base_lw:.4f}")
    print(f"    Delta:  {delta_lw:.4f}")
    print(f"    Theta:  ${theta_lw:.4f}/day")
    print(f"    Vega:   ${vega_lw:.4f} per 1% IV")

    # ── Key takeaway ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("KEY OBSERVATIONS")
    print("=" * 70)
    vol_direction = "UP" if rv_this > rv_last else "DOWN"
    print(f"\n  Realized vol moved {vol_direction}: {rv_last:.1%} -> {rv_this:.1%}")
    print(f"  IWM price moved: ${S_last_week:.2f} -> ${S_now:.2f} ({price_chg:+.2f}%)")
    if rv_this > rv_last:
        print(f"  -> Higher vol = more expensive options (both calls and puts)")
        print(f"  -> ATM call up ${c_base - c_base_lw:+.4f} (vega effect + delta)")
    else:
        print(f"  -> Lower vol = cheaper options")
        print(f"  -> Good entry for buying protection or directional calls")
    print()

    # ── Rolling vol chart data ───────────────────────────────────────
    rolling_5d = df["Close"].pct_change().rolling(5).std() * np.sqrt(252) * 100
    rolling_10d = df["Close"].pct_change().rolling(10).std() * np.sqrt(252) * 100
    rolling_21d = df["Close"].pct_change().rolling(21).std() * np.sqrt(252) * 100

    # ── Plot ─────────────────────────────────────────────────────────
    _plot_vol_comparison(
        df, this_week, last_week,
        call_data, put_data,
        rolling_5d, rolling_10d, rolling_21d,
        iv_this, iv_last, S_now, S_last_week,
    )


def _plot_vol_comparison(
    df, this_week, last_week,
    call_data, put_data,
    rolling_5d, rolling_10d, rolling_21d,
    iv_this, iv_last, S_now, S_last_week,
):
    plt.close("all")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"IWM OPTIONS: This Week vs Last Week  |  "
        f"S=${S_now:.2f}  Vol {iv_last:.0%}->{iv_this:.0%}",
        fontsize=13, fontweight="bold",
    )

    C_THIS = "#2196F3"
    C_LAST = "#FF9800"
    C_CALL = "#4CAF50"
    C_PUT = "#F44336"

    # ── Panel 1: IWM price + vol shading ─────────────────────────────
    ax = axes[0, 0]
    dates = df.index
    ax.plot(dates, df["Close"], color="#333", linewidth=1.5, label="IWM Close")

    # Shade this week / last week
    tw_start, tw_end = this_week.index[0], this_week.index[-1]
    lw_start, lw_end = last_week.index[0], last_week.index[-1]
    ax.axvspan(lw_start, lw_end, alpha=0.12, color=C_LAST, label="Last week")
    ax.axvspan(tw_start, tw_end, alpha=0.12, color=C_THIS, label="This week")

    ax.set_ylabel("IWM Price ($)", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("IWM Price + Weekly Windows", fontsize=11, fontweight="bold")

    # Annotate prices
    ax.annotate(f"${S_last_week:.2f}", xy=(lw_end, S_last_week),
                fontsize=8, color=C_LAST, fontweight="bold",
                ha="right", va="bottom")
    ax.annotate(f"${S_now:.2f}", xy=(tw_end, S_now),
                fontsize=8, color=C_THIS, fontweight="bold",
                ha="right", va="bottom")

    # ── Panel 2: Rolling vol ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(dates, rolling_5d, color=C_THIS, linewidth=1.5, label="5d RV", alpha=0.8)
    ax.plot(dates, rolling_10d, color="#9C27B0", linewidth=1.5, label="10d RV")
    ax.plot(dates, rolling_21d, color="#333", linewidth=2, label="21d RV")

    ax.axvspan(lw_start, lw_end, alpha=0.08, color=C_LAST)
    ax.axvspan(tw_start, tw_end, alpha=0.08, color=C_THIS)

    ax.set_ylabel("Annualized Vol (%)", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_title("Rolling Realized Volatility", fontsize=11, fontweight="bold")

    # Annotate this/last week vol levels
    if not rolling_5d.empty and not np.isnan(rolling_5d.iloc[-1]):
        ax.annotate(f"Now: {rolling_5d.iloc[-1]:.1f}%",
                    xy=(dates[-1], rolling_5d.iloc[-1]),
                    fontsize=8, fontweight="bold", color=C_THIS,
                    xytext=(-50, 10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=C_THIS, lw=0.8))

    # ── Panel 3: Call pricing comparison ─────────────────────────────
    ax = axes[1, 0]
    strikes = [d["pct"] * 100 for d in call_data]
    this_prices = [d["this"] for d in call_data]
    last_prices = [d["last"] for d in call_data]

    x = np.arange(len(strikes))
    width = 0.35
    ax.bar(x - width/2, last_prices, width, color=C_LAST, alpha=0.7, label="Last week")
    ax.bar(x + width/2, this_prices, width, color=C_THIS, alpha=0.7, label="This week")

    # Add change labels
    for i, d in enumerate(call_data):
        if abs(d["chg"]) > 0.001:
            ax.annotate(f"{d['chg']:+.3f}", xy=(x[i] + width/2, d["this"]),
                        fontsize=7, ha="center", va="bottom",
                        color="green" if d["chg"] > 0 else "red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"${d['strike']:.0f}\n({d['pct']:+.0%})" for d in call_data], fontsize=7)
    ax.set_ylabel("Call Premium ($)", fontsize=10)
    ax.set_title("Call Option Prices: Week-over-Week", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    # ── Panel 4: Put pricing comparison ──────────────────────────────
    ax = axes[1, 1]
    put_strikes = [d["pct"] * 100 for d in put_data]
    put_this = [d["this"] for d in put_data]
    put_last = [d["last"] for d in put_data]

    ax.bar(x - width/2, put_last, width, color=C_LAST, alpha=0.7, label="Last week")
    ax.bar(x + width/2, put_this, width, color=C_THIS, alpha=0.7, label="This week")

    for i, d in enumerate(put_data):
        if abs(d["chg"]) > 0.001:
            ax.annotate(f"{d['chg']:+.3f}", xy=(x[i] + width/2, d["this"]),
                        fontsize=7, ha="center", va="bottom",
                        color="green" if d["chg"] > 0 else "red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"${d['strike']:.0f}\n({d['pct']:+.0%})" for d in put_data], fontsize=7)
    ax.set_ylabel("Put Premium ($)", fontsize=10)
    ax.set_title("Put Option Prices: Week-over-Week", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs", "4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iwm_vol_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_path}")

    plt.show(block=True)


if __name__ == "__main__":
    run_comparison()
