#!/usr/bin/env python3
"""Generate the IWN vol analysis report with IV smile and actual options data.

Fetches IWN options chain data from the Netlify blob store for two snapshot
dates (early March and latest), computes the implied volatility smile,
and generates charts + LaTeX tables for the report.

Requires:
    ALLOC_ENGINE_SITE_ID  - Netlify site ID
    NETLIFY_AUTH_TOKEN     - Netlify PAT
"""

import math
import os
import sys
import json
import subprocess
from collections import defaultdict

import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allocation_gym.blobs import BlobClient, BlobIndex, OptionSnapshot
from allocation_gym.options.black_scholes import bs_call_price, bs_put_price

# ── Constants ─────────────────────────────────────────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "7")
TARGET_DATE_EARLY = "2026-03-04"  # Original report date
RISK_FREE = 0.045


# ── Helpers ───────────────────────────────────────────────────────────────


def derive_underlying_price(snapshots: list[OptionSnapshot]) -> float | None:
    """Derive underlying price from put-call parity on matched strike/expiry pairs."""
    calls: dict[tuple[float, str], float] = {}
    puts: dict[tuple[float, str], float] = {}
    for s in snapshots:
        if s.latest_quote is None:
            continue
        mid = s.latest_quote.mid
        if mid <= 0:
            continue
        key = (s.strike, s.expiry)
        if s.option_type == "call":
            calls[key] = mid
        elif s.option_type == "put":
            puts[key] = mid
    estimates = []
    for key in calls:
        if key in puts:
            implied_s = calls[key] - puts[key] + key[0]
            if 100 < implied_s < 400:
                estimates.append(implied_s)
    return sum(estimates) / len(estimates) if estimates else None


def compute_iv_smile(idx: BlobIndex, underlying_price: float | None) -> dict:
    """Compute IV smile by strike bucket and moneyness."""
    with_iv = idx.has_iv()
    if len(with_iv) == 0:
        return {}

    # IV by $5 strike bucket
    iv_by_strike = with_iv.aggregate(
        key_fn=lambda s: round(s.strike / 5) * 5,
        value_fn=lambda s: s.iv,
        agg="mean",
    )

    # IV by moneyness bucket (if we have underlying price)
    iv_by_moneyness = {}
    if underlying_price:
        for label, lo, hi in [
            ("Deep OTM Put", -0.30, -0.10),
            ("OTM Put", -0.10, -0.03),
            ("ATM", -0.03, 0.03),
            ("OTM Call", 0.03, 0.10),
            ("Deep OTM Call", 0.10, 0.30),
        ]:
            bucket = with_iv.where(
                lambda s, l=lo, h=hi: (
                    s.iv is not None
                    and l <= (s.strike - underlying_price) / underlying_price <= h
                )
            )
            if len(bucket) > 0:
                ivs = bucket.map(lambda s: s.iv)
                iv_by_moneyness[label] = sum(ivs) / len(ivs)

    # Calls vs puts
    calls_iv = idx.calls().has_iv()
    puts_iv = idx.puts().has_iv()
    call_avg = sum(calls_iv.map(lambda s: s.iv)) / len(calls_iv) if len(calls_iv) > 0 else None
    put_avg = sum(puts_iv.map(lambda s: s.iv)) / len(puts_iv) if len(puts_iv) > 0 else None

    # ATM IV
    atm = idx.where(
        lambda s: s.delta is not None and 0.35 < abs(s.delta) < 0.65 and s.iv is not None
    )
    atm_iv = sum(atm.map(lambda s: s.iv)) / len(atm) if len(atm) > 0 else None

    return {
        "iv_by_strike": iv_by_strike,
        "iv_by_moneyness": iv_by_moneyness,
        "call_avg_iv": call_avg,
        "put_avg_iv": put_avg,
        "atm_iv": atm_iv,
        "put_call_skew": (put_avg - call_avg) if (put_avg and call_avg) else None,
        "n_contracts": len(idx),
        "n_with_iv": len(with_iv),
    }


def compute_expiry_summary(idx: BlobIndex) -> list[dict]:
    """Summary of options by expiry date."""
    by_expiry = idx.has_iv().group_by(lambda s: s.expiry)
    rows = []
    for exp in sorted(by_expiry.keys()):
        grp = by_expiry[exp]
        ivs = grp.map(lambda s: s.iv)
        calls = grp.calls()
        puts = grp.puts()
        rows.append({
            "expiry": exp,
            "n_contracts": len(grp),
            "avg_iv": sum(ivs) / len(ivs),
            "min_iv": min(ivs),
            "max_iv": max(ivs),
            "n_calls": len(calls),
            "n_puts": len(puts),
        })
    return rows


def compute_greeks_summary(idx: BlobIndex) -> dict | None:
    """ATM Greeks summary."""
    atm = idx.where(
        lambda s: s.delta is not None and 0.35 < abs(s.delta) < 0.65
    )
    if len(atm) == 0:
        return None

    calls = atm.calls()
    puts = atm.puts()

    def avg_greek(subset, attr):
        vals = [getattr(s, attr) for s in subset if getattr(s, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "atm_call_delta": avg_greek(calls, "delta"),
        "atm_call_gamma": avg_greek(calls, "gamma"),
        "atm_call_theta": avg_greek(calls, "theta"),
        "atm_call_vega": avg_greek(calls, "vega"),
        "atm_put_delta": avg_greek(puts, "delta"),
        "atm_put_gamma": avg_greek(puts, "gamma"),
        "atm_put_theta": avg_greek(puts, "theta"),
        "atm_put_vega": avg_greek(puts, "vega"),
        "n_atm": len(atm),
    }


def generate_iv_smile_chart(
    smile_early: dict,
    smile_latest: dict,
    date_early: str,
    date_latest: str,
    out_path: str,
) -> None:
    """Generate IV smile comparison chart using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"IWN Implied Volatility Smile: {date_early} vs {date_latest}",
        fontsize=13, fontweight="bold",
    )

    C_EARLY = "#FF9800"
    C_LATEST = "#2196F3"

    # ── Panel 1: IV by strike ─────────────────────────────────────────
    ax = axes[0]
    strikes_e = sorted(smile_early.get("iv_by_strike", {}).keys())
    strikes_l = sorted(smile_latest.get("iv_by_strike", {}).keys())
    all_strikes = sorted(set(strikes_e) | set(strikes_l))

    if all_strikes:
        ivs_e = [smile_early["iv_by_strike"].get(k) for k in all_strikes]
        ivs_l = [smile_latest["iv_by_strike"].get(k) for k in all_strikes]

        # Filter to strikes that have data for at least one date
        plot_strikes, plot_e, plot_l = [], [], []
        for s, ie, il in zip(all_strikes, ivs_e, ivs_l):
            if ie is not None or il is not None:
                plot_strikes.append(s)
                plot_e.append(ie)
                plot_l.append(il)

        if plot_e and any(v is not None for v in plot_e):
            ax.plot(
                [s for s, v in zip(plot_strikes, plot_e) if v is not None],
                [v * 100 for v in plot_e if v is not None],
                "o-", color=C_EARLY, label=date_early, linewidth=2, markersize=4,
            )
        if plot_l and any(v is not None for v in plot_l):
            ax.plot(
                [s for s, v in zip(plot_strikes, plot_l) if v is not None],
                [v * 100 for v in plot_l if v is not None],
                "s-", color=C_LATEST, label=date_latest, linewidth=2, markersize=4,
            )

    ax.set_xlabel("Strike ($)", fontsize=10)
    ax.set_ylabel("Implied Volatility (%)", fontsize=10)
    ax.set_title("IV Smile by Strike", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── Panel 2: IV by moneyness bucket ───────────────────────────────
    ax = axes[1]
    labels_order = ["Deep OTM Put", "OTM Put", "ATM", "OTM Call", "Deep OTM Call"]
    m_early = smile_early.get("iv_by_moneyness", {})
    m_latest = smile_latest.get("iv_by_moneyness", {})
    labels = [l for l in labels_order if l in m_early or l in m_latest]

    if labels:
        x = np.arange(len(labels))
        width = 0.35
        vals_e = [m_early.get(l, 0) * 100 for l in labels]
        vals_l = [m_latest.get(l, 0) * 100 for l in labels]

        bars_e = ax.bar(x - width / 2, vals_e, width, color=C_EARLY, alpha=0.7, label=date_early)
        bars_l = ax.bar(x + width / 2, vals_l, width, color=C_LATEST, alpha=0.7, label=date_latest)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15)

    ax.set_ylabel("Implied Volatility (%)", fontsize=10)
    ax.set_title("IV by Moneyness", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved chart: {out_path}")


def generate_term_structure_chart(
    expiry_data_early: list[dict],
    expiry_data_latest: list[dict],
    date_early: str,
    date_latest: str,
    out_path: str,
) -> None:
    """Generate IV term structure chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"IWN IV Term Structure: {date_early} vs {date_latest}",
        fontsize=13, fontweight="bold",
    )

    C_EARLY = "#FF9800"
    C_LATEST = "#2196F3"

    if expiry_data_early:
        exps = [r["expiry"] for r in expiry_data_early]
        ivs = [r["avg_iv"] * 100 for r in expiry_data_early]
        ax.plot(exps, ivs, "o-", color=C_EARLY, label=date_early, linewidth=2, markersize=5)

    if expiry_data_latest:
        exps = [r["expiry"] for r in expiry_data_latest]
        ivs = [r["avg_iv"] * 100 for r in expiry_data_latest]
        ax.plot(exps, ivs, "s-", color=C_LATEST, label=date_latest, linewidth=2, markersize=5)

    ax.set_xlabel("Expiry Date", fontsize=10)
    ax.set_ylabel("Average IV (%)", fontsize=10)
    ax.set_title("Average IV by Expiration", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved chart: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  IWN VOL ANALYSIS REPORT GENERATOR")
    print("=" * 70)

    client = BlobClient()

    # ── Discover available dates ──────────────────────────────────────
    dates = client.list_option_dates("IWN")
    print(f"\nIWN snapshot dates available: {len(dates)}")
    if dates:
        print(f"  Newest: {dates[0]}")
        print(f"  Oldest: {dates[-1]}")

    # Find dates closest to targets
    early_dates = [d for d in dates if d.startswith("2026-03-0")]
    latest_date = dates[0] if dates else None

    # Pick the earliest March date (closest to Mar 4)
    early_date = None
    for d in sorted(early_dates):
        if d >= TARGET_DATE_EARLY:
            early_date = d
            break
    if not early_date and early_dates:
        early_date = early_dates[-1]

    if not early_date or not latest_date:
        print("ERROR: Insufficient blob data")
        return

    print(f"\n  Early snapshot:  {early_date}")
    print(f"  Latest snapshot: {latest_date}")

    # ── Fetch options chains ──────────────────────────────────────────
    print(f"\nFetching options chain for {early_date}...")
    chain_early = client.get_options_chain("IWN", date=early_date)
    idx_early = BlobIndex(chain_early.snapshots)
    price_early = derive_underlying_price(chain_early.snapshots)
    print(f"  {len(idx_early)} contracts, underlying ~${price_early:.2f}" if price_early else f"  {len(idx_early)} contracts")

    print(f"\nFetching options chain for {latest_date}...")
    chain_latest = client.get_options_chain("IWN", date=latest_date)
    idx_latest = BlobIndex(chain_latest.snapshots)
    price_latest = derive_underlying_price(chain_latest.snapshots)
    print(f"  {len(idx_latest)} contracts, underlying ~${price_latest:.2f}" if price_latest else f"  {len(idx_latest)} contracts")

    # ── Compute IV smile ──────────────────────────────────────────────
    print("\nComputing IV smile...")
    smile_early = compute_iv_smile(idx_early, price_early)
    smile_latest = compute_iv_smile(idx_latest, price_latest)

    print(f"\n  Early ({early_date}):")
    print(f"    ATM IV: {smile_early.get('atm_iv', 0):.1%}")
    print(f"    Call avg IV: {smile_early.get('call_avg_iv', 0):.1%}")
    print(f"    Put avg IV: {smile_early.get('put_avg_iv', 0):.1%}")
    print(f"    Put-Call skew: {smile_early.get('put_call_skew', 0):+.1%}")

    print(f"\n  Latest ({latest_date}):")
    print(f"    ATM IV: {smile_latest.get('atm_iv', 0):.1%}")
    print(f"    Call avg IV: {smile_latest.get('call_avg_iv', 0):.1%}")
    print(f"    Put avg IV: {smile_latest.get('put_avg_iv', 0):.1%}")
    print(f"    Put-Call skew: {smile_latest.get('put_call_skew', 0):+.1%}")

    # ── Expiry summaries ──────────────────────────────────────────────
    print("\nComputing expiry summaries...")
    expiry_early = compute_expiry_summary(idx_early)
    expiry_latest = compute_expiry_summary(idx_latest)

    # ── Greeks summaries ──────────────────────────────────────────────
    greeks_early = compute_greeks_summary(idx_early)
    greeks_latest = compute_greeks_summary(idx_latest)

    # ── Generate charts ───────────────────────────────────────────────
    os.makedirs(DOCS_DIR, exist_ok=True)

    smile_chart_path = os.path.join(DOCS_DIR, "iwn_iv_smile.png")
    generate_iv_smile_chart(
        smile_early, smile_latest,
        early_date[:10], latest_date[:10],
        smile_chart_path,
    )

    term_chart_path = os.path.join(DOCS_DIR, "iwn_iv_term_structure.png")
    generate_term_structure_chart(
        expiry_early, expiry_latest,
        early_date[:10], latest_date[:10],
        term_chart_path,
    )

    # ── Save data as JSON for LaTeX generation ────────────────────────
    report_data = {
        "early_date": early_date[:10],
        "latest_date": latest_date[:10],
        "price_early": price_early,
        "price_latest": price_latest,
        "smile_early": {
            "atm_iv": smile_early.get("atm_iv"),
            "call_avg_iv": smile_early.get("call_avg_iv"),
            "put_avg_iv": smile_early.get("put_avg_iv"),
            "put_call_skew": smile_early.get("put_call_skew"),
            "n_contracts": smile_early.get("n_contracts", 0),
            "n_with_iv": smile_early.get("n_with_iv", 0),
            "iv_by_moneyness": smile_early.get("iv_by_moneyness", {}),
        },
        "smile_latest": {
            "atm_iv": smile_latest.get("atm_iv"),
            "call_avg_iv": smile_latest.get("call_avg_iv"),
            "put_avg_iv": smile_latest.get("put_avg_iv"),
            "put_call_skew": smile_latest.get("put_call_skew"),
            "n_contracts": smile_latest.get("n_contracts", 0),
            "n_with_iv": smile_latest.get("n_with_iv", 0),
            "iv_by_moneyness": smile_latest.get("iv_by_moneyness", {}),
        },
        "expiry_early": expiry_early,
        "expiry_latest": expiry_latest,
        "greeks_early": greeks_early,
        "greeks_latest": greeks_latest,
    }

    data_path = os.path.join(DOCS_DIR, "report_data.json")
    with open(data_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n  Report data saved: {data_path}")

    # ── Print summary tables for LaTeX reference ──────────────────────
    print(f"\n{'=' * 80}")
    print("EXPIRY SUMMARY (Early)")
    print(f"{'=' * 80}")
    print(f"  {'Expiry':<12} {'Contracts':>10} {'Avg IV':>10} {'Min IV':>10} {'Max IV':>10}")
    for r in expiry_early[:8]:
        print(f"  {r['expiry']:<12} {r['n_contracts']:>10} {r['avg_iv']:>9.1%} {r['min_iv']:>9.1%} {r['max_iv']:>9.1%}")

    print(f"\n{'=' * 80}")
    print("EXPIRY SUMMARY (Latest)")
    print(f"{'=' * 80}")
    print(f"  {'Expiry':<12} {'Contracts':>10} {'Avg IV':>10} {'Min IV':>10} {'Max IV':>10}")
    for r in expiry_latest[:8]:
        print(f"  {r['expiry']:<12} {r['n_contracts']:>10} {r['avg_iv']:>9.1%} {r['min_iv']:>9.1%} {r['max_iv']:>9.1%}")

    if greeks_early:
        print(f"\nATM Greeks (Early): delta={greeks_early.get('atm_call_delta', 'n/a')}, "
              f"gamma={greeks_early.get('atm_call_gamma', 'n/a')}, "
              f"theta={greeks_early.get('atm_call_theta', 'n/a')}, "
              f"vega={greeks_early.get('atm_call_vega', 'n/a')}")

    if greeks_latest:
        print(f"ATM Greeks (Latest): delta={greeks_latest.get('atm_call_delta', 'n/a')}, "
              f"gamma={greeks_latest.get('atm_call_gamma', 'n/a')}, "
              f"theta={greeks_latest.get('atm_call_theta', 'n/a')}, "
              f"vega={greeks_latest.get('atm_call_vega', 'n/a')}")

    # ── Compile LaTeX ─────────────────────────────────────────────────
    tex_path = os.path.join(DOCS_DIR, "iwn_vol_analysis.tex")
    pdf_path = os.path.join(DOCS_DIR, "iwn_vol_analysis.pdf")
    print(f"\nCompiling LaTeX: {tex_path}")
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", DOCS_DIR, tex_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  PDF generated: {pdf_path}")
        else:
            print(f"  pdflatex returned {result.returncode}")
            # Print last few lines of output for debugging
            lines = result.stdout.strip().split("\n")
            for line in lines[-10:]:
                print(f"    {line}")
    except FileNotFoundError:
        print("  pdflatex not found — install texlive to compile")
    except subprocess.TimeoutExpired:
        print("  pdflatex timed out")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
