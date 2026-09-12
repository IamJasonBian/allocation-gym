#!/usr/bin/env python3
"""Compare IWN implied vol vs realized vol across snapshot dates.

Derives underlying price via put-call parity from options chain blobs,
computes rolling realized vol, and highlights when IV > RV (overpriced)
or IV < RV (underpriced).

Requires:
    ALLOC_ENGINE_SITE_ID  — Netlify site ID
    NETLIFY_AUTH_TOKEN     — Netlify PAT
"""

import math
from collections import defaultdict

from allocation_gym.blobs import BlobClient, BlobIndex, OptionSnapshot


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
            strike = key[0]
            implied_s = calls[key] - puts[key] + strike
            if 100 < implied_s < 400:  # IWN sanity bounds
                estimates.append(implied_s)

    if not estimates:
        return None
    return sum(estimates) / len(estimates)


def compute_realized_vol(prices: list[float], annualize: int = 252) -> float | None:
    """Compute annualized realized vol from a price series."""
    if len(prices) < 2:
        return None
    log_returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    if not log_returns:
        return None
    mean = sum(log_returns) / len(log_returns)
    var = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    daily_vol = math.sqrt(var)
    return daily_vol * math.sqrt(annualize)


def main() -> None:
    client = BlobClient()
    dates = client.list_option_dates("IWN")
    print(f"IWN snapshot dates: {len(dates)}")

    # ── Collect per-date: underlying price (via parity), avg IV, IV by bucket ──
    date_data: list[dict] = []

    for date in sorted(dates):
        chain = client.get_options_chain("IWN", date=date)
        idx = BlobIndex(chain.snapshots)
        with_iv = idx.has_iv()

        price = derive_underlying_price(chain.snapshots)
        avg_iv = None
        if len(with_iv) > 0:
            ivs = with_iv.map(lambda s: s.iv)
            avg_iv = sum(ivs) / len(ivs)

        # ATM IV (delta near 0.50)
        atm = idx.where(
            lambda s: s.delta is not None and 0.35 < abs(s.delta) < 0.65 and s.iv is not None
        )
        atm_iv = None
        if len(atm) > 0:
            atm_ivs = atm.map(lambda s: s.iv)
            atm_iv = sum(atm_ivs) / len(atm_ivs)

        # Call vs put IV
        call_iv = None
        put_iv = None
        calls_with_iv = idx.calls().has_iv()
        puts_with_iv = idx.puts().has_iv()
        if len(calls_with_iv) > 0:
            civs = calls_with_iv.map(lambda s: s.iv)
            call_iv = sum(civs) / len(civs)
        if len(puts_with_iv) > 0:
            pivs = puts_with_iv.map(lambda s: s.iv)
            put_iv = sum(pivs) / len(pivs)

        date_data.append({
            "date": date,
            "price": price,
            "avg_iv": avg_iv,
            "atm_iv": atm_iv,
            "call_iv": call_iv,
            "put_iv": put_iv,
            "n_contracts": len(idx),
            "n_with_iv": len(with_iv),
        })

    # ── Compute rolling realized vol ─────────────────────────
    prices = [d["price"] for d in date_data if d["price"] is not None]
    prices_dates = [d["date"] for d in date_data if d["price"] is not None]

    # Rolling windows: 5-day, 10-day, full
    def rolling_rv(prices_list, window):
        result = {}
        for i in range(len(prices_list)):
            start = max(0, i - window + 1)
            segment = prices_list[start : i + 1]
            rv = compute_realized_vol(segment)
            result[prices_dates[i]] = rv
        return result

    rv_5d = rolling_rv(prices, 5)
    rv_10d = rolling_rv(prices, 10)
    rv_full = compute_realized_vol(prices)

    # ── Print results ────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"IWN: Implied Vol vs Realized Vol")
    print(f"{'=' * 100}")
    print(
        f"{'Date':<12} {'Price':>8} {'Avg IV':>8} {'ATM IV':>8} "
        f"{'RV 5d':>8} {'RV 10d':>8} {'IV-RV 5d':>10} {'IV-RV 10d':>10} {'Signal':>12}"
    )
    print("-" * 100)

    for d in date_data:
        date = d["date"]
        price_str = f"${d['price']:.2f}" if d["price"] else "   n/a"
        avg_iv_str = f"{d['avg_iv']:.1%}" if d["avg_iv"] else "  n/a"
        atm_iv_str = f"{d['atm_iv']:.1%}" if d["atm_iv"] else "  n/a"

        rv5 = rv_5d.get(date)
        rv10 = rv_10d.get(date)
        rv5_str = f"{rv5:.1%}" if rv5 else "  n/a"
        rv10_str = f"{rv10:.1%}" if rv10 else "  n/a"

        # IV - RV spread (use ATM IV for cleaner signal)
        iv = d["atm_iv"] or d["avg_iv"]
        spread5 = None
        spread10 = None
        if iv is not None and rv5 is not None:
            spread5 = iv - rv5
        if iv is not None and rv10 is not None:
            spread10 = iv - rv10

        spread5_str = f"{spread5:+.1%}" if spread5 is not None else "     n/a"
        spread10_str = f"{spread10:+.1%}" if spread10 is not None else "     n/a"

        # Signal: when IV significantly differs from RV
        signal = ""
        if spread10 is not None:
            if spread10 > 0.10:
                signal = "IV >> RV"
            elif spread10 > 0.05:
                signal = "IV > RV"
            elif spread10 < -0.10:
                signal = "IV << RV"
            elif spread10 < -0.05:
                signal = "IV < RV"
            else:
                signal = "~fair"

        print(
            f"{date:<12} {price_str:>8} {avg_iv_str:>8} {atm_iv_str:>8} "
            f"{rv5_str:>8} {rv10_str:>8} {spread5_str:>10} {spread10_str:>10} {signal:>12}"
        )

    # ── Summary stats ────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"Full-period realized vol ({len(prices)} data points): ", end="")
    print(f"{rv_full:.1%}" if rv_full else "insufficient data")

    # Average IV across all dates
    all_ivs = [d["avg_iv"] for d in date_data if d["avg_iv"] is not None]
    if all_ivs:
        print(f"Average implied vol: {sum(all_ivs)/len(all_ivs):.1%}")

    if rv_full and all_ivs:
        avg_iv = sum(all_ivs) / len(all_ivs)
        vrp = avg_iv - rv_full
        print(f"Volatility Risk Premium (IV - RV): {vrp:+.1%}")
        if vrp > 0.05:
            print("  -> Options are OVERPRICED relative to realized moves (sell vol)")
        elif vrp < -0.05:
            print("  -> Options are UNDERPRICED relative to realized moves (buy vol)")
        else:
            print("  -> Options are ~fairly priced")

    # ── Call vs Put skew over time ───────────────────────────
    print(f"\n{'=' * 100}")
    print("Put-Call IV Skew Over Time:")
    print(f"{'Date':<12} {'Call IV':>10} {'Put IV':>10} {'Skew':>10}")
    print("-" * 45)
    for d in date_data:
        if d["call_iv"] and d["put_iv"]:
            skew = d["put_iv"] - d["call_iv"]
            print(f"{d['date']:<12} {d['call_iv']:>9.1%} {d['put_iv']:>9.1%} {skew:>+9.1%}")


if __name__ == "__main__":
    main()
