#!/usr/bin/env python3
"""Check IWN implied volatility across snapshots.

Fetches all available IWN options-chain blobs and analyses IV by date,
strike, expiry, and moneyness using BlobIndex lambda comparators.

Requires:
    ALLOC_ENGINE_SITE_ID  — Netlify site ID
    NETLIFY_AUTH_TOKEN     — Netlify PAT
"""

from allocation_gym.blobs import BlobClient, BlobIndex, OptionSnapshot


def main() -> None:
    client = BlobClient()

    # ── Discover what we have ────────────────────────────────
    symbols = client.list_option_symbols()
    print(f"Available symbols: {symbols}")

    if "IWN" not in symbols:
        print("IWN not found in blob store")
        return

    dates = client.list_option_dates("IWN")
    print(f"IWN snapshot dates ({len(dates)}): {dates[:10]}{'...' if len(dates) > 10 else ''}")

    # ── Load every snapshot date into one big index ──────────
    all_snapshots: list[OptionSnapshot] = []
    date_summaries: list[dict] = []

    for date in dates:
        chain = client.get_options_chain("IWN", date=date)
        idx = BlobIndex(chain.snapshots)
        with_iv = idx.has_iv()

        if len(with_iv) == 0:
            continue

        ivs = with_iv.map(lambda s: s.iv)
        avg_iv = sum(ivs) / len(ivs)
        min_iv = min(ivs)
        max_iv = max(ivs)

        date_summaries.append({
            "date": date,
            "contracts": len(idx),
            "with_iv": len(with_iv),
            "avg_iv": avg_iv,
            "min_iv": min_iv,
            "max_iv": max_iv,
        })
        all_snapshots.extend(chain.snapshots)

    # ── Daily IV summary ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"IWN IV Summary across {len(date_summaries)} dates")
    print(f"{'='*70}")
    print(f"{'Date':<14} {'Contracts':>10} {'w/ IV':>8} {'Avg IV':>10} {'Min IV':>10} {'Max IV':>10}")
    print("-" * 70)
    for s in date_summaries:
        print(f"{s['date']:<14} {s['contracts']:>10} {s['with_iv']:>8} "
              f"{s['avg_iv']:>9.1%} {s['min_iv']:>9.1%} {s['max_iv']:>9.1%}")

    # ── Build a combined index across all dates ──────────────
    idx = BlobIndex(all_snapshots)
    print(f"\nTotal snapshots loaded: {len(idx)}")

    # ── IV thresholds: when is IV > X? ───────────────────────
    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
        above = idx.where(lambda s, t=threshold: s.iv is not None and s.iv > t)
        print(f"\n--- IV > {threshold:.0%}: {len(above)} contracts ---")
        if len(above) > 0:
            # Group by expiry to see which expiries are elevated
            by_expiry = above.group_by(lambda s: s.expiry)
            for exp, grp in sorted(by_expiry.items()):
                avg = sum(grp.map(lambda s: s.iv)) / len(grp)
                print(f"  expiry={exp}  count={len(grp)}  avg_iv={avg:.1%}")

    # ── Calls vs Puts IV comparison ──────────────────────────
    calls_iv = idx.calls().has_iv()
    puts_iv = idx.puts().has_iv()
    if len(calls_iv) > 0 and len(puts_iv) > 0:
        call_avg = sum(calls_iv.map(lambda s: s.iv)) / len(calls_iv)
        put_avg = sum(puts_iv.map(lambda s: s.iv)) / len(puts_iv)
        skew = put_avg - call_avg
        print(f"\n{'='*70}")
        print(f"Call avg IV: {call_avg:.2%}  ({len(calls_iv)} contracts)")
        print(f"Put avg IV:  {put_avg:.2%}  ({len(puts_iv)} contracts)")
        print(f"Put-Call IV skew: {skew:+.2%}")

    # ── IV smile by strike bucket ────────────────────────────
    with_iv = idx.has_iv()
    if len(with_iv) > 0:
        iv_by_strike = with_iv.aggregate(
            key_fn=lambda s: round(s.strike / 5) * 5,
            value_fn=lambda s: s.iv,
            agg="mean",
        )
        print(f"\n{'='*70}")
        print("IV Smile (avg IV by $5 strike bucket):")
        for strike in sorted(iv_by_strike):
            bar = "█" * int(iv_by_strike[strike] * 100)
            print(f"  ${strike:>6.0f}  {iv_by_strike[strike]:>6.1%}  {bar}")

    # ── ATM vs OTM IV ────────────────────────────────────────
    atm = idx.atm().has_iv()
    otm = idx.otm().has_iv()
    itm = idx.itm().has_iv()
    for label, subset in [("ATM", atm), ("OTM", otm), ("ITM", itm)]:
        if len(subset) > 0:
            avg = sum(subset.map(lambda s: s.iv)) / len(subset)
            print(f"  {label}: avg IV = {avg:.2%}  ({len(subset)} contracts)")

    # ── Top 10 highest IV contracts ──────────────────────────
    top10 = with_iv.sort_by(lambda s: -s.iv)[:10]
    print(f"\n{'='*70}")
    print("Top 10 highest IV contracts:")
    print(f"  {'Symbol':<22} {'Strike':>8} {'Type':<5} {'Expiry':<12} {'IV':>8} {'Delta':>8} {'Mid':>8}")
    for s in top10:
        print(f"  {s.symbol:<22} {s.strike:>8.0f} {s.option_type:<5} {s.expiry:<12} "
              f"{s.iv:>7.1%} {s.delta or 0:>8.3f} {s.mid:>8.2f}")


if __name__ == "__main__":
    main()
