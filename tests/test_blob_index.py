"""Tests for BlobIndex lambda-based filtering and indexing."""

from allocation_gym.blobs.models import OptionSnapshot, OptionGreeks, OptionQuote
from allocation_gym.blobs.index import BlobIndex


def _make_snap(
    symbol: str = "TEST260418C00200000",
    delta: float | None = 0.5,
    iv: float | None = 0.35,
    bid: float = 2.0,
    ask: float = 2.50,
) -> OptionSnapshot:
    greeks = OptionGreeks(delta=delta, gamma=0.05, theta=-0.02, vega=0.15) if delta is not None else None
    quote = OptionQuote(bid=bid, ask=ask, bid_size=10, ask_size=12)
    return OptionSnapshot(
        symbol=symbol,
        latest_quote=quote,
        greeks=greeks,
        implied_volatility=iv,
    )


SNAPSHOTS = [
    _make_snap("CRWD260418C00350000", delta=0.6, iv=0.40, bid=12.0, ask=13.0),   # ITM call
    _make_snap("CRWD260418C00400000", delta=0.48, iv=0.35, bid=5.0, ask=5.50),    # ATM call
    _make_snap("CRWD260418C00450000", delta=0.2, iv=0.30, bid=1.0, ask=1.50),     # OTM call
    _make_snap("CRWD260418P00350000", delta=-0.4, iv=0.38, bid=3.0, ask=3.50),    # OTM put
    _make_snap("CRWD260418P00400000", delta=-0.52, iv=0.36, bid=6.0, ask=6.50),   # ITM put (ATM-ish)
    _make_snap("CRWD260418P00450000", delta=-0.8, iv=0.32, bid=15.0, ask=15.50),  # Deep ITM put
    _make_snap("CRWD260620C00400000", delta=0.55, iv=0.33, bid=10.0, ask=10.50),  # Later expiry call
    _make_snap("CRWD260620P00400000", delta=-0.45, iv=0.34, bid=8.0, ask=8.50),   # Later expiry put
    _make_snap("IWN260418C00200000", delta=0.5, iv=0.25, bid=4.0, ask=4.50),      # Different underlying
    _make_snap("IWN260418C00210000", delta=None, iv=None, bid=0.50, ask=1.0),      # No greeks
]


def test_where_calls_and_puts():
    idx = BlobIndex(SNAPSHOTS)
    calls = idx.calls()
    puts = idx.puts()
    assert len(calls) == 6  # 4 CRWD calls + 2 IWN calls
    assert len(puts) == 4
    assert all(s.is_call for s in calls)
    assert all(s.is_put for s in puts)


def test_where_lambda_strike_range():
    idx = BlobIndex(SNAPSHOTS)
    near_400 = idx.where(lambda s: 395 <= s.strike <= 405)
    assert len(near_400) == 4  # CRWD C400, P400, later expiry C400, P400


def test_chained_filters():
    idx = BlobIndex(SNAPSHOTS)
    result = (
        idx.calls()
           .where(lambda s: s.underlying == "CRWD")
           .where(lambda s: s.expiry == "2026-04-18")
           .where(lambda s: s.delta is not None and s.delta > 0.3)
    )
    assert len(result) == 2  # 350 call (0.6) and 400 call (0.48)
    assert all(s.strike in (350.0, 400.0) for s in result)


def test_moneyness_filters():
    idx = BlobIndex(SNAPSHOTS)
    atm = idx.atm()
    itm = idx.itm()
    otm = idx.otm()
    assert len(atm) >= 2   # delta near 0.5
    assert len(itm) >= 2
    assert len(otm) >= 2


def test_iv_range():
    idx = BlobIndex(SNAPSHOTS)
    high_iv = idx.iv_range(0.35, 1.0)
    assert all(s.iv >= 0.35 for s in high_iv)
    assert len(high_iv) >= 3


def test_delta_range():
    idx = BlobIndex(SNAPSHOTS)
    positive_delta = idx.delta_range(0.3, 1.0)
    assert all(s.delta >= 0.3 for s in positive_delta)


def test_has_greeks_and_iv():
    idx = BlobIndex(SNAPSHOTS)
    with_greeks = idx.has_greeks()
    with_iv = idx.has_iv()
    assert len(with_greeks) == 9  # all except the last one
    assert len(with_iv) == 9


def test_build_index_and_get():
    idx = BlobIndex(SNAPSHOTS)
    idx.build_index("by_expiry", key_fn=lambda s: s.expiry)

    apr = idx.get("by_expiry", "2026-04-18")
    jun = idx.get("by_expiry", "2026-06-20")
    assert len(apr) == 8
    assert len(jun) == 2

    keys = idx.index_keys("by_expiry")
    assert set(keys) == {"2026-04-18", "2026-06-20"}


def test_build_index_by_underlying():
    idx = BlobIndex(SNAPSHOTS)
    idx.build_index("by_underlying", key_fn=lambda s: s.underlying)

    crwd = idx.get("by_underlying", "CRWD")
    iwn = idx.get("by_underlying", "IWN")
    assert len(crwd) == 8
    assert len(iwn) == 2


def test_build_index_strike_bucket():
    idx = BlobIndex(SNAPSHOTS)
    idx.build_index("by_strike_50", key_fn=lambda s: round(s.strike / 50) * 50)
    bucket_400 = idx.get("by_strike_50", 400)
    assert len(bucket_400) >= 4


def test_sort_by():
    idx = BlobIndex(SNAPSHOTS)
    by_strike = idx.sort_by(lambda s: s.strike)
    strikes = [s.strike for s in by_strike]
    assert strikes == sorted(strikes)

    by_iv_desc = idx.has_iv().sort_by(lambda s: s.iv, reverse=True)
    ivs = [s.iv for s in by_iv_desc]
    assert ivs == sorted(ivs, reverse=True)


def test_group_by():
    idx = BlobIndex(SNAPSHOTS)
    groups = idx.group_by(lambda s: s.option_type)
    assert "call" in groups
    assert "put" in groups
    assert len(groups["call"]) == 6
    assert len(groups["put"]) == 4


def test_aggregate():
    idx = BlobIndex(SNAPSHOTS)
    avg_iv = idx.has_iv().aggregate(
        key_fn=lambda s: s.underlying,
        value_fn=lambda s: s.iv,
        agg="mean",
    )
    assert "CRWD" in avg_iv
    assert "IWN" in avg_iv
    assert 0.2 < avg_iv["CRWD"] < 0.5
    assert 0.2 < avg_iv["IWN"] < 0.3


def test_map():
    idx = BlobIndex(SNAPSHOTS)
    mids = idx.map(lambda s: s.mid)
    assert len(mids) == 10
    assert all(isinstance(m, float) for m in mids)


def test_first():
    idx = BlobIndex(SNAPSHOTS)
    first_call = idx.first(lambda s: s.is_call)
    assert first_call is not None
    assert first_call.is_call

    missing = idx.first(lambda s: s.underlying == "AAPL")
    assert missing is None


def test_expiry_filter():
    idx = BlobIndex(SNAPSHOTS)
    apr = idx.expiry("2026-04-18")
    assert len(apr) == 8


def test_empty_index():
    idx = BlobIndex([])
    assert len(idx) == 0
    assert len(idx.calls()) == 0
    assert idx.first() is None


def test_occ_parsing():
    snap = _make_snap("CRWD260418C00350000")
    assert snap.underlying == "CRWD"
    assert snap.expiry == "2026-04-18"
    assert snap.option_type == "call"
    assert snap.strike == 350.0
    assert snap.is_call
    assert not snap.is_put


def test_quote_properties():
    q = OptionQuote(bid=10.0, ask=10.50, bid_size=100, ask_size=200)
    assert q.mid == 10.25
    assert q.spread == 0.50
    assert abs(q.spread_bps - 487.8) < 1  # 0.50/10.25 * 10000


def test_to_dataframe():
    idx = BlobIndex(SNAPSHOTS)
    df = idx.to_dataframe(columns=["symbol", "strike", "option_type", "implied_volatility"])
    assert len(df) == 10
    assert list(df.columns) == ["symbol", "strike", "option_type", "implied_volatility"]


# ── Multiple records per date (duplicate symbols across intraday snapshots) ──
#
# Scenario: the blob store accumulates snapshots across multiple intraday polls.
# The same OCC symbol appears once per poll, so a "by_symbol" index built over
# the combined list contains N entries per symbol instead of 1.
#
# Expected behaviour (once fixed): build_index("by_symbol") should keep only
# the LATEST record per symbol so that callers can do:
#
#     snap = idx.get("by_symbol", "CRWD260418C00350000")[0]
#
# and be guaranteed a single result.
#
# This test FAILS with the current implementation because no deduplication is
# performed — the index accumulates all three intraday records.

def _make_intraday_snap(symbol: str, ts: str, bid: float, iv: float) -> OptionSnapshot:
    quote = OptionQuote(bid=bid, ask=bid + 0.50, bid_size=10, ask_size=10, timestamp=ts)
    greeks = OptionGreeks(delta=0.5, gamma=0.05, theta=-0.02, vega=0.15)
    return OptionSnapshot(symbol=symbol, latest_quote=quote, greeks=greeks, implied_volatility=iv)


def test_build_index_deduplicates_by_symbol():
    """build_index('by_symbol') with dedup_fn keeps only the latest record per symbol."""
    sym = "CRWD260418C00350000"

    # Three intraday polls for the same contract — latest has highest IV.
    snapshots = [
        _make_intraday_snap(sym, "2026-03-24T09:30:00Z", bid=10.0, iv=0.30),
        _make_intraday_snap(sym, "2026-03-24T11:00:00Z", bid=11.0, iv=0.33),
        _make_intraday_snap(sym, "2026-03-24T14:30:00Z", bid=12.5, iv=0.36),
    ]

    idx = BlobIndex(snapshots)
    idx.build_index(
        "by_symbol",
        key_fn=lambda s: s.symbol,
        dedup_fn=lambda s: s.latest_quote.timestamp,
    )

    result = idx.get("by_symbol", sym)

    # Expect exactly one (the latest) record per symbol.
    assert len(result) == 1, (
        f"Expected 1 deduplicated record for {sym}, got {len(result)}."
    )
    assert result[0].latest_quote.timestamp == "2026-03-24T14:30:00Z"
    assert result[0].iv == 0.36
