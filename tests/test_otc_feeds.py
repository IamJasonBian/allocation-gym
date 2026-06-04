"""Unit tests for the OTC IS-pricing feeds layer.

These tests exercise the deterministic :class:`MockL2Feed`, the order-book
schema (mid / micro-price / staleness), the datafeed-drop-resilient
:func:`build_index_price`, and the fallback behavior of
:class:`BinanceDepthFeed` *without making any network calls* (the fetch is
monkeypatched to fail).
"""

from __future__ import annotations

import pytest

from allocation_gym.otc_is_pricing.feeds import (
    BinanceDepthFeed,
    BookLevel,
    IndexResult,
    MockL2Feed,
    OrderBookSnapshot,
    build_index_price,
)


def _snap(
    *,
    symbol: str = "ALTUSDT",
    ts: float,
    bid_px: float,
    bid_sz: float,
    ask_px: float,
    ask_sz: float,
    source: str = "mock",
) -> OrderBookSnapshot:
    """Build a minimal one-level snapshot for schema tests."""
    return OrderBookSnapshot(
        symbol=symbol,
        ts=ts,
        bids=[BookLevel(price=bid_px, size=bid_sz)],
        asks=[BookLevel(price=ask_px, size=ask_sz)],
        source=source,
    )


class TestOrderBookSnapshot:
    """Schema-level invariants for OrderBookSnapshot."""

    def test_mid(self) -> None:
        s = _snap(ts=0.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0)
        assert s.mid == pytest.approx(100.0)

    def test_microprice_formula(self) -> None:
        # (bid_px*ask_sz + ask_px*bid_sz) / (bid_sz + ask_sz)
        s = _snap(ts=0.0, bid_px=99.0, bid_sz=3.0, ask_px=101.0, ask_sz=1.0)
        expected = (99.0 * 1.0 + 101.0 * 3.0) / (3.0 + 1.0)
        assert s.microprice == pytest.approx(expected)

    def test_microprice_equals_mid_when_balanced(self) -> None:
        s = _snap(ts=0.0, bid_px=99.0, bid_sz=2.0, ask_px=101.0, ask_sz=2.0)
        assert s.microprice == pytest.approx(s.mid)

    def test_microprice_zero_size_falls_back_to_mid(self) -> None:
        s = _snap(ts=0.0, bid_px=99.0, bid_sz=0.0, ask_px=101.0, ask_sz=0.0)
        assert s.microprice == pytest.approx(100.0)

    def test_one_sided_and_empty_books_do_not_raise(self) -> None:
        """Degenerate one-sided / empty books must not raise on mid/microprice.

        A snapshot with an empty side reaches :func:`build_index_price`'s
        all-stale reconstruction branch (which reads ``microprice``); the
        properties must degrade gracefully instead of raising ``IndexError``.
        """
        import math

        # Empty bids: mid / microprice fall back to the best ask price.
        empty_bids = OrderBookSnapshot(
            symbol="ALTUSDT",
            ts=0.0,
            bids=[],
            asks=[BookLevel(price=101.0, size=2.0)],
            source="mock",
        )
        assert empty_bids.mid == pytest.approx(101.0)
        assert empty_bids.microprice == pytest.approx(101.0)

        # Empty asks: mid / microprice fall back to the best bid price.
        empty_asks = OrderBookSnapshot(
            symbol="ALTUSDT",
            ts=0.0,
            bids=[BookLevel(price=99.0, size=2.0)],
            asks=[],
            source="mock",
        )
        assert empty_asks.mid == pytest.approx(99.0)
        assert empty_asks.microprice == pytest.approx(99.0)

        # Fully empty book: both default to 0.0, still finite.
        empty_both = OrderBookSnapshot(
            symbol="ALTUSDT",
            ts=0.0,
            bids=[],
            asks=[],
            source="mock",
        )
        assert empty_both.mid == pytest.approx(0.0)
        assert empty_both.microprice == pytest.approx(0.0)
        assert math.isfinite(empty_both.microprice)

    def test_build_index_price_all_stale_one_sided_book(self) -> None:
        """All-stale one-sided book yields an IndexResult, not an IndexError."""
        import math

        one_sided = OrderBookSnapshot(
            symbol="ALTUSDT",
            ts=0.0,
            bids=[],
            asks=[BookLevel(price=101.0, size=2.0)],
            source="mock",
        )
        res = build_index_price([one_sided], now=1_000_000.0, max_age_s=1.0)
        assert isinstance(res, IndexResult)
        assert res.datafeed_drop is True
        assert res.n_fresh == 0
        assert res.source == "reconstructed"
        assert math.isfinite(res.price)
        assert res.price == pytest.approx(101.0)

    def test_is_stale(self) -> None:
        s = _snap(ts=100.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0)
        assert s.is_stale(now=106.0, max_age_s=5.0) is True
        assert s.is_stale(now=104.0, max_age_s=5.0) is False
        # Boundary: now - ts == max_age_s is NOT stale (strict >).
        assert s.is_stale(now=105.0, max_age_s=5.0) is False


class TestMockL2Feed:
    """Determinism, depth shape, and drop behavior of MockL2Feed."""

    def test_deterministic_given_seed(self) -> None:
        f1 = MockL2Feed(seed=7, mid0=100.0, sigma=0.5)
        f2 = MockL2Feed(seed=7, mid0=100.0, sigma=0.5)
        s1 = [f1.snapshot("ALTUSDT") for _ in range(5)]
        s2 = [f2.snapshot("ALTUSDT") for _ in range(5)]
        for a, b in zip(s1, s2):
            assert a.ts == b.ts
            assert a.mid == pytest.approx(b.mid)
            assert a.microprice == pytest.approx(b.microprice)

    def test_different_seed_diverges(self) -> None:
        a = MockL2Feed(seed=1, mid0=100.0, sigma=0.5).snapshot()
        b = MockL2Feed(seed=2, mid0=100.0, sigma=0.5).snapshot()
        assert a.mid != pytest.approx(b.mid)

    def test_depth_shape_and_ordering(self) -> None:
        snap = MockL2Feed(seed=3, mid0=50.0, sigma=0.2).snapshot()
        assert snap.source == "mock"
        assert len(snap.bids) == 20
        assert len(snap.asks) == 20
        # Bids strictly descending, asks strictly ascending, all positive size.
        bid_px = [b.price for b in snap.bids]
        ask_px = [a.price for a in snap.asks]
        assert bid_px == sorted(bid_px, reverse=True)
        assert ask_px == sorted(ask_px)
        assert all(b.size > 0 for b in snap.bids)
        assert all(a.size > 0 for a in snap.asks)
        # Best bid below best ask.
        assert snap.bids[0].price < snap.asks[0].price

    def test_timestamps_advance_then_freeze_on_drop(self) -> None:
        f = MockL2Feed(seed=1, mid0=100.0, sigma=0.5, drop_after=3)
        snaps = [f.snapshot("ALTUSDT") for _ in range(6)]
        # First 3 fresh: strictly increasing ts.
        fresh_ts = [s.ts for s in snaps[:3]]
        assert fresh_ts == sorted(fresh_ts)
        assert len(set(fresh_ts)) == 3
        # Remaining snapshots reuse the last fresh ts (frozen / stale).
        last_fresh_ts = snaps[2].ts
        for s in snaps[3:]:
            assert s.ts == last_fresh_ts

    def test_drop_makes_snapshots_stale(self) -> None:
        f = MockL2Feed(seed=9, mid0=100.0, sigma=0.5, drop_after=2)
        snaps = [f.snapshot("ALTUSDT") for _ in range(5)]
        now = snaps[-1].ts + 10_000.0
        assert snaps[-1].is_stale(now=now, max_age_s=5.0) is True

    def test_no_drop_when_drop_after_none(self) -> None:
        f = MockL2Feed(seed=4, mid0=100.0, sigma=0.5)
        snaps = [f.snapshot() for _ in range(10)]
        ts = [s.ts for s in snaps]
        assert len(set(ts)) == 10  # all unique / advancing


class TestBuildIndexPrice:
    """Datafeed-drop-resilient index construction."""

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_index_price([], now=0.0, max_age_s=5.0)

    def test_fresh_single(self) -> None:
        f = MockL2Feed(seed=2, mid0=100.0, sigma=0.5)
        snap = f.snapshot("ALTUSDT")
        res = build_index_price([snap], now=snap.ts + 1.0, max_age_s=5.0)
        assert isinstance(res, IndexResult)
        assert res.datafeed_drop is False
        assert res.n_fresh == 1
        assert res.source == "index"
        assert res.price == pytest.approx(snap.microprice)

    def test_fresh_median_of_multiple(self) -> None:
        # Three fresh single-level snapshots with distinct micro-prices.
        snaps = [
            _snap(ts=10.0, bid_px=98.0, bid_sz=1.0, ask_px=100.0, ask_sz=1.0),
            _snap(ts=10.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0),
            _snap(ts=10.0, bid_px=100.0, bid_sz=1.0, ask_px=102.0, ask_sz=1.0),
        ]
        res = build_index_price(snaps, now=11.0, max_age_s=5.0)
        assert res.datafeed_drop is False
        assert res.n_fresh == 3
        # Median of {99, 100, 101} micro-prices == 100.
        assert res.price == pytest.approx(100.0)

    def test_ignores_stale_when_some_fresh(self) -> None:
        stale = _snap(ts=0.0, bid_px=1.0, bid_sz=1.0, ask_px=3.0, ask_sz=1.0)
        fresh = _snap(ts=100.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0)
        res = build_index_price([stale, fresh], now=101.0, max_age_s=5.0)
        assert res.datafeed_drop is False
        assert res.n_fresh == 1
        assert res.price == pytest.approx(fresh.microprice)

    def test_all_stale_uses_last_good(self) -> None:
        s = _snap(ts=0.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0)
        res = build_index_price([s], now=10_000.0, max_age_s=5.0, last_good=123.45)
        assert res.datafeed_drop is True
        assert res.n_fresh == 0
        assert res.source == "reconstructed"
        assert res.price == pytest.approx(123.45)

    def test_all_stale_falls_back_to_most_recent_microprice(self) -> None:
        old = _snap(ts=0.0, bid_px=50.0, bid_sz=1.0, ask_px=52.0, ask_sz=1.0)
        newer = _snap(ts=5.0, bid_px=99.0, bid_sz=1.0, ask_px=101.0, ask_sz=1.0)
        res = build_index_price([old, newer], now=10_000.0, max_age_s=5.0)
        assert res.datafeed_drop is True
        assert res.price == pytest.approx(newer.microprice)


class TestBinanceDepthFeedFallback:
    """BinanceDepthFeed must fall back to mock on any error (no network)."""

    def test_fallback_on_fetch_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fallback = MockL2Feed(seed=11, mid0=100.0, sigma=0.5)
        feed = BinanceDepthFeed("BTCUSDT", fallback=fallback)

        def boom() -> dict:
            raise OSError("network down")

        monkeypatch.setattr(feed, "_fetch", boom)
        snap = feed.snapshot()
        assert isinstance(snap, OrderBookSnapshot)
        assert snap.source == "mock"
        assert snap.symbol == "BTCUSDT"
        assert len(snap.bids) > 0 and len(snap.asks) > 0

    def test_fallback_on_empty_book(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feed = BinanceDepthFeed("ETHUSDT")
        monkeypatch.setattr(feed, "_fetch", lambda: {"bids": [], "asks": []})
        snap = feed.snapshot()
        assert snap.source == "mock"

    def test_success_path_parses_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "bids": [["99.5", "2.0"], ["99.0", "1.0"]],
            "asks": [["100.5", "1.5"], ["101.0", "3.0"]],
        }
        feed = BinanceDepthFeed("BTCUSDT")
        monkeypatch.setattr(feed, "_fetch", lambda: payload)
        snap = feed.snapshot()
        assert snap.source == "binance"
        assert snap.bids[0].price == pytest.approx(99.5)
        assert snap.asks[0].size == pytest.approx(1.5)
        assert snap.mid == pytest.approx(100.0)

    def test_default_fallback_is_mock_feed(self) -> None:
        feed = BinanceDepthFeed("BTCUSDT")
        assert isinstance(feed._fallback, MockL2Feed)
