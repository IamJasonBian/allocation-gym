"""Self-contained fallback L2 order-book feed.

Minimal reference implementation conforming to the frozen feed interfaces so
that the HTTP API can boot standalone before the real ``feeds`` module is
merged. Provides a mock feed and a best-effort Binance depth fetch (urllib)
that degrades to the mock feed on any error.

Frozen interfaces:
    BookLevel, OrderBookSnapshot, BinanceDepthFeed, MockL2Feed,
    build_index_price
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"

# Rough reference mid-prices used to seed the mock feed for common symbols.
_MOCK_MIDS: dict[str, float] = {
    "BTCUSDT": 100_000.0,
    "ETHUSDT": 3_500.0,
    "SOLUSDT": 150.0,
    "DOGEUSDT": 0.15,
}
_DEFAULT_MID = 100.0


@dataclass
class BookLevel:
    """A single price level in the order book."""

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """An L2 order-book snapshot at a point in time."""

    symbol: str
    ts: float
    bids: list[BookLevel]
    asks: list[BookLevel]
    source: str

    @property
    def mid(self) -> float:
        """Best-bid/best-ask midpoint."""
        if not self.bids or not self.asks:
            raise ValueError("order book has empty side; cannot compute mid")
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    @property
    def microprice(self) -> float:
        """Size-weighted micro-price using the top of book.

        Weights the best ask by the bid size and vice versa, so the price is
        pulled toward the thicker side (the standard micro-price definition).
        Falls back to the mid when top-of-book sizes are degenerate.
        """
        if not self.bids or not self.asks:
            raise ValueError("order book has empty side; cannot compute microprice")
        bid = self.bids[0]
        ask = self.asks[0]
        total = bid.size + ask.size
        if total <= 0:
            return self.mid
        return (ask.price * bid.size + bid.price * ask.size) / total

    def is_stale(self, now: float, max_age_s: float) -> bool:
        """True if the snapshot is older than ``max_age_s`` seconds."""
        return (now - self.ts) > max_age_s


def _mock_mid_for(symbol: str) -> float:
    return _MOCK_MIDS.get(symbol.upper(), _DEFAULT_MID)


def _build_snapshot(symbol: str, mid: float, source: str, ts: Optional[float] = None,
                    spread_bps: float = 2.0, levels: int = 5) -> OrderBookSnapshot:
    """Construct a synthetic but plausible snapshot around ``mid``."""
    if ts is None:
        ts = time.time()
    half_spread = mid * (spread_bps / 1e4) / 2.0
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []
    tick = max(mid * 1e-4, 1e-8)
    for i in range(levels):
        bid_px = mid - half_spread - i * tick
        ask_px = mid + half_spread + i * tick
        size = 1.0 + i  # deeper levels carry more size
        bids.append(BookLevel(price=bid_px, size=size))
        asks.append(BookLevel(price=ask_px, size=size))
    return OrderBookSnapshot(symbol=symbol.upper(), ts=ts, bids=bids, asks=asks, source=source)


class MockL2Feed:
    """Deterministic mock L2 feed that never touches the network."""

    source_name = "mock"

    def __init__(self, mids: Optional[dict[str, float]] = None, spread_bps: float = 2.0):
        self._mids = dict(_MOCK_MIDS)
        if mids:
            self._mids.update({k.upper(): v for k, v in mids.items()})
        self.spread_bps = spread_bps

    def get_book(self, symbol: str) -> OrderBookSnapshot:
        """Return a fresh mock snapshot for ``symbol``."""
        mid = self._mids.get(symbol.upper(), _DEFAULT_MID)
        return _build_snapshot(symbol, mid, source=self.source_name, spread_bps=self.spread_bps)


class BinanceDepthFeed:
    """Best-effort Binance L2 depth feed with mock fallback.

    Fetches the REST depth endpoint via :mod:`urllib`. On any network/parse
    error it logs and returns a mock snapshot so callers always get a book.
    """

    source_name = "binance"

    def __init__(self, limit: int = 10, timeout: float = 3.0,
                 fallback: Optional[MockL2Feed] = None):
        self.limit = limit
        self.timeout = timeout
        self._fallback = fallback or MockL2Feed()

    def _fetch(self, symbol: str) -> OrderBookSnapshot:
        url = f"{BINANCE_DEPTH_URL}?symbol={symbol.upper()}&limit={self.limit}"
        req = urllib.request.Request(url, headers={"User-Agent": "otc-is-pricing/0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        bids = [BookLevel(price=float(p), size=float(q)) for p, q in payload.get("bids", [])]
        asks = [BookLevel(price=float(p), size=float(q)) for p, q in payload.get("asks", [])]
        if not bids or not asks:
            raise ValueError("empty depth payload")
        return OrderBookSnapshot(
            symbol=symbol.upper(), ts=time.time(), bids=bids, asks=asks,
            source=self.source_name,
        )

    def get_book(self, symbol: str) -> OrderBookSnapshot:
        """Return a live Binance snapshot, or a mock snapshot on failure."""
        try:
            return self._fetch(symbol)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Binance depth fetch failed for %s (%s); using mock", symbol, exc)
            return self._fallback.get_book(symbol)


def build_index_price(snapshot: OrderBookSnapshot) -> float:
    """Derive a robust index price from a snapshot.

    Uses the micro-price (size-weighted top of book), which is resilient to a
    one-sided thinning of the book that would skew a naive mid.
    """
    return snapshot.microprice
