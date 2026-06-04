"""Self-contained fallback L2 order-book feed.

Minimal reference implementation conforming to the canonical ``feeds`` module
interface so the HTTP API can boot standalone before the real ``feeds`` module
is merged. Provides a deterministic mock feed and a best-effort Binance depth
fetch (urllib) that degrades to the mock feed on any error.

Canonical interface mirrored here:
    BookLevel, OrderBookSnapshot (mid/microprice/is_stale), IndexResult,
    MockL2Feed(seed, mid0, sigma, drop_after=None).snapshot(symbol),
    BinanceDepthFeed(symbol, fallback=None, timeout=3.0).snapshot(symbol),
    build_index_price(snaps, now, max_age_s, last_good=None) -> IndexResult
"""

from __future__ import annotations

import json
import logging
import statistics
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth?symbol={sym}&limit=20"


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
        """Best-bid/best-ask midpoint.

        Guards empty sides: returns the other side's best price, or 0.0 when
        both sides are empty, so it never raises IndexError.
        """
        if self.bids and self.asks:
            return 0.5 * (self.bids[0].price + self.asks[0].price)
        if self.bids:
            return self.bids[0].price
        if self.asks:
            return self.asks[0].price
        return 0.0

    @property
    def microprice(self) -> float:
        """Size-weighted micro-price using the top of book.

        Weights the best bid price by the ask size and vice versa, so the price
        leans toward the side with more resting liquidity on the opposite book.
        Falls back to :attr:`mid` when sizes are degenerate or a side is empty.
        """
        if not self.bids or not self.asks:
            return self.mid
        bid = self.bids[0]
        ask = self.asks[0]
        total = bid.size + ask.size
        if total <= 0.0:
            return self.mid
        return (bid.price * ask.size + ask.price * bid.size) / total

    def is_stale(self, now: float, max_age_s: float) -> bool:
        """True if the snapshot is older than ``max_age_s`` seconds."""
        return (now - self.ts) > max_age_s


@dataclass
class IndexResult:
    """Result of an index-price computation."""

    price: float
    source: str
    datafeed_drop: bool
    n_fresh: int


def _synthetic_depth(mid: float, rng: np.random.Generator, *, n_levels: int = 20,
                     tick_frac: float = 1e-4, base_size: float = 5.0,
                     decay: float = 0.35) -> tuple[list[BookLevel], list[BookLevel]]:
    """Build synthetic exponential-decay depth around ``mid``, fully from ``rng``."""
    tick = mid * tick_frac
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []
    for i in range(n_levels):
        decay_factor = float(np.exp(-decay * i))
        bid_jitter = 0.85 + 0.30 * float(rng.random())
        ask_jitter = 0.85 + 0.30 * float(rng.random())
        bids.append(BookLevel(price=mid - tick * (i + 1), size=base_size * decay_factor * bid_jitter))
        asks.append(BookLevel(price=mid + tick * (i + 1), size=base_size * decay_factor * ask_jitter))
    return bids, asks


class MockL2Feed:
    """Deterministic mock L2 feed with GBM mid and synthetic depth.

    Args:
        seed: Seed for the internal numpy generator.
        mid0: Initial mid price.
        sigma: Per-step lognormal volatility of the GBM mid.
        drop_after: If set, snapshots after this many calls are stale (their
            ``ts`` stops advancing) to simulate an upstream datafeed drop.
    """

    def __init__(self, seed: int, mid0: float, sigma: float,
                 drop_after: Optional[int] = None) -> None:
        self.seed = int(seed)
        self.mid0 = float(mid0)
        self.sigma = float(sigma)
        self.drop_after = drop_after
        self._rng = np.random.default_rng(self.seed)
        self._mid = float(mid0)
        self._step = 0
        # Anchor the synthetic clock to wall time so fresh snapshots read fresh
        # against a real ``time.time()`` "now" in the API. One step == one sec.
        self._t0 = time.time()
        self._last_fresh_ts = self._t0
        self.source = "mock"

    def snapshot(self, symbol: str = "ALTUSDT") -> OrderBookSnapshot:
        """Produce the next order-book snapshot for ``symbol``."""
        is_dropped = self.drop_after is not None and self._step >= self.drop_after
        if not is_dropped:
            z = float(self._rng.standard_normal())
            drift = -0.5 * self.sigma * self.sigma
            self._mid *= float(np.exp(drift + self.sigma * z))
            ts = self._t0 + self._step
            self._last_fresh_ts = ts
        else:
            ts = self._last_fresh_ts
        bids, asks = _synthetic_depth(self._mid, self._rng)
        self._step += 1
        return OrderBookSnapshot(symbol=symbol.upper(), ts=ts, bids=bids, asks=asks,
                                 source=self.source)


class BinanceDepthFeed:
    """Best-effort Binance L2 depth feed with mock fallback.

    Fetches the REST depth endpoint via :mod:`urllib`. On any network/parse
    error it logs and returns a mock snapshot (tagged ``source="mock"``).

    Args:
        symbol: Trading symbol, e.g. ``"BTCUSDT"``.
        fallback: Optional mock feed to use on error.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, symbol: str, fallback: Optional[MockL2Feed] = None,
                 timeout: float = 3.0) -> None:
        self.symbol = symbol.upper()
        self.timeout = float(timeout)
        self._fallback = fallback if fallback is not None else MockL2Feed(
            seed=0, mid0=100.0, sigma=0.5)

    def _fetch(self, sym: str) -> dict:
        url = BINANCE_DEPTH_URL.format(sym=sym)
        req = urllib.request.Request(url, headers={"User-Agent": "allocation_gym-otc"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read())

    def snapshot(self, symbol: Optional[str] = None) -> OrderBookSnapshot:
        """Return a live Binance snapshot, or a mock snapshot on any failure."""
        sym = (symbol or self.symbol).upper()
        try:
            data = self._fetch(sym)
            bids = [BookLevel(price=float(px), size=float(sz)) for px, sz in data["bids"]]
            asks = [BookLevel(price=float(px), size=float(sz)) for px, sz in data["asks"]]
            if not bids or not asks:
                raise ValueError("empty book from binance")
            return OrderBookSnapshot(symbol=sym, ts=time.time(), bids=bids, asks=asks,
                                     source="binance")
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Binance depth fetch failed for %s (%s); using mock", sym, exc)
            snap = self._fallback.snapshot(sym)
            snap.source = "mock"
            return snap


def build_index_price(snaps: list[OrderBookSnapshot], now: float, max_age_s: float,
                      last_good: Optional[float] = None) -> IndexResult:
    """Build a datafeed-drop-resilient index price from order-book snapshots.

    The index is the median micro-price of all *fresh* snapshots. If every
    snapshot is stale this signals a datafeed drop: ``datafeed_drop=True`` and
    the price falls back to ``last_good`` if given, else the micro-price of the
    most recent snapshot.

    Raises:
        ValueError: If ``snaps`` is empty.
    """
    if not snaps:
        raise ValueError("build_index_price requires at least one snapshot")
    fresh = [s for s in snaps if not s.is_stale(now, max_age_s)]
    if fresh:
        price = float(statistics.median(s.microprice for s in fresh))
        return IndexResult(price=price, source="index", datafeed_drop=False,
                           n_fresh=len(fresh))
    if last_good is not None:
        price = float(last_good)
    else:
        most_recent = max(snaps, key=lambda s: s.ts)
        price = float(most_recent.microprice)
    return IndexResult(price=price, source="reconstructed", datafeed_drop=True,
                       n_fresh=0)
