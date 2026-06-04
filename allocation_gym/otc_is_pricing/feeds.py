"""Order-book feeds and datafeed-drop resilient index pricing.

This module provides the data-ingestion layer for the OTC importance-sampling
pricer. It defines the frozen order-book snapshot schema, a deterministic mock
L2 feed (seeded GBM mid with synthetic exponential-decay depth), a real Binance
L2 depth feed that falls back to the mock feed on any error, and an index-price
builder that is resilient to upstream datafeed drops.

The index builder takes the median micro-price of *fresh* snapshots. When every
snapshot is stale (a datafeed drop), it performs a simple last-valid
reconstruction (the real importance-sampling reconstruction lives in the pricer
unit) and flags ``datafeed_drop=True``.

Only the standard library and ``numpy`` are used. No websockets, no
third-party HTTP clients.
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

# Public API of this module.
__all__ = [
    "BookLevel",
    "OrderBookSnapshot",
    "MockL2Feed",
    "BinanceDepthFeed",
    "IndexResult",
    "build_index_price",
]


@dataclass
class BookLevel:
    """A single price level in an order book.

    Attributes:
        price: Price of the level.
        size: Resting size (quantity) at this level.
    """

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """An L2 order-book snapshot for a single symbol.

    Attributes:
        symbol: Trading symbol, e.g. ``"ALTUSDT"``.
        ts: Snapshot timestamp in epoch seconds.
        bids: Bid levels in descending price order (best bid first).
        asks: Ask levels in ascending price order (best ask first).
        source: Origin of the snapshot, ``"binance"`` or ``"mock"``.
    """

    symbol: str
    ts: float
    bids: list[BookLevel]
    asks: list[BookLevel]
    source: str

    @property
    def mid(self) -> float:
        """Mid price: ``(best_bid + best_ask) / 2``.

        Robust to a one-sided or empty book: if both sides are present the
        usual mid of the touch is returned; if only one side has levels its
        best price is returned; if both sides are empty ``0.0`` is returned.
        """
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2.0
        if self.bids:
            return self.bids[0].price
        if self.asks:
            return self.asks[0].price
        return 0.0

    @property
    def microprice(self) -> float:
        """Size-weighted micro-price.

        Computed as ``(bid_px*ask_sz + ask_px*bid_sz) / (bid_sz + ask_sz)``
        using the best bid/ask levels. Weighting the bid price by the ask size
        (and vice versa) leans the micro-price toward the side with more
        resting liquidity on the *opposite* book, the standard convention.

        Falls back to :attr:`mid` if either side is empty or the combined
        top-of-book size is non-positive (which keeps a degenerate / one-sided
        book from raising and lets it degrade to the robust :attr:`mid`).
        """
        if not self.bids or not self.asks:
            return self.mid
        best_bid = self.bids[0]
        best_ask = self.asks[0]
        bid_sz = best_bid.size
        ask_sz = best_ask.size
        denom = bid_sz + ask_sz
        if denom <= 0.0:
            return self.mid
        return (best_bid.price * ask_sz + best_ask.price * bid_sz) / denom

    def is_stale(self, now: float, max_age_s: float) -> bool:
        """Return True if the snapshot is older than ``max_age_s`` seconds.

        Args:
            now: Current epoch time in seconds.
            max_age_s: Maximum tolerated age in seconds.

        Returns:
            ``True`` when ``now - ts > max_age_s``.
        """
        return (now - self.ts) > max_age_s


def _synthetic_depth(
    mid: float,
    rng: np.random.Generator,
    *,
    n_levels: int = 20,
    tick_frac: float = 1e-4,
    base_size: float = 5.0,
    decay: float = 0.35,
) -> tuple[list[BookLevel], list[BookLevel]]:
    """Build synthetic exponential-decay depth around a mid price.

    Sizes decay exponentially away from the touch and are jittered slightly so
    the two sides are not perfectly symmetric (which keeps the micro-price from
    collapsing onto the mid). Fully determined by ``rng``.

    Args:
        mid: Mid price to center the book on.
        rng: Seeded numpy random generator (advanced in place).
        n_levels: Number of levels per side.
        tick_frac: Tick size as a fraction of mid.
        base_size: Size at the touch before decay/jitter.
        decay: Exponential decay rate of size across levels.

    Returns:
        ``(bids, asks)`` with bids descending and asks ascending in price.
    """
    tick = mid * tick_frac
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []
    for i in range(n_levels):
        decay_factor = float(np.exp(-decay * i))
        # Jitter in [0.85, 1.15) keeps sides asymmetric but positive.
        bid_jitter = 0.85 + 0.30 * float(rng.random())
        ask_jitter = 0.85 + 0.30 * float(rng.random())
        bid_px = mid - tick * (i + 1)
        ask_px = mid + tick * (i + 1)
        bid_sz = base_size * decay_factor * bid_jitter
        ask_sz = base_size * decay_factor * ask_jitter
        bids.append(BookLevel(price=bid_px, size=bid_sz))
        asks.append(BookLevel(price=ask_px, size=ask_sz))
    return bids, asks


class MockL2Feed:
    """Deterministic mock L2 feed with GBM mid and synthetic depth.

    Each :meth:`snapshot` call advances the GBM mid by one step and rebuilds the
    synthetic exponential-decay depth. The feed is fully deterministic given the
    seed.

    A ``drop_after`` value simulates an upstream datafeed drop: after
    ``drop_after`` successful (fresh) snapshots, every subsequent call returns a
    *stale* snapshot carrying the timestamp of the last fresh snapshot (its
    ``ts`` no longer advances), so downstream staleness checks fire.

    Args:
        seed: Seed for the internal numpy generator.
        mid0: Initial mid price.
        sigma: Per-step lognormal volatility of the GBM mid.
        drop_after: If set, snapshots after this many calls are stale.
    """

    def __init__(
        self,
        seed: int,
        mid0: float,
        sigma: float,
        drop_after: Optional[int] = None,
    ) -> None:
        self.seed = int(seed)
        self.mid0 = float(mid0)
        self.sigma = float(sigma)
        self.drop_after = drop_after
        self._rng = np.random.default_rng(self.seed)
        self._mid = float(mid0)
        self._step = 0
        # Deterministic synthetic clock so snapshots are reproducible without
        # depending on wall-clock time. One step == one second.
        self._t0 = 1_700_000_000.0
        self._last_fresh_ts = self._t0
        self.source = "mock"

    def snapshot(self, symbol: str = "ALTUSDT") -> OrderBookSnapshot:
        """Produce the next order-book snapshot.

        Args:
            symbol: Trading symbol to stamp on the snapshot.

        Returns:
            A fresh :class:`OrderBookSnapshot`, or a stale one (old ``ts``) once
            ``drop_after`` snapshots have been produced.
        """
        is_dropped = self.drop_after is not None and self._step >= self.drop_after

        if not is_dropped:
            # Advance the GBM mid by one lognormal step.
            z = float(self._rng.standard_normal())
            drift = -0.5 * self.sigma * self.sigma
            self._mid *= float(np.exp(drift + self.sigma * z))
            ts = self._t0 + self._step
            self._last_fresh_ts = ts
        else:
            # Feed drop: do not advance mid; reuse last fresh timestamp so the
            # snapshot reads as stale relative to a moving "now".
            ts = self._last_fresh_ts

        bids, asks = _synthetic_depth(self._mid, self._rng)
        self._step += 1
        return OrderBookSnapshot(
            symbol=symbol,
            ts=ts,
            bids=bids,
            asks=asks,
            source=self.source,
        )


class BinanceDepthFeed:
    """Real Binance L2 depth feed with mock fallback.

    Fetches the top-of-book depth from the Binance REST API using only
    :mod:`urllib.request`. On *any* error (network failure, timeout, bad
    payload) it logs a warning and returns a snapshot from the fallback
    :class:`MockL2Feed`, tagged ``source="mock"``.

    Args:
        symbol: Trading symbol, e.g. ``"BTCUSDT"``.
        fallback: Optional mock feed to use on error. Defaults to a fresh
            :class:`MockL2Feed`.
        timeout: HTTP request timeout in seconds.
    """

    _URL = "https://api.binance.com/api/v3/depth?symbol={sym}&limit=20"

    def __init__(
        self,
        symbol: str,
        fallback: Optional[MockL2Feed] = None,
        timeout: float = 3.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeout = float(timeout)
        self._fallback = fallback if fallback is not None else MockL2Feed(
            seed=0, mid0=100.0, sigma=0.5
        )

    def _fetch(self) -> dict:
        """Fetch and JSON-decode the raw depth payload from Binance.

        Returns:
            The decoded JSON dict with ``bids``/``asks`` arrays.

        Raises:
            Exception: Any urllib/JSON error is allowed to propagate to the
                caller (:meth:`snapshot`), which handles the fallback.
        """
        url = self._URL.format(sym=self.symbol)
        req = urllib.request.Request(url, headers={"User-Agent": "allocation_gym-otc"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        return json.loads(raw)

    def snapshot(self, symbol: Optional[str] = None) -> OrderBookSnapshot:
        """Return a live Binance snapshot, falling back to mock on any error.

        Args:
            symbol: Optional override symbol; defaults to the configured one.

        Returns:
            An :class:`OrderBookSnapshot` with ``source="binance"`` on success,
            or the fallback mock snapshot with ``source="mock"`` on any error.
        """
        sym = (symbol or self.symbol).upper()
        try:
            data = self._fetch()
            bids = [
                BookLevel(price=float(px), size=float(sz))
                for px, sz in data["bids"]
            ]
            asks = [
                BookLevel(price=float(px), size=float(sz))
                for px, sz in data["asks"]
            ]
            if not bids or not asks:
                raise ValueError("empty book from binance")
            return OrderBookSnapshot(
                symbol=sym,
                ts=time.time(),
                bids=bids,
                asks=asks,
                source="binance",
            )
        except Exception as exc:  # noqa: BLE001 - resilience is the point.
            logger.warning(
                "Binance depth fetch failed for %s (%s); falling back to mock.",
                sym,
                exc,
            )
            snap = self._fallback.snapshot(sym)
            snap.source = "mock"
            return snap


@dataclass
class IndexResult:
    """Result of an index-price computation.

    Attributes:
        price: The computed index price.
        source: ``"index"`` for a normal median, ``"reconstructed"`` when all
            inputs were stale and a fallback value was used.
        datafeed_drop: ``True`` when every input snapshot was stale.
        n_fresh: Number of fresh (non-stale) snapshots used.
    """

    price: float
    source: str
    datafeed_drop: bool
    n_fresh: int


def build_index_price(
    snaps: list[OrderBookSnapshot],
    now: float,
    max_age_s: float,
    last_good: Optional[float] = None,
) -> IndexResult:
    """Build a datafeed-drop-resilient index price from order-book snapshots.

    The index is the median of the micro-prices of all *fresh* snapshots (those
    not stale per :meth:`OrderBookSnapshot.is_stale`). If at least one snapshot
    is fresh, ``datafeed_drop`` is ``False``.

    If every snapshot is stale, this signals a datafeed drop: ``datafeed_drop``
    is ``True`` and the price falls back to ``last_good`` when provided,
    otherwise to the micro-price of the most recent (highest ``ts``) snapshot.
    This last-valid reconstruction is an importance-sampling hook; the real IS
    reconstruction lives in the pricer unit.

    Args:
        snaps: Candidate order-book snapshots.
        now: Current epoch time in seconds, used for the staleness check.
        max_age_s: Maximum tolerated snapshot age in seconds.
        last_good: Optional last-known-good index price for fallback.

    Returns:
        An :class:`IndexResult`.

    Raises:
        ValueError: If ``snaps`` is empty.
    """
    if not snaps:
        raise ValueError("build_index_price requires at least one snapshot")

    fresh = [s for s in snaps if not s.is_stale(now, max_age_s)]

    if fresh:
        price = float(statistics.median(s.microprice for s in fresh))
        return IndexResult(
            price=price,
            source="index",
            datafeed_drop=False,
            n_fresh=len(fresh),
        )

    # All stale: datafeed drop. Reconstruct from last-good or most recent snap.
    if last_good is not None:
        price = float(last_good)
    else:
        most_recent = max(snaps, key=lambda s: s.ts)
        price = float(most_recent.microprice)
    return IndexResult(
        price=price,
        source="reconstructed",
        datafeed_drop=True,
        n_fresh=0,
    )
