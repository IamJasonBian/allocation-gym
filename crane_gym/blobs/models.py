"""Data models for Netlify blob store payloads.

Mirrors the TypeScript types in allocation-manager's blobDataService.ts,
normalising snake_case JSON from the Python allocation-engine scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re


# ── Option types ────────────────────────────────────────────


@dataclass
class OptionGreeks:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


@dataclass
class OptionQuote:
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    timestamp: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return (self.spread / mid * 10_000) if mid > 0 else 0.0


@dataclass
class OptionTrade:
    price: float = 0.0
    size: int = 0
    timestamp: str = ""


_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


@dataclass
class OptionSnapshot:
    symbol: str = ""
    latest_trade: OptionTrade | None = None
    latest_quote: OptionQuote | None = None
    greeks: OptionGreeks | None = None
    implied_volatility: float | None = None

    # Parsed fields (populated from OCC symbol)
    underlying: str = ""
    expiry: str = ""
    option_type: str = ""  # "call" or "put"
    strike: float = 0.0

    def __post_init__(self) -> None:
        if self.symbol and not self.underlying:
            self._parse_occ()

    def _parse_occ(self) -> None:
        m = _OCC_RE.match(self.symbol)
        if not m:
            return
        self.underlying = m.group(1)
        d = m.group(2)
        self.expiry = f"20{d[:2]}-{d[2:4]}-{d[4:6]}"
        self.option_type = "call" if m.group(3) == "C" else "put"
        self.strike = int(m.group(4)) / 1000

    @property
    def mid(self) -> float:
        if self.latest_quote:
            return self.latest_quote.mid
        return 0.0

    @property
    def iv(self) -> float | None:
        return self.implied_volatility

    @property
    def delta(self) -> float | None:
        return self.greeks.delta if self.greeks else None

    @property
    def gamma(self) -> float | None:
        return self.greeks.gamma if self.greeks else None

    @property
    def theta(self) -> float | None:
        return self.greeks.theta if self.greeks else None

    @property
    def vega(self) -> float | None:
        return self.greeks.vega if self.greeks else None

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_put(self) -> bool:
        return self.option_type == "put"

    @property
    def is_itm(self) -> bool:
        """Check if in-the-money (requires delta)."""
        d = self.delta
        if d is None:
            return False
        return abs(d) > 0.5

    @property
    def moneyness(self) -> str:
        """ATM / ITM / OTM classification based on delta."""
        d = self.delta
        if d is None:
            return "unknown"
        ad = abs(d)
        if ad > 0.45 and ad < 0.55:
            return "ATM"
        return "ITM" if ad > 0.5 else "OTM"


# ── Market quote type ───────────────────────────────────────


@dataclass
class MarketQuote:
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    timestamp: str = ""
    source: str = ""
    asset_class: str = ""


# ── Blob-level containers ──────────────────────────────────


@dataclass
class OptionsChainBlob:
    timestamp: str = ""
    underlying: str = ""
    blob_key: str = ""
    snapshots: list[OptionSnapshot] = field(default_factory=list)
    history_count: int = 0


@dataclass
class MarketQuotesBlob:
    timestamp: str = ""
    blob_key: str = ""
    quotes: list[MarketQuote] = field(default_factory=list)
    history_count: int = 0


# ── Deserialisers ───────────────────────────────────────────


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Get first matching key from dict (handles snake_case / camelCase)."""
    for k in keys:
        if k in d:
            return d[k]
    return default


def parse_option_snapshot(key: str, raw: dict) -> OptionSnapshot:
    quote_raw = _get(raw, "latest_quote", "latestQuote")
    trade_raw = _get(raw, "latest_trade", "latestTrade")
    greeks_raw = raw.get("greeks")

    quote = None
    if quote_raw:
        quote = OptionQuote(
            bid=quote_raw.get("bid", 0),
            ask=quote_raw.get("ask", 0),
            bid_size=_get(quote_raw, "bid_size", "bidSize", default=0),
            ask_size=_get(quote_raw, "ask_size", "askSize", default=0),
            timestamp=quote_raw.get("timestamp", ""),
        )

    trade = None
    if trade_raw:
        trade = OptionTrade(
            price=trade_raw.get("price", 0),
            size=trade_raw.get("size", 0),
            timestamp=trade_raw.get("timestamp", ""),
        )

    greeks = None
    if greeks_raw and isinstance(greeks_raw, dict):
        greeks = OptionGreeks(
            delta=greeks_raw.get("delta"),
            gamma=greeks_raw.get("gamma"),
            theta=greeks_raw.get("theta"),
            vega=greeks_raw.get("vega"),
            rho=greeks_raw.get("rho"),
        )

    return OptionSnapshot(
        symbol=key,
        latest_trade=trade,
        latest_quote=quote,
        greeks=greeks,
        implied_volatility=_get(raw, "implied_volatility", "impliedVolatility"),
    )


def parse_options_chain_blob(raw: dict) -> OptionsChainBlob:
    chain = raw.get("latest_chain", {})
    snapshots = [
        parse_option_snapshot(k, v)
        for k, v in chain.items()
        if k != "_meta" and isinstance(v, dict)
    ]
    return OptionsChainBlob(
        timestamp=raw.get("timestamp", ""),
        underlying=raw.get("underlying", ""),
        blob_key=raw.get("blob_key", ""),
        snapshots=snapshots,
        history_count=raw.get("history_count", 0),
    )


def parse_market_quote(raw: dict) -> MarketQuote:
    return MarketQuote(
        symbol=raw.get("symbol", ""),
        bid=raw.get("bid", 0),
        ask=raw.get("ask", 0),
        mid=raw.get("mid", 0),
        spread=raw.get("spread", 0),
        spread_bps=_get(raw, "spread_bps", "spreadBps", default=0),
        bid_size=_get(raw, "bid_size", "bidSize", default=0),
        ask_size=_get(raw, "ask_size", "askSize", default=0),
        timestamp=raw.get("timestamp", ""),
        source=raw.get("source", ""),
        asset_class=_get(raw, "asset_class", "assetClass", default=""),
    )


def parse_market_quotes_blob(raw: dict) -> MarketQuotesBlob:
    latest = raw.get("latest_quotes", {})
    quotes = [
        parse_market_quote(v)
        for k, v in latest.items()
        if k != "_meta" and isinstance(v, dict)
    ]
    return MarketQuotesBlob(
        timestamp=raw.get("timestamp", ""),
        blob_key=raw.get("blob_key", ""),
        quotes=quotes,
        history_count=raw.get("history_count", 0),
    )
