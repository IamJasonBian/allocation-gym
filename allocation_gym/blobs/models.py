"""Data types for Netlify blob store records and OCC symbol parsing."""

from __future__ import annotations

import re
from typing import Callable, TypedDict, TypeVar

T = TypeVar("T")
Predicate = Callable[[T], bool]

_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


class Greeks(TypedDict, total=False):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class OptionContract(TypedDict, total=False):
    symbol: str
    underlying: str
    expiry: str
    option_type: str  # "call" | "put"
    strike: float
    latest_trade: dict
    latest_quote: dict
    greeks: Greeks
    implied_volatility: float


class MarketQuote(TypedDict, total=False):
    symbol: str
    asset_class: str
    bid: float
    ask: float
    mid: float
    spread: float
    spread_bps: float
    bid_size: float
    ask_size: float
    timestamp: str
    source: str


def parse_occ(symbol: str) -> dict | None:
    """Parse an OCC option symbol like ``IWN260418C00205000``.

    Returns ``{"underlying": "IWN", "expiry": "2026-04-18",
    "option_type": "call", "strike": 205.0}`` or *None* if the
    symbol doesn't match OCC format.
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying, date_str, cp, strike_str = m.groups()
    y = 2000 + int(date_str[:2])
    mo = date_str[2:4]
    d = date_str[4:6]
    return {
        "underlying": underlying,
        "expiry": f"{y}-{mo}-{d}",
        "option_type": "call" if cp == "C" else "put",
        "strike": int(strike_str) / 1000,
    }


def apply_filters(items: list[T], predicates: tuple[Predicate[T], ...]) -> list[T]:
    """Return items that satisfy all predicates (AND-combined)."""
    if not predicates:
        return items
    out: list[T] = []
    for item in items:
        if all(p(item) for p in predicates):
            out.append(item)
    return out
