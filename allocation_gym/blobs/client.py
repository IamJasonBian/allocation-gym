"""Netlify Blob Store REST API client with lambda-based filtering."""

from __future__ import annotations

import os
from urllib.parse import quote as urlquote

import requests

from allocation_gym.blobs.models import (
    MarketQuote,
    OptionContract,
    Predicate,
    apply_filters,
    parse_occ,
)

_API_BASE = "https://api.netlify.com/api/v1/blobs"


class NetlifyBlobClient:
    """Read-only client for Netlify Blob Store REST API.

    Parameters
    ----------
    site_id : str
        The Netlify site ID that owns the blob stores.
    token : str, optional
        Personal access token.  Falls back to ``NETLIFY_TOKEN`` env var.
    """

    def __init__(self, site_id: str, token: str | None = None) -> None:
        self.site_id = site_id
        self.token = token or os.environ.get("NETLIFY_TOKEN", "")
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.token}"

    # -- low-level --------------------------------------------------------

    def list_keys(self, store: str, prefix: str = "") -> list[str]:
        """Return all blob keys in *store*, optionally filtered by *prefix*.

        Handles Netlify's cursor-based pagination automatically.
        """
        keys: list[str] = []
        cursor: str | None = None
        while True:
            params: dict[str, str] = {}
            if prefix:
                params["prefix"] = prefix
            if cursor:
                params["cursor"] = cursor
            resp = self._session.get(
                f"{_API_BASE}/{self.site_id}/{store}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            keys.extend(b["key"] for b in data.get("blobs", []))
            cursor = data.get("next_cursor")
            if not cursor:
                break
        return keys

    def get_blob(self, store: str, key: str) -> dict | list:
        """Fetch and JSON-decode a single blob by *key*.

        Key segments are URL-encoded individually to preserve literal ``/``.
        """
        encoded = "/".join(urlquote(seg, safe="") for seg in key.split("/"))
        resp = self._session.get(
            f"{_API_BASE}/{self.site_id}/{store}/{encoded}",
        )
        resp.raise_for_status()
        return resp.json()

    # -- typed accessors with lambda predicates ---------------------------

    def get_option_chain(
        self,
        key: str,
        *predicates: Predicate[OptionContract],
    ) -> list[OptionContract]:
        """Fetch an ``options-chain`` blob and return filtered contracts.

        Each contract is enriched with OCC-parsed fields (``underlying``,
        ``expiry``, ``option_type``, ``strike``) before predicates run.

        Example::

            client.get_option_chain(
                "CRWD/2026-03-20T23-20-00",
                lambda c: c["option_type"] == "call",
                lambda c: c.get("greeks", {}).get("delta", 0) > 0.3,
                lambda c: c["strike"] < 200.0,
            )
        """
        raw = self.get_blob("options-chain", key)
        if not isinstance(raw, dict):
            return []
        chain = raw.get("latest_chain", raw)
        contracts = _parse_chain(chain)
        return apply_filters(contracts, predicates)

    def get_market_quotes(
        self,
        key: str,
        *predicates: Predicate[MarketQuote],
    ) -> list[MarketQuote]:
        """Fetch a ``market-quotes`` blob and return filtered quotes.

        Example::

            client.get_market_quotes(
                "2026-03-20T23-20-00",
                lambda q: q["spread_bps"] < 50,
                lambda q: q["asset_class"] == "crypto",
            )
        """
        raw = self.get_blob("market-quotes", key)
        if not isinstance(raw, dict):
            return []
        quotes_raw = raw.get("latest_quotes", raw)
        quotes = [_map_quote(v) for k, v in quotes_raw.items() if k != "_meta"]
        return apply_filters(quotes, predicates)

    # -- bulk scan --------------------------------------------------------

    def scan_option_chains(
        self,
        prefix: str = "",
        *predicates: Predicate[OptionContract],
        limit: int | None = None,
    ) -> list[OptionContract]:
        """List keys by *prefix*, fetch each blob, filter, collect results."""
        keys = self.list_keys("options-chain", prefix)
        results: list[OptionContract] = []
        for key in keys:
            batch = self.get_option_chain(key, *predicates)
            results.extend(batch)
            if limit and len(results) >= limit:
                return results[:limit]
        return results

    def scan_market_quotes(
        self,
        prefix: str = "",
        *predicates: Predicate[MarketQuote],
        limit: int | None = None,
    ) -> list[MarketQuote]:
        """List keys by *prefix*, fetch each blob, filter, collect results."""
        keys = self.list_keys("market-quotes", prefix)
        results: list[MarketQuote] = []
        for key in keys:
            batch = self.get_market_quotes(key, *predicates)
            results.extend(batch)
            if limit and len(results) >= limit:
                return results[:limit]
        return results


# -- internal helpers -----------------------------------------------------


def _parse_chain(chain: dict) -> list[OptionContract]:
    """Convert a raw ``latest_chain`` dict into enriched OptionContract list."""
    contracts: list[OptionContract] = []
    for sym, snap in chain.items():
        if sym == "_meta" or not isinstance(snap, dict):
            continue
        parsed = parse_occ(sym)
        quote_raw = snap.get("latest_quote") or snap.get("latestQuote") or {}
        contract: OptionContract = {
            "symbol": sym,
            "underlying": parsed["underlying"] if parsed else "",
            "expiry": parsed["expiry"] if parsed else "",
            "option_type": parsed["option_type"] if parsed else "",
            "strike": parsed["strike"] if parsed else 0.0,
            "latest_trade": snap.get("latest_trade") or snap.get("latestTrade") or {},
            "latest_quote": {
                "bid": quote_raw.get("bid", 0),
                "ask": quote_raw.get("ask", 0),
                "bid_size": quote_raw.get("bid_size") or quote_raw.get("bidSize", 0),
                "ask_size": quote_raw.get("ask_size") or quote_raw.get("askSize", 0),
            },
            "greeks": snap.get("greeks") or {},
            "implied_volatility": (
                snap.get("implied_volatility")
                or snap.get("impliedVolatility")
                or 0.0
            ),
        }
        contracts.append(contract)
    return contracts


def _map_quote(raw: dict) -> MarketQuote:
    """Normalise a raw market quote dict to snake_case MarketQuote."""
    return MarketQuote(
        symbol=raw.get("symbol", ""),
        asset_class=raw.get("asset_class") or raw.get("assetClass", ""),
        bid=raw.get("bid", 0),
        ask=raw.get("ask", 0),
        mid=raw.get("mid", 0),
        spread=raw.get("spread", 0),
        spread_bps=raw.get("spread_bps") or raw.get("spreadBps", 0),
        bid_size=raw.get("bid_size") or raw.get("bidSize", 0),
        ask_size=raw.get("ask_size") or raw.get("askSize", 0),
        timestamp=raw.get("timestamp", ""),
        source=raw.get("source", ""),
    )
