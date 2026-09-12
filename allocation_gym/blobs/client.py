"""Netlify blob store REST API client.

Mirrors the vend-blobs.cjs serverless function from allocation-manager PR #62,
but calls the Netlify REST API directly so this package can run standalone
(no Netlify function proxy required).
"""

from __future__ import annotations

import os
from urllib.parse import quote as urlquote

import requests

from allocation_gym.blobs.models import (
    OptionsChainBlob,
    MarketQuotesBlob,
    parse_options_chain_blob,
    parse_market_quotes_blob,
)

NETLIFY_API = "https://api.netlify.com/api/v1"


class BlobClient:
    """Read blobs from one or more Netlify blob stores.

    Parameters
    ----------
    site_id : str
        Netlify site ID that owns the blob stores.
        Falls back to ``ALLOC_ENGINE_SITE_ID`` env var.
    token : str
        Netlify personal access token.
        Falls back to ``NETLIFY_AUTH_TOKEN`` env var.
    """

    def __init__(
        self,
        site_id: str | None = None,
        token: str | None = None,
    ) -> None:
        self.site_id = site_id or os.environ.get("ALLOC_ENGINE_SITE_ID", "")
        self.token = token or os.environ.get("NETLIFY_AUTH_TOKEN", "")
        if not self.site_id:
            raise ValueError("site_id required (or set ALLOC_ENGINE_SITE_ID)")
        if not self.token:
            raise ValueError("token required (or set NETLIFY_AUTH_TOKEN)")
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.token}"

    # ── Low-level API ───────────────────────────────────────

    def list_keys(self, store: str, prefix: str = "") -> list[str]:
        """List all blob keys in *store*, optionally filtered by *prefix*."""
        keys: list[str] = []
        cursor: str | None = None
        while True:
            params: dict[str, str] = {}
            if prefix:
                params["prefix"] = prefix
            if cursor:
                params["cursor"] = cursor
            url = f"{NETLIFY_API}/blobs/{self.site_id}/{store}"
            resp = self._session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            keys.extend(b["key"] for b in data.get("blobs", []))
            cursor = data.get("next_cursor")
            if not cursor:
                break
        return keys

    def get_blob(self, store: str, key: str) -> dict:
        """Fetch a single blob by *key* from *store*."""
        # Encode each path segment individually (keys may contain slashes)
        encoded_key = "/".join(urlquote(seg, safe="") for seg in key.split("/"))
        url = f"{NETLIFY_API}/blobs/{self.site_id}/{store}/{encoded_key}"
        resp = self._session.get(url)
        resp.raise_for_status()
        return resp.json()

    # ── Convenience: options-chain ──────────────────────────

    def list_option_symbols(self) -> list[str]:
        """Return sorted list of underlying symbols in the options-chain store."""
        keys = self.list_keys("options-chain")
        symbols: set[str] = set()
        for key in keys:
            slash = key.find("/")
            if slash > 0:
                symbols.add(key[:slash])
        return sorted(symbols)

    def list_option_dates(self, symbol: str) -> list[str]:
        """Return available snapshot dates for *symbol* (newest first)."""
        keys = self.list_keys("options-chain", prefix=f"{symbol}/")
        dates: set[str] = set()
        for key in keys:
            dates.add(_date_from_key(key))
        return sorted(dates, reverse=True)

    def get_options_chain(
        self,
        symbol: str,
        date: str | None = None,
    ) -> OptionsChainBlob:
        """Fetch an options-chain blob.

        If *date* is None, picks the richest end-of-day blob (same heuristic
        as allocation-manager's ``pickRichestKey``).
        """
        prefix = f"{symbol}/"
        if date:
            prefix = f"{symbol}/{date}"
        keys = self.list_keys("options-chain", prefix=prefix)
        if not keys:
            raise KeyError(f"No options-chain blobs for {prefix!r}")
        best = _pick_richest_key(keys) if date is None else keys[-1]
        raw = self.get_blob("options-chain", best)
        return parse_options_chain_blob(raw)

    # ── Convenience: market-quotes ──────────────────────────

    def list_market_quote_dates(self) -> list[str]:
        """Return available snapshot dates for market-quotes (newest first)."""
        keys = self.list_keys("market-quotes")
        dates: set[str] = set()
        for key in keys:
            dates.add(_date_from_key(key))
        return sorted(dates, reverse=True)

    def get_market_quotes(self, date: str | None = None) -> MarketQuotesBlob:
        """Fetch a market-quotes blob.

        If *date* is None, picks the richest end-of-day blob.
        """
        prefix = date or ""
        keys = self.list_keys("market-quotes", prefix=prefix)
        if not keys:
            raise KeyError(f"No market-quotes blobs for prefix={prefix!r}")
        best = _pick_richest_key(keys) if date is None else keys[-1]
        raw = self.get_blob("market-quotes", best)
        return parse_market_quotes_blob(raw)


# ── Helpers ─────────────────────────────────────────────────


def _date_from_key(key: str) -> str:
    ts_start = key.rfind("/") + 1 if "/" in key else 0
    return key[ts_start : ts_start + 10]


def _pick_richest_key(keys: list[str]) -> str:
    """Pick the last key from the most recent completed day (before today UTC)."""
    if len(keys) <= 1:
        return keys[-1]
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for key in reversed(keys):
        date_str = _date_from_key(key)
        if date_str < today:
            return key
    return keys[-1]
