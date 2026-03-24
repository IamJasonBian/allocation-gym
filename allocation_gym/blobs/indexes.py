"""Local JSON indexes for fast faceted blob key lookups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from allocation_gym.blobs.models import Predicate, parse_occ

if TYPE_CHECKING:
    from allocation_gym.blobs.client import NetlifyBlobClient

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data"


class LocalIndex:
    """Persisted JSON index mapping property facets to blob keys.

    On-disk structure (e.g. ``options_chain_index.json``)::

        {
          "meta": {"store": "options-chain", "built_at": "..."},
          "by_underlying": {"IWN": ["IWN/2026-03-02T...", ...], ...},
          "by_expiry":     {"2026-04-18": [...], ...},
          "by_option_type": {"call": [...], "put": [...]}
        }

    Parameters
    ----------
    name : str
        Base filename (without ``.json``).
    index_dir : Path or str, optional
        Directory to store the index file.  Defaults to ``allocation_gym/data/``.
    """

    def __init__(self, name: str, index_dir: Path | str | None = None) -> None:
        self.name = name
        self.dir = Path(index_dir) if index_dir else _DEFAULT_DIR
        self.path = self.dir / f"{name}.json"
        self._data: dict[str, Any] = {}

    # -- persistence ------------------------------------------------------

    def load(self) -> LocalIndex:
        """Load index from disk.  Returns self for chaining."""
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        return self

    def save(self) -> None:
        """Write index to disk as JSON."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    # -- lookup -----------------------------------------------------------

    @property
    def facets(self) -> list[str]:
        """Return available facet names (excluding 'meta')."""
        return [k for k in self._data if k != "meta"]

    def lookup(self, facet: str, value: str) -> list[str]:
        """Return blob keys where *facet* equals *value* exactly."""
        return list(self._data.get(facet, {}).get(value, []))

    def lookup_predicate(self, facet: str, predicate: Predicate[str]) -> list[str]:
        """Return blob keys where the facet value satisfies *predicate*.

        Example::

            idx.lookup_predicate("by_expiry", lambda d: d <= "2026-04-30")
        """
        results: list[str] = []
        for value, keys in self._data.get(facet, {}).items():
            if predicate(value):
                results.extend(keys)
        return results

    def keys(self) -> list[str]:
        """Return all unique blob keys across all facets."""
        seen: set[str] = set()
        for facet_name, mapping in self._data.items():
            if facet_name == "meta" or not isinstance(mapping, dict):
                continue
            for key_list in mapping.values():
                seen.update(key_list)
        return sorted(seen)

    # -- builders ---------------------------------------------------------

    @classmethod
    def build_options_index(
        cls,
        client: NetlifyBlobClient,
        prefix: str = "",
        index_dir: Path | str | None = None,
    ) -> LocalIndex:
        """Scan ``options-chain`` blobs and build a faceted index.

        Fetches each blob, parses OCC symbols, and indexes keys by
        ``by_underlying``, ``by_expiry``, and ``by_option_type``.
        """
        idx = cls("options_chain_index", index_dir)
        by_underlying: dict[str, set[str]] = {}
        by_expiry: dict[str, set[str]] = {}
        by_option_type: dict[str, set[str]] = {}

        keys = client.list_keys("options-chain", prefix)
        for key in keys:
            raw = client.get_blob("options-chain", key)
            if not isinstance(raw, dict):
                continue
            chain = raw.get("latest_chain", raw)
            for sym in chain:
                if sym == "_meta":
                    continue
                parsed = parse_occ(sym)
                if not parsed:
                    continue
                by_underlying.setdefault(parsed["underlying"], set()).add(key)
                by_expiry.setdefault(parsed["expiry"], set()).add(key)
                by_option_type.setdefault(parsed["option_type"], set()).add(key)

        idx._data = {
            "meta": {
                "store": "options-chain",
                "built_at": datetime.now(timezone.utc).isoformat(),
                "key_count": len(keys),
            },
            "by_underlying": {k: sorted(v) for k, v in by_underlying.items()},
            "by_expiry": {k: sorted(v) for k, v in by_expiry.items()},
            "by_option_type": {k: sorted(v) for k, v in by_option_type.items()},
        }
        return idx

    @classmethod
    def build_quotes_index(
        cls,
        client: NetlifyBlobClient,
        prefix: str = "",
        index_dir: Path | str | None = None,
    ) -> LocalIndex:
        """Scan ``market-quotes`` blobs and build a faceted index.

        Indexes keys by ``by_symbol`` and ``by_asset_class``.
        """
        idx = cls("market_quotes_index", index_dir)
        by_symbol: dict[str, set[str]] = {}
        by_asset_class: dict[str, set[str]] = {}

        keys = client.list_keys("market-quotes", prefix)
        for key in keys:
            raw = client.get_blob("market-quotes", key)
            if not isinstance(raw, dict):
                continue
            quotes = raw.get("latest_quotes", raw)
            for sym, q in quotes.items():
                if sym == "_meta" or not isinstance(q, dict):
                    continue
                by_symbol.setdefault(sym, set()).add(key)
                ac = q.get("asset_class") or q.get("assetClass", "")
                if ac:
                    by_asset_class.setdefault(ac, set()).add(key)

        idx._data = {
            "meta": {
                "store": "market-quotes",
                "built_at": datetime.now(timezone.utc).isoformat(),
                "key_count": len(keys),
            },
            "by_symbol": {k: sorted(v) for k, v in by_symbol.items()},
            "by_asset_class": {k: sorted(v) for k, v in by_asset_class.items()},
        }
        return idx
