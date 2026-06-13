"""Lambda-based blob indexing for slicing options and market data.

Build indexes over blob snapshots using arbitrary comparator functions,
then query them to slice by strike, expiry, greeks, moneyness, etc.

Example
-------
    from allocation_gym.blobs import BlobClient, BlobIndex

    client = BlobClient()
    chain = client.get_options_chain("CRWD")

    idx = BlobIndex(chain.snapshots)

    # Slice by option type
    calls = idx.where(lambda s: s.is_call)
    puts  = idx.where(lambda s: s.is_put)

    # ATM options with IV > 30%
    atm_high_iv = idx.where(
        lambda s: s.moneyness == "ATM" and s.iv is not None and s.iv > 0.30
    )

    # Strike range
    near_200 = idx.where(lambda s: 195 <= s.strike <= 205)

    # Chain multiple filters
    short_dated_calls = (
        idx.where(lambda s: s.is_call)
           .where(lambda s: s.expiry <= "2026-04-01")
           .where(lambda s: s.delta is not None and s.delta > 0.3)
    )

    # Build a named index for repeated lookups
    idx.build_index("by_expiry", key_fn=lambda s: s.expiry)
    apr_options = idx.get("by_expiry", "2026-04-18")

    idx.build_index("by_strike_bucket", key_fn=lambda s: round(s.strike / 5) * 5)
    bucket_200 = idx.get("by_strike_bucket", 200)

    # Sort
    by_iv = idx.where(lambda s: s.iv is not None).sort_by(lambda s: s.iv)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class BlobIndex(Generic[T]):
    """Filterable, indexable collection of blob items.

    Works with any item type — OptionSnapshot, MarketQuote, or raw dicts.
    All filtering is done via lambda comparators passed to ``where()``.
    """

    def __init__(self, items: list[T]) -> None:
        self._items = list(items)
        self._indexes: dict[str, dict[Any, list[T]]] = {}

    # ── Core properties ─────────────────────────────────────

    @property
    def items(self) -> list[T]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __repr__(self) -> str:
        indexes = list(self._indexes.keys())
        return f"BlobIndex({len(self._items)} items, indexes={indexes})"

    # ── Lambda filtering ────────────────────────────────────

    def where(self, predicate: Callable[[T], bool]) -> BlobIndex[T]:
        """Return a new BlobIndex with only items matching *predicate*.

        Parameters
        ----------
        predicate : callable
            A lambda or function that takes an item and returns bool.

        Returns
        -------
        BlobIndex[T]
            A new index (indexes are NOT carried over — rebuild if needed).
        """
        return BlobIndex([item for item in self._items if predicate(item)])

    def first(self, predicate: Callable[[T], bool] | None = None) -> T | None:
        """Return the first item matching *predicate*, or None."""
        source = self._items if predicate is None else (i for i in self._items if predicate(i))
        return next(iter(source), None)

    def sort_by(self, key_fn: Callable[[T], Any], reverse: bool = False) -> BlobIndex[T]:
        """Return a new BlobIndex sorted by *key_fn*."""
        return BlobIndex(sorted(self._items, key=key_fn, reverse=reverse))

    def group_by(self, key_fn: Callable[[T], Any]) -> dict[Any, BlobIndex[T]]:
        """Group items by *key_fn*, returning a dict of BlobIndex per group."""
        groups: dict[Any, list[T]] = defaultdict(list)
        for item in self._items:
            groups[key_fn(item)].append(item)
        return {k: BlobIndex(v) for k, v in groups.items()}

    def map(self, fn: Callable[[T], Any]) -> list[Any]:
        """Apply *fn* to each item and return the results."""
        return [fn(item) for item in self._items]

    def aggregate(
        self,
        key_fn: Callable[[T], Any],
        value_fn: Callable[[T], float],
        agg: str = "mean",
    ) -> dict[Any, float]:
        """Aggregate a numeric value across groups.

        Parameters
        ----------
        key_fn : callable
            Groups items (e.g. ``lambda s: s.expiry``).
        value_fn : callable
            Extracts the numeric value (e.g. ``lambda s: s.iv``).
        agg : str
            One of "mean", "sum", "min", "max", "count".
        """
        groups: dict[Any, list[float]] = defaultdict(list)
        for item in self._items:
            val = value_fn(item)
            if val is not None:
                groups[key_fn(item)].append(val)

        result: dict[Any, float] = {}
        for k, vals in groups.items():
            if not vals:
                continue
            if agg == "mean":
                result[k] = sum(vals) / len(vals)
            elif agg == "sum":
                result[k] = sum(vals)
            elif agg == "min":
                result[k] = min(vals)
            elif agg == "max":
                result[k] = max(vals)
            elif agg == "count":
                result[k] = len(vals)
            else:
                raise ValueError(f"Unknown aggregation: {agg!r}")
        return result

    # ── Named indexes for repeated lookups ──────────────────

    def build_index(
        self,
        name: str,
        key_fn: Callable[[T], Any],
        dedup_fn: Callable[[T], Any] | None = None,
    ) -> None:
        """Build a named index for O(1) lookups by *key_fn*.

        Parameters
        ----------
        name : str
            Index name (e.g. "by_expiry", "by_strike", "by_symbol").
        key_fn : callable
            Extracts the index key from each item.
        dedup_fn : callable, optional
            When provided, only the item with the **maximum** value returned by
            *dedup_fn* is kept for each key.  This is useful for building a
            ``"by_symbol"`` index where only the latest record per symbol is
            needed::

                idx.build_index(
                    "by_symbol",
                    key_fn=lambda s: s.symbol,
                    dedup_fn=lambda s: s.latest_quote.timestamp,
                )
        """
        if dedup_fn is not None:
            best: dict[Any, T] = {}
            for item in self._items:
                k = key_fn(item)
                if k not in best or dedup_fn(item) > dedup_fn(best[k]):
                    best[k] = item
            self._indexes[name] = {k: [v] for k, v in best.items()}
        else:
            idx: dict[Any, list[T]] = defaultdict(list)
            for item in self._items:
                idx[key_fn(item)].append(item)
            self._indexes[name] = dict(idx)

    def get(self, index_name: str, key: Any) -> BlobIndex[T]:
        """Look up items by *key* in a previously built named index."""
        if index_name not in self._indexes:
            raise KeyError(f"No index named {index_name!r}. Call build_index() first.")
        items = self._indexes[index_name].get(key, [])
        return BlobIndex(items)

    def index_keys(self, index_name: str) -> list[Any]:
        """Return all keys in a named index."""
        if index_name not in self._indexes:
            raise KeyError(f"No index named {index_name!r}")
        return list(self._indexes[index_name].keys())

    def drop_index(self, name: str) -> None:
        """Remove a named index."""
        self._indexes.pop(name, None)

    # ── Convenience filters for OptionSnapshot ──────────────
    #    These assume items have the relevant attributes; they
    #    silently return empty results for non-matching types.

    def calls(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "is_call", False))

    def puts(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "is_put", False))

    def expiry(self, date: str) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "expiry", None) == date)

    def strikes(self, low: float, high: float) -> BlobIndex[T]:
        return self.where(lambda s: low <= getattr(s, "strike", -1) <= high)

    def itm(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "moneyness", "") == "ITM")

    def otm(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "moneyness", "") == "OTM")

    def atm(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "moneyness", "") == "ATM")

    def has_greeks(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "greeks", None) is not None)

    def has_iv(self) -> BlobIndex[T]:
        return self.where(lambda s: getattr(s, "iv", None) is not None)

    def iv_range(self, low: float, high: float) -> BlobIndex[T]:
        """Filter by implied volatility range (as decimal, e.g. 0.30 for 30%)."""
        return self.where(
            lambda s: (iv := getattr(s, "iv", None)) is not None and low <= iv <= high
        )

    def delta_range(self, low: float, high: float) -> BlobIndex[T]:
        """Filter by delta range."""
        return self.where(
            lambda s: (d := getattr(s, "delta", None)) is not None and low <= d <= high
        )

    # ── DataFrame export ────────────────────────────────────

    def to_dataframe(self, columns: list[str] | None = None):
        """Export items to a pandas DataFrame.

        Parameters
        ----------
        columns : list[str], optional
            Attribute names to include. Defaults to all dataclass fields.
        """
        import pandas as pd
        from dataclasses import fields as dc_fields, asdict

        if not self._items:
            return pd.DataFrame()

        item = self._items[0]
        if hasattr(item, "__dataclass_fields__"):
            if columns:
                rows = [{c: getattr(i, c, None) for c in columns} for i in self._items]
            else:
                rows = [asdict(i) for i in self._items]  # type: ignore[arg-type]
        elif isinstance(item, dict):
            rows = self._items  # type: ignore[assignment]
        else:
            raise TypeError(f"Cannot convert {type(item)} to DataFrame")

        return pd.DataFrame(rows)
