"""
Read OptionsRecord from Redis using the protobuf schema.

Redis layout (per proto spec):
  options:{OCC_SYMBOL}  →  hash with flat strings + JSON sub-objects

This module hydrates the generated OptionsRecord protobuf from that hash.
"""

import json

from .generated import options_contract_pb2 as pb


def _parse_json(raw, default=None):
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _set_optional(msg, field, value):
    """Set an optional proto field if the value is not None."""
    if value is not None:
        setattr(msg, field, value)


def read_options_record(redis_client, occ_symbol):
    """Read a single OptionsRecord from Redis hash ``options:{occ_symbol}``.

    Returns an OptionsRecord protobuf, or None if key doesn't exist.
    """
    key = f"options:{occ_symbol}"
    raw = redis_client.hgetall(key)
    if not raw:
        return None

    # Normalise bytes → str
    data = {}
    for k, v in raw.items():
        k = k.decode() if isinstance(k, bytes) else k
        v = v.decode() if isinstance(v, bytes) else v
        data[k] = v

    rec = pb.OptionsRecord()

    # ── Identity (flat strings) ──
    rec.symbol = data.get("symbol", occ_symbol)
    rec.underlying = data.get("underlying", "")
    rec.expiration = data.get("expiration", "")
    rec.strike = float(data.get("strike", 0))
    rec.option_type = data.get("option_type", "")

    # ── Pricing (JSON) ──
    pricing = _parse_json(data.get("pricing"), {})
    for field in ("bid", "ask", "mid", "spread", "last_price", "limit_price",
                  "stop_price", "avg_entry"):
        _set_optional(rec.pricing, field, pricing.get(field))

    # ── Greeks (JSON) ──
    greeks = _parse_json(data.get("greeks"), {})
    for field in ("delta", "gamma", "theta", "vega", "rho", "iv"):
        _set_optional(rec.greeks, field, greeks.get(field))

    # ── Sizing (JSON) ──
    sizing = _parse_json(data.get("sizing"), {})
    for field in ("qty", "filled_qty"):
        val = sizing.get(field)
        if val is not None:
            _set_optional(rec.sizing, field, int(val))
    for field in ("volume", "open_interest"):
        val = sizing.get(field)
        if val is not None:
            _set_optional(rec.sizing, field, int(val))

    # ── PnL (JSON) ──
    pnl = _parse_json(data.get("pnl"), {})
    for field in ("unrealized_pl", "unrealized_pl_pct", "market_value"):
        _set_optional(rec.pnl, field, pnl.get(field))

    # ── Order-level flat fields ──
    for field in ("side", "order_type", "status", "order_id"):
        if field in data:
            setattr(rec, field, data[field])

    # ── Repeated orders (JSON array) ──
    orders_raw = _parse_json(data.get("orders"), [])
    for o in orders_raw:
        entry = rec.orders.add()
        entry.id = o.get("id", "")
        entry.side = o.get("side", "")
        entry.order_type = o.get("order_type", "")
        entry.qty = int(o.get("qty", 0))
        entry.status = o.get("status", "")
        entry.created_at = o.get("created_at", "")
        if o.get("limit_price") is not None:
            entry.limit_price = float(o["limit_price"])
        if o.get("stop_price") is not None:
            entry.stop_price = float(o["stop_price"])
        if o.get("filled_qty") is not None:
            entry.filled_qty = int(o["filled_qty"])
        if o.get("filled_avg_price") is not None:
            entry.filled_avg_price = float(o["filled_avg_price"])

    # ── Repeated bars (JSON array) ──
    bars_raw = _parse_json(data.get("bars"), [])
    for b in bars_raw:
        bar = rec.bars.add()
        bar.timestamp = b.get("timestamp", "")
        bar.open = float(b.get("open", 0))
        bar.high = float(b.get("high", 0))
        bar.low = float(b.get("low", 0))
        bar.close = float(b.get("close", 0))
        bar.volume = int(b.get("volume", 0))

    # ── Meta ──
    rec.updated_at = data.get("updated_at", "")
    rec.created_at = data.get("created_at", "")

    return rec


def scan_options_keys(redis_client, pattern="options:*"):
    """Yield all OCC symbols that have an options hash in Redis."""
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
        for k in keys:
            k = k.decode() if isinstance(k, bytes) else k
            yield k.removeprefix("options:")
        if cursor == 0:
            break
