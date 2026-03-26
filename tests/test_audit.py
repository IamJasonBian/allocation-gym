"""Pytest tests for the audit module — checks against mock Redis data."""

import json
import time

import pytest

from audit.reader import read_options_record, scan_options_keys
from audit.checks import (
    audit_record,
    check_identity,
    check_staleness,
    check_pricing_sanity,
    check_spread,
    check_greeks_present,
    check_order_limits,
    Finding,
)


# ─────────────────────────────────────────────────────────────
# Mock Redis
# ─────────────────────────────────────────────────────────────

class MockRedis:
    def __init__(self):
        self._h = {}

    def hset(self, key, mapping):
        self._h.setdefault(key, {}).update(mapping)

    def hgetall(self, key):
        return self._h.get(key, {})

    def scan(self, cursor, match="*", count=100):
        import fnmatch
        keys = [k for k in self._h if fnmatch.fnmatch(k, match)]
        return 0, keys


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

OCC = "TSLA260321C00250000"


@pytest.fixture
def redis_with_record():
    r = MockRedis()
    r.hset(f"options:{OCC}", {
        "symbol": OCC,
        "underlying": "TSLA",
        "expiration": "2026-03-21",
        "strike": "250.0",
        "option_type": "C",
        "pricing": json.dumps({
            "bid": 8.50, "ask": 9.20, "mid": 8.85,
            "spread": 0.70, "last_price": 8.85,
        }),
        "greeks": json.dumps({
            "iv": 0.45, "delta": 0.42, "gamma": 0.03,
            "theta": -0.15, "vega": 0.28,
        }),
        "sizing": json.dumps({"qty": 5, "volume": 3200, "open_interest": 15000}),
        "orders": json.dumps([
            {"id": "ORD001", "side": "buy", "order_type": "limit",
             "limit_price": 9.00, "qty": 5, "status": "open"},
            {"id": "ORD002", "side": "sell", "order_type": "limit",
             "limit_price": 8.00, "qty": 3, "status": "open"},
        ]),
        "updated_at": str(int(time.time() * 1000)),
    })
    return r


# ─────────────────────────────────────────────────────────────
# reader tests
# ─────────────────────────────────────────────────────────────

def test_read_options_record_parses_fields(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    assert rec is not None
    assert rec.symbol == OCC
    assert rec.underlying == "TSLA"
    assert rec.expiration == "2026-03-21"
    assert rec.strike == 250.0
    assert rec.option_type == "C"


def test_read_options_record_pricing(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    assert rec.pricing.bid == pytest.approx(8.50)
    assert rec.pricing.ask == pytest.approx(9.20)
    assert rec.pricing.mid == pytest.approx(8.85)


def test_read_options_record_greeks(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    assert rec.greeks.iv == pytest.approx(0.45)
    assert rec.greeks.delta == pytest.approx(0.42)


def test_read_options_record_orders(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    assert len(rec.orders) == 2
    assert rec.orders[0].id == "ORD001"
    assert rec.orders[1].id == "ORD002"


def test_read_options_record_missing_key(redis_with_record):
    rec = read_options_record(redis_with_record, "NONEXISTENT")
    assert rec is None


def test_scan_options_keys(redis_with_record):
    keys = list(scan_options_keys(redis_with_record))
    assert OCC in keys


# ─────────────────────────────────────────────────────────────
# checks tests
# ─────────────────────────────────────────────────────────────

def test_audit_record_returns_findings(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    findings = audit_record(rec)
    assert len(findings) > 0
    assert all(isinstance(f, Finding) for f in findings)


def test_audit_record_no_fails_on_valid_record(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    findings = audit_record(rec)
    fails = [f for f in findings if f.severity == "FAIL"]
    assert fails == [], f"Unexpected FAILs: {fails}"


def test_check_identity_passes_valid_record(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    findings = check_identity(rec)
    assert any(f.severity == "INFO" for f in findings)
    assert not any(f.severity == "FAIL" for f in findings)


def test_check_staleness_fresh_record(redis_with_record):
    rec = read_options_record(redis_with_record, OCC)
    findings = check_staleness(rec)
    assert any(f.severity == "INFO" for f in findings)
    assert not any(f.severity == "FAIL" for f in findings)


def test_check_staleness_stale_record():
    r = MockRedis()
    stale_ts = str(int((time.time() - 600) * 1000))  # 10 minutes ago
    r.hset(f"options:{OCC}", {
        "symbol": OCC, "underlying": "TSLA",
        "expiration": "2026-03-21", "strike": "250.0", "option_type": "C",
        "pricing": json.dumps({"bid": 8.50, "ask": 9.20}),
        "greeks": json.dumps({"iv": 0.45, "delta": 0.42}),
        "sizing": json.dumps({}),
        "orders": json.dumps([]),
        "updated_at": stale_ts,
    })
    rec = read_options_record(r, OCC)
    findings = check_staleness(rec, max_age_seconds=300)
    assert any(f.severity == "FAIL" for f in findings)


def test_check_pricing_sanity_crossed_market():
    r = MockRedis()
    r.hset(f"options:{OCC}", {
        "symbol": OCC, "underlying": "TSLA",
        "expiration": "2026-03-21", "strike": "250.0", "option_type": "C",
        "pricing": json.dumps({"bid": 10.0, "ask": 8.0}),  # bid > ask
        "greeks": json.dumps({"iv": 0.45, "delta": 0.42}),
        "sizing": json.dumps({}),
        "orders": json.dumps([]),
        "updated_at": str(int(time.time() * 1000)),
    })
    rec = read_options_record(r, OCC)
    findings = check_pricing_sanity(rec)
    assert any(f.severity == "FAIL" for f in findings)


def test_check_spread_illiquid():
    r = MockRedis()
    r.hset(f"options:{OCC}", {
        "symbol": OCC, "underlying": "TSLA",
        "expiration": "2026-03-21", "strike": "250.0", "option_type": "C",
        "pricing": json.dumps({"bid": 1.0, "ask": 5.0}),  # wide spread
        "greeks": json.dumps({"iv": 0.45, "delta": 0.42}),
        "sizing": json.dumps({}),
        "orders": json.dumps([]),
        "updated_at": str(int(time.time() * 1000)),
    })
    rec = read_options_record(r, OCC)
    findings = check_spread(rec, max_spread_pct=10.0)
    assert any(f.severity == "WARN" for f in findings)


def test_check_greeks_present_missing_iv():
    r = MockRedis()
    r.hset(f"options:{OCC}", {
        "symbol": OCC, "underlying": "TSLA",
        "expiration": "2026-03-21", "strike": "250.0", "option_type": "C",
        "pricing": json.dumps({"bid": 8.50, "ask": 9.20}),
        "greeks": json.dumps({"delta": 0.42}),  # no iv
        "sizing": json.dumps({}),
        "orders": json.dumps([]),
        "updated_at": str(int(time.time() * 1000)),
    })
    rec = read_options_record(r, OCC)
    findings = check_greeks_present(rec)
    assert any(f.severity == "WARN" for f in findings)
    assert any("iv" in f.detail for f in findings)
