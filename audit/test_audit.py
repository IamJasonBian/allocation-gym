"""Smoke test — runs audit checks against mock Redis data."""

import json
import sys
import time

# Mock Redis that stores everything as strings (like real Redis with decode_responses=True)
class MockRedis:
    def __init__(self):
        self._h = {}

    def hset(self, key, mapping):
        if key not in self._h:
            self._h[key] = {}
        self._h[key].update({k: v for k, v in mapping.items()})

    def hgetall(self, key):
        return self._h.get(key, {})

    def scan(self, cursor, match="*", count=100):
        import fnmatch
        keys = [k for k in self._h if fnmatch.fnmatch(k, match)]
        return 0, keys


def main():
    from reader import read_options_record
    from checks import audit_record, Finding

    r = MockRedis()
    occ = "TSLA260321C00250000"

    r.hset(f"options:{occ}", {
        "symbol": occ,
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

    rec = read_options_record(r, occ)
    assert rec is not None, "Failed to read record"
    assert rec.symbol == occ
    assert rec.underlying == "TSLA"
    assert rec.pricing.bid == 8.50
    assert rec.greeks.iv == 0.45
    assert len(rec.orders) == 2
    print(f"  [OK] read_options_record parsed {occ}")

    findings = audit_record(rec)
    severity_map = {"INFO": ".", "WARN": "!", "FAIL": "X"}
    for f in findings:
        icon = severity_map.get(f.severity, "?")
        print(f"  [{icon}] {f.check:20s} {f.detail}")

    fails = [f for f in findings if f.severity == "FAIL"]
    warns = [f for f in findings if f.severity == "WARN"]
    print(f"\n  {len(findings)} checks | {len(fails)} FAIL | {len(warns)} WARN")
    print("  PASS" if not fails else "  AUDIT FAILURES DETECTED")


if __name__ == "__main__":
    main()
