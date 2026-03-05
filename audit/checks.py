"""
Audit checks for OptionsRecord read from Redis via protobuf schema.

Each check returns a list of Finding dicts:
  {"check": str, "severity": "INFO"|"WARN"|"FAIL", "symbol": str, "detail": str}
"""

import time
from dataclasses import dataclass, field, asdict
from typing import List

from .reader import read_options_record

# ─────────────────────────────────────────────────────────────
# Finding type
# ─────────────────────────────────────────────────────────────

@dataclass
class Finding:
    check: str
    severity: str       # INFO, WARN, FAIL
    symbol: str
    detail: str

    def as_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────

def check_staleness(rec, max_age_seconds=300):
    """FAIL if updated_at is older than max_age_seconds (default 5 min)."""
    findings = []
    if not rec.updated_at:
        findings.append(Finding(
            check="staleness",
            severity="WARN",
            symbol=rec.symbol,
            detail="updated_at is missing",
        ))
        return findings

    try:
        ts = int(rec.updated_at)
        # updated_at is in millis
        if ts > 1e12:
            ts = ts / 1000
        age = time.time() - ts
        if age > max_age_seconds:
            findings.append(Finding(
                check="staleness",
                severity="FAIL",
                symbol=rec.symbol,
                detail=f"data is {age:.0f}s old (limit {max_age_seconds}s)",
            ))
        else:
            findings.append(Finding(
                check="staleness",
                severity="INFO",
                symbol=rec.symbol,
                detail=f"data is {age:.0f}s old — fresh",
            ))
    except (ValueError, TypeError):
        findings.append(Finding(
            check="staleness",
            severity="WARN",
            symbol=rec.symbol,
            detail=f"updated_at is not a valid timestamp: {rec.updated_at!r}",
        ))
    return findings


def check_pricing_sanity(rec):
    """Verify bid <= mid <= ask and spread consistency."""
    findings = []
    p = rec.pricing
    bid = p.bid if p.HasField("bid") else None
    ask = p.ask if p.HasField("ask") else None
    mid = p.mid if p.HasField("mid") else None

    if bid is None or ask is None:
        findings.append(Finding(
            check="pricing_sanity",
            severity="WARN",
            symbol=rec.symbol,
            detail="bid or ask is missing",
        ))
        return findings

    if bid < 0 or ask < 0:
        findings.append(Finding(
            check="pricing_sanity",
            severity="FAIL",
            symbol=rec.symbol,
            detail=f"negative price: bid={bid}, ask={ask}",
        ))

    if bid > ask:
        findings.append(Finding(
            check="pricing_sanity",
            severity="FAIL",
            symbol=rec.symbol,
            detail=f"bid ({bid}) > ask ({ask}) — crossed market",
        ))

    if mid is not None and ask > 0:
        expected_mid = (bid + ask) / 2
        if abs(mid - expected_mid) > 0.01:
            findings.append(Finding(
                check="pricing_sanity",
                severity="WARN",
                symbol=rec.symbol,
                detail=f"mid ({mid}) != (bid+ask)/2 ({expected_mid:.4f})",
            ))

    if not findings:
        findings.append(Finding(
            check="pricing_sanity",
            severity="INFO",
            symbol=rec.symbol,
            detail="pricing looks good",
        ))
    return findings


def check_spread(rec, max_spread_pct=10.0):
    """WARN if bid-ask spread is too wide (illiquid)."""
    findings = []
    p = rec.pricing
    bid = p.bid if p.HasField("bid") else 0
    ask = p.ask if p.HasField("ask") else 0
    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

    if mid <= 0:
        return findings

    spread_pct = (ask - bid) / mid * 100
    if spread_pct > max_spread_pct:
        findings.append(Finding(
            check="spread",
            severity="WARN",
            symbol=rec.symbol,
            detail=f"spread {spread_pct:.1f}% exceeds {max_spread_pct}% — illiquid",
        ))
    else:
        findings.append(Finding(
            check="spread",
            severity="INFO",
            symbol=rec.symbol,
            detail=f"spread {spread_pct:.1f}% is acceptable",
        ))
    return findings


def check_greeks_present(rec):
    """WARN if IV or delta are missing (needed for signal logic)."""
    findings = []
    g = rec.greeks
    missing = []
    if not g.HasField("iv"):
        missing.append("iv")
    if not g.HasField("delta"):
        missing.append("delta")
    if missing:
        findings.append(Finding(
            check="greeks_present",
            severity="WARN",
            symbol=rec.symbol,
            detail=f"missing greeks: {', '.join(missing)}",
        ))
    else:
        findings.append(Finding(
            check="greeks_present",
            severity="INFO",
            symbol=rec.symbol,
            detail=f"iv={g.iv:.2f}, delta={g.delta:.2f}",
        ))
    return findings


def check_order_limits(rec, buy_limit_drift_pct=0.20, sell_limit_drift_pct=0.20):
    """Check that open order limits haven't drifted far from current market."""
    findings = []
    p = rec.pricing
    bid = p.bid if p.HasField("bid") else 0
    ask = p.ask if p.HasField("ask") else 0

    for order in rec.orders:
        if order.status != "open":
            continue
        if not order.HasField("limit_price"):
            continue

        limit = order.limit_price
        if order.side == "buy" and ask > 0:
            drift = abs(ask - limit) / limit if limit else 0
            if drift > buy_limit_drift_pct:
                findings.append(Finding(
                    check="order_limits",
                    severity="WARN",
                    symbol=rec.symbol,
                    detail=(
                        f"order {order.id} BUY limit ${limit:.2f} "
                        f"drifted {drift:.0%} from ask ${ask:.2f}"
                    ),
                ))
        elif order.side == "sell" and bid > 0:
            drift = abs(bid - limit) / limit if limit else 0
            if drift > sell_limit_drift_pct:
                findings.append(Finding(
                    check="order_limits",
                    severity="WARN",
                    symbol=rec.symbol,
                    detail=(
                        f"order {order.id} SELL limit ${limit:.2f} "
                        f"drifted {drift:.0%} from bid ${bid:.2f}"
                    ),
                ))

    if not findings:
        open_count = sum(1 for o in rec.orders if o.status == "open")
        findings.append(Finding(
            check="order_limits",
            severity="INFO",
            symbol=rec.symbol,
            detail=f"{open_count} open order(s) within tolerance",
        ))
    return findings


def check_identity(rec):
    """Verify required identity fields are populated."""
    findings = []
    missing = []
    if not rec.symbol:
        missing.append("symbol")
    if not rec.underlying:
        missing.append("underlying")
    if not rec.expiration:
        missing.append("expiration")
    if rec.strike <= 0:
        missing.append("strike")
    if rec.option_type not in ("C", "P"):
        missing.append(f"option_type={rec.option_type!r}")

    if missing:
        findings.append(Finding(
            check="identity",
            severity="FAIL",
            symbol=rec.symbol or "UNKNOWN",
            detail=f"missing/invalid identity fields: {', '.join(missing)}",
        ))
    else:
        findings.append(Finding(
            check="identity",
            severity="INFO",
            symbol=rec.symbol,
            detail=f"{rec.underlying} {rec.expiration} {rec.option_type} ${rec.strike:.0f}",
        ))
    return findings


# ─────────────────────────────────────────────────────────────
# Run all checks
# ─────────────────────────────────────────────────────────────

ALL_CHECKS = [
    check_identity,
    check_staleness,
    check_pricing_sanity,
    check_spread,
    check_greeks_present,
    check_order_limits,
]


def audit_record(rec):
    """Run all audit checks on an OptionsRecord. Returns list of Findings."""
    findings = []
    for check_fn in ALL_CHECKS:
        findings.extend(check_fn(rec))
    return findings


def audit_symbol(redis_client, occ_symbol):
    """Read one symbol from Redis and run all audit checks."""
    rec = read_options_record(redis_client, occ_symbol)
    if rec is None:
        return [Finding(
            check="read",
            severity="FAIL",
            symbol=occ_symbol,
            detail="no data in Redis",
        )]
    return audit_record(rec)
