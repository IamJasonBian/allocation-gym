"""
CLI entry point for the audit module.

Usage:
  python -m audit                           # scan all options:* keys
  python -m audit TSLA260321C00250000       # audit a single OCC symbol
  python -m audit --redis-url redis://...   # custom Redis URL
"""

import argparse
import json
import sys

import redis

from .reader import scan_options_keys
from .checks import audit_symbol


SEVERITY_ICON = {"INFO": ".", "WARN": "!", "FAIL": "X"}


def main():
    parser = argparse.ArgumentParser(description="Audit options records in Redis")
    parser.add_argument("symbols", nargs="*", help="OCC symbol(s) to audit (default: scan all)")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    args = parser.parse_args()

    r = redis.from_url(args.redis_url, decode_responses=False)

    symbols = args.symbols
    if not symbols:
        symbols = list(scan_options_keys(r))
        if not symbols:
            print("No options:* keys found in Redis.")
            return

    all_findings = []
    for sym in symbols:
        findings = audit_symbol(r, sym)
        all_findings.extend(findings)

    if args.json_output:
        print(json.dumps([f.as_dict() for f in all_findings], indent=2))
        return

    # Pretty print
    fails = [f for f in all_findings if f.severity == "FAIL"]
    warns = [f for f in all_findings if f.severity == "WARN"]

    for f in all_findings:
        icon = SEVERITY_ICON.get(f.severity, "?")
        print(f"  [{icon}] {f.check:20s} {f.symbol}: {f.detail}")

    print()
    print(f"  {len(all_findings)} checks | {len(fails)} FAIL | {len(warns)} WARN")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
