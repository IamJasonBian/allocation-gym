"""Importance-sampling pricing of illiquid alt-coin OTC derivatives.

This package provides a self-contained system for pricing illiquid alt-coin
OTC derivatives via importance-sampling Monte Carlo, resilient to upstream
datafeed drops, with a stdlib HTTP API hooked to real Binance L2 depth and a
deterministic mock fallback.
"""
