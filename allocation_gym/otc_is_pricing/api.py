"""Stdlib HTTP API for OTC importance-sampling pricing.

Exposes a tiny JSON API over :mod:`http.server` (no third-party web deps):

    GET  /health                       -> {"status": "ok"}
    GET  /book?symbol=BTCUSDT          -> OrderBookSnapshot as JSON
    GET  /feed/status?symbol=BTCUSDT   -> feed freshness + index price
    POST /price                        -> PriceResult as JSON

Run standalone::

    python3 -m allocation_gym.otc_is_pricing.api --port 8799

The server uses defensive imports so it boots before the sibling ``feeds`` /
``pricer`` modules are merged, falling back to local reference implementations.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

try:
    from allocation_gym.otc_is_pricing.feeds import (  # type: ignore
        BinanceDepthFeed,
        MockL2Feed,
        OrderBookSnapshot,
        build_index_price,
    )
except Exception:  # pragma: no cover - exercised when sibling unmerged
    from allocation_gym.otc_is_pricing._fallback_feed import (
        BinanceDepthFeed,
        MockL2Feed,
        OrderBookSnapshot,
        build_index_price,
    )

try:
    from allocation_gym.otc_is_pricing.pricer import (  # type: ignore
        PriceResult,
        bs_price,
        price_is,
        price_plain_mc,
    )
except Exception:  # pragma: no cover - exercised when sibling unmerged
    from allocation_gym.otc_is_pricing._fallback_pricer import (
        PriceResult,
        bs_price,
        price_is,
        price_plain_mc,
    )

logger = logging.getLogger(__name__)

# Defaults / policy.
DEFAULT_SIGMA = 0.8
DEFAULT_METHOD = "drift_tilt"
DEFAULT_N_PATHS = 50_000
STALE_MAX_AGE_S = 5.0
DEFAULT_SEED = 12345


class PricingError(ValueError):
    """Raised for malformed pricing requests (maps to HTTP 400)."""


def _snapshot_to_dict(snap: OrderBookSnapshot) -> dict[str, Any]:
    """Serialise an order-book snapshot (including derived fields) to JSON."""
    try:
        mid: Optional[float] = snap.mid
        micro: Optional[float] = snap.microprice
    except Exception:
        mid = None
        micro = None
    return {
        "symbol": snap.symbol,
        "ts": snap.ts,
        "source": snap.source,
        "bids": [{"price": b.price, "size": b.size} for b in snap.bids],
        "asks": [{"price": a.price, "size": a.size} for a in snap.asks],
        "mid": mid,
        "microprice": micro,
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Serialise a PriceResult (dataclass or duck-typed) to JSON."""
    if is_dataclass(result) and not isinstance(result, type):
        return asdict(result)
    return {
        "price": getattr(result, "price", None),
        "std_error": getattr(result, "std_error", None),
        "ess": getattr(result, "ess", None),
        "n_paths": getattr(result, "n_paths", None),
        "method": getattr(result, "method", None),
    }


class PricingService:
    """Holds the feed and implements endpoint logic, free of HTTP plumbing.

    Kept separate from the request handler so it can be unit-tested directly
    without a socket.
    """

    def __init__(self, feed: Any | None = None, default_sigma: float = DEFAULT_SIGMA,
                 stale_max_age_s: float = STALE_MAX_AGE_S):
        self.feed = feed if feed is not None else BinanceDepthFeed()
        self.default_sigma = default_sigma
        self.stale_max_age_s = stale_max_age_s

    # -- feed-backed endpoints --------------------------------------------

    def get_book(self, symbol: str) -> dict[str, Any]:
        snap = self.feed.get_book(symbol)
        return _snapshot_to_dict(snap)

    def feed_status(self, symbol: str, now: Optional[float] = None) -> dict[str, Any]:
        if now is None:
            now = time.time()
        snap = self.feed.get_book(symbol)
        age = max(now - snap.ts, 0.0)
        try:
            stale = snap.is_stale(now, self.stale_max_age_s)
        except Exception:
            stale = age > self.stale_max_age_s
        # "datafeed_drop": we consider the feed dropped if the live source could
        # not be reached and we are serving a mock book, or the book is stale.
        datafeed_drop = bool(stale) or snap.source == "mock"
        try:
            index_price = build_index_price(snap)
        except Exception:
            index_price = None
        return {
            "source": snap.source,
            "age_s": age,
            "stale": bool(stale),
            "datafeed_drop": datafeed_drop,
            "index_price": index_price,
        }

    # -- pricing -----------------------------------------------------------

    def price(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise PricingError("request body must be a JSON object")

        symbol = body.get("symbol")
        kind = body.get("kind")
        if not symbol or not isinstance(symbol, str):
            raise PricingError("missing or invalid 'symbol'")
        if not kind or not isinstance(kind, str):
            raise PricingError("missing or invalid 'kind'")
        kind = kind.lower()
        if kind not in ("call", "put"):
            raise PricingError("'kind' must be 'call' or 'put'")

        try:
            K = float(body["K"])
            T = float(body["T"])
            r = float(body["r"])
        except KeyError as exc:
            raise PricingError(f"missing required field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise PricingError(f"invalid numeric field: {exc}") from exc

        if K <= 0:
            raise PricingError("'K' must be positive")
        if T < 0:
            raise PricingError("'T' must be non-negative")

        method = body.get("method") or DEFAULT_METHOD
        if not isinstance(method, str):
            raise PricingError("'method' must be a string")

        try:
            n_paths = int(body.get("n_paths", DEFAULT_N_PATHS))
        except (TypeError, ValueError) as exc:
            raise PricingError(f"invalid 'n_paths': {exc}") from exc
        if n_paths <= 0:
            raise PricingError("'n_paths' must be positive")

        sigma_raw = body.get("sigma")
        if sigma_raw is None:
            sigma = self.default_sigma
        else:
            try:
                sigma = float(sigma_raw)
            except (TypeError, ValueError) as exc:
                raise PricingError(f"invalid 'sigma': {exc}") from exc
            if sigma <= 0:
                raise PricingError("'sigma' must be positive")

        barrier = body.get("barrier")
        if barrier is not None:
            try:
                barrier = float(barrier)
            except (TypeError, ValueError) as exc:
                raise PricingError(f"invalid 'barrier': {exc}") from exc

        try:
            seed = int(body.get("seed", DEFAULT_SEED))
        except (TypeError, ValueError) as exc:
            raise PricingError(f"invalid 'seed': {exc}") from exc

        # Spot from the feed micro-price.
        snap = self.feed.get_book(symbol)
        try:
            spot = build_index_price(snap)
        except Exception as exc:
            raise PricingError(f"cannot derive spot from feed: {exc}") from exc

        kwargs: dict[str, Any] = {}
        if barrier is not None:
            kwargs["barrier"] = barrier

        if method == "plain_mc":
            result = price_plain_mc(spot, K, T, r, sigma, kind, n_paths, seed, **kwargs)
        else:
            result = price_is(spot, K, T, r, sigma, kind, n_paths, seed, method=method,
                              **kwargs)

        out = _result_to_dict(result)
        # Enrich with context useful to clients without changing PriceResult.
        out["spot"] = spot
        out["sigma"] = sigma
        out["symbol"] = snap.symbol
        out["bs_reference"] = bs_price(spot, K, T, r, sigma, kind)
        return out


def make_handler(service: PricingService) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to ``service``."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "OTCISPricing/0.1"

        # Silence default noisy logging; route through our logger instead.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            logger.debug("%s - %s", self.address_string(), fmt % args)

        # -- helpers ------------------------------------------------------

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _query(self) -> dict[str, list[str]]:
            return parse_qs(urlparse(self.path).query)

        def _symbol(self) -> str:
            q = self._query()
            vals = q.get("symbol")
            return vals[0] if vals else "BTCUSDT"

        # -- verbs --------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/health":
                    self._send_json(200, {"status": "ok"})
                elif path == "/book":
                    self._send_json(200, service.get_book(self._symbol()))
                elif path == "/feed/status":
                    self._send_json(200, service.feed_status(self._symbol()))
                else:
                    self._send_json(404, {"error": "not found", "path": path})
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("GET %s failed", path)
                self._send_json(500, {"error": "internal error", "detail": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/price":
                self._send_json(404, {"error": "not found", "path": path})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (TypeError, ValueError):
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": "invalid JSON", "detail": str(exc)})
                return
            try:
                self._send_json(200, service.price(body))
            except PricingError as exc:
                self._send_json(400, {"error": "bad request", "detail": str(exc)})
            except Exception as exc:
                logger.exception("POST /price failed")
                self._send_json(500, {"error": "internal error", "detail": str(exc)})

    return Handler


def build_server(host: str, port: int, service: Optional[PricingService] = None
                 ) -> ThreadingHTTPServer:
    """Construct (but do not start) a threading HTTP server."""
    svc = service or PricingService()
    handler = make_handler(svc)
    return ThreadingHTTPServer((host, port), handler)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OTC importance-sampling pricing API")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8799, help="bind port")
    parser.add_argument("--feed", choices=["binance", "mock"], default="binance",
                        help="order-book feed source")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="default volatility when not supplied per-request")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    feed = MockL2Feed() if args.feed == "mock" else BinanceDepthFeed()
    service = PricingService(feed=feed, default_sigma=args.sigma)
    server = build_server(args.host, args.port, service)
    logger.info("OTC IS pricing API listening on http://%s:%d (feed=%s)",
                args.host, args.port, args.feed)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
