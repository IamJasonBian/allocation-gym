"""Tests for the OTC importance-sampling pricing HTTP API.

All tests use the deterministic mock feed and never touch the network. The
server is bound on an ephemeral port in a background thread; requests go through
the real HTTP stack so handler wiring (routing, status codes, JSON) is covered.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import pytest

from allocation_gym.otc_is_pricing._fallback_feed import (
    BookLevel,
    MockL2Feed,
    OrderBookSnapshot,
    build_index_price,
)
from allocation_gym.otc_is_pricing._fallback_pricer import (
    bs_price,
    price_is,
    price_plain_mc,
)
from allocation_gym.otc_is_pricing.api import PricingService, build_server


# --------------------------------------------------------------------------
# server fixture
# --------------------------------------------------------------------------


@pytest.fixture
def server():
    service = PricingService(feed=MockL2Feed())
    srv = build_server("127.0.0.1", 0, service)  # port 0 -> ephemeral
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    base = f"http://{host}:{port}"
    try:
        yield base
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post(base: str, path: str, payload):
    data = json.dumps(payload).encode() if not isinstance(payload, (bytes, str)) else (
        payload.encode() if isinstance(payload, str) else payload
    )
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


# --------------------------------------------------------------------------
# fallback feed unit tests
# --------------------------------------------------------------------------


def test_snapshot_mid_and_microprice():
    snap = OrderBookSnapshot(
        symbol="X", ts=0.0, source="mock",
        bids=[BookLevel(99.0, 2.0)], asks=[BookLevel(101.0, 6.0)],
    )
    assert snap.mid == 100.0
    # micro pulled toward thicker ask side -> below mid.
    assert snap.microprice < snap.mid
    # (101*2 + 99*6) / 8 = 99.5
    assert math.isclose(snap.microprice, 99.5)


def test_snapshot_is_stale():
    snap = OrderBookSnapshot(symbol="X", ts=100.0, source="mock",
                             bids=[BookLevel(1, 1)], asks=[BookLevel(2, 1)])
    assert snap.is_stale(now=110.0, max_age_s=5.0) is True
    assert snap.is_stale(now=103.0, max_age_s=5.0) is False


def test_mock_feed_known_symbol():
    feed = MockL2Feed()
    snap = feed.get_book("BTCUSDT")
    assert snap.source == "mock"
    assert snap.bids and snap.asks
    assert snap.bids[0].price < snap.asks[0].price
    assert build_index_price(snap) > 0


# --------------------------------------------------------------------------
# fallback pricer unit tests
# --------------------------------------------------------------------------


def test_bs_price_matches_known_atm_call():
    # ATM, r=0: call ~ S * (2*Phi(sigma*sqrt(T)/2) - 1)
    val = bs_price(S=100, K=100, T=1.0, r=0.0, sigma=0.2, kind="call")
    assert 7.0 < val < 8.5  # ~7.97


def test_plain_mc_converges_to_bs():
    S, K, T, r, sigma = 100, 100, 0.5, 0.0, 0.4
    bs = bs_price(S, K, T, r, sigma, "call")
    mc = price_plain_mc(S, K, T, r, sigma, "call", n=200_000, seed=1)
    assert abs(mc.price - bs) < 5 * mc.std_error + 0.2
    assert mc.n_paths == 200_000


def test_is_reduces_variance_for_otm():
    # Deep OTM call: IS should cut the standard error vs plain MC.
    S, K, T, r, sigma = 100, 200, 0.5, 0.0, 0.4
    n = 50_000
    plain = price_plain_mc(S, K, T, r, sigma, "call", n=n, seed=7)
    is_res = price_is(S, K, T, r, sigma, "call", n=n, seed=7, method="drift_tilt")
    bs = bs_price(S, K, T, r, sigma, "call")
    # both unbiased & close to BS
    assert abs(is_res.price - bs) < 0.5
    # variance reduction
    assert is_res.std_error < plain.std_error
    assert 0 < is_res.ess <= n


# --------------------------------------------------------------------------
# HTTP endpoint tests
# --------------------------------------------------------------------------


def test_health(server):
    status, body = _get(server, "/health")
    assert status == 200
    assert body == {"status": "ok"}


def test_book(server):
    status, body = _get(server, "/book?symbol=BTCUSDT")
    assert status == 200
    assert body["symbol"] == "BTCUSDT"
    assert body["source"] == "mock"
    assert body["bids"] and body["asks"]
    assert body["mid"] is not None
    assert body["microprice"] is not None


def test_feed_status(server):
    status, body = _get(server, "/feed/status?symbol=BTCUSDT")
    assert status == 200
    assert set(body) == {"source", "age_s", "stale", "datafeed_drop", "index_price"}
    assert body["source"] == "mock"
    assert body["age_s"] >= 0
    assert body["stale"] is False
    # mock source counts as a datafeed drop signal
    assert body["datafeed_drop"] is True
    assert body["index_price"] > 0


def test_price_call(server):
    payload = {"symbol": "BTCUSDT", "kind": "call", "K": 120000, "T": 0.1,
               "r": 0.0, "method": "drift_tilt", "n_paths": 20000}
    status, body = _post(server, "/price", payload)
    assert status == 200
    for key in ("price", "std_error", "ess", "n_paths", "method"):
        assert key in body
    assert body["price"] >= 0
    assert body["std_error"] >= 0
    assert body["ess"] > 0
    assert body["method"] == "drift_tilt"
    assert body["spot"] > 0
    assert body["sigma"] == pytest.approx(0.8)


def test_price_default_sigma_and_method(server):
    payload = {"symbol": "BTCUSDT", "kind": "put", "K": 90000, "T": 0.2, "r": 0.0,
               "n_paths": 10000}
    status, body = _post(server, "/price", payload)
    assert status == 200
    assert body["method"] == "drift_tilt"
    assert body["sigma"] == pytest.approx(0.8)


def test_price_plain_mc_method(server):
    payload = {"symbol": "ETHUSDT", "kind": "call", "K": 3500, "T": 0.25, "r": 0.0,
               "method": "plain_mc", "n_paths": 10000, "sigma": 0.6}
    status, body = _post(server, "/price", payload)
    assert status == 200
    assert body["method"] == "plain_mc"
    assert body["sigma"] == pytest.approx(0.6)


def test_price_bad_json(server):
    req = urllib.request.Request(server + "/price", data=b"{not json",
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400
    detail = json.loads(exc.value.read().decode())
    assert detail["error"] == "invalid JSON"


def test_price_missing_field(server):
    payload = {"symbol": "BTCUSDT", "kind": "call", "T": 0.1, "r": 0.0}  # no K
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/price", payload)
    assert exc.value.code == 400
    detail = json.loads(exc.value.read().decode())
    assert detail["error"] == "bad request"


def test_price_bad_kind(server):
    payload = {"symbol": "BTCUSDT", "kind": "straddle", "K": 1, "T": 0.1, "r": 0.0}
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/price", payload)
    assert exc.value.code == 400


def test_unknown_route_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/nope")
    assert exc.value.code == 404


# --------------------------------------------------------------------------
# service-level tests (no socket)
# --------------------------------------------------------------------------


def test_service_price_direct():
    svc = PricingService(feed=MockL2Feed())
    out = svc.price({"symbol": "BTCUSDT", "kind": "call", "K": 100000, "T": 0.1,
                     "r": 0.0, "n_paths": 5000})
    assert out["price"] >= 0
    assert "bs_reference" in out
    assert out["spot"] > 0


def test_service_feed_status_stale_flag():
    svc = PricingService(feed=MockL2Feed(), stale_max_age_s=0.0)
    # now far in the future forces staleness
    st = svc.feed_status("BTCUSDT", now=2_000_000_000.0)
    assert st["stale"] is True
    assert st["datafeed_drop"] is True
