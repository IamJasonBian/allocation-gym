# eBay Auction Signals & Winning Strategy

> Reference doc for `allocation_gym` — covers signal taxonomy, scraping cadence, bid timing, and execution architecture for auction-mode listings.

---

## Background: How eBay Auctions Work

eBay uses a **proxy bidding system** (Vickrey-style). You submit a maximum bid; eBay auto-increments against competing bids up to your ceiling, revealing only the current minimum required to be winning. 

Key implication: **the displayed price is not the clearing price.** You are bidding against hidden maximums, not visible numbers. A listing showing $250 with 12 bids could have a proxy bidder sitting on a $600 max — or it could be 12 incremental bids from 3 real people with no one above $260.

This changes what signals are worth scraping.

---

## Signal Taxonomy

### 1. Listing-Level Signals
*Determines whether an auction is worth targeting.*

| Signal | What it tells you | Source |
|---|---|---|
| `time_left` | Urgency, snipe window scheduling | Countdown API `auction.time_left` |
| `bid_count` | Competition intensity | Countdown API `auction.bids` |
| `start_price` vs `current_price` | Spread from floor to now | Delta between fields |
| `buy_it_now` present on auction | Seller's own valuation ceiling | `buy_it_now: true` |
| `seller_info.positive_feedback_percent` | Delivery/legitimacy risk | `seller_info` block |
| `seller_info.review_count` | Seller volume/reliability | `seller_info` block |
| `condition` | Risk-adjusted value | `condition` field |
| `free_returns` | Downside protection | `free_returns` bool |
| `best_offer` | Negotiation floor signal | `best_offer` bool |
| `item_location` | Shipping time/cost risk | `item_location` field |

### 2. Market Signals
*Determines what fair value is — i.e., what your max bid ceiling should be.*

**Completed listings** are the only real ground truth on what items actually sell for. Not asking prices. Not BIN prices. Sold prices.

```
# eBay "sold" filter URL params
LH_Complete=1&LH_Sold=1&_sop=3   # completed, sold, most recent first
```

Compare against:
- **Active BIN floor** — the lowest current Buy It Now for the same item. Your max bid should not exceed `BIN_floor - shipping_cost`, otherwise just buy the BIN.
- **30-day sold median** — rolling window of completed auction final prices gives you a realistic ceiling.
- **Bid velocity** — bids per hour relative to time remaining. See section below.

### 3. Timing Signals
*The highest-leverage signal category on eBay. Most edge lives here.*

**Sniping window:** Optimal bid placement is **T-8s to T-15s** before auction close.
- Too early (>30s): enough time for counter-bids, drives price up, reveals your interest
- Too late (<5s): network jitter risk, bid may not register
- Sweet spot (8–15s): too late for human counter-bid, safe margin against latency

**End-time patterns:**
- Listings default to ending at the same time-of-day they were created
- Items ending between **midnight–6am ET** have fewer active snipers (thinner competition)
- Items ending on **weekday afternoons** have peak competition

**Bid increment table** — eBay uses fixed increments by price band. Knowing this lets you bid strategically:

| Current price | Increment |
|---|---|
| $0.01 – $0.99 | $0.05 |
| $1.00 – $4.99 | $0.25 |
| $5.00 – $24.99 | $0.50 |
| $25.00 – $99.99 | $1.00 |
| $100.00 – $249.99 | $2.50 |
| $250.00 – $499.99 | $5.00 |
| $500.00 – $999.99 | $10.00 |
| $1,000.00+ | $25.00 |

**Bid above round numbers.** Most bidders anchor on round numbers ($500, $250). Bidding $501 or $276 beats an equal max (earlier bid wins ties) and beats anyone who anchored just below the increment boundary.

---

## Bid Velocity Analysis

Velocity = `bid_count_delta / time_delta`

| Pattern | Interpretation |
|---|---|
| Many bids, price rising slowly | One strong proxy bidder with a high max is absorbing incremental bids |
| Many bids, price rising fast | Multiple active bidders competing — approaching real clearing price |
| Few bids, low price, <5min left | Underattended auction — high value opportunity |
| Bid count spike in last hour | Snipers activating — expect final price jump |
| Zero bids, any time remaining | Either mispriced low (value) or something wrong with listing |

Track `(bid_count, current_price, timestamp)` tuples for each watched item. The derivative matters more than the absolute value.

---

## Polling Cadence

Cadence should adapt to auction phase, not be constant. Constant polling wastes API credits and misses the critical window.

```
Phase                   Trigger                  Poll interval
──────────────────────────────────────────────────────────────
Discovery               New search results       Every 5–15 min
Watchlist (cold)        time_left > 1 hr         Every 60 sec
Watchlist (warm)        time_left 10–60 min      Every 15 sec
Active monitoring       time_left < 10 min       Every 5 sec
Snipe execution         time_left = T-10s        One-shot scheduled job
```

The snipe execution phase is **not a poller** — it's a pre-scheduled one-shot job. Calculate `end_time - 10s` when you add an item to the watchlist, schedule an APScheduler/Celery beat job to fire at that exact timestamp.

```python
# Pseudocode for snipe scheduler
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def schedule_snipe(item_id, end_time, max_bid):
    snipe_time = end_time - timedelta(seconds=10)
    scheduler.add_job(
        func=execute_bid,
        trigger='date',
        run_date=snipe_time,
        args=[item_id, max_bid],
        id=f'snipe_{item_id}',
        replace_existing=True
    )

def execute_bid(item_id, max_bid):
    # Re-validate price hasn't exceeded max before firing
    current = get_current_price(item_id)
    if current < max_bid:
        place_offer(item_id, max_bid)  # eBay Trading API PlaceOffer
```

---

## Data to Scrape Per Auction

Minimum required fields to implement the above:

```python
{
    "item_id": str,
    "title": str,
    "condition": str,
    "current_price": float,
    "start_price": float,          # for spread analysis
    "bid_count": int,
    "end_time": datetime,          # absolute UTC — critical for snipe scheduler
    "buy_it_now_price": float,     # None if not present
    "seller_feedback_pct": float,
    "seller_review_count": int,
    "free_returns": bool,
    "item_location": str,
    "listing_url": str,
    
    # Derived / tracked over time
    "bid_velocity": float,         # bids/hour rolling average
    "price_history": list,         # [(timestamp, price, bid_count), ...]
    "market_comps": {
        "sold_median_30d": float,
        "active_bin_floor": float,
    }
}
```

**Note on `end_time`:** The Countdown API does not return an absolute end timestamp — only `time_left` as a human-readable string (e.g. "2h 15m"). You need to either:
- Parse `time_left` at scrape time and compute `now + time_left` (introduces drift on re-polls)  
- Hit the eBay item page directly or use the **Shopping API `GetSingleItem`** call which returns `EndTime` as a proper UTC timestamp

`GetSingleItem` is the cleaner path and is part of eBay's legacy Shopping API (not yet decommissioned as of 2026).

---

## What Actually Wins Auctions

In order of impact:

1. **Bid timing (sniping)** — single biggest lever. 8–15 second window. Needs pre-scheduled execution, not human reaction time.

2. **Max bid calibration** — set ceiling at `sold_median_30d × risk_factor`, where risk_factor accounts for condition delta and seller trust score. Use odd amounts to beat anchored bidders.

3. **Competition read** — interpret bid velocity pattern before committing. High proxy bidder signature (many bids, flat price) means someone is sitting on a high max — either beat it or skip.

4. **Timing of auction end** — prefer off-peak end times (late night, early morning ET). Less sniper competition = lower clearing price.

5. **Seller quality filter** — hard filter on `feedback_pct < 98%` or `review_count < 100` unless discount justifies counterparty risk.

---

## Execution Layer Options

| Approach | Latency | Auth complexity | Notes |
|---|---|---|---|
| eBay Trading API `PlaceOffer` | Low (~500ms) | OAuth user token required | Cleanest path — no browser needed |
| Playwright + session cookies | Medium (2–5s) | Pre-auth session | Works without API approval |
| Selenium | High (3–8s) | Pre-auth session | Too slow for snipe window |

**Trading API** is the right target for production. `PlaceOffer` requires a user OAuth token (obtained via eBay's Auth & Auth flow), but once you have it, bid placement is a single API call with predictable sub-second latency.

---

## Module Placement in allocation_gym

```
allocation_gym/
├── scrapers/
│   ├── countdown_client.py     # Countdown API wrapper (search, item detail)
│   ├── ebay_shopping_api.py    # GetSingleItem for end_time, start_price
│   └── completed_listings.py   # Scrape sold comps for market pricing
├── pricing/
│   └── auction_valuation.py    # sold_median, BIN floor, max_bid calc
├── alerts/
│   └── snipe_scheduler.py      # APScheduler jobs, T-10s execution
├── data/
│   └── watchlist.py            # Watchlist state, bid_history, velocity
└── AUCTION_SIGNALS.md          # ← this doc
```

---

## Open Questions / TODO

- [ ] Confirm `GetSingleItem` still returns `EndTime` reliably (Shopping API decommission timeline unclear)
- [ ] Evaluate Trading API `PlaceOffer` OAuth flow — can we get user token without full app store listing?
- [ ] Build bid velocity tracker in `data/watchlist.py`
- [ ] Define `risk_factor` table for condition × seller_feedback tiers
- [ ] Location filtering: `_stpos` + `_sadis` eBay URL params via Countdown API `url` passthrough param
