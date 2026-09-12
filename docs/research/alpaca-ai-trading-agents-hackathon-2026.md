# Alpaca AI Trading Agents Hackathon — field notes (Aug 28 – Sep 4, 2026)

Research summary from the [lablab.ai × Alpaca hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon).
Judging was still in progress when these notes were written; official cash winners were not yet published.

**Relevance to allocation-gym:** the event is a live catalog of how teams wire **Alpaca
Trading API + MCP + CLI** into autonomous loops with **deterministic risk layers** —
the same separation we want between signal generation (strategies / Kelly sizers) and
broker execution (runner, forward test, future agent adapters).

---

## Executive summary

- **3,602** participants, **1,268** teams, **428** submissions (~34% ship rate).
- **352/428** (82%) stayed on the **Options Alpha Agents** track; options were mandatory.
- **Judging:** paper **P&L + creativity/engagement** on a dedicated **$100k** competition account.
- **Dominant architecture:** `LLM proposes → deterministic gates → CLI/API executes → broker reconcile → audit log`.
- **Product thesis across the field:** auditability and risk engineering beat raw return claims on one week of paper trading.

---

## Submission numbers

| Metric | Value |
|--------|-------|
| Registered participants | 3,602 |
| Teams formed | 1,268 (avg ~1.4 members) |
| Final submissions | 428 |
| Options Alpha Agents track | 352 |
| Teams reporting Alpaca stack usage | 443 |
| Community hearts (total) | 824 |

**Vote concentration:** top two projects (**Alpha Hunter**, **TradePilot AI**) held ~34% of all hearts; steep drop after #3 (**VegaGuard**, 37 hearts).

**Tooling (teams self-reporting):** Alpaca 443, Claude Code 149, Featherless 89, Streamlit 43, LangChain 22.

---

## Reported P&L (where teams published numbers)

One week of paper trading is not edge — several teams said so explicitly. Published snapshots:

| Project | Result | Notes |
|---------|--------|-------|
| Miramar Labs | +3.7% ($103,709) | Highest published return at cutoff |
| Risk Gate | +1.82% ($101,823) | Admits signal beat buy-and-hold on 0/7 names OOS |
| Glass Box | +0.58% ($100,583) | Mostly flat; one overnight gap drove gains |
| Opticycle | +$55.67 closed spread | Emphasizes payload-hash verification |
| VegaGuard | +$2 gross (1 spread) | Net P&L unknown (fees not reported) |
| Tape | −$54,152 (first account) | 0-DTE held into expiry; failure archived, book restarted |
| CrossSignal | No P&L claim | Abstained — liquidity/confidence below threshold |
| Market Jury | 0 orders | Latest decision WATCH; capital never released |

---

## Risk architecture (what the field converged on)

### Universal pattern

```
Observe → LLM structured proposal → N deterministic gates → Execute → Reconcile → Log
```

~90%+ of serious submissions **never** gave the LLM direct order-placement tools.

### Common gate types

| Gate | Typical threshold | Purpose |
|------|-------------------|---------|
| Per-trade max loss | ~2% equity | Cap single-structure worst case |
| Portfolio open risk | 8–20% NAV | Aggregate exposure ceiling |
| Daily loss halt | 2–6% | Circuit breaker |
| Kill switch / HALT | Manual + auto | Hard stop |
| Defined-risk only | No naked shorts | Structural cap on tail risk |
| Max open positions | ~5 | Concentration |
| DTE / delta bands | Strategy-specific | Options geometry |
| Liquidity / spread | Bid-ask, quote freshness | Avoid bad fills |
| Broker reconciliation | Every cycle | Journal vs `get_all_positions` |
| Idempotency | Pre-allocated `client_order_id` | Prevent duplicate submits |
| MCP read-only for LLM | Strip order tools from allowlist | Separation of powers |
| Execution via CLI/worker | Not in agent loop | Deterministic submit path |

### Named gate counts (from submission write-ups)

| Project | Gates | Standout |
|---------|-------|----------|
| TradeProof | 23 (+ 253 tests) | NL → typed policy compiler, SHA-256 audit chain |
| Volition | 16 | Monte Carlo stress + decision passport |
| AlphaSwarm | 18 | 1k-path VaR, ATR trailing stops |
| Tape | 13 | 22/33 runs blocked; 15k MCP calls journaled |
| FLINCH | 11 | Code proposes; LLM clamped to VETO / SIZE_DOWN / APPROVE |
| Miramar Labs | 8 | Synthetic SL/TP (no Alpaca option brackets) |
| SENTINEL | 6 | “CRO” metaphor, 108 tests |

### Control-flow variants

1. **Standard:** model proposes, code validates (most teams).
2. **FLINCH inversion:** deterministic engine proposes; LLM only approves/vetoes.
3. **Market Jury:** zero trades by design — analysis ≠ capital ≠ execution authority.

---

## Key ideas (recurring themes)

### 1. Auditability over alpha

Winning narrative: *verify every decision*, not *we beat the market*.

- Hash-chained receipts (TradeProof, Opticycle)
- Append-only journals with gate verdicts (Glass Box, Tape)
- Public “reconcile wall” vs broker state (Tape, Volition)

### 2. Multi-agent debate

| Project | Structure |
|---------|-----------|
| TradePilot AI | Market + news agents → decision agent |
| Options Sentinel | Bull vs bear → decision |
| Alpha Hunter (#1 community votes) | Discover → adversarial break → score → allocate |
| Miramar Labs | Analyst → Dealer (MCP chain read) → Floor Broker (execute) |
| CrossSignal | Six cross-asset lenses + adversarial confidence penalty |

### 3. Options-specific intelligence

Hard problem: picking expiration, strike, and structure — not direction.

- LLM reads chains via **Alpaca MCP** (Greeks, quotes, expirations)
- **Reconciliation overwrites** model numbers with broker truth (Miramar)
- **Custom Greeks for 0DTE** where Alpaca returns none (FLINCH)
- Structures: debit spreads, iron condors, credit verticals — almost always defined-risk

### 4. Abstention as feature

Strong agents refuse more than they trade: NO_TRADE, WATCH, gate-blocked cycles, position caps.

### 5. Infra separation (Alpaca stack)

| Layer | Tool |
|-------|------|
| LLM market access | MCP server (read-only allowlist) |
| Order submission | CLI or REST worker |
| Scheduling | Cron / k8s / 15-min loops |
| Observability | Streamlit, Next.js, SSE dashboards |

---

## Three archetypes

```
A. Trading floor   — multi-agent + MCP chain reader + broker executor (Miramar, TradePilot)
B. Proof machine   — gates + hash audit + abstention (TradeProof, Tape, Glass Box)
C. Research scientist — discover → adversarial test → allocate (Alpha Hunter, CrossSignal)
```

---

## Community top 10 (by hearts, pre-official judging)

1. Alpha Hunter — Autonomous AI Trading Scientist (146)
2. TradePilot AI (135)
3. VegaGuard: Auditable AI Options Agent (37)
4. QASIX-Alpaca AI Trading Agent (37)
5. AlphaPilot AI (27)
6. SentryTheta AI (24)
7. Synthetix Alpha (22)
8. Elite-Bot: Multi-Asset AI Agent Trading Hub (20)
9. trdrbot — self-improving options agent (15)
10. Alpacca trading bot (15)

---

## Implications for allocation-gym

1. **Keep the split we already have:** strategies/sizers propose weights; runner and
   credentials own the data/execution boundary — mirror “LLM proposes, code disposes.”
2. **If we add an agent layer:** MCP read-only for research; CLI/API worker for orders;
   Kelly and variance metrics stay in deterministic code, not prompts.
3. **Options path:** defined-risk geometry checks before submit; reconcile journal vs
   Alpaca positions each cycle; idempotent client order IDs.
4. **Evaluation:** one-week paper P&L is insufficient; prefer audit trails, gate
   block rates, and forward-test / Monte Carlo (already in `allocation_gym.simulation`).

---

## References

- [Hackathon recap page](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon)
- [Live dashboard / final stats](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon/live)
- [Alpaca MCP Server](https://github.com/alpacahq/alpaca-mcp-server)
- [Alpaca agentic integrations](https://github.com/alpacahq/agentic)
- Notable submissions: [Miramar Labs](https://lablab.ai/submissions/w6elie9s0fcfo1p7iq0pd68l), [TradeProof](https://lablab.ai/submissions/pn00mvb44g5pd60w5q1yt0z8), [TradePilot](https://lablab.ai/submissions/m49lau2dgw8hi7zsqymz176z), [Alpha Hunter](https://lablab.ai/submissions/dfdw7wkv91cnyhglp84gx6vm)

*Compiled Sep 12, 2026 from lablab public submission pages and live dashboard.*
