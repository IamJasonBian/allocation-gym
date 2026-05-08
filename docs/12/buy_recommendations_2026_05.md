---
title: "Buy Recommendations: Trailing 3M + Forward 3M (May 2026)"
subtitle: "Allocation Engine - Current-Market Overlay on the Variance-Kelly Framework"
date: May 7, 2026
author: "Allocation Engine Research"
---

# Buy Recommendations: Trailing 3M + Forward 3M (May 2026)

**Subtitle:** Allocation Engine - Current-Market Overlay on the Variance-Kelly Framework
**Date:** May 7, 2026
**Author:** Allocation Engine Research

---

## 1. Executive Summary

The variance-Kelly codebase's primary expression - long BTC paired with long IWN puts - has a current-market analog that adds an energy / commodity call leg and tightens the put hedge into a dead-cheap volatility regime. With **VIX at 17.39** (down 39% from the March panic at 31.05) and **BTC at $81,022** (down 14% from the codebase's reference $94K print), options are inexpensive on both sides of the book. The recommendation set below targets a total option theta budget of approximately **1% of NAV**, with the highest-conviction names being **IWN puts** (the codebase's already-validated tail hedge, sized for a fresh small-cap debt-wall regime) and **XLE calls** (the most glaring blind spot in the codebase's universe model). The IBIT calls replace the codebase's GBTC Mini Trust spot scalping with defined-risk exposure into the June FOMC and the Powell-replacement catalyst.

| Trade | Direction | Tenor | Conviction | Sizing %NAV |
|---|---|---|---|---|
| IWN 3M ATM puts | LONG VOL (Put) | 3M | High | 0.25% |
| XLY 3M ATM puts | LONG VOL (Put) | 3M | Med-High | 0.25% |
| XLE 3M ATM calls | LONG VOL (Call) | 3M | High | 0.40% |
| IBIT 3M ATM calls | LONG VOL (Call) | 3M | High | 0.30% |
| EWZ 2-3M slightly-OTM calls | LONG VOL (Call) | 2-3M | Med | 0.20% |
| DBC 3M ATM calls | LONG VOL (Call) | 3M | Med | 0.05% |
| GLD 3M ATM calls | LONG VOL (Call) | 3M | Med | 0.05% |
| IBIT spot | LONG | open | Med (dip-buy) | 30-45% w/ hedge |

---

## 2. Codebase audit recap

`allocation-gym` is a Backtrader-driven variance-Kelly backtesting framework. The signal stack centers on **Yang-Zhang variance**, the **Variance Ratio (VR)**, the **Efficiency Ratio (ER)**, a regime classifier (CHOP / TREND / MIXED), vol-of-vol, and downside semivariance. Four strategies sit on top of those signals:

- **momentum** - enter when VR >= 1.1, ER >= 0.4, and price > SMA50; exit when price < SMA50 or regime flips to CHOP
- **mean_reversion** - enter when VR < 0.9 and RSI < 30; exit when RSI > 70
- **variance_kelly** - inverse-covariance Kelly rebalance, with drift > 5% triggering a re-weight
- **momentum_dca** - Ornstein-Uhlenbeck band selection on entries, exiting at the OU equilibrium epsilon\*

Sizing is governed by a **quarter-Kelly sizer** (fraction = 0.25), with stop distances scaled to downside-semivariance rather than total variance.

Realized YTD P&L between **Feb 21 and Mar 25, 2026** came in at **+$19,965** total: **$15,770** from BTC bracket trading on the GBTC Mini Trust plus **$4,195** from options round-trips. Specific options trades worth flagging for sizing reference:

- IWN $262P 3/11 expiry: **+57%**
- IWN $247P 3/13 expiry: **+103%**
- CRWD $390P 3/06 expiry: **+58%**
- CRWD $367.5P 3/20 expiry: **-15%**

The structural pair finding is the load-bearing result: **BTC <-> IWN 30-day correlation +0.61**, with **+0.28 residual** after VIX is partialled out (so VIX explains roughly 48% of the joint move). Critically, **IWN 3M ATM puts cost 4.4% of spot vs BTC puts at 15.4%** - **3.5x cheaper** for equivalent BTC tail-protection notional. The codebase also publishes two ranking tables: a top-10 long thesis screen of negative-BTC-correlation diversifiers and a top-10 short thesis screen of positive-BTC-correlation correlated-drawdown plays (reproduced verbatim in Section 7).

---

## 3. Current market state (May 7, 2026)

| Indicator | Value | Move since codebase analysis |
|---|---|---|
| BTC/USD | **$81,022** | -14% from ~$94K |
| VIX | **17.39** | -39% from March peak 31.05 |
| EWT | $94.41 | **+33% in 5 weeks** ($70.77 on Apr 2) |
| EWZ | ~$39 | +60% trailing 1Y |
| S&P 500 (Q1) | -4.3% | growth selloff |
| Nasdaq 100 (Q1) | -5.8% | tech leadership broken |
| XLE Energy (Q1) | **+37.9%** | oil +84%, regime change |
| DBC Commodities (Q1) | **+29.5%** | rotation winner |
| XLK / XLY / XLF (Q1) | each **<-7%** | rotation losers |
| Russell 2000 (IWM) | broke Jan ATH | +8.6% in 2 weeks late Q1 |

The single most important fact for sizing: **VIX has collapsed from 31 to 17 in one month**, which means options on both sides of the book are cheap, and the entire put-and-call program below is being entered at depressed implied vol. The second most important fact is that the codebase has **no energy / commodity exposure model** - the entire Q1 rotation winner cohort (XLE, DBC, materials) is missing from the universe, which is the visible blind spot we are correcting via overlay.

---

## 4. Trailing 3-month review (Feb-May 2026)

- **Q1 2026 closed S&P -4.3%, Nasdaq -5.8%** - growth led the selloff
- **Sector winners:** XLE +37.9% (oil +84%), DBC +29.5%, materials
- **Sector losers:** XLK, XLY, XLF all down >7%
- **Earnings growth was a divergence:** Comm Services +53.2%, Tech +50.0%, Cons Disc +39.0% - strong earnings paired with weak prices, the classic multiple-compression signature
- **VIX path:** 17 -> 31 (March panic) -> 17 (May complacency); a one-month 39% volatility decline coming off the spike
- **Russell 2000** broke its January ATH after rallying 8.6% in two weeks late Q1 - the "great rotation" trade
- **BTC** drew down ~$94K -> $81K, -14%
- The codebase's IWN puts trades **closed +57% and +103%** during the March vol spike - this is exactly the codebase playbook executing in production

---

## 5. Forward 3-month outlook (May-Aug 2026)

**Catalysts:**

- **Powell's Fed Chair tenure ends May 2026** - the replacement is widely expected to lean dovish
- **Clarity Act** (US crypto regulatory framework) potentially comes to vote
- Three Fed cuts on consensus for 2026 but timing is uncertain - the **June FOMC is THE marker**
- **$368B small-cap maturity wall in 2026**; Russell 2000 carries **32% floating-rate debt vs 6% for the S&P 500**
- Geopolitical oil premium not pricing out

**Consensus price paths:**

- **BTC:** June high $83.9K / low $70.3K, fading to a $71-78K range Jul-Aug
- **IWM:** extended at ATH; debt rollover at ~6.5% rates is a structural drag, not a transient one
- **Energy:** momentum intact, no clean reversal signal yet
- **EM equities (EWT, EWZ):** positive but **EWT is parabolic** and unsuitable as a fresh entry

---

## 6. Recommendations

### 6.1 Call book (forward upside, cheap IV)

**XLE 3M ATM calls - 0.40% NAV.** This is the codebase blind spot. Q1 2026 closed +37.9% with WTI up 84% YTD and the geopolitical oil premium not pricing out. Energy is the only sector whose earnings, prices, and macro setup are all aligned. We size this the largest because it has the cleanest single-name/single-thesis profile and because the codebase's universe model would have systematically underweighted it.

- Rotation winner with intact momentum
- Geopolitical bid persistent
- Implied vol cheap relative to realized
- *Prefer 3M expiry through June FOMC*

**IBIT 3M ATM calls - 0.30% NAV.** This replaces the codebase's GBTC Mini Trust spot scalping with defined-risk into the Powell-replacement Fed catalyst. Consensus targets are **$83-85K June high**, with downside risk to ~$70K well-defined.

- Defined risk vs uncapped spot
- IV cheap; cheaper carry than spot at low IV
- Catalyst-rich tenor (June FOMC, Clarity Act)

**EWZ 2-3M slightly-OTM calls - 0.20% NAV.** Brazil tracks Petrobras / Vale / Itau, has rallied +60% trailing 1Y, and is materially less extended than EWT. The codebase's negative-BTC-correlation ranking placed EWZ at #2 in the long-thesis screen, so this is a codebase-native call.

**DBC 3M ATM calls - 0.05% NAV.** Cleaner momentum than picking single ag commodities. Small token sizing because of the diversified composition; this is a regime-confirmation trade more than a thesis trade.

**GLD 3M ATM calls - 0.05% NAV.** Asymmetric upside on geopolitical tail risk at VIX 17. Token size, but a cheap option on a multi-vector tail.

### 6.2 Put book (forward downside, structural)

**IWN 3M ATM puts - 0.25% NAV.** This is the codebase's primary BTC tail-hedge expression and the current setup is materially **better than the codebase's original entry**: small caps are at all-time highs, the **$368B debt wall** sits at 6.5% rollover rates, Fed cuts are being delayed, and IV is at a multi-month low. Reference round-trips: **+57% and +103%** on the Mar 11 and Mar 13 expiries during the Feb -> Mar vol spike.

**XLY 3M puts - 0.25% NAV.** Consumer discretionary closed Q1 down -7%; Fed-cut delays squeeze cyclicals; the codebase's long-thesis ranking placed XLY at #4 (negative BTC correlation), which is the same structural feature that makes it a clean put candidate when the underlying weakens.

### 6.3 Pair trade - modernized codebase pair

The codebase's BTC-long XOR IWN-puts pair has two natural successors at the current vol regime:

1. **Long XLE 3M calls + Long IWN 3M puts** - both legs monetize a stagflation / oil-spike / small-cap-debt-wall scenario. Cheap on both legs given VIX 17.
2. **Long IBIT 3M calls + Long IWN 3M puts** - preserves the structural BTC-vs-small-cap pair the codebase identified, but expresses BTC long with **defined risk** into the Fed catalyst rather than uncapped spot.

### 6.4 Avoid (or trim if held)

- **EWT calls** - parabolic, +33% in 5 weeks, mean-reversion candidate
- **XLK / QQQ calls** - losing sector with no near-term catalyst
- **RBC** (+120% 1Y from codebase short-thesis list) - overbought; trim on strength
- **BTC calls past 3M tenor** - Jul-Aug fades on consensus

---

## 7. Universe (codebase reference tables)

### 7.1 Top 10 BTC-LONG thesis (negative BTC correlation = diversifier candidates)

| Rank | Ticker | Spot | 30d Vol | 3M Corr | 1Y Return | Score |
|---|---|---|---|---|---|---|
| 1 | WEAT | $21.99 | 23.0% | -0.333 | -16.7% | 0.523 |
| 2 | EWZ  | $38.64 | 27.1% | -0.253 | +33.5% | 0.495 |
| 3 | EWT  | $75.07 | 24.0% | -0.257 | +71.5% | 0.479 |
| 4 | XLY  | $115.42 | 16.4% | -0.269 | +28.6% | 0.479 |
| 5 | EEM  | $61.50 | 18.5% | -0.212 | +59.7% | 0.473 |
| 6 | MOO  | $85.20 | 14.2% | -0.198 | +25.4% | 0.469 |
| 7 | REXR | $37.65 | 28.3% | -0.233 | -22.6% | 0.469 |
| 8 | HYG  | $80.28 |  2.6% | -0.201 | +16.4% | 0.455 |
| 9 | VWO  | $57.24 | 15.1% | -0.165 | +46.8% | 0.454 |
| 10 | SOYB | $23.69 |  9.3% | -0.167 |  -2.7% | 0.450 |

### 7.2 Top 10 BTC-SHORT thesis (positive BTC correlation = correlated drawdown plays)

| Rank | Ticker | Spot | 30d Vol | 3M Corr | 1Y Return | Score |
|---|---|---|---|---|---|---|
| 1 | GMED | $93.80 | 23.7% | +0.154 | +69.9% | 0.436 |
| 2 | RJF  | $154.42 | 35.9% | +0.158 | +31.3% | 0.434 |
| 3 | MIDD | $164.99 | 29.4% | +0.139 |  +7.1% | 0.424 |
| 4 | PODD | $245.45 | 32.7% | +0.099 | +46.3% | 0.421 |
| 5 | CFLT | $30.70 |  3.3% | +0.128 | -11.1% | 0.416 |
| 6 | RBC  | $584.89 | 18.7% | +0.042 | +120.1% | 0.415 |
| 7 | URBN | $65.69 | 35.9% | +0.089 | +52.4% | 0.413 |
| 8 | WES  | $42.09 | 22.9% | +0.136 | +49.3% | 0.412 |
| 9 | IWO  | $338.51 | 21.3% | +0.049 | +27.7% | 0.404 |
| 10 | GGG  | $94.82 | 18.8% | +0.040 |  +6.3% | 0.401 |

---

## 8. Caveats and known risks

- Codebase data is roughly **60 days stale**; correlations may have shifted in the post-VIX-collapse risk-on regime
- Codebase has **no energy / commodity exposure model** - XLE / DBC / oil are blind spots; treat those calls as a regime overlay, not a framework recommendation
- **EWT was the codebase's #3 hero negative-corr trade** - it is now too extended; do not conflate prior recommendation with current entry
- Calls + puts together at low IV is a long-gamma straddle book - it bleeds via theta if vol stays pinned at 17
- The **June FOMC + Powell-replacement announcement** is the single biggest risk-event in the next 3 months - time call expiries **through it, not into it**
- Variance-Kelly's quarter-Kelly was sized for trending regimes; sustained chop will degrade Sharpe

---

## 9. Sources

- Bitcoin price May 7, 2026 - Fortune (https://fortune.com/article/price-of-bitcoin-05-07-2026/)
- Q1 2026 Recap - Seeking Alpha (https://seekingalpha.com/article/4887991-q1-2026-recap)
- Russell 2000 surges 8.9% - FinancialContent (https://markets.financialcontent.com/stocks/article/marketminute-2026-3-11-the-great-rotation-russell-2000-surges-89-as-small-cap-value-dethrones-tech-giants-in-2026)
- Small caps under siege as yields surge - FinancialContent (https://markets.financialcontent.com/stocks/article/marketminute-2026-3-23-small-caps-under-siege-russell-2000-enters-correction-as-yields-surge)
- Bitcoin price predictions 2026 - Changelly (https://changelly.com/blog/bitcoin-price-prediction/)
- EWT Stock Analysis - stockanalysis.com (https://stockanalysis.com/etf/ewt/)
- VIX falls below 17 - 24/7 Wall St. (https://247wallst.com/investing/2026/05/01/the-cboe-vix-falls-to-16-level-as-risk-on-trade-returns-to-market/)
- EWZ commodity tailwinds - 24/7 Wall St. (https://247wallst.com/investing/2026/05/07/how-ewz-and-eww-investors-are-riding-commodity-and-supply-chain-tailwinds/)
