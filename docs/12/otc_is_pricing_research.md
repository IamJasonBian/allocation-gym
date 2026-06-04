# Importance-Sampling Pricing for Alt-Coin OTC Derivatives: A Research Survey

*Research writeup for the `otc_is_pricing` module. Pure survey of the "galaxy" of
importance-sampling (IS) approaches relevant to marking illiquid alt-coin OTC
derivatives under unreliable upstream datafeeds.*

---

## 1. The Problem

A derivatives desk that writes over-the-counter (OTC) contracts on alt-coins must
**mark its book continuously** — every position needs a fair value on demand for
P&L, margin, and risk. For liquid majors (BTC, ETH) this is nearly free: deep,
two-sided L2 order books and dense listed-option surfaces pin down the mark. For
alt-coins it is genuinely hard, for three compounding reasons.

**Thin, illiquid L2 books.** An alt-coin's central-limit order book is often a
handful of resting orders with wide spreads and shallow depth. The mid-price is
noisy, the microprice jumps on single cancellations, and there is frequently no
listed-option surface at all to imply a volatility from. The mark must be *modelled*
rather than *read off*.

**Unreliable upstream feeds.** The pricing pipeline depends on exchange websockets
and REST snapshots that drop, lag, or return stale depth. A **datafeed drop is an
acute risk** precisely because the desk cannot stop marking: a contract still needs
a number while the feed is dark. Marking off a stale book silently introduces a
bias toward the last-seen state; marking off nothing is not an option.

**Rare-event payoffs.** The interesting OTC structures — deep out-of-the-money
(OTM) European options, knock-in/knock-out barriers, and digital/binary payoffs —
have value concentrated in tail scenarios that a plain simulation almost never
visits. For a deep-OTM call struck at $K \gg S_0$, almost every simulated path
expires worthless; the few in-the-money paths carry the entire price, so the naive
Monte Carlo estimator has enormous relative variance and needs an impractical
number of paths to converge. This is the regime where **importance sampling** earns
its keep.

---

## 2. Importance Sampling Fundamentals

Importance sampling reduces variance by **sampling from a different distribution**
and correcting with a likelihood ratio. We want a price

$$ \mu = \mathbb{E}_p[h(X)] = \int h(x)\, f(x)\, dx, $$

where $f$ is the (risk-neutral) density of the path/terminal state $X$ and $h$ is
the discounted payoff. Pick a **proposal** density $g$ that is positive wherever
$h\cdot f \neq 0$. Then

$$ \mu = \int h(x)\,\frac{f(x)}{g(x)}\, g(x)\, dx
     = \mathbb{E}_q\!\left[ h(X)\,\frac{f(X)}{g(X)} \right]
     = \mathbb{E}_q\!\left[ h(X)\, w(X) \right], $$

with the **likelihood ratio** (Radon–Nikodym derivative) $w(x) = f(x)/g(x)$. The
estimator is $\hat\mu = \frac{1}{N}\sum_{i=1}^N h(X_i)\, w(X_i)$ with
$X_i \sim g$. It is unbiased for any valid $g$; the *art* is choosing $g$ to put
simulation mass in the **important region** where $|h|\cdot f$ is large.

**Change of measure.** When $X$ is a diffusion (e.g. geometric Brownian motion),
shifting the proposal is a Girsanov change of measure: tilt the Brownian drift and
the likelihood ratio becomes an exponential martingale. This is what every tilting
scheme below does concretely.

**Zero-variance optimal proposal.** The variance of $h\cdot w$ is minimized (to
zero, for a nonnegative payoff) by

$$ g^\*(x) \;\propto\; |h(x)|\, f(x). $$

This is not usable directly — its normalizing constant *is* the price $\mu$ we are
trying to compute — but it is the north star: a good proposal should look like
$f$ reweighted toward the payoff support.

**Effective sample size and degeneracy.** A proposal that is too aggressive
produces a few enormous weights and many near-zero ones. The diagnostic is the
**effective sample size**

$$ \mathrm{ESS} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} \in [1, N]. $$

$\mathrm{ESS} \approx N$ means weights are even and the estimator is efficient;
$\mathrm{ESS} \ll N$ signals **weight degeneracy** — one path dominates and the
variance is no better (often worse) than naive MC. ESS is the primary guardrail to
monitor when tuning a tilt.

**Self-normalized IS.** When $f$ or $g$ is known only up to a constant — exactly
the case for an **order-book-implied density**, which is a shape with no analytic
normalizer — use the self-normalized estimator

$$ \hat\mu_{\mathrm{SNIS}}
   = \frac{\sum_i h(X_i)\, w(X_i)}{\sum_i w(X_i)}. $$

This divides out the unknown constant. It is slightly biased ($O(1/N)$) but
consistent, and it is the workhorse for any proposal derived from empirical market
microstructure.

---

## 3. The Galaxy of Approaches

### 3.1 Plain Monte Carlo (baseline / control)

Sample $X_i \sim f$ directly and average $h(X_i)$. **Mechanism:** none — it is the
$g = f$, $w \equiv 1$ special case. **When it wins:** at-the-money or in-the-money
payoffs where most paths contribute, and as the *control* against which any IS
scheme must be validated (same $\mu$, smaller variance). The error decays as
$O(N^{-1/2})$ regardless of dimension. **Tradeoff:** for a deep-OTM or barrier
payoff with hit probability $p$, the relative standard error scales like
$\sqrt{(1-p)/(Np)}$, so achieving fixed relative accuracy needs $N \propto 1/p$
paths — catastrophic when $p \sim 10^{-4}$.

### 3.2 Drift-Tilt / Esscher Exponential Tilting (OTM European)

For a European call on GBM, $\log S_T \sim \mathcal{N}\big(\log S_0 + (r-\tfrac12\sigma^2)T,\ \sigma^2 T\big)$.
**Mechanism:** shift the terminal log-return mean by $\theta\,\sigma\sqrt{T}$ (an
Esscher / exponential tilt of the Gaussian, equivalently a Girsanov drift change)
so simulated paths land *near the strike* instead of near the forward. The natural
choice centers the proposal at $\log K$:

$$ \theta = \frac{\log(K/S_0) - (r - \tfrac12\sigma^2)T}{\sigma\sqrt{T}}. $$

The likelihood ratio for a Gaussian mean shift is $w = \exp\!\big(-\theta Z - \tfrac12\theta^2\big)$
with $Z$ the standard normal driving the path. **When it wins:** moderately to
deeply OTM European options — variance reductions of one to several orders of
magnitude are typical. **Tradeoff:** $\theta$ must be tuned (too large over-tilts
and degrades ESS); the closed-form $\theta$ above is exact only for log-normal
terminals and is an approximation under stochastic volatility — which is precisely
the gap the Heston-specific large-deviation schemes of arXiv:2511.19826 close.

### 3.3 Rare-Event / Barrier (Knock-In) IS

A knock-in option pays only if the path crosses a barrier $B$ — a path-dependent
rare event when $B$ is far from $S_0$. **Mechanism:** apply a Girsanov drift tilt
that pushes paths *toward the barrier*, making the crossing typical rather than
rare, and weight each path by the path-space likelihood ratio (an exponential of
the tilted increments). This is the financial-engineering face of **tail-probability
IS**: estimating $p_t = \mathbb{P}(X \ge t)$ for large $t$ by tilting the
distribution so that $t$ sits near the proposal's mean. **When it wins:** knock-in,
one-touch, and other barrier structures whose trigger probability is small.
**Tradeoff:** for two-sided or knock-out features a single static drift can
*increase* variance in some regions; state-dependent / adaptive tilts (tilt
strength as a function of distance-to-barrier and time-to-maturity) are needed, and
correctness of the discrete-monitoring likelihood ratio is fiddly.

### 3.4 Digital / Binary Option IS

A cash-or-nothing digital pays a fixed amount iff $S_T > K$ — the payoff is the pure
**indicator** $h(x) = \mathbf{1}\{x > K\}$. **Mechanism:** the same drift tilt as
§3.2, centering the proposal at $K$. **When it wins:** this is where IS delivers its
*maximal* benefit. Because the payoff is constant on its support, the optimal
proposal $g^\* \propto \mathbf{1}\{x>K\}\, f(x)$ is just $f$ truncated to the
in-the-money region; a well-centered tilt approximates it closely and the estimator
variance collapses. Equivalently, pricing a digital *is* estimating the tail
probability $p = \mathbb{P}(S_T > K)$, the canonical rare-event IS problem.
**Tradeoff:** sharp sensitivity near the strike (the price is essentially a CDF
evaluation) means the tilt center must track $K$ accurately; small mis-centering
shows up directly in ESS.

### 3.5 L2-Density / Order-Book-Implied Proposal (Self-Normalized IS)

Rather than tilt an assumed parametric $f$, **read a proposal shape off the live L2
book**. The resting depth on each side, smoothed and reflected through the
microprice, gives an empirical, unnormalized density $\tilde g(x)$ over near-term
terminal prices that already encodes where the market thinks liquidity (and thus
realized moves) will land. **Mechanism:** simulate from $\tilde g$ and price with
the **self-normalized** estimator of §2, since $\tilde g$ has no analytic
normalizer. The risk-neutral $f$ supplies the weights $w = f/\tilde g$. **When it
wins:** short-dated marks where the book genuinely carries information the
parametric model misses, and as a market-consistent *control variate*. **Tradeoff:**
a thin alt-coin book gives a noisy, possibly bimodal $\tilde g$; ESS must be watched
closely and the proposal floored/regularized so that $\tilde g > 0$ everywhere $f$
has support (otherwise the estimator is biased, not just noisy).

### 3.6 Feed-Drop Reconstruction

When the upstream L2 feed drops, the last good snapshot is **stale but
informative**. **Mechanism:** treat the stale L2 snapshot as the proposal $g$ (it
describes a plausible recent state of the world) and **tilt back toward the
risk-neutral $f$** via the likelihood ratio $w = f/g$, so the mark remains an
unbiased estimate of the *current* risk-neutral value even though the samples come
from a slightly out-of-date distribution. This mirrors how exchanges build a robust
**mark price** under bad data: take the **median** of an index-price leg, an
index-plus-basis leg, and the last traded price; **drop any source that is stale
beyond a threshold (~10 s)**; and **fall back to the last traded price when depth
is insufficient** to form the basis. Bybit's documented scheme does exactly this —
mark = median(index·funding-adjust, index + 2.5-min basis MA, last price), with
fallback to the platform's last traded price whenever the index is abnormal or the
basis cannot be computed. The IS analogue is principled: the median/fallback logic
chooses *which* proposal to trust, and the likelihood-ratio reweighting removes the
staleness bias from the chosen proposal. **Tradeoff:** as the snapshot ages, $g$
drifts from $f$, weights spread, and ESS decays — so the scheme self-reports its own
degradation, and beyond a staleness budget the desk should widen the mark's
uncertainty band rather than trust a single number.

---

## 4. System Design Note

These approaches map directly onto the implementation under
`allocation_gym/otc_is_pricing/`. A **feeds** layer pulls Binance L2 depth and
synthesizes a robust snapshot, with a **mock fallback** so the pricer never blocks
on a dropped websocket (the §3.6 stale-snapshot-as-proposal path). An **IS
sampler + pricer** implements the tilts of §3.2–§3.5: it builds a proposal
(parametric drift tilt or order-book-implied density), draws paths, computes
likelihood ratios and the self-normalized estimate, and reports **ESS** alongside
every price as a confidence diagnostic. A small **stdlib-only HTTP API** exposes
quote requests. Separately, an **audit script** in `docs/12/` runs the sampler
across payoff types and tilt settings and emits an **expected-vs-actual PDF**,
overlaying the IS proposal $g$, the target $f$, and realized weight/ESS statistics
so a reviewer can see the variance reduction (and any degeneracy) at a glance.

---

## 5. References

- [Importance sampling — Wikipedia](https://en.wikipedia.org/wiki/Importance_sampling) — estimator, likelihood ratio, optimal proposal $g^\* \propto |h|f$, ESS and weight degeneracy.
- Glasserman, *Monte Carlo Methods in Financial Engineering* (Springer, 2003), importance-sampling chapter — change of measure for diffusions, drift tilting for OTM options, and rare-event / barrier estimators.
- [Efficient Importance Sampling under the Heston Model: Short-Maturity and Deep Out-of-the-Money Options — arXiv:2511.19826](https://arxiv.org/abs/2511.19826) — large-deviation, state-dependent change of measure achieving logarithmic efficiency; variance reduction exceeding several orders of magnitude in both regimes.
- [Mark Price Calculation (Perpetual and Expiry Contracts) — Bybit Help Center](https://www.bybit.com/en/help-center/article/Mark-Price-Calculation-Perpetual-Expiry-Contracts) — median of index / index-plus-basis / last-price, with fallback to last traded price under abnormal or insufficient data (the datafeed-fallback mechanism of §3.6).
