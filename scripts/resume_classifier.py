#!/usr/bin/env python3
"""Simple résumé classification system: score a résumé against THIS system.

"Analyze the current resume against system" — we derive a *system skill profile*
by scanning the ``allocation_gym`` codebase (imports + a domain lexicon), then
classify a résumé against it:

  1. Predicted role bucket (quant-dev / quant-research / data-eng / ml / swe)
     with a softmax confidence.
  2. A fit score (0-100) = weighted coverage of the system's most-demanded skills.
  3. Matched skills (with evidence weight) and the top missing-skill gaps.

Dependency-light: standard library only. Runs out-of-the-box on a bundled
sample résumé; pass ``--resume PATH`` (.txt/.md) to analyse a real one.

Usage:
    python3 scripts/resume_classifier.py                 # sample résumé vs this repo
    python3 scripts/resume_classifier.py --resume cv.txt # your résumé
    python3 scripts/resume_classifier.py --resume cv.txt --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Skill lexicon: canonical skill -> regex alias alternatives --------------
# Each alias is matched case-insensitively with word boundaries against text.
SKILL_LEXICON: dict[str, list[str]] = {
    "python": [r"python"],
    "numpy": [r"numpy"],
    "pandas": [r"pandas"],
    "matplotlib": [r"matplotlib", r"plotting"],
    "monte carlo": [r"monte[\s-]?carlo"],
    "importance sampling": [r"importance[\s-]?sampling", r"variance[\s-]?reduction"],
    "options pricing": [r"option[s]?\b", r"black[\s-]?scholes", r"derivative[s]?", r"greeks"],
    "volatility modeling": [r"volatilit", r"vol[\s-]?of[\s-]?vol", r"yang[\s-]?zhang", r"garch", r"implied vol"],
    "backtesting": [r"backtest", r"backtrader"],
    "stochastic calculus": [r"stochastic", r"brownian", r"gbm", r"ito"],
    "statistics": [r"regression", r"bayesian", r"hypothesis", r"statistic"],
    "kelly sizing": [r"kelly"],
    "rest api": [r"fastapi", r"flask", r"\brest\b", r"\bapi\b", r"http\.server", r"endpoint"],
    "websockets": [r"websocket", r"\bwss?\b", r"streaming"],
    "sql": [r"\bsql\b", r"postgres", r"mysql", r"bigquery"],
    "etl pipelines": [r"\betl\b", r"pipeline", r"ingest", r"airflow"],
    "machine learning": [r"machine[\s-]?learning", r"\bml\b", r"pytorch", r"tensorflow", r"scikit", r"neural"],
    "testing": [r"pytest", r"unittest", r"\btdd\b", r"\btest[s]?\b"],
    "git": [r"\bgit\b", r"github", r"version control"],
    "cloud": [r"\baws\b", r"\bgcp\b", r"azure", r"netlify", r"lambda", r"serverless"],
    "crypto markets": [r"crypto", r"bitcoin", r"\bbtc\b", r"altcoin", r"order[\s-]?book", r"l2 depth"],
}

# --- Role taxonomy: role -> the skills that define it ------------------------
ROLE_TAXONOMY: dict[str, set[str]] = {
    "Quant Developer": {
        "python", "numpy", "pandas", "monte carlo", "options pricing",
        "backtesting", "importance sampling", "volatility modeling",
        "rest api", "websockets", "testing", "crypto markets",
    },
    "Quant Researcher": {
        "monte carlo", "options pricing", "volatility modeling",
        "importance sampling", "statistics", "kelly sizing",
        "stochastic calculus", "backtesting",
    },
    "Data Engineer": {
        "python", "sql", "etl pipelines", "rest api", "websockets",
        "pandas", "cloud", "testing",
    },
    "ML Engineer": {
        "python", "numpy", "machine learning", "statistics", "pandas", "cloud",
    },
    "Software Engineer": {
        "python", "rest api", "websockets", "testing", "git", "cloud", "sql",
    },
}


def count_aliases(text: str, aliases: list[str]) -> int:
    """Total case-insensitive, word-boundary-ish matches of any alias in text."""
    total = 0
    for alias in aliases:
        total += len(re.findall(alias, text, flags=re.IGNORECASE))
    return total


def extract_skills(text: str) -> dict[str, int]:
    """Map each canonical skill to its hit count in ``text`` (0 if absent)."""
    return {skill: count_aliases(text, aliases) for skill, aliases in SKILL_LEXICON.items()}


# --- Holistic text similarity (résumé vs JD) --------------------------------
_STOPWORDS = {
    "the", "and", "for", "with", "you", "are", "our", "your", "will", "have",
    "has", "this", "that", "from", "into", "via", "per", "but", "not", "all",
    "any", "can", "out", "who", "use", "used", "using", "their", "they", "them",
    "a", "an", "to", "of", "in", "on", "as", "at", "or", "by", "is", "be", "we",
    "experience", "years", "team", "work", "working", "role", "ability", "strong",
    "including", "etc", "across", "within", "over", "plus", "such", "more",
}


def _content_tokens(text: str) -> list[str]:
    """Lowercased alphabetic content tokens (len>=3, stopwords removed)."""
    toks = re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", text.lower())
    return [t for t in toks if t not in _STOPWORDS]


def text_cosine(a: str, b: str) -> float:
    """Bag-of-words term-frequency cosine similarity between two texts (0-1)."""
    from collections import Counter

    ca, cb = Counter(_content_tokens(a)), Counter(_content_tokens(b))
    if not ca or not cb:
        return 0.0
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def build_system_profile(repo_root: Path) -> dict[str, int]:
    """Derive the system's skill demand by scanning the codebase.

    Aggregates skill-alias hit counts across the repo's Python and Markdown so
    the most-referenced skills (e.g. python, monte carlo, options pricing) carry
    the most weight — this is the "system" a résumé is scored against.
    """
    blob_parts: list[str] = []
    for path in repo_root.rglob("*"):
        if path.suffix.lower() not in {".py", ".md"}:
            continue
        # Skip our own vendored/worktree copies and caches to avoid double counting.
        parts = set(path.parts)
        if {".git", "__pycache__", ".claude", "vendor"} & parts:
            continue
        try:
            blob_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    blob = "\n".join(blob_parts)
    profile = {skill: c for skill, c in extract_skills(blob).items() if c > 0}
    return dict(sorted(profile.items(), key=lambda kv: kv[1], reverse=True))


def softmax(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a label->score mapping."""
    if not scores:
        return {}
    mx = max(scores.values())
    exps = {k: math.exp(v - mx) for k, v in scores.items()}
    denom = sum(exps.values()) or 1.0
    return {k: v / denom for k, v in exps.items()}


@dataclass
class Classification:
    role: str
    role_confidence: float
    role_scores: dict[str, float]
    fit_score: float                       # 0-100 coverage of the demand profile
    demand_label: str = "system"           # "system" or "job description"
    text_similarity: float | None = None   # TF cosine vs JD (JD mode only)
    matched: list[tuple[str, int]] = field(default_factory=list)
    gaps: list[tuple[str, int]] = field(default_factory=list)


def classify(
    resume_text: str,
    demand_profile: dict[str, int],
    top_n: int = 12,
    demand_label: str = "system",
) -> Classification:
    """Classify a résumé into a role and score its fit against a demand profile.

    ``demand_profile`` is a skill->weight mapping the résumé is scored against —
    either the codebase-derived system profile or a job description's skills.
    """
    resume_skills = extract_skills(resume_text)
    present = {s for s, c in resume_skills.items() if c > 0}

    # Role scoring: weight each present skill by log(1+resume hits) so repetition
    # helps but with diminishing returns; argmax role, softmax confidence.
    role_scores: dict[str, float] = {}
    for role, skills in ROLE_TAXONOMY.items():
        role_scores[role] = sum(
            math.log1p(resume_skills[s]) for s in skills if resume_skills.get(s, 0) > 0
        )
    role_probs = softmax(role_scores)
    best_role = max(role_probs, key=role_probs.get) if role_probs else "Unknown"

    # Fit: weighted coverage of the demand profile's top-N skills.
    top_skills = list(demand_profile.items())[:top_n]
    demand_total = sum(w for _, w in top_skills) or 1
    covered = sum(w for s, w in top_skills if s in present)
    fit = 100.0 * covered / demand_total

    matched = sorted(
        ((s, resume_skills[s]) for s in present), key=lambda kv: kv[1], reverse=True
    )
    gaps = [(s, w) for s, w in top_skills if s not in present]

    return Classification(
        role=best_role,
        role_confidence=role_probs.get(best_role, 0.0),
        role_scores=role_probs,
        fit_score=fit,
        demand_label=demand_label,
        matched=matched,
        gaps=gaps,
    )


SAMPLE_RESUME = """
Jordan Quant — Quantitative Developer

Summary: Python engineer with 5 years building systematic trading and
derivatives-pricing systems for crypto and equities.

Experience:
- Built a Monte Carlo options-pricing engine in Python/NumPy with importance
  sampling for deep out-of-the-money and barrier payoffs; cut variance ~30x.
- Designed a Backtrader backtesting framework with Yang-Zhang volatility,
  variance-ratio regime classification, and Kelly position sizing.
- Implemented a real-time market-data service over WebSockets and a REST API
  (FastAPI) consuming Binance L2 order-book depth for altcoin pricing.
- Modeled implied and realized volatility; wrote pytest suites; shipped via git.

Skills: Python, NumPy, pandas, Monte Carlo, Black-Scholes, stochastic calculus,
volatility modeling, backtesting, websockets, REST APIs, AWS.
"""


def load_resume(path: str | None) -> tuple[str, str]:
    """Return (label, text) for the résumé to analyse."""
    if path is None:
        return "bundled sample résumé", SAMPLE_RESUME
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"résumé file not found: {path}")
    if p.suffix.lower() == ".pdf":
        raise SystemExit(
            "PDF not supported directly — convert to .txt/.md first "
            "(e.g. `pdftotext cv.pdf cv.txt`), then re-run."
        )
    return p.name, p.read_text(encoding="utf-8", errors="ignore")


def render(label: str, demand_profile: dict[str, int], c: Classification, demand_src: str) -> str:
    """Human-readable report. ``demand_src`` describes the demand profile origin."""
    jd_mode = c.demand_label == "job description"
    bar = "=" * 64
    title = "candidate vs. job description" if jd_mode else "candidate vs. allocation_gym system"
    demand_hdr = (
        "JOB DESCRIPTION — required skills detected:" if jd_mode
        else "TOP SYSTEM DEMAND (most-referenced skills in the codebase):"
    )
    lines = [
        bar,
        f"RÉSUMÉ CLASSIFICATION  —  {title}",
        bar,
        f"Résumé source : {label}",
        f"Scored against: {demand_src} ({len(demand_profile)} skills)",
        "",
        demand_hdr,
    ]
    for skill, w in list(demand_profile.items())[:12]:
        lines.append(f"    {skill:<22} weight {w}")
    lines += [
        "",
        f"PREDICTED ROLE : {c.role}   (confidence {c.role_confidence*100:.0f}%)",
        "  role distribution:",
    ]
    for role, p in sorted(c.role_scores.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"    {role:<20} {p*100:5.1f}%")
    fit_hdr = "JD MATCH (skill coverage)" if jd_mode else "SYSTEM FIT SCORE"
    lines += ["", f"{fit_hdr} : {c.fit_score:.0f}/100"]
    if c.text_similarity is not None:
        lines.append(f"JD TEXT SIMILARITY (TF cosine) : {c.text_similarity*100:.0f}/100")
        blended = 0.6 * c.fit_score + 0.4 * (c.text_similarity * 100)
        lines.append(f"OVERALL JD MATCH (0.6·coverage + 0.4·cosine) : {blended:.0f}/100")
    gap_hdr = "TOP GAPS (JD requires, résumé lacks):" if jd_mode else "TOP GAPS (system demands, résumé lacks):"
    lines += [
        "",
        f"MATCHED SKILLS ({len(c.matched)}):",
        "    " + (", ".join(f"{s}×{n}" for s, n in c.matched) or "(none)"),
        "",
        gap_hdr,
        "    " + (", ".join(s for s, _ in c.gaps) or "(none — full coverage)"),
        bar,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify a résumé against the system or a job description.")
    ap.add_argument("--resume", help="path to a .txt/.md résumé (default: bundled sample)")
    ap.add_argument("--jd", help="path to a .txt/.md job description; score the résumé against it instead of the repo")
    ap.add_argument("--top", type=int, default=12, help="number of top demand skills to score against")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    label, text = load_resume(args.resume)

    similarity: float | None = None
    if args.jd:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            raise SystemExit(f"job-description file not found: {args.jd}")
        if jd_path.suffix.lower() == ".pdf":
            raise SystemExit("PDF JD not supported — convert to .txt/.md first (pdftotext).")
        jd_text = jd_path.read_text(encoding="utf-8", errors="ignore")
        demand_profile = {s: c for s, c in extract_skills(jd_text).items() if c > 0}
        demand_profile = dict(sorted(demand_profile.items(), key=lambda kv: kv[1], reverse=True))
        demand_src = f"job description: {jd_path.name}"
        demand_label = "job description"
        similarity = text_cosine(text, jd_text)
    else:
        demand_profile = build_system_profile(REPO_ROOT)
        demand_src = "allocation_gym codebase"
        demand_label = "system"

    c = classify(text, demand_profile, top_n=args.top, demand_label=demand_label)
    c.text_similarity = similarity

    if args.json:
        out = {
            "resume": label,
            "scored_against": demand_src,
            "demand_skills": demand_profile,
            "predicted_role": c.role,
            "role_confidence": round(c.role_confidence, 4),
            "role_distribution": {k: round(v, 4) for k, v in c.role_scores.items()},
            "fit_score": round(c.fit_score, 1),
            "matched_skills": dict(c.matched),
            "gaps": [s for s, _ in c.gaps],
        }
        if similarity is not None:
            out["jd_text_similarity"] = round(similarity * 100, 1)
            out["overall_jd_match"] = round(0.6 * c.fit_score + 0.4 * similarity * 100, 1)
        print(json.dumps(out, indent=2))
    else:
        print(render(label, demand_profile, c, demand_src))


if __name__ == "__main__":
    main()
