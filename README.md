# crane-gym

Hardware arbitrage toolkit for eBay GPUs, storage drives, and other scalping opportunities.

## Modules

| Module | Purpose |
|---|---|
| `crane_gym.scrapers` | eBay sold-listing scrapers (GPUs, SSDs, HDDs, etc.) |
| `crane_gym.pricing` | Price trend analysis, margin calculation, deal scoring |
| `crane_gym.alerts` | Threshold-based deal alerts |
| `crane_gym.data` | Historical price data storage and retrieval |

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Scrape recent sold listings for RTX 4090
python -m crane_gym.scrapers.ebay --query "RTX 4090" --category gpu

# Analyze price trends
python -m crane_gym.pricing.trends --item "RTX 4090" --days 30
```
