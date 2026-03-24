# allocation-gym

Hardware arbitrage toolkit for eBay GPUs, storage drives, and other scalping opportunities.

## Modules

| Module | Purpose |
|---|---|
| `allocation_gym.scrapers` | eBay sold-listing scrapers (GPUs, SSDs, HDDs, etc.) |
| `allocation_gym.pricing` | Price trend analysis, margin calculation, deal scoring |
| `allocation_gym.alerts` | Threshold-based deal alerts |
| `allocation_gym.data` | Historical price data storage and retrieval |

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Scrape recent sold listings for RTX 4090
python -m allocation_gym.scrapers.ebay --query "RTX 4090" --category gpu

# Analyze price trends
python -m allocation_gym.pricing.trends --item "RTX 4090" --days 30
```
