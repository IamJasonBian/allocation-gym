"""
Credential resolution: env vars -> GitHub repo variables -> error.

Usage:
    from allocation_gym.credentials import get_alpaca_keys
    api_key, secret_key = get_alpaca_keys()
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

GH_REPO = "IamJasonBian/allocation-gym"


def _gh_variable_get(name: str) -> str:
    """Fetch a variable from GitHub repo via `gh` CLI."""
    try:
        result = subprocess.run(
            ["gh", "variable", "get", name, "--repo", GH_REPO],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def get_alpaca_keys() -> tuple[str, str]:
    """
    Resolve Alpaca API credentials.

    Priority:
      1. ALPACA_API_KEY / ALPACA_SECRET_KEY env vars
      2. GitHub repo variables via `gh variable get`
    """
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if api_key and secret_key:
        return api_key, secret_key

    logger.info("Env vars not set, trying GitHub repo variables...")
    if not api_key:
        api_key = _gh_variable_get("ALPACA_API_KEY")
    if not secret_key:
        secret_key = _gh_variable_get("ALPACA_SECRET_KEY")

    if api_key and secret_key:
        logger.info("Loaded Alpaca keys from GitHub repo variables")
    elif not api_key or not secret_key:
        logger.warning(
            "Alpaca keys not found. Set ALPACA_API_KEY/ALPACA_SECRET_KEY "
            "env vars or store as GitHub variables in %s", GH_REPO
        )

    return api_key, secret_key
