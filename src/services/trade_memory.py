"""
Supermemory integration for trading bot long-term memory.

Stores closed trade outcomes and recalls similar past trades to inform
LLM gate decisions. Uses Supermemory v3 API with httpx (no extra deps).
"""
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.supermemory.ai"
_TIMEOUT = 5.0


def _headers() -> dict[str, str]:
    key = os.getenv("SUPERMEMORY_API_KEY", "")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def store_trade(
    strategy: str,
    symbol: str,
    signal: str,
    pnl: float,
    regime: str,
    params: dict,
    price: float,
) -> None:
    """Persist a closed trade outcome to Supermemory."""
    if not os.getenv("SUPERMEMORY_API_KEY"):
        return
    outcome = "WIN" if pnl > 0 else "LOSS"
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = (
        f"[TRADE] {date_str} | {strategy} | {signal} {symbol} @ {price:.2f} | "
        f"PnL: ${pnl:.4f} ({outcome}) | Regime: {regime} | "
        f"cooldown={params.get('cooldown_seconds', '?')}s "
        f"max_hr={params.get('max_trades_per_hour', '?')} "
        f"min_signal={params.get('min_signal_strength', '?')}"
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{_BASE}/v3/documents",
                headers=_headers(),
                json={"content": content, "tags": ["trading-bot", strategy, symbol]},
            )
            if r.status_code not in (200, 201):
                logger.debug("Supermemory store non-200: %s %s", r.status_code, r.text[:80])
    except Exception as e:
        logger.debug("Supermemory store_trade failed (non-fatal): %s", e)


async def recall_trades(
    strategy: str, symbol: str, regime: str, limit: int = 5
) -> str:
    """Recall similar past trades from Supermemory for LLM gate context."""
    if not os.getenv("SUPERMEMORY_API_KEY"):
        return ""
    query = f"{strategy} {symbol} {regime} trade PnL WIN LOSS"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                f"{_BASE}/v3/search",
                headers=_headers(),
                json={"q": query, "limit": limit},
            )
            if r.status_code != 200:
                return ""
            items = r.json().get("results", [])
            if not items:
                return ""
            return "\n".join(item.get("content", "") for item in items[:limit])
    except Exception as e:
        logger.debug("Supermemory recall_trades failed (non-fatal): %s", e)
        return ""
