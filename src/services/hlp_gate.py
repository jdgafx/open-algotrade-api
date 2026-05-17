"""
HLP Sentiment Gate — Gate 4.6 in the trade entry pipeline.

MoonDev Edge B: HLP (Hyperliquid's market-maker vault) tends to be wrong at
extremes. When HLP has a large concentrated losing position in an asset, fading
it (going opposite direction) has historically high win rates.

Gate logic:
- Cache HLP positions every 5 minutes from the public Hyperliquid API
- For each asset, compute uPnL as % of notional
- Block entries that go IN THE SAME DIRECTION as HLP when HLP is losing > threshold
"""

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HLP_ADDRESS = "0x010461C14e146ac35Fe42271BDC1134EE31C703a"
HL_API = "https://api.hyperliquid.xyz/info"
CACHE_TTL = 300          # seconds between refreshes
BLOCK_THRESHOLD = -0.02  # block when HLP uPnL/notional < -2%


class HLPSentimentGate:
    """Blocks entries aligned with HLP's underwater concentrated positions."""

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._cache_time: float = 0.0
        self._refreshing: bool = False  # simple single-process reentrancy guard

    async def _refresh(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    HL_API,
                    json={"type": "clearinghouseState", "user": HLP_ADDRESS},
                )
            positions = resp.json().get("assetPositions", [])
            cache: dict[str, dict] = {}
            for p in positions:
                pos = p.get("position", {})
                coin = pos.get("coin", "")
                size = float(pos.get("szi", 0) or 0)
                entry = float(pos.get("entryPx") or 0)
                upnl = float(pos.get("unrealizedPnl") or 0)
                if not coin or entry == 0 or size == 0:
                    continue
                notional = abs(size) * entry
                upnl_pct = upnl / notional if notional > 0 else 0.0
                cache[coin.upper()] = {
                    "direction": "long" if size > 0 else "short",
                    "upnl_pct": upnl_pct,
                    "notional": notional,
                }
            self._cache = cache
            self._cache_time = time.monotonic()
            logger.info("HLP gate refreshed: %d positions tracked", len(cache))
        except Exception as exc:
            logger.warning("HLP gate refresh failed (gate inactive this cycle): %s", exc)

    async def should_block(self, symbol: str, is_long: bool) -> tuple[bool, str]:
        """Returns (block, reason). Block when our direction matches HLP's losing position."""
        if time.monotonic() - self._cache_time > CACHE_TTL and not self._refreshing:
            self._refreshing = True
            try:
                await self._refresh()
            finally:
                self._refreshing = False

        coin = symbol.replace("-USD", "").replace("/USD", "").replace("USDT", "").upper()
        data = self._cache.get(coin)
        if not data:
            return False, ""

        our_dir = "long" if is_long else "short"
        hlp_dir = data["direction"]
        upnl_pct = data["upnl_pct"]

        if our_dir == hlp_dir and upnl_pct < BLOCK_THRESHOLD:
            reason = (
                f"HLP {hlp_dir} {coin} uPnL={upnl_pct*100:.1f}% "
                f"(threshold={BLOCK_THRESHOLD*100:.0f}%) — fading HLP"
            )
            return True, reason

        return False, ""

    def get_signal(self, symbol: str) -> Optional[str]:
        """Returns 'fade_long', 'fade_short', or None — for logging/dashboards."""
        coin = symbol.replace("-USD", "").replace("/USD", "").upper()
        data = self._cache.get(coin)
        if not data or data["upnl_pct"] >= BLOCK_THRESHOLD:
            return None
        return "fade_long" if data["direction"] == "long" else "fade_short"
