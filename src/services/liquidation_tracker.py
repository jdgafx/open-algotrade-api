"""
Liquidation Intelligence Pipeline — Tracks positions approaching liquidation
across HyperLiquid, aggregates liquidation data, and provides anti-liquidation
guard integration for the strategy engine.

Uses the HL Info API to:
- Poll tracked addresses for positions near liquidation
- Aggregate liquidation levels by symbol/price
- Track real-time liquidation events via WebSocket
- Calculate liquidation volume over time windows
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class NearLiquidation:
    """A position that is approaching its liquidation price."""
    address: str
    symbol: str
    side: str  # "long" or "short"
    size: float
    size_usd: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    proximity_pct: float  # how close to liquidation (0-100%)
    leverage: float
    unrealized_pnl: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LiquidationEvent:
    """A confirmed liquidation event."""
    address: str
    symbol: str
    side: str
    size: float
    size_usd: float
    price: float
    timestamp: datetime
    is_cascade: bool = False  # part of a cascade event


@dataclass
class LiquidationHeatmapLevel:
    """Aggregated liquidation volume at a price level."""
    symbol: str
    price_level: float
    long_liq_usd: float  # total USD in long liquidations at this level
    short_liq_usd: float  # total USD in short liquidations at this level
    total_positions: int


@dataclass
class LiquidationThreshold:
    """Safety threshold status for trading."""
    safe_to_trade: bool
    reason: str
    total_liq_5min: float
    total_liq_15min: float
    total_liq_30min: float
    total_liq_1hr: float
    threshold_usd: float
    risk_level: str  # "low", "medium", "high", "extreme"


class LiquidationTracker:
    """
    Tracks positions approaching liquidation across HyperLiquid.

    Polls tracked addresses for positions, estimates liquidation prices,
    and maintains a real-time view of liquidation risk across the market.
    """

    def __init__(
        self,
        base_url: str,
        tracked_addresses: Optional[List[str]] = None,
        liq_threshold_usd: float = 250_000.0,
        poll_interval: int = 30,
        price_bucket_pct: float = 0.5,
    ):
        self.base_url = base_url
        self.tracked_addresses: List[str] = tracked_addresses or []
        self.liq_threshold_usd = liq_threshold_usd
        self.poll_interval = poll_interval
        self.price_bucket_pct = price_bucket_pct

        self._near_liquidations: List[NearLiquidation] = []
        self._liquidation_events: List[LiquidationEvent] = []
        self._heatmap_cache: Dict[str, List[LiquidationHeatmapLevel]] = {}
        self._mid_prices: Dict[str, float] = {}
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._last_poll: Optional[datetime] = None

        logger.info(
            "LiquidationTracker initialized | tracked=%d addresses | threshold=$%.0f",
            len(self.tracked_addresses),
            self.liq_threshold_usd,
        )

    # ──────────────────────────────────────────────
    # HL API Helpers
    # ──────────────────────────────────────────────

    def _info_request(self, data: Dict) -> Any:
        """Make a POST request to the HL Info API."""
        url = f"{self.base_url}/info"
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        resp.raise_for_status()
        return resp.json()

    def _get_all_mids(self) -> Dict[str, float]:
        """Get all mid prices."""
        raw = self._info_request({"type": "allMids"})
        return {k: float(v) for k, v in raw.items()}

    def _get_meta_and_ctxs(self) -> Tuple[List[Dict], List[Dict]]:
        """Get meta + asset contexts (OI, funding, etc.)."""
        result = self._info_request({"type": "metaAndAssetCtxs"})
        if isinstance(result, list) and len(result) >= 2:
            return result[0].get("universe", []), result[1]
        return [], []

    def _get_user_state(self, address: str) -> Dict:
        """Get full user state for an address."""
        return self._info_request({"type": "clearinghouseState", "user": address})

    def _estimate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        is_long: bool,
        maintenance_margin: float = 0.03,
    ) -> float:
        """
        Estimate liquidation price from entry, leverage, and maintenance margin.

        For longs: liq_price = entry * (1 - 1/leverage + maintenance_margin)
        For shorts: liq_price = entry * (1 + 1/leverage - maintenance_margin)
        """
        if leverage <= 0:
            leverage = 1.0
        if is_long:
            return entry_price * (1.0 - (1.0 / leverage) + maintenance_margin)
        else:
            return entry_price * (1.0 + (1.0 / leverage) - maintenance_margin)

    # ──────────────────────────────────────────────
    # Core Scanning
    # ──────────────────────────────────────────────

    def scan_address(self, address: str) -> List[NearLiquidation]:
        """Scan a single address for positions near liquidation."""
        near_liqs = []
        try:
            state = self._get_user_state(address)
            positions = state.get("assetPositions", [])

            for pos_wrapper in positions:
                pos = pos_wrapper.get("position", {})
                size = float(pos.get("szi", 0))
                if size == 0:
                    continue

                symbol = pos.get("coin", "")
                entry_px = float(pos.get("entryPx", 0))
                is_long = size > 0
                leverage_info = pos.get("leverage", {})
                leverage = float(leverage_info.get("value", 1)) if isinstance(leverage_info, dict) else float(leverage_info)
                liq_px = float(pos.get("liquidationPx", 0) or 0)

                # If HL doesn't provide liq price, estimate it
                if liq_px == 0:
                    liq_px = self._estimate_liquidation_price(entry_px, leverage, is_long)

                mark_px = self._mid_prices.get(symbol, float(pos.get("markPx", 0) or entry_px))
                if mark_px == 0:
                    continue

                # Calculate proximity to liquidation
                if is_long:
                    proximity_pct = ((mark_px - liq_px) / mark_px) * 100 if mark_px > 0 else 100
                else:
                    proximity_pct = ((liq_px - mark_px) / mark_px) * 100 if mark_px > 0 else 100

                proximity_pct = max(0, proximity_pct)
                size_usd = abs(size) * mark_px
                unrealized_pnl = float(pos.get("unrealizedPnl", 0))

                # Only track positions within 10% of liquidation
                if proximity_pct <= 10.0:
                    near_liqs.append(NearLiquidation(
                        address=address,
                        symbol=symbol,
                        side="long" if is_long else "short",
                        size=abs(size),
                        size_usd=size_usd,
                        entry_price=entry_px,
                        mark_price=mark_px,
                        liquidation_price=liq_px,
                        proximity_pct=proximity_pct,
                        leverage=leverage,
                        unrealized_pnl=unrealized_pnl,
                    ))
        except Exception as e:
            logger.warning("Failed to scan address %s: %s", address[:10], e)

        return near_liqs

    async def scan_all(self) -> List[NearLiquidation]:
        """Scan all tracked addresses for near-liquidation positions."""
        self._mid_prices = await asyncio.to_thread(self._get_all_mids)
        all_near = []

        for address in self.tracked_addresses:
            near = await asyncio.to_thread(self.scan_address, address)
            all_near.extend(near)
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)

        self._near_liquidations = sorted(all_near, key=lambda x: x.proximity_pct)
        self._last_poll = datetime.now(timezone.utc)
        self._rebuild_heatmap()

        logger.info(
            "Scan complete | %d positions near liquidation | %d addresses scanned",
            len(all_near),
            len(self.tracked_addresses),
        )
        return self._near_liquidations

    # ──────────────────────────────────────────────
    # Heatmap
    # ──────────────────────────────────────────────

    def _rebuild_heatmap(self) -> None:
        """Rebuild the liquidation heatmap from current near-liquidation data."""
        self._heatmap_cache.clear()
        buckets: Dict[str, Dict[float, Dict]] = defaultdict(lambda: defaultdict(
            lambda: {"long_usd": 0.0, "short_usd": 0.0, "count": 0}
        ))

        for nl in self._near_liquidations:
            # Bucket liq price to nearest price_bucket_pct
            bucket_size = nl.liquidation_price * (self.price_bucket_pct / 100)
            if bucket_size > 0:
                bucket_price = round(nl.liquidation_price / bucket_size) * bucket_size
            else:
                bucket_price = nl.liquidation_price

            entry = buckets[nl.symbol][bucket_price]
            if nl.side == "long":
                entry["long_usd"] += nl.size_usd
            else:
                entry["short_usd"] += nl.size_usd
            entry["count"] += 1

        for symbol, price_buckets in buckets.items():
            levels = []
            for price_level, data in sorted(price_buckets.items()):
                levels.append(LiquidationHeatmapLevel(
                    symbol=symbol,
                    price_level=round(price_level, 2),
                    long_liq_usd=round(data["long_usd"], 2),
                    short_liq_usd=round(data["short_usd"], 2),
                    total_positions=data["count"],
                ))
            self._heatmap_cache[symbol] = levels

    def get_heatmap(self, symbol: str) -> List[LiquidationHeatmapLevel]:
        """Get liquidation heatmap levels for a symbol."""
        return self._heatmap_cache.get(symbol, [])

    # ──────────────────────────────────────────────
    # Liquidation Events
    # ──────────────────────────────────────────────

    def record_liquidation(self, event: LiquidationEvent) -> None:
        """Record a liquidation event."""
        self._liquidation_events.append(event)
        # Keep only 24hr of events
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._liquidation_events = [
            e for e in self._liquidation_events if e.timestamp > cutoff
        ]
        logger.info(
            "Liquidation recorded | %s %s %s | $%.0f @ %.2f",
            event.address[:10], event.symbol, event.side,
            event.size_usd, event.price,
        )

    def get_liquidation_volume(
        self, minutes: int = 30, symbol: Optional[str] = None
    ) -> float:
        """Get total liquidation volume in USD over the last N minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        total = 0.0
        for event in self._liquidation_events:
            if event.timestamp > cutoff:
                if symbol is None or event.symbol == symbol:
                    total += event.size_usd
        return total

    # ──────────────────────────────────────────────
    # Safety Threshold (Anti-Liquidation Guard)
    # ──────────────────────────────────────────────

    def get_threshold_status(self) -> LiquidationThreshold:
        """
        Check if it's safe to trade based on recent liquidation volume.
        Used by strategies as an anti-liquidation guard.
        """
        vol_5m = self.get_liquidation_volume(5)
        vol_15m = self.get_liquidation_volume(15)
        vol_30m = self.get_liquidation_volume(30)
        vol_1h = self.get_liquidation_volume(60)

        # Risk level based on 30-min volume vs threshold
        ratio = vol_30m / self.liq_threshold_usd if self.liq_threshold_usd > 0 else 0

        if ratio < 0.25:
            risk_level = "low"
        elif ratio < 0.5:
            risk_level = "medium"
        elif ratio < 1.0:
            risk_level = "high"
        else:
            risk_level = "extreme"

        safe = vol_30m < self.liq_threshold_usd

        reason = "Market conditions normal" if safe else (
            f"Liquidation volume ${vol_30m:,.0f} exceeds "
            f"threshold ${self.liq_threshold_usd:,.0f} in 30min"
        )

        return LiquidationThreshold(
            safe_to_trade=safe,
            reason=reason,
            total_liq_5min=vol_5m,
            total_liq_15min=vol_15m,
            total_liq_30min=vol_30m,
            total_liq_1hr=vol_1h,
            threshold_usd=self.liq_threshold_usd,
            risk_level=risk_level,
        )

    def is_safe_to_trade(self, symbol: Optional[str] = None) -> bool:
        """
        Quick check: is it safe to trade right now?
        Used by strategies before entering positions.
        """
        vol = self.get_liquidation_volume(30, symbol)
        return vol < self.liq_threshold_usd

    # ──────────────────────────────────────────────
    # Near Liquidation Queries
    # ──────────────────────────────────────────────

    def get_near_liquidations(
        self,
        symbol: Optional[str] = None,
        max_proximity_pct: float = 5.0,
        sort_by: str = "proximity",
        limit: int = 100,
    ) -> List[NearLiquidation]:
        """
        Get positions near liquidation, filtered and sorted.

        Args:
            symbol: Filter by symbol (None = all)
            max_proximity_pct: Only positions within this % of liquidation
            sort_by: "proximity", "size", or "leverage"
            limit: Max results
        """
        results = self._near_liquidations

        if symbol:
            results = [nl for nl in results if nl.symbol == symbol]

        results = [nl for nl in results if nl.proximity_pct <= max_proximity_pct]

        if sort_by == "size":
            results.sort(key=lambda x: x.size_usd, reverse=True)
        elif sort_by == "leverage":
            results.sort(key=lambda x: x.leverage, reverse=True)
        else:
            results.sort(key=lambda x: x.proximity_pct)

        return results[:limit]

    def get_aggregated_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated statistics about positions near liquidation."""
        near = self._near_liquidations
        if symbol:
            near = [nl for nl in near if nl.symbol == symbol]

        longs = [nl for nl in near if nl.side == "long"]
        shorts = [nl for nl in near if nl.side == "short"]

        return {
            "total_near_liq": len(near),
            "total_long_near_liq": len(longs),
            "total_short_near_liq": len(shorts),
            "total_long_usd_at_risk": sum(nl.size_usd for nl in longs),
            "total_short_usd_at_risk": sum(nl.size_usd for nl in shorts),
            "within_1pct": len([nl for nl in near if nl.proximity_pct <= 1.0]),
            "within_2pct": len([nl for nl in near if nl.proximity_pct <= 2.0]),
            "within_5pct": len([nl for nl in near if nl.proximity_pct <= 5.0]),
            "symbols": list(set(nl.symbol for nl in near)),
            "last_updated": self._last_poll.isoformat() if self._last_poll else None,
        }

    # ──────────────────────────────────────────────
    # Historical Data
    # ──────────────────────────────────────────────

    def get_recent_events(
        self, minutes: int = 60, symbol: Optional[str] = None, limit: int = 100
    ) -> List[LiquidationEvent]:
        """Get recent liquidation events."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        events = [e for e in self._liquidation_events if e.timestamp > cutoff]
        if symbol:
            events = [e for e in events if e.symbol == symbol]
        return sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]

    # ──────────────────────────────────────────────
    # Address Management
    # ──────────────────────────────────────────────

    def add_address(self, address: str) -> None:
        """Add an address to track."""
        if address not in self.tracked_addresses:
            self.tracked_addresses.append(address)
            logger.info("Added tracked address: %s", address[:10])

    def remove_address(self, address: str) -> None:
        """Remove an address from tracking."""
        if address in self.tracked_addresses:
            self.tracked_addresses.remove(address)
            logger.info("Removed tracked address: %s", address[:10])

    # ──────────────────────────────────────────────
    # Background Polling
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("LiquidationTracker started")

    async def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("LiquidationTracker stopped")

    async def _poll_loop(self) -> None:
        """Background loop that periodically scans for near-liquidation positions."""
        while self._running:
            try:
                await self.scan_all()
            except Exception as e:
                logger.error("Poll loop error: %s", e)
            await asyncio.sleep(self.poll_interval)
