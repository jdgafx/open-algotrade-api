"""
Whale Tracker — Monitors top depositors on HyperLiquid, tracks their
positions, PnL, and provides whale trade alerts.

Key features:
- Track top N whale addresses (configurable)
- Per-whale: positions, entry prices, unrealized PnL, liquidation levels
- Whale labeling/tagging system
- Whale trade alerts (open/close notifications)
- Survival statistics (% that have blown up)
- Historical performance tracking
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


@dataclass
class WhalePosition:
    """A single whale's position."""
    symbol: str
    side: str
    size: float
    size_usd: float
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    leverage: float
    return_on_equity: float


@dataclass
class WhaleProfile:
    """Tracked whale with metadata."""
    address: str
    label: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    alert_enabled: bool = False
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: Optional[datetime] = None

    # Snapshot data (refreshed on poll)
    account_value: float = 0.0
    positions: List[WhalePosition] = field(default_factory=list)
    total_unrealized_pnl: float = 0.0
    total_position_value: float = 0.0
    position_count: int = 0

    # Historical tracking
    peak_account_value: float = 0.0
    initial_deposit_estimate: float = 0.0
    is_blown_up: bool = False  # account value dropped to ~0
    blown_up_at: Optional[datetime] = None


@dataclass
class WhaleTradeAlert:
    """Alert when a whale opens or closes a position."""
    address: str
    label: Optional[str]
    action: str  # "opened", "closed", "increased", "decreased"
    symbol: str
    side: str
    size: float
    size_usd: float
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WhaleSurvivalStats:
    """Aggregated survival statistics for tracked whales."""
    total_tracked: int
    still_active: int  # account value > $1K
    blown_up: int  # account value ~0
    survival_rate: float  # % still active
    blow_up_rate: float  # % blown up
    total_whale_equity: float
    avg_account_value: float
    median_account_value: float
    top_whale_value: float
    total_positions: int


class WhaleTracker:
    """
    Tracks whale addresses on HyperLiquid with position monitoring,
    labeling, alerts, and survival statistics.
    """

    def __init__(
        self,
        base_url: str,
        max_whales: int = 1000,
        poll_interval: int = 60,
        blown_up_threshold: float = 1000.0,
    ):
        self.base_url = base_url
        self.max_whales = max_whales
        self.poll_interval = poll_interval
        self.blown_up_threshold = blown_up_threshold

        self._whales: Dict[str, WhaleProfile] = {}
        self._alerts: List[WhaleTradeAlert] = []
        self._previous_positions: Dict[str, Dict[str, WhalePosition]] = {}
        self._alert_callbacks: List[Callable] = []
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._mid_prices: Dict[str, float] = {}

        logger.info(
            "WhaleTracker initialized | max_whales=%d | poll_interval=%ds",
            max_whales,
            poll_interval,
        )

    # ──────────────────────────────────────────────
    # HL API Helpers
    # ──────────────────────────────────────────────

    def _info_request(self, data: Dict) -> Any:
        url = f"{self.base_url}/info"
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=data)
        resp.raise_for_status()
        return resp.json()

    def _get_all_mids(self) -> Dict[str, float]:
        raw = self._info_request({"type": "allMids"})
        return {k: float(v) for k, v in raw.items()}

    def _get_user_state(self, address: str) -> Dict:
        return self._info_request({"type": "clearinghouseState", "user": address})

    def _get_user_fills(self, address: str) -> List[Dict]:
        return self._info_request({"type": "userFills", "user": address})

    # ──────────────────────────────────────────────
    # Whale Management
    # ──────────────────────────────────────────────

    def add_whale(
        self,
        address: str,
        label: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> WhaleProfile:
        """Add a whale address to track."""
        if address in self._whales:
            # Update existing
            whale = self._whales[address]
            if label is not None:
                whale.label = label
            if notes is not None:
                whale.notes = notes
            if tags is not None:
                whale.tags = tags
            return whale

        whale = WhaleProfile(
            address=address,
            label=label,
            notes=notes,
            tags=tags or [],
        )
        self._whales[address] = whale
        logger.info("Added whale: %s (%s)", address[:10], label or "unlabeled")
        return whale

    def remove_whale(self, address: str) -> bool:
        """Remove a whale from tracking."""
        if address in self._whales:
            del self._whales[address]
            self._previous_positions.pop(address, None)
            logger.info("Removed whale: %s", address[:10])
            return True
        return False

    def set_label(self, address: str, label: str, notes: Optional[str] = None) -> bool:
        """Set label and optional notes for a whale."""
        whale = self._whales.get(address)
        if not whale:
            return False
        whale.label = label
        if notes is not None:
            whale.notes = notes
        return True

    def toggle_alert(self, address: str, enabled: bool) -> bool:
        """Enable or disable trade alerts for a whale."""
        whale = self._whales.get(address)
        if not whale:
            return False
        whale.alert_enabled = enabled
        logger.info("Alerts %s for whale %s", "enabled" if enabled else "disabled", address[:10])
        return True

    # ──────────────────────────────────────────────
    # Whale Scanning
    # ──────────────────────────────────────────────

    def _estimate_liq_price(
        self, entry_px: float, leverage: float, is_long: bool
    ) -> float:
        if leverage <= 0:
            leverage = 1.0
        maint = 0.03
        if is_long:
            return entry_px * (1.0 - (1.0 / leverage) + maint)
        else:
            return entry_px * (1.0 + (1.0 / leverage) - maint)

    def scan_whale(self, address: str) -> Optional[WhaleProfile]:
        """Fetch and update a single whale's data."""
        whale = self._whales.get(address)
        if not whale:
            return None

        try:
            state = self._get_user_state(address)
            margin = state.get("marginSummary", {})
            account_value = float(margin.get("accountValue", 0))

            positions = []
            total_pnl = 0.0
            total_pos_value = 0.0

            for pos_wrapper in state.get("assetPositions", []):
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
                if liq_px == 0:
                    liq_px = self._estimate_liq_price(entry_px, leverage, is_long)

                mark_px = self._mid_prices.get(symbol, entry_px)
                unrealized_pnl = float(pos.get("unrealizedPnl", 0))
                roe = float(pos.get("returnOnEquity", 0)) * 100
                size_usd = abs(size) * mark_px
                total_pnl += unrealized_pnl
                total_pos_value += size_usd

                positions.append(WhalePosition(
                    symbol=symbol,
                    side="long" if is_long else "short",
                    size=abs(size),
                    size_usd=size_usd,
                    entry_price=entry_px,
                    mark_price=mark_px,
                    liquidation_price=liq_px,
                    unrealized_pnl=unrealized_pnl,
                    leverage=leverage,
                    return_on_equity=roe,
                ))

            # Detect position changes for alerts
            if whale.alert_enabled:
                self._detect_trade_changes(whale, positions)

            # Store current positions for next diff
            self._previous_positions[address] = {
                p.symbol + p.side: p for p in positions
            }

            # Update whale profile
            whale.account_value = account_value
            whale.positions = positions
            whale.total_unrealized_pnl = total_pnl
            whale.total_position_value = total_pos_value
            whale.position_count = len(positions)
            whale.last_updated = datetime.now(timezone.utc)

            # Track peak and blown-up status
            if account_value > whale.peak_account_value:
                whale.peak_account_value = account_value
            if whale.initial_deposit_estimate == 0 and account_value > 0:
                whale.initial_deposit_estimate = account_value

            if account_value < self.blown_up_threshold and whale.peak_account_value > self.blown_up_threshold * 10:
                if not whale.is_blown_up:
                    whale.is_blown_up = True
                    whale.blown_up_at = datetime.now(timezone.utc)
                    logger.warning(
                        "Whale BLOWN UP: %s (%s) | peak=$%.0f now=$%.0f",
                        address[:10], whale.label or "?",
                        whale.peak_account_value, account_value,
                    )

            return whale

        except Exception as e:
            logger.warning("Failed to scan whale %s: %s", address[:10], e)
            return None

    def _detect_trade_changes(
        self, whale: WhaleProfile, current_positions: List[WhalePosition]
    ) -> None:
        """Detect new/closed positions and generate alerts."""
        prev = self._previous_positions.get(whale.address, {})
        curr = {p.symbol + p.side: p for p in current_positions}

        # New positions (in current but not in previous)
        for key, pos in curr.items():
            if key not in prev:
                alert = WhaleTradeAlert(
                    address=whale.address,
                    label=whale.label,
                    action="opened",
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.size,
                    size_usd=pos.size_usd,
                    price=pos.mark_price,
                )
                self._alerts.append(alert)
                self._fire_alert(alert)
            elif abs(curr[key].size - prev[key].size) > 0.001:
                action = "increased" if curr[key].size > prev[key].size else "decreased"
                alert = WhaleTradeAlert(
                    address=whale.address,
                    label=whale.label,
                    action=action,
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.size,
                    size_usd=pos.size_usd,
                    price=pos.mark_price,
                )
                self._alerts.append(alert)
                self._fire_alert(alert)

        # Closed positions (in previous but not in current)
        for key, pos in prev.items():
            if key not in curr:
                alert = WhaleTradeAlert(
                    address=whale.address,
                    label=whale.label,
                    action="closed",
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.size,
                    size_usd=pos.size_usd,
                    price=pos.mark_price,
                )
                self._alerts.append(alert)
                self._fire_alert(alert)

    def _fire_alert(self, alert: WhaleTradeAlert) -> None:
        """Fire alert to registered callbacks."""
        logger.info(
            "Whale alert: %s (%s) %s %s %s | $%.0f",
            alert.address[:10], alert.label or "?",
            alert.action, alert.side, alert.symbol,
            alert.size_usd,
        )
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.error("Alert callback error: %s", e)

    def on_alert(self, callback: Callable[[WhaleTradeAlert], None]) -> None:
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    # ──────────────────────────────────────────────
    # Batch Scanning
    # ──────────────────────────────────────────────

    async def scan_all(self) -> List[WhaleProfile]:
        """Scan all tracked whales."""
        self._mid_prices = await asyncio.to_thread(self._get_all_mids)
        results = []

        for address in list(self._whales.keys()):
            whale = await asyncio.to_thread(self.scan_whale, address)
            if whale:
                results.append(whale)
            await asyncio.sleep(0.1)  # rate limit

        # Trim old alerts (keep 24hr)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self._alerts = [a for a in self._alerts if a.timestamp > cutoff]

        logger.info("Whale scan complete | %d whales updated", len(results))
        return results

    # ──────────────────────────────────────────────
    # Queries
    # ──────────────────────────────────────────────

    def get_whale(self, address: str) -> Optional[WhaleProfile]:
        """Get a specific whale's profile."""
        return self._whales.get(address)

    def list_whales(
        self,
        sort_by: str = "account_value",
        limit: int = 100,
        offset: int = 0,
    ) -> List[WhaleProfile]:
        """List tracked whales, sorted by field."""
        whales = list(self._whales.values())

        if sort_by == "pnl":
            whales.sort(key=lambda w: w.total_unrealized_pnl, reverse=True)
        elif sort_by == "positions":
            whales.sort(key=lambda w: w.position_count, reverse=True)
        elif sort_by == "label":
            whales.sort(key=lambda w: w.label or "zzz")
        else:
            whales.sort(key=lambda w: w.account_value, reverse=True)

        return whales[offset:offset + limit]

    def get_survival_stats(self) -> WhaleSurvivalStats:
        """
        Get survival statistics for all tracked whales.
        The '75% blow up' metric from MoonDev's research.
        """
        whales = list(self._whales.values())
        total = len(whales)

        if total == 0:
            return WhaleSurvivalStats(
                total_tracked=0, still_active=0, blown_up=0,
                survival_rate=0, blow_up_rate=0, total_whale_equity=0,
                avg_account_value=0, median_account_value=0,
                top_whale_value=0, total_positions=0,
            )

        active = [w for w in whales if w.account_value > self.blown_up_threshold]
        blown = [w for w in whales if w.is_blown_up]

        values = sorted([w.account_value for w in whales])
        median_val = values[len(values) // 2] if values else 0
        total_equity = sum(w.account_value for w in whales)
        total_positions = sum(w.position_count for w in whales)

        return WhaleSurvivalStats(
            total_tracked=total,
            still_active=len(active),
            blown_up=len(blown),
            survival_rate=round(len(active) / total * 100, 1) if total > 0 else 0,
            blow_up_rate=round(len(blown) / total * 100, 1) if total > 0 else 0,
            total_whale_equity=round(total_equity, 2),
            avg_account_value=round(total_equity / total, 2) if total > 0 else 0,
            median_account_value=round(median_val, 2),
            top_whale_value=round(max(values) if values else 0, 2),
            total_positions=total_positions,
        )

    def get_recent_alerts(
        self, minutes: int = 60, address: Optional[str] = None, limit: int = 100
    ) -> List[WhaleTradeAlert]:
        """Get recent whale trade alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        alerts = [a for a in self._alerts if a.timestamp > cutoff]
        if address:
            alerts = [a for a in alerts if a.address == address]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    # ──────────────────────────────────────────────
    # Background Polling
    # ──────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("WhaleTracker started")

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("WhaleTracker stopped")

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.scan_all()
            except Exception as e:
                logger.error("Whale poll error: %s", e)
            await asyncio.sleep(self.poll_interval)
