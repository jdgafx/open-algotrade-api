"""
Risk Controller — Layer 0: The Seatbelt (Guardian).

Always-on position monitoring service that protects traders from blowing up.
This is the MOST IMPORTANT feature of the entire platform.

Monitors:
- Daily P&L vs max daily loss threshold
- Per-position leverage limits
- Margin usage limits
- Global stop loss / take profit
- Trailing stops
- Anti-tilt lockout
- Liquidation cascade detection

Actions:
- Auto-close positions when limits breached
- Auto-withdraw to wallet on max daily loss
- Lock trading after tilt detection
- Log every risk action for audit trail
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from src.services.risk_models import (
    RiskConfig,
    RiskEvent,
    RiskEventType,
    RiskSeverity,
    RiskSnapshot,
    RiskStatus,
)

logger = logging.getLogger(__name__)


class RiskController:
    """
    Always-on risk monitoring service.

    Connects to HyperLiquid via the existing client and executor.
    Runs a monitoring loop checking positions every 1-5 seconds.
    """

    def __init__(
        self,
        client: Any,  # HyperliquidClient
        executor: Any,  # HyperliquidVaultExecutor
        config: Optional[RiskConfig] = None,
    ):
        self.client = client
        self.executor = executor
        self.config = config or RiskConfig()

        # State
        self._status = RiskStatus.INACTIVE
        self._task: Optional[asyncio.Task] = None
        self._events: List[RiskEvent] = []
        self._event_counter = 0
        self._start_of_day_equity: float = 0.0
        self._day_start: Optional[datetime] = None
        self._lockout_until: Optional[datetime] = None
        self._trailing_stops: Dict[str, float] = {}  # symbol -> highest/lowest mark
        self._last_check: Optional[datetime] = None
        self._subscribers: List[Callable] = []

        logger.info("RiskController initialized | config=%s", self.config.model_dump())

    @property
    def status(self) -> RiskStatus:
        return self._status

    @property
    def is_active(self) -> bool:
        return self._status in (RiskStatus.MONITORING, RiskStatus.LOCKED_OUT)

    def update_config(self, config: RiskConfig) -> None:
        self.config = config
        self._log_event(
            RiskEventType.CONFIG_UPDATED,
            RiskSeverity.INFO,
            "Risk configuration updated",
            details=config.model_dump(),
        )
        logger.info("RiskController config updated")

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to risk events (for WebSocket broadcasting)."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        self._subscribers = [s for s in self._subscribers if s is not callback]

    async def start(self) -> None:
        """Start the risk monitoring loop."""
        if self._task and not self._task.done():
            logger.warning("RiskController already running")
            return

        # Initialize start-of-day equity
        try:
            self._start_of_day_equity = await self.executor.get_account_value()
        except Exception as e:
            logger.warning("Could not get initial account value: %s", e)
            self._start_of_day_equity = 0.0

        self._day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._status = RiskStatus.MONITORING
        self._task = asyncio.create_task(self._monitoring_loop())

        self._log_event(
            RiskEventType.MONITORING_STARTED,
            RiskSeverity.INFO,
            f"Risk monitoring started | SOD equity=${self._start_of_day_equity:.2f}",
            details={
                "start_of_day_equity": self._start_of_day_equity,
                "config": self.config.model_dump(),
            },
        )
        logger.info(
            "RiskController STARTED | SOD=$%.2f | interval=%.1fs",
            self._start_of_day_equity,
            self.config.check_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the risk monitoring loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._status = RiskStatus.INACTIVE
        self._log_event(
            RiskEventType.MONITORING_STOPPED,
            RiskSeverity.INFO,
            "Risk monitoring stopped",
        )
        logger.info("RiskController STOPPED")

    def get_snapshot(self) -> RiskSnapshot:
        """Get current risk state snapshot."""
        return RiskSnapshot(
            status=self._status,
            account_value=self._start_of_day_equity,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            start_of_day_equity=self._start_of_day_equity,
            total_margin_used=0.0,
            margin_usage_pct=0.0,
            max_position_leverage=0.0,
            open_positions=0,
            lockout_until=self._lockout_until,
            trailing_stops=dict(self._trailing_stops),
            last_check=self._last_check,
        )

    def get_events(self, limit: int = 100) -> List[RiskEvent]:
        """Get recent risk events."""
        return list(reversed(self._events[-limit:]))

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop — runs every check_interval_seconds."""
        logger.info("Risk monitoring loop started")

        while True:
            try:
                await self._check_cycle()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Risk monitoring loop cancelled")
                break
            except Exception as e:
                logger.error("Risk monitoring error: %s", e, exc_info=True)
                await asyncio.sleep(self.config.check_interval_seconds)

    async def _check_cycle(self) -> None:
        """Single check cycle — the core of risk monitoring."""
        now = datetime.now(timezone.utc)
        self._last_check = now

        # Reset SOD equity at midnight UTC
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._day_start and today_start > self._day_start:
            self._day_start = today_start
            try:
                self._start_of_day_equity = await self.executor.get_account_value()
            except Exception:
                pass
            self._trailing_stops.clear()
            logger.info("Day reset | new SOD=$%.2f", self._start_of_day_equity)

        # Check lockout
        if self._lockout_until and now < self._lockout_until:
            return  # Still locked out, skip all checks

        if self._lockout_until and now >= self._lockout_until:
            self._lockout_until = None
            self._status = RiskStatus.MONITORING
            self._log_event(
                RiskEventType.TILT_LOCKOUT,
                RiskSeverity.INFO,
                "Anti-tilt lockout expired. Trading re-enabled.",
            )

        # Get current state
        try:
            account_value = await self.executor.get_account_value()
            positions = await self.executor.get_all_positions()
        except Exception as e:
            logger.warning("Failed to get account state: %s", e)
            return

        # Build snapshot
        daily_pnl = account_value - self._start_of_day_equity
        daily_pnl_pct = (
            (daily_pnl / self._start_of_day_equity * 100)
            if self._start_of_day_equity > 0
            else 0.0
        )

        total_margin = 0.0
        max_leverage = 0.0
        for pos in positions:
            pos_margin = abs(pos.get("entry_px", 0) * pos.get("size", 0))
            lev = pos.get("leverage", 1.0)
            total_margin += pos_margin / lev if lev > 0 else pos_margin
            max_leverage = max(max_leverage, lev)

        margin_pct = (
            (total_margin / account_value * 100) if account_value > 0 else 0.0
        )

        # Update snapshot and broadcast
        snapshot = RiskSnapshot(
            status=self._status,
            account_value=account_value,
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl_pct, 2),
            start_of_day_equity=self._start_of_day_equity,
            total_margin_used=round(total_margin, 2),
            margin_usage_pct=round(margin_pct, 2),
            max_position_leverage=max_leverage,
            open_positions=len(positions),
            lockout_until=self._lockout_until,
            trailing_stops=dict(self._trailing_stops),
            last_check=self._last_check,
        )
        await self._broadcast(snapshot)

        # === RISK CHECKS (ordered by severity) ===

        # 1. MAX DAILY LOSS — the big one
        if (
            self._start_of_day_equity > 0
            and daily_pnl_pct <= -self.config.max_daily_loss_pct
        ):
            await self._handle_max_daily_loss(account_value, daily_pnl, daily_pnl_pct)
            return  # Everything closed, no further checks needed

        # 2. Per-position checks
        for pos in positions:
            symbol = pos.get("coin", "")
            size = pos.get("size", 0)
            entry_px = pos.get("entry_px", 0)
            leverage = pos.get("leverage", 1.0)
            pnl_pct = pos.get("return_on_equity", 0.0)
            is_long = pos.get("long", True)

            # 2a. LEVERAGE CHECK
            if leverage > self.config.max_leverage:
                await self._handle_leverage_exceeded(symbol, leverage)

            # 2b. MARGIN CHECK
            pos_margin = abs(entry_px * size) / leverage if leverage > 0 else abs(entry_px * size)
            pos_margin_pct = (pos_margin / account_value * 100) if account_value > 0 else 0.0
            if pos_margin_pct > self.config.max_margin_pct:
                await self._handle_margin_exceeded(symbol, pos_margin_pct)

            # 2c. STOP LOSS
            if pnl_pct <= -self.config.stop_loss_pct:
                await self._handle_stop_loss(symbol, pnl_pct)
                continue

            # 2d. TAKE PROFIT
            if pnl_pct >= self.config.take_profit_pct:
                await self._handle_take_profit(symbol, pnl_pct)
                continue

            # 2e. TRAILING STOP
            if self.config.trailing_stop_pct is not None and self.config.trailing_stop_pct > 0:
                await self._handle_trailing_stop(symbol, pnl_pct, is_long)

    # ──────────────────────────────────────────────
    # Risk Action Handlers
    # ──────────────────────────────────────────────

    async def _handle_max_daily_loss(
        self, account_value: float, daily_pnl: float, daily_pnl_pct: float
    ) -> None:
        """Max daily loss hit — close everything, withdraw, lockout."""
        logger.critical(
            "MAX DAILY LOSS HIT | pnl=$%.2f (%.2f%%) | CLOSING ALL POSITIONS",
            daily_pnl,
            daily_pnl_pct,
        )

        self._status = RiskStatus.EMERGENCY

        # Close all positions
        try:
            await self.executor.emergency_close_all()
        except Exception as e:
            logger.error("Emergency close failed: %s", e)

        self._log_event(
            RiskEventType.MAX_LOSS_HIT,
            RiskSeverity.CRITICAL,
            f"Max daily loss hit: ${daily_pnl:.2f} ({daily_pnl_pct:.2f}%). "
            f"All positions closed.",
            details={
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct,
                "account_value": account_value,
                "threshold": self.config.max_daily_loss_pct,
            },
        )

        # Auto-withdrawal
        if self.config.auto_withdraw_on_max_loss:
            await self._auto_withdraw(account_value)

        # Anti-tilt lockout
        if self.config.anti_tilt_lockout_hours > 0:
            self._lockout_until = datetime.now(timezone.utc) + timedelta(
                hours=self.config.anti_tilt_lockout_hours
            )
            self._status = RiskStatus.LOCKED_OUT

            self._log_event(
                RiskEventType.TILT_LOCKOUT,
                RiskSeverity.CRITICAL,
                f"Anti-tilt lockout activated for {self.config.anti_tilt_lockout_hours}h. "
                f"Trading locked until {self._lockout_until.isoformat()}",
                details={"lockout_until": self._lockout_until.isoformat()},
            )

    async def _handle_leverage_exceeded(self, symbol: str, leverage: float) -> None:
        """Position leverage exceeds max — close position."""
        logger.warning(
            "LEVERAGE EXCEEDED | %s | leverage=%.1fx > max=%.1fx | CLOSING",
            symbol,
            leverage,
            self.config.max_leverage,
        )

        try:
            positions = await self.executor.get_all_positions()
            for pos in positions:
                if pos.get("coin") == symbol:
                    await asyncio.to_thread(
                        self.client.kill_switch, symbol, 3, self.client.vault_address
                    )
                    break
        except Exception as e:
            logger.error("Failed to close overleveraged position %s: %s", symbol, e)

        self._log_event(
            RiskEventType.LEVERAGE_EXCEEDED,
            RiskSeverity.WARNING,
            f"Leverage exceeded on {symbol}: {leverage:.1f}x > {self.config.max_leverage:.1f}x max. Position closed.",
            details={"symbol": symbol, "leverage": leverage, "max": self.config.max_leverage},
        )

    async def _handle_margin_exceeded(self, symbol: str, margin_pct: float) -> None:
        """Position margin exceeds max % of account — close position."""
        logger.warning(
            "MARGIN EXCEEDED | %s | margin=%.1f%% > max=%.1f%% | CLOSING",
            symbol,
            margin_pct,
            self.config.max_margin_pct,
        )

        try:
            await asyncio.to_thread(
                self.client.kill_switch, symbol, 3, self.client.vault_address
            )
        except Exception as e:
            logger.error("Failed to close over-margin position %s: %s", symbol, e)

        self._log_event(
            RiskEventType.MARGIN_EXCEEDED,
            RiskSeverity.WARNING,
            f"Margin exceeded on {symbol}: {margin_pct:.1f}% > {self.config.max_margin_pct:.1f}% max. Position closed.",
            details={"symbol": symbol, "margin_pct": margin_pct, "max": self.config.max_margin_pct},
        )

    async def _handle_stop_loss(self, symbol: str, pnl_pct: float) -> None:
        """Global stop loss hit on position."""
        logger.warning(
            "STOP LOSS HIT | %s | pnl=%.2f%% | threshold=%.2f%%",
            symbol,
            pnl_pct,
            -self.config.stop_loss_pct,
        )

        try:
            await asyncio.to_thread(
                self.client.kill_switch, symbol, 3, self.client.vault_address
            )
        except Exception as e:
            logger.error("Stop loss execution failed for %s: %s", symbol, e)

        self._log_event(
            RiskEventType.STOP_LOSS_HIT,
            RiskSeverity.WARNING,
            f"Stop loss hit on {symbol}: {pnl_pct:.2f}% (threshold: -{self.config.stop_loss_pct:.1f}%)",
            details={"symbol": symbol, "pnl_pct": pnl_pct, "threshold": self.config.stop_loss_pct},
        )

    async def _handle_take_profit(self, symbol: str, pnl_pct: float) -> None:
        """Global take profit hit on position."""
        logger.info(
            "TAKE PROFIT HIT | %s | pnl=%.2f%% | threshold=%.2f%%",
            symbol,
            pnl_pct,
            self.config.take_profit_pct,
        )

        try:
            await asyncio.to_thread(
                self.client.kill_switch, symbol, 3, self.client.vault_address
            )
        except Exception as e:
            logger.error("Take profit execution failed for %s: %s", symbol, e)

        self._log_event(
            RiskEventType.TAKE_PROFIT_HIT,
            RiskSeverity.INFO,
            f"Take profit hit on {symbol}: {pnl_pct:.2f}% (threshold: {self.config.take_profit_pct:.1f}%)",
            details={"symbol": symbol, "pnl_pct": pnl_pct, "threshold": self.config.take_profit_pct},
        )

    async def _handle_trailing_stop(
        self, symbol: str, pnl_pct: float, is_long: bool
    ) -> None:
        """Trailing stop logic — tracks peak PnL and triggers if drawdown exceeds threshold."""
        trail_pct = self.config.trailing_stop_pct
        if trail_pct is None:
            return

        peak = self._trailing_stops.get(symbol, pnl_pct)

        # Update peak
        if is_long:
            if pnl_pct > peak:
                self._trailing_stops[symbol] = pnl_pct
                peak = pnl_pct
        else:
            # For shorts, "peak" is most negative (best short PnL is most positive)
            if pnl_pct > peak:
                self._trailing_stops[symbol] = pnl_pct
                peak = pnl_pct

        # Only trigger trailing stop if we've been profitable
        if peak > 0 and (peak - pnl_pct) >= trail_pct:
            logger.info(
                "TRAILING STOP HIT | %s | peak=%.2f%% current=%.2f%% trail=%.2f%%",
                symbol,
                peak,
                pnl_pct,
                trail_pct,
            )

            try:
                await asyncio.to_thread(
                    self.client.kill_switch, symbol, 3, self.client.vault_address
                )
                self._trailing_stops.pop(symbol, None)
            except Exception as e:
                logger.error("Trailing stop execution failed for %s: %s", symbol, e)

            self._log_event(
                RiskEventType.TRAILING_STOP_HIT,
                RiskSeverity.INFO,
                f"Trailing stop hit on {symbol}: peak={peak:.2f}%, current={pnl_pct:.2f}%, trail={trail_pct:.1f}%",
                details={
                    "symbol": symbol,
                    "peak_pnl_pct": peak,
                    "current_pnl_pct": pnl_pct,
                    "trail_pct": trail_pct,
                },
            )

    async def _auto_withdraw(self, account_value: float) -> None:
        """Auto-withdraw available balance to wallet."""
        try:
            withdrawable = await asyncio.to_thread(self.client.get_withdrawable)
            if withdrawable <= 0:
                logger.info("No withdrawable balance for auto-withdrawal")
                return

            # Use the exchange's withdraw method
            # HyperLiquid withdraw: withdraw USDC to the connected wallet
            result = await asyncio.to_thread(
                self.client.exchange.usd_class_transfer,
                int(withdrawable),  # amount in USD (integer)
                False,  # toPerp=False means withdraw from perp to spot
            )

            self._log_event(
                RiskEventType.AUTO_WITHDRAWAL,
                RiskSeverity.CRITICAL,
                f"Auto-withdrawal triggered: ${withdrawable:.2f} sent to wallet",
                details={
                    "amount": withdrawable,
                    "account_value_before": account_value,
                    "result": str(result),
                },
            )
            logger.info("Auto-withdrawal: $%.2f to wallet", withdrawable)

        except Exception as e:
            logger.error("Auto-withdrawal failed: %s", e)
            self._log_event(
                RiskEventType.AUTO_WITHDRAWAL,
                RiskSeverity.CRITICAL,
                f"Auto-withdrawal FAILED: {e}",
                details={"error": str(e)},
            )

    # ──────────────────────────────────────────────
    # Event Management
    # ──────────────────────────────────────────────

    def _log_event(
        self,
        event_type: RiskEventType,
        severity: RiskSeverity,
        message: str,
        details: Optional[dict] = None,
    ) -> RiskEvent:
        """Log a risk event and notify subscribers."""
        self._event_counter += 1
        event = RiskEvent(
            id=self._event_counter,
            event_type=event_type,
            severity=severity,
            message=message,
            details=details,
        )
        self._events.append(event)

        # Keep max 1000 events in memory
        if len(self._events) > 1000:
            self._events = self._events[-500:]

        # Notify subscribers synchronously (they can queue for async delivery)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as e:
                logger.warning("Subscriber notification failed: %s", e)

        return event

    async def _broadcast(self, snapshot: RiskSnapshot) -> None:
        """Broadcast snapshot to WebSocket subscribers."""
        # This is called by the monitoring loop and picked up by the
        # WebSocket handler in the risk routes
        pass
