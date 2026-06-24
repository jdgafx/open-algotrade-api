"""
BaseStrategy ABC — The common interface all Open Algotrade strategies implement.

Every strategy in the system (all 36+ from ALGOS/) gets ported to this interface.
The orchestrator runs strategies through this uniform lifecycle.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class PositionDict(TypedDict, total=False):
    """Shape of the per-strategy position dict that ``should_exit`` consumes.

    The canonical producer is ``PaperTradingExecutor.get_position``; the
    circuit-breaker shadow evaluator (``ShadowRecoveryEvaluator.synthetic_position``)
    builds the same shape for halted-strategy recovery eval. Declaring it once keeps
    the two producers from silently drifting. ``total=False`` because some failure
    branches omit ``mark_price``/``unrealized_pnl``."""
    symbol: str
    size: float
    entry_px: float
    mark_price: Optional[float]
    pnl_perc: float
    unrealized_pnl: float
    is_long: bool
    side: str


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    CLOSE_ALL = "close_all"
    NONE = "none"


class StrategyTier(Enum):
    A = "hl_native"
    B = "bonus_algos"
    C = "bootcamp_bots"
    D = "backtested"
    E = "ai_pipeline"
    F = "infrastructure"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    strength: float = 1.0
    price: Optional[float] = None
    size_usd: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StrategyConfig:
    name: str
    symbol: str
    tier: StrategyTier = StrategyTier.A
    timeframe: str = "1h"
    leverage: int = 3
    size_usd: float = 50.0  # Reduced from $100 (MoonDev: "tiny size, tiny size" for forward testing)
    max_positions: int = 1
    target_pct: float = 9.0  # MoonDev uses 9% global target
    max_loss_pct: float = -8.0  # MoonDev uses -8% global stop
    lookback_days: int = 7
    interval_seconds: int = 30
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    is_running: bool = False
    iterations: int = 0
    last_signal: Optional[Signal] = None
    last_iteration: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    errors: int = 0
    consecutive_errors: int = 0
    start_time: Optional[datetime] = None
    # Circuit breaker state
    consecutive_losses: int = 0
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason: str = ""
    # Anti-overtrading state
    last_trade_close_time: Optional[datetime] = None
    trades_this_hour: int = 0
    hour_start: Optional[datetime] = None
    entry_bar_count: int = 0

    def reset_circuit_breaker(self) -> None:
        """Clear the FULL circuit-breaker state so a halted strategy returns to a
        genuinely clean slate.

        Single source of truth for "what a clean breaker state is" — shared by the
        manual reset endpoint and the condition-aware auto-recovery path so the two
        can never drift. Must clear every input the breaker trips on
        (``_apply_trade_accounting``): the consecutive-loss counter AND the drawdown
        inputs. ``max_drawdown`` is derived as ``peak_pnl - total_pnl``, so leaving
        ``peak_pnl`` stale would immediately recompute a large drawdown and re-trip the
        breaker on the first losing trade — the exact bug this fixes. ``total_pnl`` and
        the trade counters are deliberately preserved: a reset clears the *halt*, not
        the strategy's realized track record.
        """
        self.circuit_breaker_triggered = False
        self.circuit_breaker_reason = ""
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_pnl = self.total_pnl


class BaseStrategy(ABC):
    """
    All 36+ paid strategies from ALGOS/ implement this interface.

    Lifecycle: on_start() → [run_iteration() loop] → on_stop()
    Each iteration: get data → should_enter()/should_exit() → emit Signal
    The executor translates Signals into HyperliquidClient calls.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.state = StrategyState()
        self._logger = logging.getLogger(f"strategy.{config.name}")

    @abstractmethod
    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        """Analyze market data and return an entry signal, or None."""
        ...

    @abstractmethod
    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        """Analyze market data + current position, return exit signal or None."""
        ...

    def _risk_reflex_exit(self, position: Dict[str, Any]) -> Optional[Signal]:
        """State-independent stop-loss / take-profit from the live position's PnL and
        this config's ``max_loss_pct`` / ``target_pct``.

        Strategies that track entry state internally (e.g. ``_entry_bar``) and early-return
        None from ``should_exit`` should call this FIRST, so an open position is still
        protected when that internal state is unavailable — most importantly after a process
        restart, where the strategy has no record of the entry but the executor still holds a
        real position. Returns a CLOSE signal, or None when within the configured band.
        """
        pnl_perc = position.get("pnl_perc")
        if pnl_perc is None:
            return None
        is_long = position.get("is_long", position.get("size", 0) > 0)
        close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
        if pnl_perc <= self.config.max_loss_pct:
            return Signal(
                signal_type=close_type, symbol=self.config.symbol,
                reason=f"max-loss reflex {pnl_perc:.1f}% <= {self.config.max_loss_pct:.1f}%",
            )
        if pnl_perc >= self.config.target_pct:
            return Signal(
                signal_type=close_type, symbol=self.config.symbol,
                reason=f"target reflex {pnl_perc:.1f}% >= {self.config.target_pct:.1f}%",
            )
        return None

    async def on_start(self) -> None:
        self.state.is_running = True
        self.state.start_time = datetime.now(timezone.utc)
        self._logger.info("Started | %s on %s", self.config.name, self.config.symbol)

    async def on_stop(self) -> None:
        self.state.is_running = False
        self._logger.info(
            "Stopped | %s | trades=%d pnl=%.2f win_rate=%.1f%%",
            self.config.name,
            self.state.total_trades,
            self.state.total_pnl,
            self.win_rate,
        )

    async def on_error(self, error: Exception) -> None:
        self.state.errors += 1
        self.state.consecutive_errors += 1
        self._logger.error(
            "Error | %s | %s: %s", self.config.name, type(error).__name__, error
        )

    async def run_iteration(
        self, data: pd.DataFrame, position: Optional[Dict[str, Any]] = None
    ) -> Optional[Signal]:
        self.state.iterations += 1
        self.state.last_iteration = datetime.now(timezone.utc)
        now = datetime.now(timezone.utc)

        # ── Circuit breaker check ──
        if self.state.circuit_breaker_triggered:
            self._logger.warning(
                "HALTED by circuit breaker | %s | reason: %s",
                self.config.name,
                self.state.circuit_breaker_reason,
            )
            return None

        # ── Anti-overtrading parameters (configurable per strategy) ──
        min_hold_bars = self.config.params.get("min_hold_bars", 3)
        cooldown_seconds = self.config.params.get("cooldown_seconds", 300)
        max_trades_per_hour = self.config.params.get("max_trades_per_hour", 4)
        min_signal_strength = self.config.params.get("min_signal_strength", 0.5)

        # ── Rolling hourly trade counter ──
        if self.state.hour_start is None:
            self.state.hour_start = now
        elif (now - self.state.hour_start).total_seconds() >= 3600:
            self.state.trades_this_hour = 0
            self.state.hour_start = now

        try:
            has_position = position and position.get("size", 0) != 0

            if has_position:
                # Track how many bars we've been in this position
                self.state.entry_bar_count += 1

                # Check if min hold period has elapsed (unless stop-loss)
                pnl_pct = position.get("pnl_perc", 0)
                is_stop_loss = pnl_pct <= self.config.max_loss_pct

                if self.state.entry_bar_count < min_hold_bars and not is_stop_loss:
                    self._logger.debug(
                        "Hold | %s | bar %d/%d (min hold not met)",
                        self.config.name,
                        self.state.entry_bar_count,
                        min_hold_bars,
                    )
                    return None

                signal = await self.should_exit(data, position)

                # If we got an exit signal, reset bar count
                # (trade counting is handled by record_trade() called from executor)
                if signal and signal.signal_type != SignalType.NONE:
                    self.state.entry_bar_count = 0
            else:
                # Reset bar count when flat
                self.state.entry_bar_count = 0

                # ── Cooldown check: must wait after closing a trade ──
                if self.state.last_trade_close_time is not None:
                    elapsed = (now - self.state.last_trade_close_time).total_seconds()
                    if elapsed < cooldown_seconds:
                        self._logger.debug(
                            "Cooldown | %s | %.0fs / %ds remaining",
                            self.config.name,
                            cooldown_seconds - elapsed,
                            cooldown_seconds,
                        )
                        return None

                # ── Max trades per hour check ──
                if self.state.trades_this_hour >= max_trades_per_hour:
                    self._logger.debug(
                        "Rate limit | %s | %d/%d trades this hour",
                        self.config.name,
                        self.state.trades_this_hour,
                        max_trades_per_hour,
                    )
                    return None

                signal = await self.should_enter(data)

                # ── Minimum signal strength filter ──
                if signal and signal.signal_type != SignalType.NONE:
                    if signal.strength < min_signal_strength:
                        self._logger.debug(
                            "Weak signal filtered | %s | strength=%.2f < %.2f",
                            self.config.name,
                            signal.strength,
                            min_signal_strength,
                        )
                        signal = None
                    else:
                        # Count this entry toward hourly rate limit BEFORE execution
                        # so concurrent open positions count against max_trades_per_hour.
                        # (record_trade only fires on close, which would bypass the cap.)
                        self.state.trades_this_hour += 1

            if signal and signal.signal_type != SignalType.NONE:
                self.state.last_signal = signal
                self.state.consecutive_errors = 0
                self._logger.info(
                    "Signal | %s | %s %s | reason=%s",
                    self.config.name,
                    signal.signal_type.value,
                    signal.symbol,
                    signal.reason,
                )

            return signal

        except Exception as e:
            await self.on_error(e)
            return None

    def _apply_trade_accounting(self, pnl: float) -> None:
        """Update the durable trade counters + circuit-breaker state from one
        realized pnl.

        Shared by record_trade() (live exits) and restore_from_pnls() (redeploy
        rehydration). Deliberately excludes the wall-clock anti-overtrading state
        (last_trade_close_time / trades_this_hour / hour_start) so that replaying
        historical trades on boot does not impose a spurious cooldown or inflate
        the hourly trade counter.
        """
        self.state.total_trades += 1
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1

        if self.state.total_pnl > self.state.peak_pnl:
            self.state.peak_pnl = self.state.total_pnl

        drawdown = self.state.peak_pnl - self.state.total_pnl
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

        # ── Circuit breaker checks ──
        max_consecutive_losses = self.config.params.get("max_consecutive_losses", 5)
        max_strategy_drawdown = self.config.params.get("max_strategy_drawdown", 25.0)

        if self.state.consecutive_losses >= max_consecutive_losses:
            self.state.circuit_breaker_triggered = True
            self.state.circuit_breaker_reason = (
                f"{self.state.consecutive_losses} consecutive losses "
                f"(limit: {max_consecutive_losses})"
            )
            self._logger.critical(
                "CIRCUIT BREAKER TRIPPED | %s | %s",
                self.config.name,
                self.state.circuit_breaker_reason,
            )

        if self.state.max_drawdown >= max_strategy_drawdown:
            self.state.circuit_breaker_triggered = True
            self.state.circuit_breaker_reason = (
                f"max drawdown ${self.state.max_drawdown:.2f} "
                f"exceeds limit ${max_strategy_drawdown:.2f}"
            )
            self._logger.critical(
                "CIRCUIT BREAKER TRIPPED | %s | %s",
                self.config.name,
                self.state.circuit_breaker_reason,
            )

    def record_trade(self, pnl: float) -> None:
        now = datetime.now(timezone.utc)

        self._apply_trade_accounting(pnl)

        # Anti-overtrading: update close time and hourly counter
        self.state.last_trade_close_time = now
        if self.state.hour_start is None:
            self.state.hour_start = now
        elif (now - self.state.hour_start).total_seconds() >= 3600:
            self.state.trades_this_hour = 1
            self.state.hour_start = now
        else:
            self.state.trades_this_hour += 1

    def restore_from_pnls(self, pnls: List[float]) -> None:
        """Rebuild durable StrategyState (trade counters + circuit-breaker state)
        by replaying realized pnls in chronological order after a redeploy.

        Used by the paper executor on boot to restore per-strategy state — most
        importantly a tripped circuit breaker — that would otherwise reset to zero
        every redeploy. Excludes wall-clock anti-overtrading state by construction
        (see _apply_trade_accounting), so a restart imposes no spurious cooldown.
        """
        for pnl in pnls:
            self._apply_trade_accounting(pnl)

    @property
    def win_rate(self) -> float:
        if self.state.total_trades == 0:
            return 0.0
        return (self.state.winning_trades / self.state.total_trades) * 100

    @property
    def is_healthy(self) -> bool:
        max_consecutive = self.config.params.get("max_consecutive_errors", 10)
        return self.state.consecutive_errors < max_consecutive

    def get_stats(self) -> Dict[str, Any]:
        uptime = ""
        if self.state.start_time:
            delta = datetime.now(timezone.utc) - self.state.start_time
            hours = delta.total_seconds() / 3600
            uptime = f"{hours:.1f}h"

        return {
            "name": self.config.name,
            "symbol": self.config.symbol,
            "tier": self.config.tier.value,
            "is_running": self.state.is_running,
            "is_healthy": self.is_healthy,
            "iterations": self.state.iterations,
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "losing_trades": self.state.losing_trades,
            "win_rate": round(self.win_rate, 1),
            "total_pnl": round(self.state.total_pnl, 2),
            "max_drawdown": round(self.state.max_drawdown, 2),
            "errors": self.state.errors,
            "consecutive_losses": self.state.consecutive_losses,
            "circuit_breaker_triggered": self.state.circuit_breaker_triggered,
            "circuit_breaker_reason": self.state.circuit_breaker_reason,
            "uptime": uptime,
            "trades_this_hour": self.state.trades_this_hour,
            "entry_bar_count": self.state.entry_bar_count,
            "cooldown_active": (
                self.state.last_trade_close_time is not None
                and (datetime.now(timezone.utc) - self.state.last_trade_close_time).total_seconds()
                < self.config.params.get("cooldown_seconds", 300)
            ),
            "last_signal": (
                {
                    "type": self.state.last_signal.signal_type.value,
                    "reason": self.state.last_signal.reason,
                    "timestamp": self.state.last_signal.timestamp.isoformat(),
                }
                if self.state.last_signal
                else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.config.name!r} "
            f"symbol={self.config.symbol!r} "
            f"running={self.state.is_running}>"
        )
