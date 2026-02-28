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
from typing import Any, Dict, List, Optional

import pandas as pd


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
    target_pct: float = 5.0
    max_loss_pct: float = -10.0
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
    # Anti-overtrading state
    last_trade_close_time: Optional[datetime] = None
    trades_this_hour: int = 0
    hour_start: Optional[datetime] = None
    entry_bar_count: int = 0


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

                # If we got an exit signal, record the close timestamp
                if signal and signal.signal_type != SignalType.NONE:
                    self.state.last_trade_close_time = now
                    self.state.trades_this_hour += 1
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

    def record_trade(self, pnl: float) -> None:
        now = datetime.now(timezone.utc)
        self.state.total_trades += 1
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1

        if self.state.total_pnl > self.state.peak_pnl:
            self.state.peak_pnl = self.state.total_pnl

        drawdown = self.state.peak_pnl - self.state.total_pnl
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown

        # Anti-overtrading: update close time and hourly counter
        self.state.last_trade_close_time = now
        if self.state.hour_start is None:
            self.state.hour_start = now
        elif (now - self.state.hour_start).total_seconds() >= 3600:
            self.state.trades_this_hour = 1
            self.state.hour_start = now
        else:
            self.state.trades_this_hour += 1

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
