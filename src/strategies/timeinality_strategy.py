"""
Timeinality Strategy — time-of-day + day-of-week short bias.

MoonDev source: "No One Trades Timeinality So I Built A Bot To Trade It" (5lbObSjHxIw).
Claimed metrics: 257% return short-only, 71% WR, max DD 23%, 52 weeks (SUI/crypto 1hr).

Logic:
1. Gate: current UTC hour must be in bear_hours AND weekday in bear_days.
2. Confirmation (optional): last candle must be bearish (close < open).
3. Always SHORT — the edge is that these time windows have persistent downward
   bias in crypto (Asian session wind-down + US open reversal + weekend drift).
4. Exit on: profit target, stop loss, OR hold_cap_bars exceeded.

Default bear windows (UTC):
  hours: 2-6 (Asian session ending, pre-European open weakness)
         14-15 (US market open often reverses crypto pump)
  days:  Monday (0), Sunday (6) — weekend hangover / start-of-week deleveraging
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class TimeinailityStrategy(BaseStrategy):
    """
    Short-only strategy gated purely on time-of-day and day-of-week seasonality.

    State machine: IDLE → ENTERED (SHORT on bear window) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _is_bear_time(self, now: datetime) -> bool:
        p = self.config.params
        bear_hours = list(p.get("bear_hours", [2, 3, 4, 5, 6, 14, 15]))
        bear_days = list(p.get("bear_days", [0, 6]))
        return now.hour in bear_hours and now.weekday() in bear_days

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        if len(data) < 3:
            return None

        now = datetime.now(timezone.utc)
        if not self._is_bear_time(now):
            return None

        p = self.config.params
        require_bearish = bool(p.get("require_bearish_candle", True))

        last_open = float(data["open"].iloc[-1])
        last_close = float(data["close"].iloc[-1])

        if require_bearish and last_close >= last_open:
            return None

        price = last_close
        if price <= 0:
            return None

        # Strength: base 0.65 + bonus for more extreme bear alignment
        bear_hours = list(p.get("bear_hours", [2, 3, 4, 5, 6, 14, 15]))
        hour_score = len(bear_hours) - bear_hours.index(now.hour) if now.hour in bear_hours else 0
        bearish_body = abs(last_close - last_open) / last_open if last_open > 0 else 0.0
        strength = min(0.65 + bearish_body * 5.0 + hour_score * 0.005, 0.92)

        self._entry_bar = self._bar_count - 1

        logger.info("[Timeinality] SHORT | price=%.4f hour=%d day=%d strength=%.2f",
                    price, now.hour, now.weekday(), strength)

        return Signal(
            signal_type=SignalType.SHORT,
            symbol=self.config.symbol,
            price=price,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"Timeinality SHORT: hour={now.hour} day={now.weekday()}",
            metadata={"hour": now.hour, "weekday": now.weekday()},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 4))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Timeinality hold cap: {bars_held} bars",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Timeinality TP: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Timeinality SL: {pnl_pct:.1f}%",
            )

        return None
