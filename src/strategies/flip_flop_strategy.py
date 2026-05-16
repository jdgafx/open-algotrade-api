"""
Flip-Flop Regime Switcher — Always-in-market SuperTrend reversal strategy.

MoonDev source: "I Found a Strategy That Makes Buy & Hold Look Like a Joke (14x Better)"
(YouTube PvTKTQikbEY). Backtested: 529% return vs B&H 33%, WR 81%, Sortino 7.7.

Logic:
- Computes SuperTrend (Wilder ATR bands, period=10, multiplier=3.0 default).
- Flips LONG when SuperTrend direction turns UP, SHORT when it turns DOWN.
- Always in the market — each flip closes the prior leg and opens the next.
- Hard stops/targets act as safety nets; the trend flip is the primary exit.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class FlipFlopStrategy(BaseStrategy):

    def __init__(self, config: "StrategyConfig") -> None:
        super().__init__(config)
        self._entered_direction: int = 0  # 0=none, 1=long, -1=short

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
        n = len(df)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        prev_close = np.empty(n)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )

        # Wilder's smoothed ATR (RMA)
        atr = np.zeros(n)
        if n >= period:
            atr[period - 1] = tr[:period].mean()
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl2 = (high + low) / 2.0
        raw_upper = hl2 + multiplier * atr
        raw_lower = hl2 - multiplier * atr

        direction = np.zeros(n, dtype=int)
        upper = raw_upper.copy()
        lower = raw_lower.copy()

        for i in range(1, n):
            if atr[i] == 0:
                direction[i] = direction[i - 1]
                upper[i] = upper[i - 1]
                lower[i] = lower[i - 1]
                continue

            # Bands only tighten within the current trend leg
            if direction[i - 1] >= 0:
                lower[i] = max(raw_lower[i], lower[i - 1])
            if direction[i - 1] <= 0:
                upper[i] = min(raw_upper[i], upper[i - 1])

            if close[i] > upper[i - 1]:
                direction[i] = 1
            elif close[i] < lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        out = df.copy()
        out["st_direction"] = direction
        out["st_line"] = np.where(direction >= 0, lower, upper)
        return out

    def _compute(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        p = self.config.params
        period = int(p.get("atr_period", 10))
        multiplier = float(p.get("multiplier", 3.0))
        if len(data) < period + 4:
            return None
        return self._supertrend(data, period, multiplier)

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        df = self._compute(data)
        if df is None:
            return None

        curr_dir = int(df["st_direction"].iloc[-1])
        if curr_dir == 0:
            return None

        # Re-enter if direction changed since last entry OR no position yet.
        # This makes the strategy truly always-in-market: after a safety stop,
        # it re-enters on the next bar in whatever direction ST shows.
        if curr_dir == self._entered_direction:
            return None

        price = float(df["close"].iloc[-1])
        st_line = float(df["st_line"].iloc[-1])

        if curr_dir == 1:
            self._entered_direction = 1
            logger.info("[FlipFlop] LONG at %.4f (ST=%.4f)", price, st_line)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.85,
                reason=f"FlipFlop LONG: ST UP at {price:.4f}",
                metadata={"st_direction": curr_dir, "st_line": st_line},
            )

        self._entered_direction = -1
        logger.info("[FlipFlop] SHORT at %.4f (ST=%.4f)", price, st_line)
        return Signal(
            signal_type=SignalType.SHORT,
            symbol=self.config.symbol,
            price=price,
            size_usd=self.config.size_usd,
            strength=0.85,
            reason=f"FlipFlop SHORT: ST DOWN at {price:.4f}",
            metadata={"st_direction": curr_dir, "st_line": st_line},
        )

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        df = self._compute(data)
        if df is None:
            return None

        curr_dir = int(df["st_direction"].iloc[-1])
        prev_dir = int(df["st_direction"].iloc[-2])
        price = float(df["close"].iloc[-1])

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = float(position.get("pnl_perc", 0))

        if is_long and curr_dir == -1 and prev_dir == 1:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"FlipFlop exit LONG: ST flipped DOWN at {price:.4f}",
            )
        if not is_long and curr_dir == 1 and prev_dir == -1:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"FlipFlop exit SHORT: ST flipped UP at {price:.4f}",
            )

        # Safety net stops — reset direction so we can re-enter on the next bar
        if pnl_pct >= self.config.target_pct:
            self._entered_direction = 0
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"FlipFlop target: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            self._entered_direction = 0
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"FlipFlop stop: {pnl_pct:.1f}%",
            )

        return None
