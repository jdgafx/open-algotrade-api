"""
Consecutive Down Bars Mean-Reversion Strategy.

MoonDev sources:
  - "Claude Code Just Made Finding Winning Trading Strategies EASY" (RkKB725yyn4)
  - "The $500,000,000 Trading Bot" (U4hzjr1YGjs)

Logic:
  Count N consecutive bearish bars (close < open). After N bars downward, the
  selling pressure is exhausted → buy the bounce. Optional VWAP filter and RSI
  max cap to avoid chasing into strong downtrends.

Edge: Short-term mean reversion after a concentrated sell sequence. The longer
the streak, the higher the probability of a bounce in crypto (momentum overshoots
then snaps back). Works across all liquid crypto perp pairs on 1h–4h timeframes.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ConsecutiveDownStrategy(BaseStrategy):
    """
    Long-only mean-reversion on N consecutive bearish candles.

    State: IDLE → ENTERED (LONG after streak) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _count_consecutive_down(self, df: pd.DataFrame) -> int:
        """Count how many trailing bars are bearish (close < open)."""
        opens = df["open"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        count = 0
        for i in range(len(df) - 1, -1, -1):
            if closes[i] < opens[i]:
                count += 1
            else:
                break
        return count

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        """Session VWAP = sum(typical_price × volume) / sum(volume)."""
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)
        vols = df["volume"].to_numpy(float)
        tp = (highs + lows + closes) / 3.0
        total_vol = vols.sum()
        if total_vol <= 0:
            return float(closes[-1])
        return float((tp * vols).sum() / total_vol)

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        n = len(closes)
        if n < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        return 100.0 - (100.0 / (1.0 + rs))

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        if self._entry_bar is not None:
            return None  # already in position

        p = self.config.params
        rsi_period = int(p.get("rsi_period", 14))
        n_down = int(p.get("n_down_bars", 4))
        min_bars = max(n_down + 5, rsi_period + 2)

        if len(data) < min_bars:
            return None

        streak = self._count_consecutive_down(data)
        if streak < n_down:
            return None

        rsi_max = float(p.get("rsi_max", 45.0))
        closes = data["close"].to_numpy(float)
        rsi_val = self._compute_rsi(closes, rsi_period)
        if rsi_val > rsi_max:
            return None

        use_vwap = bool(p.get("use_vwap_filter", True))
        current_close = float(data["close"].iloc[-1])
        if use_vwap:
            vwap = self._compute_vwap(data)
            if current_close >= vwap:
                return None  # price above VWAP — not oversold enough

        # Strength: deeper streak and more oversold → higher conviction
        streak_score = min((streak - n_down) / max(n_down, 1), 1.0)
        rsi_depth = max(0.0, rsi_max - rsi_val) / rsi_max
        strength = min(0.58 + streak_score * 0.15 + rsi_depth * 0.20, 0.90)

        self._entry_bar = self._bar_count - 1

        logger.info(
            "[ConsecDown] LONG | price=%.4f streak=%d rsi=%.1f strength=%.2f",
            current_close, streak, rsi_val, strength,
        )
        return Signal(
            signal_type=SignalType.LONG,
            symbol=self.config.symbol,
            price=current_close,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"ConsecDown LONG: {streak} red bars, rsi={rsi_val:.1f}",
            metadata={"streak": streak, "rsi": rsi_val, "strategy": "consecutive_down"},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 8))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"ConsecDown hold cap: {bars_held} bars",
            )
        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"ConsecDown TP: {pnl_pct:.1f}%",
            )
        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"ConsecDown SL: {pnl_pct:.1f}%",
            )
        return None
