"""
Donchian Channel Breakout Strategy.

MoonDev source: "This New Claude Agent Just Changed Trading Forever (182%)" (sJfwegO57U4).
Claimed metrics: 563% return (top of 100-strategy sweep, BTC/ETH futures).

Logic:
1. Compute the N-bar Donchian channel: upper = max(high[-N:-1]), lower = min(low[-N:-1])
2. Enter LONG when current close breaks above the upper channel (new N-bar high).
3. Exit on: profit target (3%), stop loss (1.5%), or hold_cap_bars (12 bars).

Edge: Channel breakouts capture the start of sustained trending moves. When price
prints a new N-bar high it signals that buyers have overwhelmed prior resistance —
a classic trend-following entry with strong momentum backing. Top performer in
MoonDev's automated 100-strategy competitive sweep.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class DonchianChannelStrategy(BaseStrategy):
    """
    Long-only Donchian channel breakout strategy.

    State machine: IDLE → ENTERED (LONG on upper-channel break) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0
        self._entry_price: Optional[float] = None

    def _channel_bounds(self, df: pd.DataFrame, period: int) -> tuple[float, float]:
        """Return (upper, lower) Donchian channel using prior N bars (excludes last)."""
        window = df.iloc[-period - 1:-1] if len(df) > period else df.iloc[:-1]
        upper = float(window["high"].max())
        lower = float(window["low"].min())
        return upper, lower

    def _compute_atr(self, df: pd.DataFrame, period: int) -> float:
        """Average True Range over the last N bars."""
        n = min(len(df), period + 1)
        highs = df["high"].to_numpy(float)[-n:]
        lows = df["low"].to_numpy(float)[-n:]
        closes = df["close"].to_numpy(float)[-n:]
        trs = []
        for i in range(1, len(highs)):
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
        return float(np.mean(trs)) if trs else 0.0

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        p = self.config.params
        channel_period = int(p.get("channel_period", 20))
        min_bars = channel_period + 2

        if len(data) < min_bars:
            return None

        upper, lower = self._channel_bounds(data, channel_period)
        current_close = float(data["close"].iloc[-1])
        current_high = float(data["high"].iloc[-1])

        # Breakout: close AND high both exceed upper channel
        if current_close <= upper or current_high <= upper:
            return None

        # Momentum confirmation: current bar must be bullish
        current_open = float(data["open"].iloc[-1])
        if current_close <= current_open:
            return None

        # Strength based on how far above the channel we broke
        breakout_pct = (current_close - upper) / upper if upper > 0 else 0.0
        atr = self._compute_atr(data, int(p.get("atr_period", 14)))
        atr_pct = atr / current_close if current_close > 0 else 0.01
        # Stronger breakout relative to ATR = higher conviction
        strength = min(0.60 + min(breakout_pct / atr_pct, 2.0) * 0.15, 0.92)

        self._entry_bar = self._bar_count - 1
        self._entry_price = current_close

        logger.info("[Donchian] LONG | price=%.4f upper=%.4f breakout=%.3f%% strength=%.2f",
                    current_close, upper, breakout_pct * 100, strength)

        return Signal(
            signal_type=SignalType.LONG,
            symbol=self.config.symbol,
            price=current_close,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"Donchian LONG: {channel_period}-bar breakout above {upper:.2f}",
            metadata={"upper_channel": upper, "lower_channel": lower,
                      "breakout_pct": breakout_pct, "strategy": "donchian_channel"},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 12))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            self._entry_price = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Donchian hold cap: {bars_held} bars",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            self._entry_price = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Donchian TP: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            self._entry_price = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Donchian SL: {pnl_pct:.1f}%",
            )

        return None
