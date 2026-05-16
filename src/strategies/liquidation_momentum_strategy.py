"""
Pure Liquidation Momentum Strategy.

MoonDev source: "Last Trading Bot Video That Got Me Banned" (qsgtmPWdC2U).
Claimed metrics: unoptimized 200% return, optimized 500% / 1.5 yr;
Sortino 5.8, max DD 12%.

Logic:
1. Detect a major liquidation event: volume spike (> vol_spike_mult × N-bar avg)
   AND candle range ≥ momentum_candle_pct × close (big move confirms real liquidity event).
2. Enter WITH momentum direction — bearish candle → SHORT, bullish candle → LONG.
   (Opposite of LiquidationAdxStrategy which fades the move with divergence.)
3. Exit on: profit target, stop loss, OR 2-hr hold cap (hold_cap_bars).

Insight: After a large liquidation cascade the surviving traders are all positioned
in one direction. Momentum continuation is more probable than immediate reversal
for the next 1-2 hours.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class LiquidationMomentumStrategy(BaseStrategy):
    """
    Momentum-follow after a liquidation cascade.

    State machine: IDLE → ENTERED (on liquidation spike) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _detect_liquidation_spike(self, df: pd.DataFrame) -> bool:
        p = self.config.params
        vol_lookback = int(p.get("vol_lookback", 20))
        vol_mult = float(p.get("vol_spike_mult", 2.5))
        momentum_pct = float(p.get("momentum_candle_pct", 0.003))

        if len(df) < vol_lookback + 1:
            return False

        vol = df["volume"].to_numpy(float)
        avg_vol = vol[-vol_lookback - 1:-1].mean()
        if avg_vol <= 0:
            return False

        volume_spike = vol[-1] / avg_vol >= vol_mult

        close = float(df["close"].iloc[-1])
        hl_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
        big_candle = hl_range >= momentum_pct * close if close > 0 else False

        return bool(volume_spike and big_candle)

    def _spike_direction(self, df: pd.DataFrame) -> Optional[str]:
        """Return 'LONG', 'SHORT', or None based on the last candle's direction."""
        last_open = float(df["open"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_close < last_open:
            return "SHORT"
        if last_close > last_open:
            return "LONG"
        return None

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        vol_lookback = int(p.get("vol_lookback", 20))
        min_bars = vol_lookback + 2

        self._bar_count += 1

        if len(data) < min_bars:
            return None

        if not self._detect_liquidation_spike(data):
            return None

        direction = self._spike_direction(data)
        if direction is None:
            return None

        vol = data["volume"].to_numpy(float)
        avg_vol = vol[-vol_lookback - 1:-1].mean()
        vol_ratio = vol[-1] / avg_vol if avg_vol > 0 else 1.0
        strength = min(0.5 + (vol_ratio - float(p.get("vol_spike_mult", 2.5))) / 10.0, 0.95)
        strength = max(strength, 0.5)

        price = float(data["close"].iloc[-1])
        signal_type = SignalType.LONG if direction == "LONG" else SignalType.SHORT
        self._entry_bar = self._bar_count - 1

        logger.info("[LiqMom] %s | price=%.4f vol_ratio=%.1fx strength=%.2f",
                    direction, price, vol_ratio, strength)

        return Signal(
            signal_type=signal_type,
            symbol=self.config.symbol,
            price=price,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"LiqMom {direction}: vol_ratio={vol_ratio:.1f}x",
            metadata={"vol_ratio": vol_ratio, "direction": direction},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 8))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        is_long = position.get("is_long", position.get("size", 0) > 0)
        close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT

        bars_held = self._bar_count - self._entry_bar
        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqMom hold cap: {bars_held} bars > {hold_cap_bars}",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqMom target: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqMom stop: {pnl_pct:.1f}%",
            )

        return None
