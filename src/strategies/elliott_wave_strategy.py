"""
Elliott Wave Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_elliot_waves

- Simplified Elliott Wave pattern detection via swing high/low analysis
- Identifies potential Wave 2/4 retracements for entry (Wave 3/5 trades)
- Uses Fibonacci retracement levels (38.2%, 50%, 61.8%) for validation
- Exit: price reversal or Fibonacci extension targets
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ElliottWaveStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        swing_lookback = p.get("swing_lookback", 5)
        fib_min = p.get("fib_retracement_min", 0.382)
        fib_max = p.get("fib_retracement_max", 0.618)
        min_swing_pct = p.get("min_swing_pct", 0.5)

        min_len = swing_lookback * 6
        if len(data) < min_len:
            return None

        price = data["close"].iloc[-1]
        swings = self._detect_swings(data, swing_lookback)

        if len(swings) < 4:
            return None

        # Check for bullish Wave 2 retracement (buy opportunity for Wave 3)
        wave_signal = self._check_wave_pattern(swings, price, fib_min, fib_max, min_swing_pct, bullish=True)
        if wave_signal:
            # Strength based on retracement depth (closer to 0.618 = golden ratio = stronger)
            ret = wave_signal['retracement']
            strength = min(0.5 + abs(ret - 0.5) * 2, 0.9)  # Peaks at 0.382 and 0.618
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Elliott LONG: Wave retracement at {ret:.1%}, expecting impulse wave",
                metadata=wave_signal,
            )

        # Check for bearish Wave 2 retracement (sell opportunity for Wave 3)
        wave_signal = self._check_wave_pattern(swings, price, fib_min, fib_max, min_swing_pct, bullish=False)
        if wave_signal:
            ret = wave_signal['retracement']
            strength = min(0.5 + abs(ret - 0.5) * 2, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Elliott SHORT: Wave retracement at {ret:.1%}, expecting impulse wave",
                metadata=wave_signal,
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        swing_lookback = p.get("swing_lookback", 5)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)
        entry_price = position.get("entry_price", data["close"].iloc[-1])

        price = data["close"].iloc[-1]

        # Exit on reversal: price moves against position by a significant amount
        reversal_pct = p.get("reversal_exit_pct", 1.5)
        if is_long and price < entry_price * (1 - reversal_pct / 100):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Elliott exit: reversal {((price/entry_price)-1)*100:.1f}%",
            )
        if not is_long and price > entry_price * (1 + reversal_pct / 100):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Elliott exit: reversal {((entry_price/price)-1)*100:.1f}%",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _detect_swings(df: pd.DataFrame, lookback: int) -> List[Tuple[int, float, str]]:
        """Detect swing highs and lows. Returns list of (index, price, 'high'|'low')."""
        swings = []
        highs = df["high"].values
        lows = df["low"].values

        for i in range(lookback, len(df) - lookback):
            if highs[i] == max(highs[i - lookback : i + lookback + 1]):
                swings.append((i, highs[i], "high"))
            if lows[i] == min(lows[i - lookback : i + lookback + 1]):
                swings.append((i, lows[i], "low"))

        swings.sort(key=lambda x: x[0])
        return swings

    @staticmethod
    def _check_wave_pattern(
        swings: List[Tuple[int, float, str]],
        current_price: float,
        fib_min: float,
        fib_max: float,
        min_swing_pct: float,
        bullish: bool,
    ) -> Optional[Dict[str, Any]]:
        """Check if recent swings form a wave retracement pattern."""
        recent = swings[-4:]

        if bullish:
            # Look for: low -> high -> retracement low (Wave 1 up, Wave 2 retrace)
            lows = [(i, p) for i, p, t in recent if t == "low"]
            highs = [(i, p) for i, p, t in recent if t == "high"]
            if len(lows) < 2 or len(highs) < 1:
                return None

            wave1_start = lows[-2][1]
            wave1_end = highs[-1][1]
            wave2_end = lows[-1][1]

            wave1_range = wave1_end - wave1_start
            if wave1_range <= 0 or (wave1_range / wave1_start * 100) < min_swing_pct:
                return None

            retracement = (wave1_end - wave2_end) / wave1_range
            if fib_min <= retracement <= fib_max and current_price > wave2_end:
                return {"retracement": retracement, "wave1_start": wave1_start, "wave1_end": wave1_end}
        else:
            # Look for: high -> low -> retracement high (Wave 1 down, Wave 2 retrace)
            highs = [(i, p) for i, p, t in recent if t == "high"]
            lows = [(i, p) for i, p, t in recent if t == "low"]
            if len(highs) < 2 or len(lows) < 1:
                return None

            wave1_start = highs[-2][1]
            wave1_end = lows[-1][1]
            wave2_end = highs[-1][1]

            wave1_range = wave1_start - wave1_end
            if wave1_range <= 0 or (wave1_range / wave1_start * 100) < min_swing_pct:
                return None

            retracement = (wave2_end - wave1_end) / wave1_range
            if fib_min <= retracement <= fib_max and current_price < wave2_end:
                return {"retracement": retracement, "wave1_start": wave1_start, "wave1_end": wave1_end}

        return None
