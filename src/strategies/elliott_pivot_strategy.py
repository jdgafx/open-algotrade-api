"""
Elliott Waves + Pivot Lines Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_elliot_waves_pivot_lines

- Combines Elliott Wave swing pattern detection with pivot point levels
- Long: bullish wave pattern + price above pivot point + near S1 support
- Short: bearish wave pattern + price below pivot point + near R1 resistance
- Exit at pivot targets (R1/S1) or wave invalidation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ElliottPivotStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        swing_lookback = p.get("swing_lookback", 5)
        pivot_lookback = p.get("pivot_lookback", 24)

        min_len = max(swing_lookback * 6, pivot_lookback + 3)
        if len(data) < min_len:
            return None

        data = self._add_pivots(data, pivot_lookback)
        current = data.iloc[-1]
        price = current["close"]
        pp = current.get("pivot")
        r1 = current.get("r1")
        s1 = current.get("s1")

        if any(pd.isna(v) for v in [pp, r1, s1]):
            return None

        # Detect wave swings
        wave_bias = self._detect_wave_bias(data, swing_lookback)

        # Long: bullish wave + price above pivot
        if wave_bias == "bullish" and price > pp:
            # Strength: distance above pivot relative to PP-S1 range
            pivot_range = r1 - s1 if r1 > s1 else price * 0.01
            dist = (price - pp) / pivot_range if pivot_range > 0 else 0.5
            strength = min(0.6 + dist * 0.2, 0.9)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Elliott+Pivot LONG: bullish wave, price {price:.2f} above PP {pp:.2f}",
                metadata={"pivot": pp, "r1": r1, "s1": s1, "wave_bias": wave_bias},
            )

        # Short: bearish wave + price below pivot
        if wave_bias == "bearish" and price < pp:
            pivot_range = r1 - s1 if r1 > s1 else price * 0.01
            dist = (pp - price) / pivot_range if pivot_range > 0 else 0.5
            strength = min(0.6 + dist * 0.2, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Elliott+Pivot SHORT: bearish wave, price {price:.2f} below PP {pp:.2f}",
                metadata={"pivot": pp, "r1": r1, "s1": s1, "wave_bias": wave_bias},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        pivot_lookback = p.get("pivot_lookback", 24)

        data = self._add_pivots(data, pivot_lookback)
        current = data.iloc[-1]

        price = current["close"]
        r1 = current.get("r1", 0)
        s1 = current.get("s1", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long at R1 or below S1
        if is_long:
            if price >= r1:
                return Signal(signal_type=SignalType.CLOSE_LONG, symbol=self.config.symbol, reason=f"Elliott+Pivot: hit R1 {r1:.2f}")
            if price <= s1:
                return Signal(signal_type=SignalType.CLOSE_LONG, symbol=self.config.symbol, reason=f"Elliott+Pivot: below S1 {s1:.2f}")

        # Exit short at S1 or above R1
        if not is_long:
            if price <= s1:
                return Signal(signal_type=SignalType.CLOSE_SHORT, symbol=self.config.symbol, reason=f"Elliott+Pivot: hit S1 {s1:.2f}")
            if price >= r1:
                return Signal(signal_type=SignalType.CLOSE_SHORT, symbol=self.config.symbol, reason=f"Elliott+Pivot: above R1 {r1:.2f}")

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_pivots(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        if "pivot" in df.columns:
            return df
        df = df.copy()
        prev_high = df["high"].rolling(window=lookback).max().shift(1)
        prev_low = df["low"].rolling(window=lookback).min().shift(1)
        prev_close = df["close"].shift(1)
        df["pivot"] = (prev_high + prev_low + prev_close) / 3
        df["r1"] = 2 * df["pivot"] - prev_low
        df["s1"] = 2 * df["pivot"] - prev_high
        return df

    @staticmethod
    def _detect_wave_bias(df: pd.DataFrame, lookback: int) -> str:
        """Simplified wave bias detection via higher-highs/higher-lows pattern."""
        if len(df) < lookback * 4:
            return "neutral"

        closes = df["close"].values
        recent = closes[-lookback * 4:]

        # Split into segments and check trend
        seg_size = lookback
        segments = [recent[i * seg_size:(i + 1) * seg_size] for i in range(4)]
        segment_means = [s.mean() for s in segments]

        # Higher means = bullish, lower means = bearish
        rising = all(segment_means[i] < segment_means[i + 1] for i in range(3))
        falling = all(segment_means[i] > segment_means[i + 1] for i in range(3))

        if rising:
            return "bullish"
        elif falling:
            return "bearish"
        return "neutral"
