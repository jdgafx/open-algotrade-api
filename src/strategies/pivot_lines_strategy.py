"""
Pivot Lines Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_pivot_lines

- Classic pivot point calculation (PP, R1, R2, S1, S2)
- Long when price moves above pivot point with S1 as stop, R1 as target
- Short when price moves below pivot point with R1 as stop, S1 as target
- Exit at support/resistance levels or target/stop
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class PivotLinesStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        pivot_lookback = p.get("pivot_lookback", 24)

        if len(data) < pivot_lookback + 3:
            return None

        data = self._add_pivots(data, pivot_lookback)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        pp = current.get("pivot")
        r1 = current.get("r1")
        s1 = current.get("s1")
        prev_price = prev["close"]
        prev_pp = prev.get("pivot")

        if any(pd.isna(v) for v in [pp, r1, s1, prev_pp]):
            return None

        # Long: price crosses above pivot point
        if prev_price <= prev_pp and price > pp:
            # Strength based on distance to target vs distance to stop
            reward = r1 - price if r1 > price else 0
            risk = price - s1 if price > s1 else price * 0.01
            rr_ratio = reward / risk if risk > 0 else 1.0
            strength = min(0.5 + rr_ratio * 0.15, 0.9)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Pivot LONG: price {price:.2f} crossed above PP {pp:.2f}, target R1={r1:.2f}",
                metadata={"pivot": pp, "r1": r1, "s1": s1},
            )

        # Short: price crosses below pivot point
        if prev_price >= prev_pp and price < pp:
            reward = price - s1 if price > s1 else 0
            risk = r1 - price if r1 > price else price * 0.01
            rr_ratio = reward / risk if risk > 0 else 1.0
            strength = min(0.5 + rr_ratio * 0.15, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Pivot SHORT: price {price:.2f} crossed below PP {pp:.2f}, target S1={s1:.2f}",
                metadata={"pivot": pp, "r1": r1, "s1": s1},
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
        high = current.get("high", price)
        low = current.get("low", price)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: hit R1 (target) or dropped below S1 (stop)
        if is_long:
            if high >= r1:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Pivot exit: hit R1 resistance {r1:.2f}",
                )
            if low <= s1:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Pivot exit: hit S1 support {s1:.2f}",
                )

        # Exit short: hit S1 (target) or rose above R1 (stop)
        if not is_long:
            if low <= s1:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Pivot exit: hit S1 support {s1:.2f}",
                )
            if high >= r1:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Pivot exit: hit R1 resistance {r1:.2f}",
                )

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
        # Use rolling window to calculate pivot from prior period
        prev_high = df["high"].rolling(window=lookback).max().shift(1)
        prev_low = df["low"].rolling(window=lookback).min().shift(1)
        prev_close = df["close"].shift(1)

        df["pivot"] = (prev_high + prev_low + prev_close) / 3
        df["r1"] = 2 * df["pivot"] - prev_low
        df["s1"] = 2 * df["pivot"] - prev_high
        df["r2"] = df["pivot"] + (prev_high - prev_low)
        df["s2"] = df["pivot"] - (prev_high - prev_low)
        return df
