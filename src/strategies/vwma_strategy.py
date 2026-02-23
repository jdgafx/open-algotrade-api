"""
VWMA Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/9_more_indicators/9_vwma.py

- Volume-Weighted Moving Average with multiple periods (20, 41, 75)
- Long when all three VWMAs align bullish (fast > mid > slow)
- Short when all three VWMAs align bearish (fast < mid < slow)
- Exit: VWMA alignment breaks
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class VWMAStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        fast = p.get("fast_period", 20)
        mid = p.get("mid_period", 41)
        slow = p.get("slow_period", 75)

        if len(data) < slow + 2:
            return None

        data = self._add_vwma(data, fast, mid, slow)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]

        vwma_fast = current.get("vwma_fast", None)
        vwma_mid = current.get("vwma_mid", None)
        vwma_slow = current.get("vwma_slow", None)

        if any(pd.isna(v) for v in [vwma_fast, vwma_mid, vwma_slow]):
            return None

        prev_fast = prev.get("vwma_fast", vwma_fast)
        prev_mid = prev.get("vwma_mid", vwma_mid)

        # Long: bullish alignment — fast > mid > slow and price above fast
        bullish_aligned = vwma_fast > vwma_mid > vwma_slow
        was_not_aligned = not (prev_fast > prev_mid)

        if bullish_aligned and was_not_aligned and price > vwma_fast:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"VWMA alignment LONG: fast {vwma_fast:.2f} > mid {vwma_mid:.2f} > slow {vwma_slow:.2f}",
                metadata={"vwma_fast": vwma_fast, "vwma_mid": vwma_mid, "vwma_slow": vwma_slow},
            )

        # Short: bearish alignment — fast < mid < slow and price below fast
        bearish_aligned = vwma_fast < vwma_mid < vwma_slow
        was_not_bearish = not (prev_fast < prev_mid)

        if bearish_aligned and was_not_bearish and price < vwma_fast:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"VWMA alignment SHORT: fast {vwma_fast:.2f} < mid {vwma_mid:.2f} < slow {vwma_slow:.2f}",
                metadata={"vwma_fast": vwma_fast, "vwma_mid": vwma_mid, "vwma_slow": vwma_slow},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        fast = p.get("fast_period", 20)
        mid = p.get("mid_period", 41)
        slow = p.get("slow_period", 75)

        data = self._add_vwma(data, fast, mid, slow)
        current = data.iloc[-1]

        vwma_fast = current.get("vwma_fast", None)
        vwma_mid = current.get("vwma_mid", None)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if vwma_fast is not None and vwma_mid is not None:
            # Exit when alignment breaks
            if is_long and vwma_fast < vwma_mid:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"VWMA alignment broken: fast {vwma_fast:.2f} < mid {vwma_mid:.2f}",
                )
            if not is_long and vwma_fast > vwma_mid:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"VWMA alignment broken: fast {vwma_fast:.2f} > mid {vwma_mid:.2f}",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_vwma(df: pd.DataFrame, fast: int, mid: int, slow: int) -> pd.DataFrame:
        if "vwma_fast" in df.columns:
            return df
        df = df.copy()
        vol = df["volume"].fillna(0)

        for period, name in [(fast, "vwma_fast"), (mid, "vwma_mid"), (slow, "vwma_slow")]:
            pv = df["close"] * vol
            df[name] = pv.rolling(window=period).sum() / vol.rolling(window=period).sum().replace(0, 1)

        return df
