"""
EMA + Bollinger Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_ema_bollinger

- Combined EMA crossover trend detection with Bollinger Band squeeze
- Long: short EMA crosses above long EMA AND price near lower BB
- Short: long EMA crosses above short EMA AND price near upper BB
- Exit: price hits opposite BB or EMA crossover reverses
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class EMABollingerStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        short_ema = p.get("short_ema_period", 50)
        long_ema = p.get("long_ema_period", 200)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)

        min_len = max(long_ema, bb_period) + 5
        if len(data) < min_len:
            return None

        data = self._add_indicators(data, short_ema, long_ema, bb_period, bb_std)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        ema_s = current.get("ema_short")
        ema_l = current.get("ema_long")
        bb_upper = current.get("bb_upper")
        bb_lower = current.get("bb_lower")
        prev_ema_s = prev.get("ema_short")
        prev_ema_l = prev.get("ema_long")

        if any(pd.isna(v) for v in [ema_s, ema_l, bb_upper, bb_lower, prev_ema_s, prev_ema_l]):
            return None

        # BB proximity: price within 20% of band distance from the band
        bb_range = bb_upper - bb_lower
        bb_proximity = bb_range * 0.2 if bb_range > 0 else 0

        # Long: bullish EMA cross + price near lower band (within proximity)
        if prev_ema_s <= prev_ema_l and ema_s > ema_l and price <= bb_lower + bb_proximity:
            strength = min(0.6 + (bb_lower + bb_proximity - price) / max(bb_range, 1) * 2, 0.9)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"EMA+BB LONG: EMA cross bullish, price {price:.2f} near lower BB {bb_lower:.2f}",
                metadata={"ema_short": ema_s, "ema_long": ema_l, "bb_lower": bb_lower},
            )

        # Short: bearish EMA cross + price near upper band (within proximity)
        if prev_ema_l <= prev_ema_s and ema_l > ema_s and price >= bb_upper - bb_proximity:
            strength = min(0.6 + (price - bb_upper + bb_proximity) / max(bb_range, 1) * 2, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"EMA+BB SHORT: EMA cross bearish, price {price:.2f} near upper BB {bb_upper:.2f}",
                metadata={"ema_short": ema_s, "ema_long": ema_l, "bb_upper": bb_upper},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        short_ema = p.get("short_ema_period", 50)
        long_ema = p.get("long_ema_period", 200)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)

        data = self._add_indicators(data, short_ema, long_ema, bb_period, bb_std)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        ema_s = current.get("ema_short", 0)
        ema_l = current.get("ema_long", 0)
        bb_upper = current.get("bb_upper", 0)
        bb_lower = current.get("bb_lower", 0)
        prev_ema_s = prev.get("ema_short", 0)
        prev_ema_l = prev.get("ema_long", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: price hits upper BB or bearish EMA cross
        if is_long and (price >= bb_upper or (prev_ema_l <= prev_ema_s and ema_l > ema_s)):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"EMA+BB exit long: price={price:.2f} upper_bb={bb_upper:.2f}",
            )

        # Exit short: price hits lower BB or bullish EMA cross
        if not is_long and (price <= bb_lower or (prev_ema_s <= prev_ema_l and ema_s > ema_l)):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"EMA+BB exit short: price={price:.2f} lower_bb={bb_lower:.2f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame, short_p: int, long_p: int, bb_p: int, bb_std: float) -> pd.DataFrame:
        if "ema_short" in df.columns:
            return df
        df = df.copy()
        df["ema_short"] = df["close"].ewm(span=short_p, adjust=False).mean()
        df["ema_long"] = df["close"].ewm(span=long_p, adjust=False).mean()
        df["bb_mid"] = df["close"].rolling(window=bb_p).mean()
        bb_std_val = df["close"].rolling(window=bb_p).std()
        df["bb_upper"] = df["bb_mid"] + bb_std * bb_std_val
        df["bb_lower"] = df["bb_mid"] - bb_std * bb_std_val
        return df
