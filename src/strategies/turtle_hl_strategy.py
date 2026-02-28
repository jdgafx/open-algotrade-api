"""
Turtle Trading Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/1_turtle_trending_algo/

Classic 55-bar breakout system:
- Entry: Price breaks above 55-bar high (long) or below 55-bar low (short)
- Confirms with current candle open < breakout level and close > previous close
- Stop: 2x ATR trailing stop
- Take profit: 0.2% from entry
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class TurtleHLStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        lookback = p.get("lookback_period", 55)
        atr_period = p.get("atr_period", 20)

        if len(data) < lookback + 2:
            return None

        data = self._add_atr(data, atr_period)

        lookback_slice = data.iloc[-(lookback + 1):-1]
        highest_high = lookback_slice["high"].max()
        lowest_low = lookback_slice["low"].min()

        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]

        # Calculate breakout strength: how far price exceeded the level relative to ATR
        atr = current.get("atr", 0)
        if pd.isna(atr) or atr <= 0:
            atr = (highest_high - lowest_low) / lookback  # fallback

        if (
            price >= highest_high
            and current["open"] < highest_high
            and current["close"] > prev["close"]
        ):
            # Signal strength based on breakout magnitude vs ATR
            breakout_size = (price - highest_high) / atr if atr > 0 else 0.5
            strength = min(0.5 + breakout_size * 0.3, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Turtle breakout LONG: price {price:.2f} > {lookback}-bar high {highest_high:.2f}",
                metadata={"atr": float(atr), "highest_high": highest_high},
            )

        if (
            price <= lowest_low
            and current["open"] > lowest_low
            and current["close"] < prev["close"]
        ):
            breakout_size = (lowest_low - price) / atr if atr > 0 else 0.5
            strength = min(0.5 + breakout_size * 0.3, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Turtle breakout SHORT: price {price:.2f} < {lookback}-bar low {lowest_low:.2f}",
                metadata={"atr": float(atr), "lowest_low": lowest_low},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        atr_period = p.get("atr_period", 20)
        atr_mult = p.get("atr_multiplier", 2.0)
        tp_pct = p.get("take_profit_pct", 0.002)

        data = self._add_atr(data, atr_period)
        current = data.iloc[-1]
        price = current["close"]
        atr = current.get("atr", 0)

        entry_px = position.get("entry_px", price)
        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if is_long:
            stop = entry_px - (atr * atr_mult)
            tp = entry_px * (1 + tp_pct)
            if price <= stop:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Turtle stop loss: {price:.2f} <= {stop:.2f}",
                )
            if price >= tp:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Turtle take profit: {price:.2f} >= {tp:.2f}",
                )
        else:
            stop = entry_px + (atr * atr_mult)
            tp = entry_px * (1 - tp_pct)
            if price >= stop:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Turtle stop loss: {price:.2f} >= {stop:.2f}",
                )
            if price <= tp:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Turtle take profit: {price:.2f} <= {tp:.2f}",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"PnL target hit: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Max loss hit: {pnl_pct:.1f}%",
            )

        return None

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
        if "atr" in df.columns:
            return df
        df = df.copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(window=period).mean()
        return df
