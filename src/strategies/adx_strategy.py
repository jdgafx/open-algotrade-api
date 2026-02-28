"""
ADX Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_adx

- ADX (Average Directional Index) for trend strength
- +DI/-DI crossover for direction
- Long when +DI crosses above -DI and ADX > threshold
- Short when -DI crosses above +DI and ADX > threshold
- Exit: opposite DI crossover or ADX drops below exit threshold
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ADXStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        adx_period = p.get("adx_period", 14)
        di_period = p.get("di_period", 14)
        adx_threshold = p.get("adx_threshold", 25)

        min_len = max(adx_period, di_period) + 5
        if len(data) < min_len:
            return None

        data = self._add_adx(data, adx_period, di_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        adx = current.get("adx")
        plus_di = current.get("plus_di")
        minus_di = current.get("minus_di")
        prev_plus_di = prev.get("plus_di")
        prev_minus_di = prev.get("minus_di")

        if any(pd.isna(v) for v in [adx, plus_di, minus_di, prev_plus_di, prev_minus_di]):
            return None

        price = current["close"]

        # Long: +DI crosses above -DI with strong trend
        if prev_plus_di <= prev_minus_di and plus_di > minus_di and adx > adx_threshold:
            # Strength: higher ADX = stronger trend signal
            strength = min(0.5 + (adx - adx_threshold) / 50, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"ADX LONG: +DI {plus_di:.1f} crossed above -DI {minus_di:.1f}, ADX={adx:.1f}",
                metadata={"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
            )

        # Short: -DI crosses above +DI with strong trend
        if prev_minus_di <= prev_plus_di and minus_di > plus_di and adx > adx_threshold:
            strength = min(0.5 + (adx - adx_threshold) / 50, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"ADX SHORT: -DI {minus_di:.1f} crossed above +DI {plus_di:.1f}, ADX={adx:.1f}",
                metadata={"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        adx_period = p.get("adx_period", 14)
        di_period = p.get("di_period", 14)
        exit_threshold = p.get("exit_threshold", 20)

        data = self._add_adx(data, adx_period, di_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        adx = current.get("adx", 0)
        plus_di = current.get("plus_di", 0)
        minus_di = current.get("minus_di", 0)
        prev_plus_di = prev.get("plus_di", 0)
        prev_minus_di = prev.get("minus_di", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: -DI crosses above +DI or ADX weakens
        if is_long and (
            (prev_minus_di <= prev_plus_di and minus_di > plus_di)
            or adx < exit_threshold
        ):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"ADX exit long: ADX={adx:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}",
            )

        # Exit short: +DI crosses above -DI or ADX weakens
        if not is_long and (
            (prev_plus_di <= prev_minus_di and plus_di > minus_di)
            or adx < exit_threshold
        ):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"ADX exit short: ADX={adx:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_adx(df: pd.DataFrame, adx_period: int, di_period: int) -> pd.DataFrame:
        if "adx" in df.columns:
            return df
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = tr.rolling(window=di_period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=di_period).mean() / atr.replace(0, 1)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=di_period).mean() / atr.replace(0, 1)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
        adx = dx.rolling(window=adx_period).mean()

        df["adx"] = adx
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        return df
