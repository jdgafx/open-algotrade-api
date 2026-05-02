"""
SMA Crossover Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/6_sma.py

- SMA with support/resistance levels
- Long when price crosses above SMA near support
- Short when price crosses below SMA near resistance
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SMAStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_period", 20)
        support_lookback = p.get("support_lookback", 20)
        adx_period = p.get("adx_period", 14)
        adx_threshold = p.get("adx_threshold", 25)

        if len(data) < max(sma_period, adx_period * 2) + 2:
            return None

        data = self._add_indicators(data, sma_period, support_lookback, adx_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        sma = current["sma"]
        adx = current.get("adx", 0)
        support = current.get("support", 0)
        resistance = current.get("resistance", float("inf"))

        if pd.isna(sma) or pd.isna(adx):
            return None

        # Require trending market (ADX > threshold) to avoid whipsaws in ranging conditions
        if adx <= adx_threshold:
            return None

        # Long: price crosses above SMA with trend confirmation
        if prev["close"] < prev.get("sma", sma) and price > sma:
            near_support = support > 0 and (price - support) / price < 0.02
            strength = min(0.65 + (adx - adx_threshold) / 100, 0.95)
            if near_support:
                strength = min(strength + 0.10, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"SMA cross LONG: {price:.2f} > SMA {sma:.2f}, ADX {adx:.1f}" + (" (near support)" if near_support else ""),
                metadata={"sma": sma, "adx": adx, "support": support, "resistance": resistance},
            )

        # Short: price crosses below SMA with trend confirmation
        if prev["close"] > prev.get("sma", sma) and price < sma:
            near_resistance = resistance > 0 and (resistance - price) / price < 0.02
            strength = min(0.65 + (adx - adx_threshold) / 100, 0.95)
            if near_resistance:
                strength = min(strength + 0.10, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"SMA cross SHORT: {price:.2f} < SMA {sma:.2f}, ADX {adx:.1f}" + (" (near resistance)" if near_resistance else ""),
                metadata={"sma": sma, "adx": adx, "support": support, "resistance": resistance},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_period", 20)

        data = self._add_indicators(data, sma_period, 20)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        sma = current.get("sma", price)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit on SMA cross against position
        if is_long and prev["close"] > prev.get("sma", sma) and price < sma:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"SMA exit: price {price:.2f} crossed below SMA {sma:.2f}",
            )
        if not is_long and prev["close"] < prev.get("sma", sma) and price > sma:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"SMA exit: price {price:.2f} crossed above SMA {sma:.2f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame, sma_period: int, lookback: int, adx_period: int = 14) -> pd.DataFrame:
        if "sma" in df.columns and "adx" in df.columns:
            return df
        df = df.copy()
        df["sma"] = df["close"].rolling(window=sma_period).mean()
        df["support"] = df["low"].rolling(window=lookback).min()
        df["resistance"] = df["high"].rolling(window=lookback).max()

        # Wilder ADX — same approach as bollinger_strategy
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        atr_safe = atr.replace(0, np.nan)
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr_safe
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr_safe
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        df["adx"] = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()
        return df
