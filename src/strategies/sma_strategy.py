"""
SMA Crossover Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/6_sma.py

- SMA with support/resistance levels
- Long when price crosses above SMA near support
- Short when price crosses below SMA near resistance
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SMAStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_period", 20)
        support_lookback = p.get("support_lookback", 20)

        if len(data) < sma_period + 2:
            return None

        data = self._add_indicators(data, sma_period, support_lookback)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        sma = current["sma"]
        support = current.get("support", 0)
        resistance = current.get("resistance", float("inf"))

        if pd.isna(sma):
            return None

        # Long: price crosses above SMA
        if prev["close"] < prev.get("sma", sma) and price > sma:
            # Stronger signal near support
            near_support = support > 0 and (price - support) / price < 0.02
            strength = 0.80 if near_support else 0.65
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"SMA cross LONG: {price:.2f} > SMA {sma:.2f}" + (" (near support)" if near_support else ""),
                metadata={"sma": sma, "support": support, "resistance": resistance},
            )

        # Short: price crosses below SMA
        if prev["close"] > prev.get("sma", sma) and price < sma:
            near_resistance = resistance > 0 and (resistance - price) / price < 0.02
            strength = 0.80 if near_resistance else 0.65
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"SMA cross SHORT: {price:.2f} < SMA {sma:.2f}" + (" (near resistance)" if near_resistance else ""),
                metadata={"sma": sma, "support": support, "resistance": resistance},
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
    def _add_indicators(df: pd.DataFrame, sma_period: int, lookback: int) -> pd.DataFrame:
        if "sma" in df.columns:
            return df
        df = df.copy()
        df["sma"] = df["close"].rolling(window=sma_period).mean()
        df["support"] = df["low"].rolling(window=lookback).min()
        df["resistance"] = df["high"].rolling(window=lookback).max()
        return df
