"""
VWAP Bot Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/12_day12_bots/day12_hyperliquid/12_vwap_bot.py

- Calculates VWAP from intraday volume-weighted price
- 70% probability bias above VWAP (trend continuation)
- 30% probability bias below VWAP (reversal)
- Long when price above VWAP with bullish momentum
- Short when price below VWAP with bearish momentum
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class VWAPBotStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        if len(data) < 5:
            return None

        data = self._add_vwap(data)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        vwap = current["vwap"]

        if pd.isna(vwap) or vwap <= 0:
            return None

        price_vs_vwap = (price - vwap) / vwap

        # Long: price crosses above VWAP with momentum
        if (
            prev["close"] <= prev.get("vwap", vwap)
            and price > vwap
            and current["close"] > current["open"]
        ):
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"VWAP crossover LONG: price {price:.2f} > VWAP {vwap:.2f} ({price_vs_vwap:+.4f})",
                metadata={"vwap": vwap, "price_vs_vwap": price_vs_vwap},
            )

        # Short: price crosses below VWAP with momentum
        if (
            prev["close"] >= prev.get("vwap", vwap)
            and price < vwap
            and current["close"] < current["open"]
        ):
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"VWAP crossover SHORT: price {price:.2f} < VWAP {vwap:.2f} ({price_vs_vwap:+.4f})",
                metadata={"vwap": vwap, "price_vs_vwap": price_vs_vwap},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        data = self._add_vwap(data)
        current = data.iloc[-1]
        price = current["close"]
        vwap = current.get("vwap", price)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"VWAP target: {pnl_pct:.1f}%")

        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"VWAP stop: {pnl_pct:.1f}%")

        # Exit on VWAP cross against position
        if is_long and price < vwap * 0.998:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"VWAP exit: price {price:.2f} fell below VWAP {vwap:.2f}",
            )
        if not is_long and price > vwap * 1.002:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"VWAP exit: price {price:.2f} rose above VWAP {vwap:.2f}",
            )

        return None

    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        if "vwap" in df.columns:
            return df
        df = df.copy()
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"].fillna(0)
        df["cum_tp_vol"] = (typical_price * vol).cumsum()
        df["cum_vol"] = vol.cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, 1)
        return df
