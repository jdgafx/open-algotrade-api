"""
RSI Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/7_rsi.py

- RSI indicator for overbought/oversold detection
- Long when RSI drops below oversold (30) then crosses back up
- Short when RSI rises above overbought (70) then crosses back down
- Exit: RSI crosses 50 or target/stop
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)
        oversold = p.get("oversold", 30)
        overbought = p.get("overbought", 70)

        if len(data) < rsi_period + 3:
            return None

        data = self._add_rsi(data, rsi_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        rsi = current["rsi"]
        prev_rsi = prev.get("rsi", 50)

        if pd.isna(rsi) or pd.isna(prev_rsi):
            return None

        # Long: RSI was oversold, now crossing back up
        if prev_rsi < oversold and rsi >= oversold:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"RSI oversold reversal LONG: RSI {prev_rsi:.1f} -> {rsi:.1f} crossing above {oversold}",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi},
            )

        # Short: RSI was overbought, now crossing back down
        if prev_rsi > overbought and rsi <= overbought:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"RSI overbought reversal SHORT: RSI {prev_rsi:.1f} -> {rsi:.1f} crossing below {overbought}",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)

        data = self._add_rsi(data, rsi_period)
        current = data.iloc[-1]
        rsi = current.get("rsi", 50)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit when RSI crosses neutral (50)
        if is_long and rsi >= 50:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"RSI exit: RSI {rsi:.1f} reached neutral zone",
            )
        if not is_long and rsi <= 50:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"RSI exit: RSI {rsi:.1f} reached neutral zone",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
        if "rsi" in df.columns:
            return df
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df
