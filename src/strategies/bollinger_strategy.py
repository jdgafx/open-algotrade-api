"""
Bollinger Band Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/10_day10_bots/day10_hyperliquid/10_bollinger_bot.py

- Entry: Bollinger Band squeeze (tight bands) then breakout
- Long when price breaks above upper band from squeeze
- Short when price breaks below lower band from squeeze
- Exit: Middle band (SMA) or opposite band
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class BollingerStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        period = p.get("bb_period", 20)
        std_dev = p.get("bb_std", 2.0)
        squeeze_threshold = p.get("squeeze_threshold", 0.03)

        if len(data) < period + 2:
            return None

        data = self._add_bollinger(data, period, std_dev)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]

        bb_width = current["bb_width"]
        prev_width = prev["bb_width"]

        is_squeeze = prev_width < squeeze_threshold
        expanding = bb_width > prev_width

        if is_squeeze and expanding and price > current["bb_upper"]:
            # Strength: tighter the prior squeeze, stronger the breakout signal
            expansion_ratio = bb_width / max(prev_width, 0.001)
            strength = min(0.5 + expansion_ratio * 0.1, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"BB squeeze breakout LONG: width {prev_width:.4f} -> {bb_width:.4f}, price {price:.2f} > upper {current['bb_upper']:.2f}",
                metadata={"bb_width": bb_width, "squeeze": True},
            )

        if is_squeeze and expanding and price < current["bb_lower"]:
            expansion_ratio = bb_width / max(prev_width, 0.001)
            strength = min(0.5 + expansion_ratio * 0.1, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"BB squeeze breakout SHORT: width {prev_width:.4f} -> {bb_width:.4f}, price {price:.2f} < lower {current['bb_lower']:.2f}",
                metadata={"bb_width": bb_width, "squeeze": True},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        period = p.get("bb_period", 20)
        std_dev = p.get("bb_std", 2.0)

        data = self._add_bollinger(data, period, std_dev)
        current = data.iloc[-1]
        price = current["close"]
        sma = current["bb_mid"]

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if is_long:
            if price <= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"BB exit: price {price:.2f} crossed below SMA {sma:.2f}",
                )
        else:
            if price >= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"BB exit: price {price:.2f} crossed above SMA {sma:.2f}",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target hit: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_bollinger(df: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
        if "bb_upper" in df.columns:
            return df
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        df["bb_std"] = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * std_dev)
        df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * std_dev)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        return df
