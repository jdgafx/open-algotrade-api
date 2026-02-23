"""
Consolidation Pop Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/3_consolidation_pop_algo/

- Detects consolidation via True Range deviance (low ATR relative to range)
- When deviance is below threshold (tight range), waits for breakout
- Buy at lower 1/3 of range, sell at upper 1/3
- TP: 0.3%, SL: 0.25%
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class ConsolidationPopStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        atr_period = p.get("atr_period", 14)
        deviance_threshold = p.get("deviance_threshold", 0.4)
        buy_zone = p.get("range_position_buy", 0.33)
        sell_zone = p.get("range_position_sell", 0.67)

        if len(data) < atr_period + 2:
            return None

        data = self._add_consolidation_indicators(data, atr_period)
        current = data.iloc[-1]
        price = current["close"]

        deviance = current.get("tr_deviance", 1.0)
        range_pos = current.get("range_position", 0.5)

        if pd.isna(deviance) or pd.isna(range_pos):
            return None

        is_consolidating = deviance < deviance_threshold

        if is_consolidating:
            if range_pos <= buy_zone:
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"Consolidation LONG: deviance {deviance:.3f} < {deviance_threshold}, price at {range_pos:.2f} of range (lower {buy_zone})",
                    metadata={"deviance": deviance, "range_position": range_pos},
                )

            if range_pos >= sell_zone:
                return Signal(
                    signal_type=SignalType.SHORT,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"Consolidation SHORT: deviance {deviance:.3f} < {deviance_threshold}, price at {range_pos:.2f} of range (upper {sell_zone})",
                    metadata={"deviance": deviance, "range_position": range_pos},
                )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        tp_pct = p.get("tp_pct", 0.003)
        sl_pct = p.get("sl_pct", 0.0025)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        entry_px = position.get("entry_px", 0)
        price = data.iloc[-1]["close"]

        if entry_px > 0:
            pnl = (price - entry_px) / entry_px if is_long else (entry_px - price) / entry_px

            if pnl >= tp_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"Consolidation TP: {pnl:.4f} >= {tp_pct}",
                )
            if pnl <= -sl_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"Consolidation SL: {pnl:.4f} <= -{sl_pct}",
                )

        pnl_pct = position.get("pnl_perc", 0)
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_consolidation_indicators(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
        if "tr_deviance" in df.columns:
            return df
        df = df.copy()

        # True Range
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = pd.concat([
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs(),
        ], axis=1).max(axis=1)

        # ATR
        df["atr"] = df["tr"].rolling(window=atr_period).mean()

        # Range over lookback
        rolling_high = df["high"].rolling(window=atr_period).max()
        rolling_low = df["low"].rolling(window=atr_period).min()
        total_range = rolling_high - rolling_low

        # TR deviance: how tight is current TR relative to the range
        df["tr_deviance"] = df["tr"] / total_range.replace(0, 1)

        # Position in range (0 = bottom, 1 = top)
        df["range_position"] = (df["close"] - rolling_low) / total_range.replace(0, 1)

        return df
