"""
Quarter Theory Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_quarter_theory

- Based on price levels divisible by quarter increments of major levels
- Identifies quarter-point price levels as support/resistance
- Long when price breaks above a quarter level
- Short when price breaks below a quarter level
- Uses dynamic quarter sizing based on price magnitude
"""

import logging
import math
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class QuarterTheoryStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        quarter_size = p.get("quarter_size", None)  # Auto-detect if None
        breakout_pct = p.get("breakout_pct", 0.1)

        if len(data) < 5:
            return None

        price = data["close"].iloc[-1]
        prev_price = data["close"].iloc[-2]

        # Auto-detect quarter size based on price magnitude
        if quarter_size is None:
            quarter_size = self._auto_quarter_size(price)

        # Compute quarter boundaries from PREVIOUS bar — the level that was (or wasn't) crossed
        prev_lower = math.floor(prev_price / quarter_size) * quarter_size
        prev_upper = prev_lower + quarter_size
        threshold = quarter_size * breakout_pct / 100

        # Long: price crossed above prev_upper (prev bar was at or below the level; current bar above)
        if prev_price <= prev_upper and price > prev_upper + threshold:
            level_crossed = prev_upper
            breakout_dist = (price - level_crossed) / quarter_size
            strength = min(0.65 + breakout_dist * 0.5, 0.9)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Quarter LONG: price {price:.2f} broke above {level_crossed:.2f}",
                metadata={"quarter_level": level_crossed, "quarter_size": quarter_size},
            )

        # Short: price crossed below prev_lower
        if prev_price >= prev_lower and price < prev_lower - threshold:
            level_crossed = prev_lower
            breakout_dist = (level_crossed - price) / quarter_size
            strength = min(0.65 + breakout_dist * 0.5, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Quarter SHORT: price {price:.2f} broke below {level_crossed:.2f}",
                metadata={"quarter_level": level_crossed, "quarter_size": quarter_size},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        quarter_size = p.get("quarter_size", None)
        tp_quarters = p.get("take_profit_quarters", 1)
        sl_quarters = p.get("stop_loss_quarters", 1)

        price = data["close"].iloc[-1]
        entry_price = position.get("entry_px", position.get("entry_price", price))
        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if quarter_size is None:
            quarter_size = self._auto_quarter_size(price)

        tp_dist = quarter_size * tp_quarters
        sl_dist = quarter_size * sl_quarters

        if is_long:
            if price >= entry_price + tp_dist:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Quarter TP: price {price:.2f} hit target +{tp_quarters} quarters",
                )
            if price <= entry_price - sl_dist:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Quarter SL: price {price:.2f} hit stop -{sl_quarters} quarters",
                )
        else:
            if price <= entry_price - tp_dist:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Quarter TP: price {price:.2f} hit target -{tp_quarters} quarters",
                )
            if price >= entry_price + sl_dist:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Quarter SL: price {price:.2f} hit stop +{sl_quarters} quarters",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _auto_quarter_size(price: float) -> float:
        """Auto-detect appropriate quarter size based on price magnitude."""
        if price > 10000:
            return 250.0
        elif price > 1000:
            return 25.0
        elif price > 100:
            return 2.5
        elif price > 10:
            return 0.25
        else:
            return 0.025
