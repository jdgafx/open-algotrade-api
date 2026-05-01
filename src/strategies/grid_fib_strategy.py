"""
Grid Trading + Fibonacci Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_grid_trading_fibonacci

- Calculates Fibonacci retracement levels from recent swing high/low
- Places grid entries at Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Long when price retraces to a Fibonacci support level in uptrend
- Short when price retraces to a Fibonacci resistance level in downtrend
- Exit at next Fibonacci extension or stop loss
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)

FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]


class GridFibStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        lookback = p.get("fib_lookback", 50)
        proximity_pct = p.get("proximity_pct", 0.3)
        trend_period = p.get("trend_period", 20)

        if len(data) < lookback + 5:
            return None

        recent = data.iloc[-lookback:]
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        fib_range = swing_high - swing_low

        if fib_range <= 0:
            return None

        price = data["close"].iloc[-1]
        sma = data["close"].iloc[-trend_period:].mean()
        threshold = fib_range * proximity_pct / 100

        prev_price = data["close"].iloc[-2]

        # Uptrend: price above SMA, look for fib support bounces
        if price > sma:
            for fib in FIB_LEVELS:
                level = swing_high - fib_range * fib
                if abs(price - level) <= threshold and price > level and price > prev_price:
                    # Golden ratio levels (0.382, 0.618) get stronger signals
                    fib_strength = 0.7 if fib in (0.382, 0.618) else 0.6
                    strength = min(fib_strength + (1 - abs(price - level) / threshold) * 0.2, 0.9)
                    return Signal(
                        signal_type=SignalType.LONG,
                        symbol=self.config.symbol,
                        price=price,
                        size_usd=self.config.size_usd,
                        strength=strength,
                        reason=f"Grid+Fib LONG: bounce at {fib:.1%} level ({level:.2f}), uptrend",
                        metadata={"fib_level": fib, "price_level": level, "swing_high": swing_high, "swing_low": swing_low},
                    )

        # Downtrend: price below SMA, look for fib resistance rejections
        if price < sma:
            for fib in FIB_LEVELS:
                level = swing_low + fib_range * fib
                if abs(price - level) <= threshold and price < level and price < prev_price:
                    fib_strength = 0.7 if fib in (0.382, 0.618) else 0.6
                    strength = min(fib_strength + (1 - abs(price - level) / threshold) * 0.2, 0.9)
                    return Signal(
                        signal_type=SignalType.SHORT,
                        symbol=self.config.symbol,
                        price=price,
                        size_usd=self.config.size_usd,
                        strength=strength,
                        reason=f"Grid+Fib SHORT: rejection at {fib:.1%} level ({level:.2f}), downtrend",
                        metadata={"fib_level": fib, "price_level": level, "swing_high": swing_high, "swing_low": swing_low},
                    )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        lookback = p.get("fib_lookback", 50)
        tp_fib = p.get("take_profit_fib", 0.618)
        sl_fib = p.get("stop_loss_fib", 1.0)

        recent = data.iloc[-lookback:]
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        fib_range = swing_high - swing_low

        price = data["close"].iloc[-1]
        entry_price = position.get("entry_px", position.get("entry_price", price))
        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if fib_range > 0:
            if is_long:
                tp_level = entry_price + fib_range * tp_fib
                sl_level = entry_price - fib_range * sl_fib
                if price >= tp_level:
                    return Signal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.config.symbol,
                        reason=f"Grid+Fib TP: price {price:.2f} hit {tp_fib:.1%} extension",
                    )
                if price <= sl_level:
                    return Signal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.config.symbol,
                        reason=f"Grid+Fib SL: price {price:.2f} below stop",
                    )
            else:
                tp_level = entry_price - fib_range * tp_fib
                sl_level = entry_price + fib_range * sl_fib
                if price <= tp_level:
                    return Signal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.config.symbol,
                        reason=f"Grid+Fib TP: price {price:.2f} hit {tp_fib:.1%} extension",
                    )
                if price >= sl_level:
                    return Signal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.config.symbol,
                        reason=f"Grid+Fib SL: price {price:.2f} above stop",
                    )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None
