"""
Market Maker Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/5_market_maker/market-maker.py

- Places bid and ask orders around mid price with configurable spread
- ATR-based no-trade zones (avoids volatile periods)
- Kill switch on excessive position size
- Refreshes orders on each iteration
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MarketMakerStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self._last_side = None

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        spread = p.get("spread", 0.001)
        atr_period = p.get("atr_period", 14)
        max_position_usd = p.get("max_position_usd", 1000.0)

        if len(data) < atr_period + 2:
            return None

        data = self._add_atr(data, atr_period)
        current = data.iloc[-1]
        price = current["close"]
        atr = current.get("atr", 0)

        if pd.isna(atr) or atr <= 0:
            return None

        # No-trade zone: ATR too high relative to price (volatile)
        atr_pct = atr / price
        if atr_pct > spread * 3:
            logger.debug("MM no-trade zone: ATR %.4f%% > threshold", atr_pct * 100)
            return None

        # For market making, prefer buying when price is below short-term average
        # and selling when above -- this gives the MM an edge vs blind alternating
        sma_short = data["close"].iloc[-5:].mean()
        bid_price = price * (1 - spread / 2)
        ask_price = price * (1 + spread / 2)

        # Low volatility = better for MM, higher signal strength
        vol_quality = max(0.3, 1.0 - atr_pct / (spread * 3))

        if price < sma_short and self._last_side != "buy":
            self._last_side = "buy"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=bid_price,
                size_usd=min(self.config.size_usd, max_position_usd),
                strength=vol_quality,
                reason=f"MM bid: {bid_price:.2f} (spread={spread:.4f}, ATR%={atr_pct:.4f})",
                metadata={"spread": spread, "atr_pct": atr_pct, "order_type": "limit"},
            )
        elif price > sma_short and self._last_side != "sell":
            self._last_side = "sell"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=ask_price,
                size_usd=min(self.config.size_usd, max_position_usd),
                strength=vol_quality,
                reason=f"MM ask: {ask_price:.2f} (spread={spread:.4f}, ATR%={atr_pct:.4f})",
                metadata={"spread": spread, "atr_pct": atr_pct, "order_type": "limit"},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        kill_size_usd = p.get("kill_size_usd", 2000.0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        size = abs(position.get("size", 0))
        entry_px = position.get("entry_px", 0)
        pnl_pct = position.get("pnl_perc", 0)
        price = data.iloc[-1]["close"]

        position_usd = size * price

        # Kill switch: position too large
        if position_usd > kill_size_usd:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"MM kill switch: position ${position_usd:.0f} > ${kill_size_usd:.0f}",
            )

        # Take profit on spread capture
        spread = p.get("spread", 0.001)
        if entry_px > 0:
            pnl_from_entry = (price - entry_px) / entry_px if is_long else (entry_px - price) / entry_px
            if pnl_from_entry >= spread * 0.8:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"MM spread captured: {pnl_from_entry:.4f}",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

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
