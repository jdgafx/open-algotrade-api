"""
Mean Reversion Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/6_mean_reversion/74_tickers_mean_reversion.py

- Multi-timeframe: 4h SMA trend filter, 15m SMA entry signal
- Long when 4h trend is bullish and 15m price below SMA (mean reversion buy)
- Short when 4h trend is bearish and 15m price above SMA (mean reversion sell)
- Originally traded 74+ tickers — here adapted for single-symbol BaseStrategy
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_entry_period", 20)
        reversion_target = p.get("reversion_target_pct", 0.003)

        if len(data) < sma_period + 2:
            return None

        data = self._add_sma(data, sma_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        sma = current["sma"]

        if pd.isna(sma):
            return None

        deviation = (price - sma) / sma

        # Check for trend filter from higher timeframe (if available)
        trend_bullish = current.get("trend_bullish", None)
        if trend_bullish is None:
            # Infer trend from SMA slope
            if len(data) >= sma_period + 5:
                sma_5_ago = data.iloc[-5].get("sma", sma)
                trend_bullish = sma > sma_5_ago if not pd.isna(sma_5_ago) else True
            else:
                trend_bullish = True

        # Mean reversion long: price below SMA in uptrend
        if trend_bullish and deviation < -0.005 and current["close"] > prev["close"]:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"Mean reversion LONG: deviation {deviation:.4f}, price {price:.2f} < SMA {sma:.2f}",
                metadata={"deviation": deviation, "sma": sma, "trend": "bullish"},
            )

        # Mean reversion short: price above SMA in downtrend
        if not trend_bullish and deviation > 0.005 and current["close"] < prev["close"]:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"Mean reversion SHORT: deviation {deviation:.4f}, price {price:.2f} > SMA {sma:.2f}",
                metadata={"deviation": deviation, "sma": sma, "trend": "bearish"},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_entry_period", 20)
        reversion_target = p.get("reversion_target_pct", 0.003)

        data = self._add_sma(data, sma_period)
        price = data.iloc[-1]["close"]
        sma = data.iloc[-1].get("sma", price)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        entry_px = position.get("entry_px", 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit when price reverts to mean (SMA)
        if is_long and price >= sma:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Mean reversion exit: price {price:.2f} >= SMA {sma:.2f}",
            )
        if not is_long and price <= sma:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Mean reversion exit: price {price:.2f} <= SMA {sma:.2f}",
            )

        # Also check reversion target from entry
        if entry_px > 0:
            pnl_from_entry = (price - entry_px) / entry_px if is_long else (entry_px - price) / entry_px
            if pnl_from_entry >= reversion_target:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"Mean reversion target: {pnl_from_entry:.4f} >= {reversion_target}",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
        if "sma" in df.columns:
            return df
        df = df.copy()
        df["sma"] = df["close"].rolling(window=period).mean()
        return df
