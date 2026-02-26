"""
RSI + VWAP Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_rsi_vwap

- Combined RSI and VWAP signals for confluence-based entries
- Long: RSI exiting oversold + price crosses above VWAP
- Short: RSI exiting overbought + price crosses below VWAP
- Exit: RSI extreme in opposite direction or price crosses VWAP against position
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class RSIVWAPStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)
        oversold = p.get("oversold", 30)
        overbought = p.get("overbought", 70)

        min_len = rsi_period + 5
        if len(data) < min_len:
            return None

        data = self._add_indicators(data, rsi_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        rsi = current.get("rsi")
        vwap = current.get("vwap")
        prev_price = prev["close"]
        prev_vwap = prev.get("vwap")

        if any(pd.isna(v) for v in [rsi, vwap, prev_vwap]):
            return None

        # Long: RSI above oversold (recovering) + price crosses above VWAP
        if rsi > oversold and prev_price < prev_vwap and price > vwap:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"RSI+VWAP LONG: RSI={rsi:.1f} recovering, price crossed above VWAP {vwap:.2f}",
                metadata={"rsi": rsi, "vwap": vwap},
            )

        # Short: RSI below overbought (declining) + price crosses below VWAP
        if rsi < overbought and prev_price > prev_vwap and price < vwap:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"RSI+VWAP SHORT: RSI={rsi:.1f} declining, price crossed below VWAP {vwap:.2f}",
                metadata={"rsi": rsi, "vwap": vwap},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)
        overbought = p.get("overbought", 70)
        oversold = p.get("oversold", 30)

        data = self._add_indicators(data, rsi_period)
        current = data.iloc[-1]

        price = current["close"]
        rsi = current.get("rsi", 50)
        vwap = current.get("vwap", price)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: RSI overbought or price below VWAP
        if is_long and (rsi > overbought or price < vwap):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"RSI+VWAP exit long: RSI={rsi:.1f}, price vs VWAP={price - vwap:.2f}",
            )

        # Exit short: RSI oversold or price above VWAP
        if not is_long and (rsi < oversold or price > vwap):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"RSI+VWAP exit short: RSI={rsi:.1f}, price vs VWAP={price - vwap:.2f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
        if "rsi" in df.columns and "vwap" in df.columns:
            return df
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, 1)
        df["rsi"] = 100 - (100 / (1 + rs))

        # VWAP (cumulative typical price * volume / cumulative volume)
        if "volume" in df.columns and df["volume"].sum() > 0:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            cum_tp_vol = (typical_price * df["volume"]).cumsum()
            cum_vol = df["volume"].cumsum().replace(0, 1)
            df["vwap"] = cum_tp_vol / cum_vol
        else:
            # Fallback: use rolling mean as VWAP proxy
            df["vwap"] = df["close"].rolling(window=20).mean()

        return df
