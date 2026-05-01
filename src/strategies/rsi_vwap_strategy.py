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

        prev_rsi = prev.get("rsi", 50)
        if pd.isna(prev_rsi):
            return None

        # Long: RSI in oversold zone (or recovering from it within 3 bars) AND price above VWAP
        # Relaxed from strict simultaneous crossover to improve signal frequency
        rsi_oversold_now = rsi <= oversold + 5
        rsi_recovering = rsi > prev_rsi  # RSI is rising
        price_above_vwap = price >= vwap * 0.997  # within 0.3% of VWAP counts

        if (rsi_oversold_now or (prev_rsi < oversold and rsi_recovering)) and price_above_vwap:
            depth = max(oversold - min(rsi, prev_rsi), 0)
            strength = min(0.5 + depth / 20, 0.92)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=max(strength, 0.5),
                reason=f"RSI+VWAP LONG: RSI {rsi:.1f} (oversold zone), price {price:.2f} at/above VWAP {vwap:.2f}",
                metadata={"rsi": rsi, "vwap": vwap},
            )

        # Short: RSI in overbought zone (or falling from it) AND price below VWAP
        rsi_overbought_now = rsi >= overbought - 5
        rsi_falling = rsi < prev_rsi
        price_below_vwap = price <= vwap * 1.003

        if (rsi_overbought_now or (prev_rsi > overbought and rsi_falling)) and price_below_vwap:
            depth = max(max(rsi, prev_rsi) - overbought, 0)
            strength = min(0.5 + depth / 20, 0.92)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=max(strength, 0.5),
                reason=f"RSI+VWAP SHORT: RSI {rsi:.1f} (overbought zone), price {price:.2f} at/below VWAP {vwap:.2f}",
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

        # Exit long: RSI overbought AND price losing VWAP (0.3% buffer to avoid noise exits)
        if is_long and (rsi > overbought or price < vwap * 0.997):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"RSI+VWAP exit long: RSI={rsi:.1f}, price vs VWAP={price - vwap:.2f}",
            )

        # Exit short: RSI oversold AND price regaining VWAP (0.3% buffer)
        if not is_long and (rsi < oversold or price > vwap * 1.003):
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
