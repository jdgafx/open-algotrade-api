"""
SMA + ADX + Bollinger + Volume Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_sma_adx_crossovers_bollinger_volume

- Multi-indicator confirmation system:
  1. SMA crossover for trend direction
  2. ADX > threshold for trend strength
  3. Bollinger Band contraction for volatility squeeze
  4. Volume above threshold for confirmation
- Long: price crosses above SMA + ADX strong + BB squeeze + volume
- Short: price crosses below SMA + ADX strong + BB squeeze + volume
- Exit: opposite SMA cross, ADX weakens, or BB expands
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SMAAdxBBVolStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_period", 20)
        adx_period = p.get("adx_period", 14)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        min_adx = p.get("min_adx", 20)
        vol_multiplier = p.get("volume_multiplier", 1.5)

        min_len = max(sma_period, adx_period, bb_period) + 10
        if len(data) < min_len:
            return None

        data = self._add_indicators(data, sma_period, adx_period, bb_period, bb_std)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        sma = current.get("sma")
        prev_price = prev["close"]
        prev_sma = prev.get("sma")
        adx = current.get("adx")
        bb_width = current.get("bb_width")
        avg_bb_width = current.get("avg_bb_width")
        volume = current.get("volume", 0)
        avg_volume = data["volume"].iloc[-sma_period:].mean() if "volume" in data.columns else 0

        if any(pd.isna(v) for v in [sma, prev_sma, adx, bb_width, avg_bb_width]):
            return None

        vol_ok = avg_volume == 0 or volume > avg_volume * vol_multiplier
        squeeze = bb_width < avg_bb_width
        trend_strong = adx > min_adx

        # Long: price crosses above SMA + all confirmations
        if prev_price <= prev_sma and price > sma and trend_strong and squeeze and vol_ok:
            # Multi-confirmation = strong signal. Scale by ADX strength.
            strength = min(0.6 + (adx - min_adx) / 40, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Multi LONG: SMA cross + ADX={adx:.1f} + BB squeeze + vol confirmed",
                metadata={"sma": sma, "adx": adx, "bb_width": bb_width},
            )

        # Short: price crosses below SMA + all confirmations
        if prev_price >= prev_sma and price < sma and trend_strong and squeeze and vol_ok:
            strength = min(0.6 + (adx - min_adx) / 40, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Multi SHORT: SMA cross + ADX={adx:.1f} + BB squeeze + vol confirmed",
                metadata={"sma": sma, "adx": adx, "bb_width": bb_width},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_period", 20)
        adx_period = p.get("adx_period", 14)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        min_adx = p.get("min_adx", 20)

        data = self._add_indicators(data, sma_period, adx_period, bb_period, bb_std)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        sma = current.get("sma", 0)
        prev_price = prev["close"]
        prev_sma = prev.get("sma", 0)
        adx = current.get("adx", 0)
        bb_width = current.get("bb_width", 0)
        avg_bb_width = current.get("avg_bb_width", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: SMA cross down or ADX weakens significantly
        # NOTE: Removed BB expansion exit -- it triggers immediately after squeeze
        # breakout entries, causing instant exits. Squeeze -> expansion IS the trade.
        if is_long and (
            (prev_price >= prev_sma and price < sma)
            or adx < min_adx * 0.7  # ADX must drop well below threshold, not just touch it
        ):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Multi exit long: SMA cross or ADX weak ({adx:.1f})",
            )

        # Exit short: SMA cross up or ADX weakens significantly
        if not is_long and (
            (prev_price <= prev_sma and price > sma)
            or adx < min_adx * 0.7
        ):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Multi exit short: SMA cross or ADX weak ({adx:.1f})",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_indicators(df: pd.DataFrame, sma_p: int, adx_p: int, bb_p: int, bb_std: float) -> pd.DataFrame:
        if "sma" in df.columns and "adx" in df.columns:
            return df
        df = df.copy()

        # SMA
        df["sma"] = df["close"].rolling(window=sma_p).mean()

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(window=bb_p).mean()
        std = df["close"].rolling(window=bb_p).std()
        df["bb_upper"] = df["bb_mid"] + bb_std * std
        df["bb_lower"] = df["bb_mid"] - bb_std * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, 1)
        df["avg_bb_width"] = df["bb_width"].rolling(window=bb_p).mean()

        # ADX
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        atr = tr.rolling(window=adx_p).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=adx_p).mean() / atr.replace(0, 1)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=adx_p).mean() / atr.replace(0, 1)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1) * 100
        df["adx"] = dx.rolling(window=adx_p).mean()

        return df
