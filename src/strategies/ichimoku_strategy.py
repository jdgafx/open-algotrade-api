"""
Ichimoku Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_ichimoku

- Full Ichimoku Cloud system (Tenkan, Kijun, Senkou A/B, Chikou)
- Long: price above cloud + Tenkan crosses above Kijun
- Short: price below cloud + Kijun crosses above Tenkan
- Exit: price re-enters cloud or opposite TK cross
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class IchimokuStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        tenkan_period = p.get("tenkan_period", 9)
        kijun_period = p.get("kijun_period", 26)
        senkou_b_period = p.get("senkou_b_period", 52)

        min_len = senkou_b_period + 5
        if len(data) < min_len:
            return None

        data = self._add_ichimoku(data, tenkan_period, kijun_period, senkou_b_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        tenkan = current.get("tenkan")
        kijun = current.get("kijun")
        span_a = current.get("senkou_a")
        span_b = current.get("senkou_b")
        prev_tenkan = prev.get("tenkan")
        prev_kijun = prev.get("kijun")

        if any(pd.isna(v) for v in [tenkan, kijun, span_a, span_b, prev_tenkan, prev_kijun]):
            return None

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        cloud_thickness = abs(span_a - span_b)

        # Long: price above cloud + Tenkan above Kijun (bullish alignment)
        # Crossover (just_crossed) gives stronger signal; already-aligned is a valid entry too
        if price > cloud_top and tenkan > kijun:
            just_crossed = prev_tenkan <= prev_kijun
            dist_above = (price - cloud_top) / price if price > 0 else 0
            strength = min(0.75 if just_crossed else 0.6 + dist_above * 10 + cloud_thickness / price * 5, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Ichimoku LONG: {'TK cross ' if just_crossed else ''}bullish, price {price:.2f} above cloud [{cloud_bottom:.2f}-{cloud_top:.2f}]",
                metadata={"tenkan": tenkan, "kijun": kijun, "cloud_top": cloud_top},
            )

        # Short: price below cloud + Kijun above Tenkan (bearish alignment)
        if price < cloud_bottom and kijun > tenkan:
            just_crossed = prev_kijun <= prev_tenkan
            dist_below = (cloud_bottom - price) / price if price > 0 else 0
            strength = min(0.75 if just_crossed else 0.6 + dist_below * 10 + cloud_thickness / price * 5, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Ichimoku SHORT: TK cross bearish, price {price:.2f} below cloud [{cloud_bottom:.2f}-{cloud_top:.2f}]",
                metadata={"tenkan": tenkan, "kijun": kijun, "cloud_bottom": cloud_bottom},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        tenkan_period = p.get("tenkan_period", 9)
        kijun_period = p.get("kijun_period", 26)
        senkou_b_period = p.get("senkou_b_period", 52)

        data = self._add_ichimoku(data, tenkan_period, kijun_period, senkou_b_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        price = current["close"]
        tenkan = current.get("tenkan", 0)
        kijun = current.get("kijun", 0)
        span_b = current.get("senkou_b", 0)
        prev_tenkan = prev.get("tenkan", 0)
        prev_kijun = prev.get("kijun", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: price falls below Kijun or below cloud, or bearish TK cross
        if is_long and (
            price < kijun
            or price < span_b
            or (prev_kijun <= prev_tenkan and kijun > tenkan)
        ):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Ichimoku exit long: price={price:.2f} kijun={kijun:.2f}",
            )

        # Exit short: price rises above Kijun or above cloud, or bullish TK cross
        if not is_long and (
            price > kijun
            or price > current.get("senkou_a", 0)
            or (prev_tenkan <= prev_kijun and tenkan > kijun)
        ):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"Ichimoku exit short: price={price:.2f} kijun={kijun:.2f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_ichimoku(df: pd.DataFrame, tenkan_p: int, kijun_p: int, senkou_b_p: int) -> pd.DataFrame:
        if "tenkan" in df.columns:
            return df
        df = df.copy()
        high = df["high"]
        low = df["low"]

        df["tenkan"] = (high.rolling(tenkan_p).max() + low.rolling(tenkan_p).min()) / 2
        df["kijun"] = (high.rolling(kijun_p).max() + low.rolling(kijun_p).min()) / 2
        df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2)
        df["senkou_b"] = (high.rolling(senkou_b_p).max() + low.rolling(senkou_b_p).min()) / 2
        return df
