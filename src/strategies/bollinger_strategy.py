"""
Bollinger Band Strategy — BaseStrategy implementation.

Rewritten to match MoonDev's "Harvard Algorithmic Trading with AI" approach:
detect a Bollinger Band squeeze using Keltner Channels (BB inside KC = low
volatility), then trade the breakout once the squeeze releases, confirmed by
ADX > 25.

Logic:
- Squeeze:        upper_bb < upper_kc AND lower_bb > lower_kc
- Squeeze release: previous bar in squeeze, current bar NOT in squeeze
- Long:  squeeze_release AND close > bb_upper AND ADX > 25
- Short: squeeze_release AND close < bb_lower AND ADX > 25
- Exit: price crosses back through BB middle (SMA), or TP/SL fallback
  from self.config.target_pct / self.config.max_loss_pct (defaults 5% / -3%
  per MoonDev's spec, but driven by config so global overrides still apply).

Defaults (MoonDev's exact params):
- BB period=20, std=2.0
- Keltner period=20, ATR multiplier=1.5
- ADX period=14, threshold=25
- TP 5%, SL 3% (set via params or config)
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
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        kc_period = p.get("kc_period", 20)
        kc_mult = p.get("kc_mult", 1.5)
        adx_period = p.get("adx_period", 14)
        adx_threshold = p.get("adx_threshold", 25.0)

        min_bars = max(bb_period, kc_period, adx_period * 2) + 2
        if len(data) < min_bars:
            return None

        data = self._add_indicators(
            data, bb_period, bb_std, kc_period, kc_mult, adx_period
        )
        current = data.iloc[-1]
        prev = data.iloc[-2]

        # Need clean indicator values
        for col in ("bb_upper", "bb_lower", "kc_upper", "kc_lower", "adx"):
            if pd.isna(current[col]) or pd.isna(prev[col]):
                return None

        price = current["close"]
        adx = current["adx"]

        prev_squeeze = bool(prev["kc_squeeze"])
        current_squeeze = bool(current["kc_squeeze"])
        squeeze_release = prev_squeeze and not current_squeeze

        if not squeeze_release or adx <= adx_threshold:
            return None

        # Strength: 0.65 + (adx-25)/100, capped at 0.95
        strength = min(0.65 + (adx - adx_threshold) / 100.0, 0.95)

        if price > current["bb_upper"]:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"BB squeeze release LONG: ADX {adx:.1f} > {adx_threshold}, "
                    f"price {price:.2f} > upper {current['bb_upper']:.2f}"
                ),
                metadata={
                    "adx": adx,
                    "bb_upper": current["bb_upper"],
                    "bb_lower": current["bb_lower"],
                    "kc_upper": current["kc_upper"],
                    "kc_lower": current["kc_lower"],
                    "squeeze_release": True,
                },
            )

        if price < current["bb_lower"]:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"BB squeeze release SHORT: ADX {adx:.1f} > {adx_threshold}, "
                    f"price {price:.2f} < lower {current['bb_lower']:.2f}"
                ),
                metadata={
                    "adx": adx,
                    "bb_upper": current["bb_upper"],
                    "bb_lower": current["bb_lower"],
                    "kc_upper": current["kc_upper"],
                    "kc_lower": current["kc_lower"],
                    "squeeze_release": True,
                },
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        kc_period = p.get("kc_period", 20)
        kc_mult = p.get("kc_mult", 1.5)
        adx_period = p.get("adx_period", 14)

        data = self._add_indicators(
            data, bb_period, bb_std, kc_period, kc_mult, adx_period
        )
        current = data.iloc[-1]
        price = current["close"]
        sma = current["bb_mid"]

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Primary exit: price crosses back through the BB middle (SMA)
        if not pd.isna(sma):
            if is_long and price <= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"BB exit: price {price:.2f} crossed back below SMA {sma:.2f}",
                )
            if (not is_long) and price >= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"BB exit: price {price:.2f} crossed back above SMA {sma:.2f}",
                )

        # TP / SL fallback driven by strategy config (MoonDev: 5% TP, 3% SL)
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Target hit: {pnl_pct:.2f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Max loss: {pnl_pct:.2f}%",
            )

        return None

    @staticmethod
    def _add_indicators(
        df: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        adx_period: int = 14,
    ) -> pd.DataFrame:
        """Compute Bollinger Bands, Keltner Channels, squeeze flag, and ADX.

        Idempotent: if the columns are already present, returns df unchanged.
        Pure numpy/pandas (no TA-Lib).
        """
        required = {
            "bb_upper",
            "bb_lower",
            "bb_mid",
            "kc_upper",
            "kc_lower",
            "kc_mid",
            "kc_squeeze",
            "adx",
        }
        if required.issubset(df.columns):
            return df

        df = df.copy()

        # ---- Bollinger Bands (SMA + std) ----
        df["bb_mid"] = df["close"].rolling(window=bb_period).mean()
        bb_std_series = df["close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_mid"] + bb_std_series * bb_std
        df["bb_lower"] = df["bb_mid"] - bb_std_series * bb_std

        # ---- True Range / ATR (Wilder's smoothing) ----
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Wilder's ATR for both KC (uses kc/adx period — they share 14 by default
        # in classic ADX, but Keltner here uses its own period via ewm).
        atr_kc = tr.ewm(alpha=1.0 / kc_period, adjust=False).mean()
        atr_adx = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()

        # ---- Keltner Channels: EMA(close) ± mult * ATR ----
        df["kc_mid"] = close.ewm(span=kc_period, adjust=False).mean()
        df["kc_upper"] = df["kc_mid"] + kc_mult * atr_kc
        df["kc_lower"] = df["kc_mid"] - kc_mult * atr_kc

        # ---- Squeeze flag: BB inside KC ----
        df["kc_squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (
            df["bb_lower"] > df["kc_lower"]
        )

        # ---- ADX (Wilder, 14-period default) ----
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        plus_dm_smooth = plus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean()

        # Avoid divide-by-zero
        atr_safe = atr_adx.replace(0, np.nan)
        plus_di = 100.0 * (plus_dm_smooth / atr_safe)
        minus_di = 100.0 * (minus_dm_smooth / atr_safe)

        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100.0 * (di_diff / di_sum.replace(0, np.nan))
        df["adx"] = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()

        return df
