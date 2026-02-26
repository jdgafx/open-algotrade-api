"""
MACD Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_macd

- MACD line (12 EMA - 26 EMA) and Signal line (9 EMA of MACD)
- Long when MACD crosses above Signal + histogram confirms + price above MA filter
- Short when Signal crosses above MACD + histogram confirms
- Exit: opposite crossover, stop loss, or take profit
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MACDStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        fast = p.get("fast_period", 12)
        slow = p.get("slow_period", 26)
        signal_period = p.get("signal_period", 9)
        ma_filter = p.get("ma_filter_period", 50)
        confirm_bars = p.get("confirmation_bars", 2)

        min_len = max(slow, ma_filter) + signal_period + 5
        if len(data) < min_len:
            return None

        data = self._add_macd(data, fast, slow, signal_period, ma_filter)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        macd = current.get("macd")
        sig = current.get("macd_signal")
        hist = current.get("macd_hist")
        prev_macd = prev.get("macd")
        prev_sig = prev.get("macd_signal")
        ma = current.get("ma_filter")

        if any(pd.isna(v) for v in [macd, sig, hist, prev_macd, prev_sig, ma]):
            return None

        price = current["close"]

        # Histogram confirmation: last N bars positive/negative
        hist_series = data["macd_hist"].iloc[-confirm_bars:]
        if hist_series.isna().any():
            return None

        # Long: MACD crosses above signal, histogram positive, price above MA
        if prev_macd <= prev_sig and macd > sig and (hist_series > 0).all() and price > ma:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"MACD LONG: MACD {macd:.4f} crossed above signal {sig:.4f}, price above MA {ma:.2f}",
                metadata={"macd": macd, "signal": sig, "histogram": hist},
            )

        # Short: signal crosses above MACD, histogram negative
        if prev_sig <= prev_macd and sig > macd and (hist_series < 0).all():
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"MACD SHORT: signal {sig:.4f} crossed above MACD {macd:.4f}",
                metadata={"macd": macd, "signal": sig, "histogram": hist},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        fast = p.get("fast_period", 12)
        slow = p.get("slow_period", 26)
        signal_period = p.get("signal_period", 9)
        ma_filter = p.get("ma_filter_period", 50)

        data = self._add_macd(data, fast, slow, signal_period, ma_filter)
        current = data.iloc[-1]
        prev = data.iloc[-2]

        macd = current.get("macd", 0)
        sig = current.get("macd_signal", 0)
        prev_macd = prev.get("macd", 0)
        prev_sig = prev.get("macd_signal", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Exit long: signal crosses above MACD
        if is_long and prev_sig <= prev_macd and sig > macd:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"MACD exit: signal crossed above MACD",
            )

        # Exit short: MACD crosses above signal
        if not is_long and prev_macd <= prev_sig and macd > sig:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"MACD exit: MACD crossed above signal",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int, ma_period: int) -> pd.DataFrame:
        if "macd" in df.columns:
            return df
        df = df.copy()
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["ma_filter"] = df["close"].ewm(span=ma_period, adjust=False).mean()
        return df
