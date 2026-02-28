"""
MACD Strategy — BaseStrategy implementation.
Ported from ALGOS/bt_code/bt_macd

Entry modes (all configurable via params):
1. Classic crossover: MACD crosses signal line + histogram confirmation + MA filter (original)
2. Histogram momentum: Histogram positive and growing = long, negative and shrinking = short
3. Zero-line cross: MACD crossing above/below zero line
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
        ma_filter_period = p.get("ma_filter_period", 50)
        confirm_bars = p.get("confirmation_bars", 1)  # Reduced from 2 to 1
        use_ma_filter = p.get("use_ma_filter", True)
        histogram_mode = p.get("histogram_mode", True)
        zero_cross_mode = p.get("zero_cross_mode", True)
        histogram_growth_bars = p.get("histogram_growth_bars", 3)

        min_len = max(slow, ma_filter_period) + signal_period + 5
        if len(data) < min_len:
            return None

        data = self._add_macd(data, fast, slow, signal_period, ma_filter_period)
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

        # --- Mode 1: Classic MACD crossover (original, with optional MA filter) ---
        ma_long_ok = (not use_ma_filter) or (price > ma)
        ma_short_ok = (not use_ma_filter) or (price < ma)

        if prev_macd <= prev_sig and macd > sig and (hist_series > 0).all() and ma_long_ok:
            logger.info(
                "[MACD] Classic crossover LONG: MACD %.4f crossed above signal %.4f, "
                "hist confirmed %d bars, MA filter %s",
                macd, sig, confirm_bars, "pass" if ma_long_ok else "skipped",
            )
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.9,
                reason=(
                    f"MACD LONG: MACD {macd:.4f} crossed above signal {sig:.4f}"
                    f"{f', price above MA {ma:.2f}' if use_ma_filter else ''}"
                ),
                metadata={
                    "macd": macd, "signal": sig, "histogram": hist,
                    "mode": "classic_crossover",
                },
            )

        if prev_sig <= prev_macd and sig > macd and (hist_series < 0).all() and ma_short_ok:
            logger.info(
                "[MACD] Classic crossover SHORT: signal %.4f crossed above MACD %.4f, "
                "hist confirmed %d bars",
                sig, macd, confirm_bars,
            )
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.9,
                reason=f"MACD SHORT: signal {sig:.4f} crossed above MACD {macd:.4f}",
                metadata={
                    "macd": macd, "signal": sig, "histogram": hist,
                    "mode": "classic_crossover",
                },
            )

        # --- Mode 2: Zero-line cross ---
        if zero_cross_mode:
            zero_signal = self._check_zero_cross(data, price, macd, prev_macd, sig)
            if zero_signal is not None:
                return zero_signal

        # --- Mode 3: Histogram momentum ---
        if histogram_mode:
            hist_signal = self._check_histogram_momentum(
                data, price, macd, sig, histogram_growth_bars,
            )
            if hist_signal is not None:
                return hist_signal

        return None

    def _check_zero_cross(
        self, data: pd.DataFrame, price: float,
        macd: float, prev_macd: float, sig: float,
    ) -> Optional[Signal]:
        """MACD crossing above zero = bullish, below zero = bearish."""

        # Long: MACD crosses above zero and MACD > signal (aligned)
        if prev_macd <= 0 and macd > 0 and macd > sig:
            logger.info(
                "[MACD] Zero-line cross LONG: MACD %.4f crossed above 0 (signal %.4f)",
                macd, sig,
            )
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.75,
                reason=f"MACD zero-cross LONG: MACD {macd:.4f} crossed above zero, signal {sig:.4f}",
                metadata={"macd": macd, "signal": sig, "mode": "zero_cross"},
            )

        # Short: MACD crosses below zero and MACD < signal (aligned)
        if prev_macd >= 0 and macd < 0 and macd < sig:
            logger.info(
                "[MACD] Zero-line cross SHORT: MACD %.4f crossed below 0 (signal %.4f)",
                macd, sig,
            )
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.75,
                reason=f"MACD zero-cross SHORT: MACD {macd:.4f} crossed below zero, signal {sig:.4f}",
                metadata={"macd": macd, "signal": sig, "mode": "zero_cross"},
            )

        return None

    def _check_histogram_momentum(
        self, data: pd.DataFrame, price: float,
        macd: float, sig: float, growth_bars: int,
    ) -> Optional[Signal]:
        """Histogram momentum: positive and growing = long bias,
        negative and growing (more negative) = short bias."""

        if len(data) < growth_bars + 1:
            return None

        hist_recent = data["macd_hist"].iloc[-(growth_bars + 1):]
        if hist_recent.isna().any():
            return None

        hist_values = hist_recent.values
        current_hist = hist_values[-1]

        # Check if histogram is consistently growing (each bar larger than previous)
        hist_diffs = [hist_values[i + 1] - hist_values[i] for i in range(len(hist_values) - 1)]

        # Long: histogram positive and each bar is bigger than the last
        if current_hist > 0 and all(d > 0 for d in hist_diffs):
            avg_growth = sum(hist_diffs) / len(hist_diffs)
            logger.info(
                "[MACD] Histogram momentum LONG: hist %.4f, growing for %d bars "
                "(avg growth %.4f)",
                current_hist, growth_bars, avg_growth,
            )
            strength = min(0.5 + abs(current_hist) * 10, 0.8)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"MACD histogram momentum LONG: hist {current_hist:.4f} positive and "
                    f"growing for {growth_bars} bars"
                ),
                metadata={
                    "macd": macd, "signal": sig, "histogram": current_hist,
                    "hist_growth": avg_growth, "mode": "histogram_momentum",
                },
            )

        # Short: histogram negative and each bar is more negative than the last
        if current_hist < 0 and all(d < 0 for d in hist_diffs):
            avg_growth = sum(hist_diffs) / len(hist_diffs)
            logger.info(
                "[MACD] Histogram momentum SHORT: hist %.4f, growing negative for %d bars "
                "(avg growth %.4f)",
                current_hist, growth_bars, avg_growth,
            )
            strength = min(0.5 + abs(current_hist) * 10, 0.8)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"MACD histogram momentum SHORT: hist {current_hist:.4f} negative and "
                    f"growing for {growth_bars} bars"
                ),
                metadata={
                    "macd": macd, "signal": sig, "histogram": current_hist,
                    "hist_growth": avg_growth, "mode": "histogram_momentum",
                },
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
        hist = current.get("macd_hist", 0)
        prev_hist = prev.get("macd_hist", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)
        entry_mode = position.get("metadata", {}).get("mode", "classic")

        # Exit long: signal crosses above MACD (classic exit)
        if is_long and prev_sig <= prev_macd and sig > macd:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"MACD exit: signal crossed above MACD",
            )

        # Exit short: MACD crosses above signal (classic exit)
        if not is_long and prev_macd <= prev_sig and macd > sig:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"MACD exit: MACD crossed above signal",
            )

        # Additional exit for histogram momentum entries: histogram reversal
        if entry_mode == "histogram_momentum":
            if is_long and hist < prev_hist and hist < 0:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"MACD histogram reversal exit: hist {hist:.4f} turned negative and shrinking",
                )
            if not is_long and hist > prev_hist and hist > 0:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"MACD histogram reversal exit: hist {hist:.4f} turned positive and growing",
                )

        # Additional exit for zero-cross entries: MACD crosses back through zero
        if entry_mode == "zero_cross":
            if is_long and macd < 0:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"MACD zero-cross exit: MACD {macd:.4f} fell back below zero",
                )
            if not is_long and macd > 0:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"MACD zero-cross exit: MACD {macd:.4f} rose back above zero",
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
