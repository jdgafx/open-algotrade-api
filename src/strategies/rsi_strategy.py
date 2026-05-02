"""
RSI Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/7_rsi.py

Entry modes (all configurable via params):
1. Classic oversold/overbought reversal (original)
2. Trend mode: RSI above/below 50 and moving = trend-following signal
3. Divergence mode: Price-RSI divergence detection for early entries
4. Momentum confirmation: RSI direction + rate of change
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):

    _entry_mode: str = "classic"

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)
        oversold = p.get("oversold", 30)
        overbought = p.get("overbought", 70)
        trend_mode = p.get("trend_mode", True)
        divergence_mode = p.get("divergence_mode", True)
        trend_long_threshold = p.get("trend_rsi_threshold_long", 55)
        trend_short_threshold = p.get("trend_rsi_threshold_short", 45)
        divergence_lookback = p.get("divergence_lookback", 14)
        rsi_momentum_period = p.get("rsi_momentum_period", 3)

        min_len = rsi_period + max(divergence_lookback, 5) + 3
        if len(data) < min_len:
            return None

        data = self._add_rsi(data, rsi_period)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        rsi = current["rsi"]
        prev_rsi = prev.get("rsi", 50)

        if pd.isna(rsi) or pd.isna(prev_rsi):
            return None

        # --- Mode 1: Classic oversold/overbought reversal (original logic) ---
        if prev_rsi < oversold and rsi >= oversold:
            logger.info(
                "[RSI] Classic oversold reversal LONG: RSI %.1f -> %.1f crossing above %d",
                prev_rsi, rsi, oversold,
            )
            self._entry_mode = "classic"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.9,
                reason=f"RSI oversold reversal LONG: RSI {prev_rsi:.1f} -> {rsi:.1f} crossing above {oversold}",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi, "mode": "classic_oversold"},
            )

        if prev_rsi > overbought and rsi <= overbought:
            logger.info(
                "[RSI] Classic overbought reversal SHORT: RSI %.1f -> %.1f crossing below %d",
                prev_rsi, rsi, overbought,
            )
            self._entry_mode = "classic"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.9,
                reason=f"RSI overbought reversal SHORT: RSI {prev_rsi:.1f} -> {rsi:.1f} crossing below {overbought}",
                metadata={"rsi": rsi, "prev_rsi": prev_rsi, "mode": "classic_overbought"},
            )

        # --- Mode 2: Divergence detection ---
        if divergence_mode:
            divergence_signal = self._check_divergence(
                data, price, rsi, divergence_lookback,
            )
            if divergence_signal is not None:
                return divergence_signal

        # --- Mode 3: Trend-following mode ---
        if trend_mode:
            trend_signal = self._check_trend_mode(
                data, price, rsi, prev_rsi,
                trend_long_threshold, trend_short_threshold,
                rsi_momentum_period,
            )
            if trend_signal is not None:
                return trend_signal

        return None

    def _check_divergence(
        self, data: pd.DataFrame, price: float, rsi: float, lookback: int
    ) -> Optional[Signal]:
        """Detect RSI divergence: price makes new low but RSI makes higher low (bullish)
        or price makes new high but RSI makes lower high (bearish)."""
        recent = data.iloc[-lookback:]

        if len(recent) < lookback or recent["rsi"].isna().any():
            return None

        prices = recent["close"].values
        rsis = recent["rsi"].values

        # Find the lowest price point in the lookback window (excluding last bar)
        price_min_idx = prices[:-1].argmin()
        price_min = prices[price_min_idx]
        rsi_at_price_min = rsis[price_min_idx]

        # Bullish divergence: current price near/below prior low, but RSI is higher
        if price <= price_min * 1.002 and rsi > rsi_at_price_min + 3:
            logger.info(
                "[RSI] Bullish divergence detected: price %.2f near low %.2f, "
                "RSI %.1f > prior RSI %.1f (higher low)",
                price, price_min, rsi, rsi_at_price_min,
            )
            self._entry_mode = "divergence"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.85,
                reason=(
                    f"RSI bullish divergence LONG: price {price:.2f} at low but "
                    f"RSI {rsi:.1f} > prior {rsi_at_price_min:.1f} (higher low)"
                ),
                metadata={
                    "rsi": rsi, "rsi_at_prior_low": rsi_at_price_min,
                    "price_low": price_min, "mode": "divergence_bullish",
                },
            )

        # Find the highest price point in the lookback window (excluding last bar)
        price_max_idx = prices[:-1].argmax()
        price_max = prices[price_max_idx]
        rsi_at_price_max = rsis[price_max_idx]

        # Bearish divergence: current price near/above prior high, but RSI is lower
        if price >= price_max * 0.998 and rsi < rsi_at_price_max - 3:
            logger.info(
                "[RSI] Bearish divergence detected: price %.2f near high %.2f, "
                "RSI %.1f < prior RSI %.1f (lower high)",
                price, price_max, rsi, rsi_at_price_max,
            )
            self._entry_mode = "divergence"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=0.85,
                reason=(
                    f"RSI bearish divergence SHORT: price {price:.2f} at high but "
                    f"RSI {rsi:.1f} < prior {rsi_at_price_max:.1f} (lower high)"
                ),
                metadata={
                    "rsi": rsi, "rsi_at_prior_high": rsi_at_price_max,
                    "price_high": price_max, "mode": "divergence_bearish",
                },
            )

        return None

    def _check_trend_mode(
        self, data: pd.DataFrame, price: float, rsi: float, prev_rsi: float,
        long_threshold: float, short_threshold: float, momentum_period: int,
    ) -> Optional[Signal]:
        """Trend-following: RSI above threshold and rising = long,
        RSI below threshold and falling = short. Confirmed by momentum."""

        # Calculate RSI rate of change over momentum_period bars
        if len(data) < momentum_period + 2:
            return None

        rsi_values = data["rsi"].iloc[-(momentum_period + 1):]
        if rsi_values.isna().any():
            return None

        rsi_roc = rsi - rsi_values.iloc[0]  # RSI change over momentum_period
        rsi_rising = rsi > prev_rsi and rsi_roc > 0
        rsi_falling = rsi < prev_rsi and rsi_roc < 0

        # Long: RSI above threshold and consistently rising
        if rsi > long_threshold and rsi_rising and rsi_roc > 2:
            logger.info(
                "[RSI] Trend mode LONG: RSI %.1f > %d, rising (ROC %.1f over %d bars)",
                rsi, long_threshold, rsi_roc, momentum_period,
            )
            # Scale strength by how far above threshold
            strength = min(0.5 + (rsi - long_threshold) / 50, 0.8)
            self._entry_mode = "trend"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"RSI trend LONG: RSI {rsi:.1f} above {long_threshold}, "
                    f"rising (ROC {rsi_roc:.1f} over {momentum_period} bars)"
                ),
                metadata={
                    "rsi": rsi, "rsi_roc": rsi_roc, "prev_rsi": prev_rsi,
                    "mode": "trend_long",
                },
            )

        # Short: RSI below threshold and consistently falling
        if rsi < short_threshold and rsi_falling and rsi_roc < -2:
            logger.info(
                "[RSI] Trend mode SHORT: RSI %.1f < %d, falling (ROC %.1f over %d bars)",
                rsi, short_threshold, rsi_roc, momentum_period,
            )
            strength = min(0.5 + (short_threshold - rsi) / 50, 0.8)
            self._entry_mode = "trend"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"RSI trend SHORT: RSI {rsi:.1f} below {short_threshold}, "
                    f"falling (ROC {rsi_roc:.1f} over {momentum_period} bars)"
                ),
                metadata={
                    "rsi": rsi, "rsi_roc": rsi_roc, "prev_rsi": prev_rsi,
                    "mode": "trend_short",
                },
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        rsi_period = p.get("rsi_period", 14)

        data = self._add_rsi(data, rsi_period)
        current = data.iloc[-1]
        rsi = current.get("rsi", 50)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Determine entry mode from position metadata to use appropriate exit logic
        entry_mode = self._entry_mode

        if entry_mode.startswith("trend"):
            # Trend exits: RSI reversal against trend direction
            if is_long and rsi < 45:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"RSI trend exit: RSI {rsi:.1f} fell below 45 (trend weakening)",
                )
            if not is_long and rsi > 55:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"RSI trend exit: RSI {rsi:.1f} rose above 55 (trend weakening)",
                )
        else:
            # Classic/divergence exits: RSI crosses neutral (50)
            if is_long and rsi >= 50:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"RSI exit: RSI {rsi:.1f} reached neutral zone",
                )
            if not is_long and rsi <= 50:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"RSI exit: RSI {rsi:.1f} reached neutral zone",
                )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
        if "rsi" in df.columns:
            return df
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df
