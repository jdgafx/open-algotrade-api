"""
Mean Reversion Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/6_mean_reversion/74_tickers_mean_reversion.py

Entry modes (all configurable via params):
1. Original: Multi-timeframe SMA trend filter + deviation entry
2. Single-timeframe Bollinger Bands: Price below lower band = long, above upper = short
3. Z-score based: Enter when price is >N std devs from SMA
4. Dynamic position sizing based on distance from mean
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    _entry_mode: str = "original"

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_entry_period", 20)
        reversion_target = p.get("reversion_target_pct", 0.003)
        single_timeframe = p.get("single_timeframe", True)
        zscore_entry = p.get("zscore_entry", 1.5)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        dynamic_sizing = p.get("dynamic_sizing", True)

        min_len = max(sma_period, bb_period) + 5
        if len(data) < min_len:
            return None

        data = self._add_indicators(data, sma_period, bb_period, bb_std)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]
        sma = current["sma"]

        if pd.isna(sma):
            return None

        deviation = (price - sma) / sma

        # --- Mode 1 (single-timeframe): Z-score based entries ---
        if single_timeframe:
            zscore = current.get("zscore", None)
            if zscore is not None and not pd.isna(zscore):
                zscore_signal = self._check_zscore_entry(
                    price, sma, zscore, zscore_entry, deviation, dynamic_sizing,
                )
                if zscore_signal is not None:
                    return zscore_signal

            # Bollinger Band entries
            bb_upper = current.get("bb_upper", None)
            bb_lower = current.get("bb_lower", None)
            if bb_upper is not None and bb_lower is not None:
                bb_signal = self._check_bollinger_entry(
                    data, price, sma, bb_upper, bb_lower, deviation, dynamic_sizing,
                )
                if bb_signal is not None:
                    return bb_signal

        # --- Mode 2: Original SMA deviation + trend filter ---
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
            size = self._dynamic_size(deviation, dynamic_sizing)
            logger.info(
                "[MeanRev] Original mode LONG: deviation %.4f, price %.2f < SMA %.2f, "
                "trend bullish, size_usd %.2f",
                deviation, price, sma, size,
            )
            self._entry_mode = "original"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=min(0.5 + abs(deviation) * 20, 0.9),
                reason=f"Mean reversion LONG: deviation {deviation:.4f}, price {price:.2f} < SMA {sma:.2f}",
                metadata={"deviation": deviation, "sma": sma, "trend": "bullish", "mode": "original"},
            )

        # Mean reversion short: price above SMA in downtrend
        if not trend_bullish and deviation > 0.005 and current["close"] < prev["close"]:
            size = self._dynamic_size(deviation, dynamic_sizing)
            logger.info(
                "[MeanRev] Original mode SHORT: deviation %.4f, price %.2f > SMA %.2f, "
                "trend bearish, size_usd %.2f",
                deviation, price, sma, size,
            )
            self._entry_mode = "original"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=min(0.5 + abs(deviation) * 20, 0.9),
                reason=f"Mean reversion SHORT: deviation {deviation:.4f}, price {price:.2f} > SMA {sma:.2f}",
                metadata={"deviation": deviation, "sma": sma, "trend": "bearish", "mode": "original"},
            )

        return None

    def _check_zscore_entry(
        self, price: float, sma: float, zscore: float,
        zscore_threshold: float, deviation: float, dynamic_sizing: bool,
    ) -> Optional[Signal]:
        """Enter mean reversion when z-score exceeds threshold."""

        # Long: price is significantly below mean (negative z-score)
        if zscore <= -zscore_threshold:
            size = self._dynamic_size(deviation, dynamic_sizing)
            strength = min(0.6 + abs(zscore) / 10, 0.95)
            logger.info(
                "[MeanRev] Z-score LONG: zscore %.2f <= -%.2f, price %.2f, SMA %.2f, "
                "size_usd %.2f",
                zscore, zscore_threshold, price, sma, size,
            )
            self._entry_mode = "zscore"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=strength,
                reason=(
                    f"Mean reversion z-score LONG: z={zscore:.2f} below -{zscore_threshold}, "
                    f"price {price:.2f} far below SMA {sma:.2f}"
                ),
                metadata={
                    "zscore": zscore, "deviation": deviation, "sma": sma,
                    "mode": "zscore",
                },
            )

        # Short: price is significantly above mean (positive z-score)
        if zscore >= zscore_threshold:
            size = self._dynamic_size(deviation, dynamic_sizing)
            strength = min(0.6 + abs(zscore) / 10, 0.95)
            logger.info(
                "[MeanRev] Z-score SHORT: zscore %.2f >= %.2f, price %.2f, SMA %.2f, "
                "size_usd %.2f",
                zscore, zscore_threshold, price, sma, size,
            )
            self._entry_mode = "zscore"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=strength,
                reason=(
                    f"Mean reversion z-score SHORT: z={zscore:.2f} above {zscore_threshold}, "
                    f"price {price:.2f} far above SMA {sma:.2f}"
                ),
                metadata={
                    "zscore": zscore, "deviation": deviation, "sma": sma,
                    "mode": "zscore",
                },
            )

        return None

    def _check_bollinger_entry(
        self, data: pd.DataFrame, price: float, sma: float,
        bb_upper: float, bb_lower: float, deviation: float, dynamic_sizing: bool,
    ) -> Optional[Signal]:
        """Enter when price pierces Bollinger Band and starts reverting."""

        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None

        prev = data.iloc[-2]
        prev_price = prev["close"]
        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return None

        # Long: price was below lower band and is now bouncing back
        if price <= bb_lower and price > prev_price:
            size = self._dynamic_size(deviation, dynamic_sizing)
            logger.info(
                "[MeanRev] Bollinger LONG: price %.2f <= lower band %.2f, bouncing "
                "(prev %.2f), SMA %.2f, size_usd %.2f",
                price, bb_lower, prev_price, sma, size,
            )
            self._entry_mode = "bollinger"
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=0.8,
                reason=(
                    f"Bollinger band LONG: price {price:.2f} at lower band {bb_lower:.2f}, "
                    f"bouncing toward SMA {sma:.2f}"
                ),
                metadata={
                    "bb_lower": bb_lower, "bb_upper": bb_upper, "sma": sma,
                    "deviation": deviation, "mode": "bollinger",
                },
            )

        # Short: price was above upper band and is now falling back
        if price >= bb_upper and price < prev_price:
            size = self._dynamic_size(deviation, dynamic_sizing)
            logger.info(
                "[MeanRev] Bollinger SHORT: price %.2f >= upper band %.2f, falling "
                "(prev %.2f), SMA %.2f, size_usd %.2f",
                price, bb_upper, prev_price, sma, size,
            )
            self._entry_mode = "bollinger"
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=size,
                strength=0.8,
                reason=(
                    f"Bollinger band SHORT: price {price:.2f} at upper band {bb_upper:.2f}, "
                    f"falling toward SMA {sma:.2f}"
                ),
                metadata={
                    "bb_upper": bb_upper, "bb_lower": bb_lower, "sma": sma,
                    "deviation": deviation, "mode": "bollinger",
                },
            )

        return None

    def _dynamic_size(self, deviation: float, enabled: bool) -> float:
        """Scale position size by distance from mean. Further = larger (more confident
        in reversion). Caps at 2x base size."""
        if not enabled:
            return self.config.size_usd
        abs_dev = abs(deviation)
        # Scale: 1x at 0.5% deviation, up to 2x at 2%+ deviation
        scale = min(1.0 + abs_dev * 50, 2.0)
        sized = self.config.size_usd * scale
        logger.debug(
            "[MeanRev] Dynamic sizing: deviation %.4f -> scale %.2fx -> size_usd %.2f",
            deviation, scale, sized,
        )
        return sized

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sma_period = p.get("sma_entry_period", 20)
        bb_period = p.get("bb_period", 20)
        bb_std = p.get("bb_std", 2.0)
        reversion_target = p.get("reversion_target_pct", 0.003)
        zscore_exit = p.get("zscore_exit", 0.5)

        data = self._add_indicators(data, sma_period, bb_period, bb_std)
        current = data.iloc[-1]
        price = current["close"]
        sma = current.get("sma", price)
        zscore = current.get("zscore", 0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        entry_px = position.get("entry_px", 0)
        pnl_pct = position.get("pnl_perc", 0)
        entry_mode = self._entry_mode

        # Z-score based exit: price has reverted close to mean
        if entry_mode == "zscore" and not pd.isna(zscore):
            if is_long and zscore >= -zscore_exit:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Z-score exit: z={zscore:.2f} reverted above -{zscore_exit}",
                )
            if not is_long and zscore <= zscore_exit:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Z-score exit: z={zscore:.2f} reverted below {zscore_exit}",
                )

        # Bollinger exit: price returns to SMA (middle band)
        if entry_mode == "bollinger":
            if is_long and price >= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Bollinger exit: price {price:.2f} reached SMA {sma:.2f}",
                )
            if not is_long and price <= sma:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"Bollinger exit: price {price:.2f} reached SMA {sma:.2f}",
                )

        # Original exit: price reverts to mean (SMA)
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
    def _add_indicators(
        df: pd.DataFrame, sma_period: int, bb_period: int, bb_std: float,
    ) -> pd.DataFrame:
        """Add SMA, Bollinger Bands, and z-score indicators."""
        if "sma" in df.columns and "bb_upper" in df.columns:
            return df
        df = df.copy()

        # SMA
        df["sma"] = df["close"].rolling(window=sma_period).mean()

        # Bollinger Bands
        bb_sma = df["close"].rolling(window=bb_period).mean()
        bb_rolling_std = df["close"].rolling(window=bb_period).std()
        df["bb_upper"] = bb_sma + bb_std * bb_rolling_std
        df["bb_lower"] = bb_sma - bb_std * bb_rolling_std

        # Z-score: how many std devs price is from SMA
        df["zscore"] = (df["close"] - df["sma"]) / bb_rolling_std.replace(0, 1)

        return df

    @staticmethod
    def _add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Kept for backward compatibility."""
        if "sma" in df.columns:
            return df
        df = df.copy()
        df["sma"] = df["close"].rolling(window=period).mean()
        return df
