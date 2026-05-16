"""
Liquidation ADX + RSI Divergence Strategy.

MoonDev source: "$1.7 Billion Liquidated So I'm Building A Liquidation Bot"
(YouTube zyYsX_1PqMI). Best variant: 494% return, max DD 6.67%, avg DD 1.05%,
Sharpe 2.81. Unoptimized 340% vs B&H 65%.

Logic:
1. Detect a major liquidation event via volume spike proxy
   (candle volume > vol_spike_mult × N-bar average — same proxy as liqdip).
2. Once a liquidation event fires, arm the strategy for 1 entry within max_arm_bars.
3. On armed bars, require ALL three confirmations before entering:
   a. ADX > adx_threshold (trend is strong enough to ride)
   b. +DI > -DI → LONG; -DI > +DI → SHORT (direction from directional index)
   c. RSI divergence confirmation: price new-low + RSI higher-low → bullish;
      price new-high + RSI lower-high → bearish (matches the signal direction)
4. Exit: ADX falls below exit_threshold OR stop/target hit.

Parameter defaults are MoonDev-informed but intentionally tunable for RBI.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


def _wilder_rma(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Running Moving Average (used by ADX/RSI)."""
    n = len(series)
    result = np.zeros(n)
    if n < period:
        return result
    result[period - 1] = series[:period].mean()
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + series[i]) / period
    return result


def _adx_di(df: pd.DataFrame, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ADX, +DI, -DI) arrays."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(close)

    prev_high = np.empty(n)
    prev_low = np.empty(n)
    prev_close = np.empty(n)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_high[1:] = high[:-1]
    prev_low[1:] = low[:-1]
    prev_close[1:] = close[:-1]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    dm_plus = np.where(
        (high - prev_high) > (prev_low - low),
        np.maximum(high - prev_high, 0.0), 0.0,
    )
    dm_minus = np.where(
        (prev_low - low) > (high - prev_high),
        np.maximum(prev_low - low, 0.0), 0.0,
    )

    tr_rma = _wilder_rma(tr, period)
    dp_rma = _wilder_rma(dm_plus, period)
    dm_rma = _wilder_rma(dm_minus, period)

    di_plus = np.where(tr_rma > 0, 100 * dp_rma / tr_rma, 0.0)
    di_minus = np.where(tr_rma > 0, 100 * dm_rma / tr_rma, 0.0)

    dx = np.where(
        (di_plus + di_minus) > 0,
        100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0.0,
    )
    adx = _wilder_rma(dx, period)
    return adx, di_plus, di_minus


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


class LiquidationAdxStrategy(BaseStrategy):
    """
    Liquidation-triggered ADX + RSI divergence entry.

    State machine: IDLE → ARMED (on volume spike) → ENTERED → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._armed_bars_remaining: int = 0
        self._liq_events: List[Dict] = []

    def inject_liquidation_data(self, events: List[Dict]) -> None:
        """Called by orchestrator to inject real liquidation events when available."""
        self._liq_events = events

    def _detect_liquidation(self, df: pd.DataFrame, vol_mult: float, vol_lookback: int) -> bool:
        """Volume spike proxy for a major liquidation event."""
        if len(df) < vol_lookback + 1:
            return False
        vol = df["volume"].to_numpy(float)
        current_vol = vol[-1]
        avg_vol = vol[-vol_lookback - 1:-1].mean()
        if avg_vol <= 0:
            return False
        spike = current_vol / avg_vol >= vol_mult
        hl_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
        avg_range = float((df["high"] - df["low"]).iloc[-vol_lookback - 1:-1].mean())
        big_candle = hl_range >= 1.5 * avg_range if avg_range > 0 else False
        return spike and big_candle

    def _rsi_divergence(self, close: np.ndarray, rsi: np.ndarray, lookback: int, direction: str) -> bool:
        """
        Check for RSI divergence over last `lookback` bars.
        LONG: price lower-low but RSI higher-low (bullish divergence).
        SHORT: price higher-high but RSI lower-high (bearish divergence).
        """
        if len(close) < lookback + 1:
            return False
        window_close = close[-lookback - 1:]
        window_rsi = rsi[-lookback - 1:]
        if direction == "LONG":
            price_new_low = window_close[-1] < window_close[:-1].min()
            rsi_higher_low = window_rsi[-1] > window_rsi[:-1].min()
            return bool(price_new_low and rsi_higher_low)
        if direction == "SHORT":
            price_new_high = window_close[-1] > window_close[:-1].max()
            rsi_lower_high = window_rsi[-1] < window_rsi[:-1].max()
            return bool(price_new_high and rsi_lower_high)
        return False

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        adx_period = int(p.get("adx_period", 14))
        adx_threshold = float(p.get("adx_threshold", 25.0))
        rsi_period = int(p.get("rsi_period", 14))
        div_lookback = int(p.get("divergence_lookback", 8))
        vol_mult = float(p.get("vol_spike_mult", 2.5))
        vol_lookback = int(p.get("vol_lookback", 20))
        max_arm_bars = int(p.get("max_arm_bars", 6))
        min_bars = adx_period * 2 + max_arm_bars + 5

        if len(data) < min_bars:
            return None

        close = data["close"].to_numpy(float)
        adx_arr, di_plus_arr, di_minus_arr = _adx_di(data, adx_period)
        rsi_arr = _rsi(close, rsi_period)

        adx_now = adx_arr[-1]
        di_plus_now = di_plus_arr[-1]
        di_minus_now = di_minus_arr[-1]

        # Arm on volume spike (liquidation proxy)
        if self._detect_liquidation(data, vol_mult, vol_lookback):
            self._armed_bars_remaining = max_arm_bars
            logger.info("[LiqAdx] Armed after volume spike. ADX=%.1f", adx_now)

        if self._armed_bars_remaining <= 0:
            return None

        self._armed_bars_remaining -= 1

        if adx_now < adx_threshold:
            return None

        price = float(close[-1])

        if di_plus_now > di_minus_now:
            if self._rsi_divergence(close, rsi_arr, div_lookback, "LONG"):
                strength = min(0.5 + (adx_now - adx_threshold) / 50.0, 0.95)
                self._armed_bars_remaining = 0
                logger.info("[LiqAdx] LONG | price=%.4f ADX=%.1f +DI=%.1f RSI=%.1f",
                            price, adx_now, di_plus_now, rsi_arr[-1])
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    strength=strength,
                    reason=f"LiqAdx LONG: ADX={adx_now:.1f} +DI>{di_minus_now:.1f} RSI-div",
                    metadata={"adx": adx_now, "di_plus": di_plus_now, "di_minus": di_minus_now,
                               "rsi": float(rsi_arr[-1])},
                )

        elif di_minus_now > di_plus_now:
            if self._rsi_divergence(close, rsi_arr, div_lookback, "SHORT"):
                strength = min(0.5 + (adx_now - adx_threshold) / 50.0, 0.95)
                self._armed_bars_remaining = 0
                logger.info("[LiqAdx] SHORT | price=%.4f ADX=%.1f -DI=%.1f RSI=%.1f",
                            price, adx_now, di_minus_now, rsi_arr[-1])
                return Signal(
                    signal_type=SignalType.SHORT,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    strength=strength,
                    reason=f"LiqAdx SHORT: ADX={adx_now:.1f} -DI>{di_plus_now:.1f} RSI-div",
                    metadata={"adx": adx_now, "di_plus": di_plus_now, "di_minus": di_minus_now,
                               "rsi": float(rsi_arr[-1])},
                )

        return None

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        p = self.config.params
        adx_period = int(p.get("adx_period", 14))
        exit_threshold = float(p.get("exit_threshold", 18.0))

        if len(data) < adx_period * 2:
            return None

        adx_arr, _, _ = _adx_di(data, adx_period)
        adx_now = adx_arr[-1]
        pnl_pct = float(position.get("pnl_perc", 0))
        is_long = position.get("is_long", position.get("size", 0) > 0)

        if adx_now < exit_threshold:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqAdx trend exhausted: ADX={adx_now:.1f} < {exit_threshold}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqAdx target: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"LiqAdx stop: {pnl_pct:.1f}%",
            )

        return None
