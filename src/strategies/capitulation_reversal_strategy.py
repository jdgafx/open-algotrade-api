"""
Capitulation Reversal Strategy.

MoonDev source: "Clawdbot For Trading is UNSTOPPABLE (71,057% ROI)" (gej2RO55zxQ).
Claimed metrics: 536% return, 68% WR, -34% DD (ETH 1-hr).

Logic:
1. Detect capitulation: RSI < oversold_threshold AND price dropped > drop_pct%
   from the N-bar high AND volume spike on the sell-off bar.
2. Confirm reversal: current candle is bullish (close > open) — the market is
   absorbing the capitulation and beginning to recover.
3. Enter LONG — fade the extreme move, ride the mean-reversion.
4. Exit on: profit target (2%), stop loss (1.5%), or hold_cap_bars (8 bars).

Edge: When all three conditions align (oversold RSI + significant price drop +
volume spike + bullish reversal candle), the probability of at least a short-term
bounce is significantly elevated. Most capitulation events see 2-5% recoveries
within 4-8 hours on crypto.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class CapitulationReversalStrategy(BaseStrategy):
    """
    Long-only strategy that enters at the bottom of capitulation sell-offs.

    State machine: IDLE → ENTERED (LONG on confirmed reversal) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Wilder's smoothed RSI."""
        n = len(closes)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()

        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def _is_capitulation(self, df: pd.DataFrame) -> bool:
        p = self.config.params
        rsi_period = int(p.get("rsi_period", 14))
        rsi_oversold = float(p.get("rsi_oversold", 25))
        drop_lookback = int(p.get("drop_lookback", 20))
        drop_pct = float(p.get("drop_pct", 0.03))
        vol_mult = float(p.get("vol_spike_mult", 1.5))
        vol_lookback = int(p.get("vol_lookback", 20))

        min_bars = max(rsi_period + 2, drop_lookback + 2, vol_lookback + 2)
        if len(df) < min_bars:
            return False

        closes = df["close"].to_numpy(float)
        rsi = self._compute_rsi(closes, rsi_period)
        if rsi[-1] >= rsi_oversold:
            return False

        recent_high = float(df["high"].iloc[-drop_lookback - 1:-1].max())
        current_close = float(df["close"].iloc[-1])
        if recent_high <= 0:
            return False
        if (recent_high - current_close) / recent_high < drop_pct:
            return False

        vol = df["volume"].to_numpy(float)
        avg_vol = vol[-vol_lookback - 1:-1].mean()
        if avg_vol <= 0 or vol[-1] / avg_vol < vol_mult:
            return False

        return True

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        p = self.config.params
        min_bars = max(
            int(p.get("rsi_period", 14)) + 2,
            int(p.get("drop_lookback", 20)) + 2,
            int(p.get("vol_lookback", 20)) + 2,
        )

        if len(data) < min_bars:
            return None

        if not self._is_capitulation(data):
            return None

        require_bullish = bool(p.get("require_bullish_confirm", True))
        last_open = float(data["open"].iloc[-1])
        last_close = float(data["close"].iloc[-1])

        if require_bullish and last_close <= last_open:
            return None

        rsi_period = int(p.get("rsi_period", 14))
        closes = data["close"].to_numpy(float)
        rsi_val = float(self._compute_rsi(closes, rsi_period)[-1])
        rsi_oversold = float(p.get("rsi_oversold", 25))

        # Strength: deeper oversold + stronger reversal candle = higher conviction
        rsi_depth = max(0.0, rsi_oversold - rsi_val) / rsi_oversold
        body = abs(last_close - last_open) / last_open if last_open > 0 else 0.0
        strength = min(0.60 + rsi_depth * 0.25 + body * 5.0, 0.93)

        price = last_close
        self._entry_bar = self._bar_count - 1

        logger.info("[CapRev] LONG | price=%.4f rsi=%.1f strength=%.2f",
                    price, rsi_val, strength)

        return Signal(
            signal_type=SignalType.LONG,
            symbol=self.config.symbol,
            price=price,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"CapRev LONG: rsi={rsi_val:.1f} (oversold)",
            metadata={"rsi": rsi_val, "strategy": "capitulation_reversal"},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        # State-independent risk reflex first: honor max_loss / target from the live
        # position even if entry tracking is unavailable (e.g. after a restart).
        reflex = self._risk_reflex_exit(position)
        if reflex is not None:
            return reflex
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 8))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"CapRev hold cap: {bars_held} bars",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"CapRev TP: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"CapRev SL: {pnl_pct:.1f}%",
            )

        return None
