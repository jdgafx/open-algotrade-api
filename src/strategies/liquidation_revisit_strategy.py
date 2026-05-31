"""
Liquidation Revisit Momentum Strategy.

MoonDev source: "I Found 5 Strategies Nobody's Talking About (514%)" (zNtSOxKlNl8).
Claimed metrics: 514% return, profit factor 5.33, expectancy 5+, vs 33% B&H.

Logic (3-state machine):
  IDLE → WATCHING: Detect a liquidation candle (large red body + volume spike).
                   Record the liquidation low as the target revisit price.
  WATCHING → IDLE: If price doesn't revisit the low within revisit_window_bars, reset.
  WATCHING → ENTERED: Price returns to within tolerance_pct of the liq low
                       AND RSI is oversold → LONG. Fade the liquidation sweep.
  ENTERED → IDLE: TP / SL / hold_cap exit.

Edge: After a liquidation cascade, price frequently returns to test the liquidation
low as a support zone. Institutional buyers accumulate at these levels. The revisit
+ RSI oversold confirmation gives a high-probability fade entry with good R:R.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class LiquidationRevisitStrategy(BaseStrategy):
    """
    Long-only liquidation revisit strategy.

    State: IDLE → WATCHING (after liq candle) → ENTERED (on revisit) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._liq_low: Optional[float] = None
        self._liq_bar: Optional[int] = None
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _is_liquidation_candle(self, row: pd.Series, avg_vol: float) -> bool:
        """True when the bar is a large bearish candle with volume spike."""
        open_px = float(row["open"])
        close_px = float(row["close"])
        vol = float(row["volume"])
        if open_px <= 0:
            return False
        body_pct = (open_px - close_px) / open_px  # positive when bearish
        p = self.config.params
        liq_pct = float(p.get("liq_candle_pct", 0.02))
        vol_mult = float(p.get("vol_spike_mult", 1.5))
        return body_pct >= liq_pct and (avg_vol <= 0 or vol / avg_vol >= vol_mult)

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        n = len(closes)
        if n < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        return 100.0 - (100.0 / (1.0 + rs))

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        p = self.config.params
        vol_lookback = int(p.get("vol_lookback", 20))
        rsi_period = int(p.get("rsi_period", 14))
        min_bars = max(vol_lookback + 2, rsi_period + 2)

        if len(data) < min_bars:
            return None

        revisit_window = int(p.get("revisit_window_bars", 20))
        tolerance_pct = float(p.get("revisit_tolerance_pct", 0.005))
        rsi_oversold = float(p.get("rsi_oversold", 35))

        last_row = data.iloc[-1]
        current_close = float(last_row["close"])
        current_low = float(last_row["low"])

        # === WATCHING: check for expiry or revisit ===
        if self._liq_low is not None and self._liq_bar is not None:
            bars_waiting = self._bar_count - self._liq_bar

            # Expired — reset to IDLE
            if bars_waiting > revisit_window:
                logger.debug("[LiqRevisit] WATCHING expired after %d bars", bars_waiting)
                self._liq_low = None
                self._liq_bar = None
            else:
                # Check revisit: price low touched within tolerance of liq low
                revisit_zone = self._liq_low * (1 + tolerance_pct)
                if current_low <= revisit_zone:
                    closes = data["close"].to_numpy(float)
                    rsi_val = self._compute_rsi(closes, rsi_period)
                    if rsi_val <= rsi_oversold:
                        depth = max(0.0, rsi_oversold - rsi_val) / rsi_oversold
                        strength = min(0.62 + depth * 0.25, 0.90)
                        self._entry_bar = self._bar_count - 1
                        self._liq_low = None
                        self._liq_bar = None
                        logger.info(
                            "[LiqRevisit] LONG | price=%.4f rsi=%.1f strength=%.2f",
                            current_close, rsi_val, strength,
                        )
                        return Signal(
                            signal_type=SignalType.LONG,
                            symbol=self.config.symbol,
                            price=current_close,
                            size_usd=self.config.size_usd,
                            strength=strength,
                            reason=f"LiqRevisit LONG: rsi={rsi_val:.1f} revisit={current_low:.4f}",
                            metadata={"rsi": rsi_val, "strategy": "liquidation_revisit"},
                        )
                return None

        # === IDLE: scan for liquidation candle ===
        avg_vol = float(data["volume"].iloc[-vol_lookback - 1:-1].mean())
        if self._is_liquidation_candle(last_row, avg_vol):
            self._liq_low = float(last_row["low"])
            self._liq_bar = self._bar_count - 1
            logger.info("[LiqRevisit] WATCHING | liq_low=%.4f", self._liq_low)

        return None

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
                reason=f"LiqRevisit hold cap: {bars_held} bars",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"LiqRevisit TP: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"LiqRevisit SL: {pnl_pct:.1f}%",
            )

        return None
