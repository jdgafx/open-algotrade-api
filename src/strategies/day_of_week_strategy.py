"""
Day-of-Week Bias Strategy.

MoonDev source: "Trading Bots Predicting Price & New Polymarket Updates" (rKpnPyzH6r4).
Claimed metrics: 83% WR Thursdays, Sharpe 3.1, DD −5%, 24% return.

Logic:
1. Gate: only enter when current UTC day is in trade_days (default: Wed=2, Thu=3).
2. Momentum filter: RSI > rsi_min AND price > SMA(sma_period).
3. Bullish candle confirm: close > open on entry bar.
4. Enter LONG — capture the statistically favorable day-of-week drift.
5. Exit on: profit target (2%), stop loss (1.5%), or hold_cap_bars (6 bars).

Edge: Crypto markets show measurable day-of-week seasonality. Thursdays and
Wednesdays historically see institutional rebalancing and positive drift in
BTC/ETH. Combining the day gate with momentum confirmation reduces false entries.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class DayOfWeekBiasStrategy(BaseStrategy):
    """
    Long-only strategy that only enters on statistically favorable days of week.

    State machine: IDLE → ENTERED (LONG on favorable day + momentum) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    def _is_trade_day(self, now: datetime) -> bool:
        trade_days = list(self.config.params.get("trade_days", [2, 3]))
        return now.weekday() in trade_days

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
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

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        p = self.config.params
        rsi_period = int(p.get("rsi_period", 14))
        sma_period = int(p.get("sma_period", 20))
        min_bars = max(rsi_period + 2, sma_period + 2)

        if len(data) < min_bars:
            return None

        now = datetime.now(tz=timezone.utc)
        if not self._is_trade_day(now):
            return None

        closes = data["close"].to_numpy(float)
        rsi = self._compute_rsi(closes, rsi_period)
        rsi_val = float(rsi[-1])
        rsi_min = float(p.get("rsi_min", 45))
        if rsi_val < rsi_min:
            return None

        sma = float(np.mean(closes[-sma_period:]))
        current_close = float(closes[-1])
        if current_close <= sma:
            return None

        require_bullish = bool(p.get("require_bullish_candle", True))
        last_open = float(data["open"].iloc[-1])
        if require_bullish and current_close <= last_open:
            return None

        # Strength: RSI headroom + price vs SMA distance
        rsi_room = min((rsi_val - rsi_min) / (100.0 - rsi_min), 1.0)
        sma_dist = min((current_close - sma) / sma, 0.05) / 0.05
        day = now.weekday()
        # Thursday (3) gets slight conviction bonus over Wednesday (2)
        day_bonus = 0.05 if day == 3 else 0.0
        strength = min(0.55 + rsi_room * 0.20 + sma_dist * 0.15 + day_bonus, 0.90)

        self._entry_bar = self._bar_count - 1

        logger.info("[DayOfWeek] LONG | day=%d rsi=%.1f price=%.4f sma=%.4f strength=%.2f",
                    day, rsi_val, current_close, sma, strength)

        return Signal(
            signal_type=SignalType.LONG,
            symbol=self.config.symbol,
            price=current_close,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=f"DayOfWeek LONG: day={day} rsi={rsi_val:.1f}",
            metadata={"rsi": rsi_val, "sma": sma, "day_of_week": day,
                      "strategy": "day_of_week_bias"},
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
        hold_cap_bars = int(p.get("hold_cap_bars", 6))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"DayOfWeek hold cap: {bars_held} bars",
            )

        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"DayOfWeek TP: {pnl_pct:.1f}%",
            )

        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"DayOfWeek SL: {pnl_pct:.1f}%",
            )

        return None
