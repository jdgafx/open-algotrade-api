"""
Liquidation Double Dip Strategy — MoonDev's primary alpha source.

Based on MoonDev's 200-video analysis:
- After a massive liquidation event (>$500K long liquidations), price bounces
- Price ALWAYS revisits the liquidation low within hours
- Most traders buy the first dip and get smoked on the revisit
- The Double Dip buys at the SECOND touch of the liquidation level

Variants:
1. Double Dip (default): Wait for revisit of liq level, then long
2. Flip-Flop: Long the initial bounce, then short back to liq level
3. Momentum Confirm: Double dip with RSI oversold OR MACD bullish confirmation

MoonDev's parameters:
- Liquidation threshold: $500K (long liquidations)
- Lookback: 3 bars on 5-minute data (15 minutes)
- Take profit: 1.5%
- Stop loss: 10% (wide — liquidation bounces are volatile)
- Leverage: 5x max
- Limit orders only (loop until filled)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class LiquidationDipStrategy(BaseStrategy):
    """
    Liquidation Double Dip — buys at the second touch of liquidation lows.

    This strategy maintains internal state:
    - _liq_events: Recent liquidation events (from orchestrator injection)
    - _liq_low_price: The price level where the large liquidation occurred
    - _liq_low_time: When the liquidation happened
    - _initial_bounce_seen: Whether price has bounced from the liq level
    - _ready_for_dip: Whether we're watching for the second touch
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Liquidation event tracking
        self._liq_events: List[Dict] = []
        self._liq_low_price: Optional[float] = None
        self._liq_low_time: Optional[datetime] = None
        self._initial_bounce_seen: bool = False
        self._ready_for_dip: bool = False
        self._bounce_high: float = 0.0

    def inject_liquidation_data(self, events: List[Dict]):
        """Called by orchestrator to inject recent liquidation events."""
        self._liq_events = events

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        liq_threshold_usd = p.get("liq_threshold_usd", 500_000)  # $500K minimum
        bounce_pct = p.get("bounce_pct", 0.005)  # 0.5% bounce needed to confirm initial bounce
        dip_tolerance_pct = p.get("dip_tolerance_pct", 0.002)  # 0.2% tolerance for revisit
        max_age_hours = p.get("max_liq_age_hours", 4)  # Only trade liq events within 4 hours
        mode = p.get("mode", "double_dip")  # "double_dip", "flip_flop", "momentum_confirm"

        if len(data) < 20:
            return None

        current = data.iloc[-1]
        price = current["close"]
        low = current["low"]

        # Step 1: Detect new large liquidation events
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=max_age_hours)

        # Check for large long liquidation events (from injected data or simulated)
        large_liq = self._find_large_liquidation(data, liq_threshold_usd, cutoff)

        if large_liq and self._liq_low_price is None:
            self._liq_low_price = large_liq["price"]
            self._liq_low_time = large_liq.get("time", now)
            self._initial_bounce_seen = False
            self._ready_for_dip = False
            self._bounce_high = 0.0
            logger.info(
                "[LIQ DIP] Large liquidation detected: $%.0f at $%.2f | watching for double dip",
                large_liq.get("volume_usd", 0),
                self._liq_low_price,
            )

        # If no active liquidation level, nothing to do
        if self._liq_low_price is None:
            return None

        # Expire old liquidation levels
        if self._liq_low_time and (now - self._liq_low_time) > timedelta(hours=max_age_hours):
            logger.info("[LIQ DIP] Liquidation level expired (>%dh old)", max_age_hours)
            self._reset_state()
            return None

        # Step 2: Track the initial bounce
        if not self._initial_bounce_seen:
            # Price must move UP from the liq level by bounce_pct
            bounce_target = self._liq_low_price * (1 + bounce_pct)
            if price > bounce_target:
                self._initial_bounce_seen = True
                self._bounce_high = price
                logger.info(
                    "[LIQ DIP] Initial bounce confirmed: price $%.2f > bounce target $%.2f",
                    price,
                    bounce_target,
                )
            return None

        # Track the bounce high
        if price > self._bounce_high:
            self._bounce_high = price

        # Step 3: Watch for the REVISIT (second dip back to liq level)
        revisit_zone = self._liq_low_price * (1 + dip_tolerance_pct)

        if low <= revisit_zone:
            # Price has revisited the liquidation level!
            self._ready_for_dip = True

            # Calculate signal strength based on:
            # 1. How much price bounced before revisiting (bigger bounce = stronger signal)
            # 2. How precisely we're at the liq level
            bounce_size = (
                (self._bounce_high - self._liq_low_price) / self._liq_low_price
                if self._liq_low_price > 0
                else 0
            )
            precision = (
                1.0 - abs(low - self._liq_low_price) / (self._liq_low_price * dip_tolerance_pct * 5)
                if self._liq_low_price > 0
                else 0.5
            )

            strength = min(0.6 + bounce_size * 2 + max(0, precision) * 0.2, 0.95)

            if mode == "momentum_confirm":
                # Require RSI oversold OR MACD bullish confirmation
                confirmed = self._check_momentum_confirmation(data)
                if not confirmed:
                    logger.debug("[LIQ DIP] Waiting for momentum confirmation at liq level")
                    return None
                strength = min(strength + 0.1, 0.98)

            # Capture bounce_high before reset
            bounce_high = self._bounce_high
            entry_price = self._liq_low_price

            logger.info(
                "[LIQ DIP] DOUBLE DIP ENTRY! Price $%.2f revisited liq level $%.2f "
                "(bounced to $%.2f first) | strength=%.2f",
                price,
                entry_price,
                bounce_high,
                strength,
            )

            # Reset state after entry
            self._reset_state()

            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=entry_price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"Liq Double Dip: price revisited liq level ${entry_price:.2f} "
                    f"after bouncing to ${bounce_high:.2f}"
                ),
                metadata={
                    "liq_level": entry_price,
                    "bounce_high": bounce_high,
                    "bounce_pct": bounce_size,
                    "mode": mode,
                },
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        take_profit_pct = p.get("take_profit_pct", 1.5)  # MoonDev: 1.5%
        stop_loss_pct = p.get("stop_loss_pct", 10.0)  # MoonDev: 10% (wide)
        time_limit_hours = p.get("time_limit_hours", 24)  # Max 24h hold (MoonDev)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # Take profit
        if pnl_pct >= take_profit_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Liq Dip TP: {pnl_pct:.2f}% >= {take_profit_pct}%",
            )

        # Stop loss
        if pnl_pct <= -stop_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Liq Dip SL: {pnl_pct:.2f}% <= -{stop_loss_pct}%",
            )

        # Time-based exit (MoonDev: don't hold more than 24h after liq event)
        time_limit_bars = int(
            time_limit_hours * 3600 / max(self.config.interval_seconds, 1)
        )
        if self.state.entry_bar_count > time_limit_bars:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Liq Dip time limit: {self.state.entry_bar_count} bars > {time_limit_bars}",
            )

        # Global target/stop
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Target: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Max loss: {pnl_pct:.1f}%",
            )

        return None

    def _find_large_liquidation(
        self, data: pd.DataFrame, threshold_usd: float, cutoff: datetime
    ) -> Optional[Dict]:
        """
        Detect large liquidation events from price action.

        Since we may not have direct liquidation feed data in OHLCV,
        we detect "liquidation-like" events from sharp price drops with high volume.
        A sharp drop + volume spike on 5m data strongly correlates with liquidation cascades.

        If liquidation data is injected via inject_liquidation_data(), use that instead.
        """
        # First check injected liquidation data
        for event in self._liq_events:
            evt_time = event.get("timestamp")
            if evt_time and isinstance(evt_time, str):
                try:
                    evt_time = datetime.fromisoformat(evt_time.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

            if evt_time and evt_time > cutoff:
                if (
                    event.get("side", "").lower() == "long"
                    and event.get("size_usd", 0) >= threshold_usd
                ):
                    return {
                        "price": event.get("price", 0),
                        "volume_usd": event.get("size_usd", 0),
                        "time": evt_time,
                    }

        # Fallback: detect from price action (sharp drop + volume spike)
        if len(data) < 10:
            return None

        lookback = min(20, len(data))
        recent = data.tail(lookback)

        # Look for bars with > 2x average volume AND significant price drop
        if "volume" not in recent.columns:
            return None

        avg_volume = recent["volume"].mean()
        if avg_volume <= 0:
            return None

        for i in range(len(recent) - 1, max(len(recent) - 5, -1), -1):
            bar = recent.iloc[i]

            bar_volume = bar.get("volume", 0)
            bar_drop = (bar["close"] - bar["open"]) / bar["open"] if bar["open"] > 0 else 0

            # Large red candle with volume spike = potential liquidation cascade
            if bar_volume > avg_volume * 2.5 and bar_drop < -0.01:
                estimated_liq_usd = (
                    bar_volume * bar["close"] * 0.1
                )  # Rough estimate: 10% of volume is liquidation
                if estimated_liq_usd >= threshold_usd * 0.5:  # Lower threshold for simulated detection
                    return {
                        "price": bar["low"],
                        "volume_usd": estimated_liq_usd,
                        "time": datetime.now(timezone.utc),
                    }

        return None

    def _check_momentum_confirmation(self, data: pd.DataFrame) -> bool:
        """Check RSI oversold OR MACD bullish crossover for momentum confirmation."""
        if len(data) < 26:
            return False

        # RSI check
        delta = data["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if not pd.isna(current_rsi) and current_rsi < 30:
            return True

        # MACD check
        ema12 = data["close"].ewm(span=12).mean()
        ema26 = data["close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        if len(macd) >= 2 and len(signal) >= 2:
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                return True  # Bullish crossover

        return False

    def _reset_state(self):
        """Reset liquidation tracking state."""
        self._liq_low_price = None
        self._liq_low_time = None
        self._initial_bounce_seen = False
        self._ready_for_dip = False
        self._bounce_high = 0.0
