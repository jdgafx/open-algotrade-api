"""
Correlation Trading Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/2_correlation_algo/

- ETH (leader) moves first, altcoins (followers) lag behind
- Detects when leader has moved but follower hasn't caught up
- Enters follower in leader's direction
- SL: 0.2%, TP: 0.25%
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class CorrelationStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        lag_threshold = p.get("lag_threshold", 0.002)
        correlation_window = p.get("correlation_window", 20)

        if len(data) < correlation_window + 2:
            return None

        current = data.iloc[-1]
        price = current["close"]

        # Check for leader price data in the DataFrame
        leader_col = "leader_close"
        if leader_col not in data.columns:
            # Without leader data, use the strategy's own momentum detection
            return self._momentum_entry(data)

        # Calculate returns
        data = data.copy()
        data["follower_ret"] = data["close"].pct_change()
        data["leader_ret"] = data[leader_col].pct_change()

        recent = data.tail(5)
        leader_move = recent["leader_ret"].sum()
        follower_move = recent["follower_ret"].sum()
        lag = leader_move - follower_move

        if abs(lag) >= lag_threshold:
            if lag > 0:
                # Leader went up more — follower should catch up (long)
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"Correlation lag LONG: leader +{leader_move:.4f} vs follower +{follower_move:.4f}, lag={lag:.4f}",
                    metadata={"leader_move": leader_move, "follower_move": follower_move, "lag": lag},
                )
            else:
                # Leader went down more — follower should catch up (short)
                return Signal(
                    signal_type=SignalType.SHORT,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"Correlation lag SHORT: leader {leader_move:.4f} vs follower {follower_move:.4f}, lag={lag:.4f}",
                    metadata={"leader_move": leader_move, "follower_move": follower_move, "lag": lag},
                )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sl_pct = p.get("sl_pct", 0.002)
        tp_pct = p.get("tp_pct", 0.0025)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        entry_px = position.get("entry_px", 0)
        price = data.iloc[-1]["close"]

        if entry_px > 0:
            pnl_from_entry = (price - entry_px) / entry_px if is_long else (entry_px - price) / entry_px

            if pnl_from_entry >= tp_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"Correlation TP: {pnl_from_entry:.4f} >= {tp_pct}",
                )
            if pnl_from_entry <= -sl_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=f"Correlation SL: {pnl_from_entry:.4f} <= -{sl_pct}",
                )

        pnl_pct = position.get("pnl_perc", 0)
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    def _momentum_entry(self, data: pd.DataFrame) -> Optional[Signal]:
        """Fallback without leader data: use momentum mean reversion."""
        p = self.config.params
        window = p.get("correlation_window", 20)

        if len(data) < window:
            return None

        recent = data.tail(window)
        sma = recent["close"].mean()
        price = data.iloc[-1]["close"]
        deviation = (price - sma) / sma

        # Mean reversion: buy when below SMA, sell when above
        if deviation < -0.01:
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"Correlation momentum LONG: {deviation:.4f} below SMA",
            )
        if deviation > 0.01:
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                reason=f"Correlation momentum SHORT: {deviation:.4f} above SMA",
            )

        return None
