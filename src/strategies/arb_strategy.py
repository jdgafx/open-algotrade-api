"""
Funding Rate Arbitrage Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/HyperLiquid-Trading-Bots/arb.py

- Monitors funding rates between two correlated assets (e.g. BTC and ETH)
- Goes long on the asset with lower funding (getting paid) and short on higher
- Manages combined PnL across both legs
- Exit: Combined PnL target or divergence reversal
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class FundingArbStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        funding_threshold = p.get("funding_threshold", 0.0005)

        # The data should contain funding rate info in metadata columns
        # or we detect from the last rows
        if len(data) < 2:
            return None

        current = data.iloc[-1]

        # Check for funding rate columns
        funding_a = current.get("funding_rate_a", None)
        funding_b = current.get("funding_rate_b", None)

        if funding_a is None or funding_b is None:
            # Funding data not available in OHLCV — strategy needs enriched data
            # Fall back to price divergence as a proxy
            return self._price_divergence_entry(data)

        spread = funding_a - funding_b

        if abs(spread) >= funding_threshold:
            if spread > 0:
                # Asset A has higher funding — short A, long B
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=self.config.symbol,
                    size_usd=self.config.size_usd,
                    reason=f"Funding arb: spread {spread:.6f} > threshold, long {self.config.symbol}",
                    metadata={"funding_spread": spread, "direction": "long_b_short_a"},
                )
            else:
                return Signal(
                    signal_type=SignalType.SHORT,
                    symbol=self.config.symbol,
                    size_usd=self.config.size_usd,
                    reason=f"Funding arb: spread {spread:.6f} < -threshold, short {self.config.symbol}",
                    metadata={"funding_spread": spread, "direction": "short_b_long_a"},
                )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        combined_target = p.get("combined_target_pct", 0.8)  # MoonDev uses 0.6%, we use 0.8% (slightly looser)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if pnl_pct >= combined_target:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Arb target hit: {pnl_pct:.1f}% >= {combined_target}%",
            )

        # Arb-specific tight stop (MoonDev uses -0.9%, we're slightly looser at -1.5%)
        arb_max_loss = p.get("arb_max_loss_pct", -1.5)
        if pnl_pct <= arb_max_loss:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Arb tight stop: {pnl_pct:.1f}% <= {arb_max_loss}%",
            )

        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Arb max loss: {pnl_pct:.1f}%",
            )

        return None

    def _price_divergence_entry(self, data: pd.DataFrame) -> Optional[Signal]:
        """Fallback: use price momentum divergence as arb proxy."""
        if len(data) < 20:
            return None

        p = self.config.params
        threshold = p.get("momentum_threshold", 0.015)  # Slightly more sensitive (was 2%, now 1.5%)

        # Use 10-bar momentum with confirmation (last 3 bars reversing)
        recent = data.tail(10)
        momentum = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

        # Require momentum to be strong enough
        if abs(momentum) < threshold:
            return None

        # Require early reversal confirmation (last bar started moving back)
        last_bar_change = (data["close"].iloc[-1] - data["close"].iloc[-2]) / data["close"].iloc[-2]

        if momentum < -threshold and last_bar_change > 0:
            # Price dropped significantly but last bar is green -- mean reversion long
            strength = min(0.5 + abs(momentum) * 10, 0.9)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Arb momentum divergence LONG: {momentum:.4f}",
                metadata={"momentum": momentum},
            )
        if momentum > threshold and last_bar_change < 0:
            # Price surged but last bar is red -- mean reversion short
            strength = min(0.5 + abs(momentum) * 10, 0.9)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"Arb momentum divergence SHORT: {momentum:.4f}",
                metadata={"momentum": momentum},
            )

        return None
