"""
Supply/Demand Zone Strategy — BaseStrategy implementation.
Ported from ALGOS/ATC Bootcamp Code 2025/11_day11_bots/day11_hyperliquid/11_sdz_bot.py

- Detects supply and demand zones from OHLCV pivot points
- Long when price enters a demand zone (buying pressure area)
- Short when price enters a supply zone (selling pressure area)
- Exit: Target % or stop loss
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SupplyDemandZoneStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        if len(data) < 10:
            return None

        zones = self._calculate_zones(data)
        current = data.iloc[-1]
        price = current["close"]

        for zone_high, zone_low in zones["demand"]:
            if zone_low <= price <= zone_high:
                return Signal(
                    signal_type=SignalType.LONG,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"SDZ demand zone LONG: price {price:.2f} in zone [{zone_low:.2f}, {zone_high:.2f}]",
                    metadata={"zone_type": "demand", "zone_high": zone_high, "zone_low": zone_low},
                )

        for zone_high, zone_low in zones["supply"]:
            if zone_low <= price <= zone_high:
                return Signal(
                    signal_type=SignalType.SHORT,
                    symbol=self.config.symbol,
                    price=price,
                    size_usd=self.config.size_usd,
                    reason=f"SDZ supply zone SHORT: price {price:.2f} in zone [{zone_low:.2f}, {zone_high:.2f}]",
                    metadata={"zone_type": "supply", "zone_high": zone_high, "zone_low": zone_low},
                )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"SDZ target hit: {pnl_pct:.1f}%")

        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"SDZ stop loss: {pnl_pct:.1f}%")

        zones = self._calculate_zones(data)
        price = data.iloc[-1]["close"]

        if is_long:
            for zone_high, zone_low in zones["supply"]:
                if zone_low <= price <= zone_high:
                    return Signal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.config.symbol,
                        reason=f"SDZ long exit at supply zone: {price:.2f}",
                    )
        else:
            for zone_high, zone_low in zones["demand"]:
                if zone_low <= price <= zone_high:
                    return Signal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.config.symbol,
                        reason=f"SDZ short exit at demand zone: {price:.2f}",
                    )

        return None

    @staticmethod
    def _calculate_zones(df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
        supply_zones = []
        demand_zones = []

        for i in range(2, len(df) - 2):
            # Demand: price drops then reverses up strongly
            if (
                df.iloc[i]["low"] < df.iloc[i - 1]["low"]
                and df.iloc[i]["low"] < df.iloc[i + 1]["low"]
                and df.iloc[i + 1]["close"] > df.iloc[i]["high"]
            ):
                demand_zones.append(
                    (float(df.iloc[i]["high"]), float(df.iloc[i]["low"]))
                )

            # Supply: price rises then reverses down strongly
            if (
                df.iloc[i]["high"] > df.iloc[i - 1]["high"]
                and df.iloc[i]["high"] > df.iloc[i + 1]["high"]
                and df.iloc[i + 1]["close"] < df.iloc[i]["low"]
            ):
                supply_zones.append(
                    (float(df.iloc[i]["high"]), float(df.iloc[i]["low"]))
                )

        return {"supply": supply_zones[-5:], "demand": demand_zones[-5:]}
