"""
Closed-Market Overnight Strategy.

Edge (MoonDev video 10 — "Claude Code Found a 5.37 Sharpe Strategy"):
when US equity markets are CLOSED, crypto runs on different flow than
when they are open. Less institutional activity, more retail-driven
trends overnight that often mean-revert at market open. Calendar filter
alone reportedly produces a 5.37 Sharpe; we add a simple momentum
breakout entry on top of the gate.

Active window (US Eastern Time):
  - Weekdays 16:00 ET → 09:30 ET next day
  - All day Saturday and Sunday
NYSE holidays are NOT handled here (would need an exchange calendar
dependency); marginal effect for crypto.

Entry: simple lookback-window breakout (close > max(high) of last N
bars) once the calendar gate is open.

Exit: TP/SL/time-stop/market-open. We force-close any position at the
US market open (09:30 ET) regardless of PnL — the edge is overnight,
holding through the open re-introduces the regime we are avoiding.
"""

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


# US Eastern Time offset from UTC. Naive: assumes EDT (-4); EST is -5.
# For a long-running deployment use zoneinfo("America/New_York"); a
# constant offset is fine for forward testing in May (DST in effect).
_ET_OFFSET = timedelta(hours=-4)

_NYSE_OPEN = time(9, 30)
_NYSE_CLOSE = time(16, 0)


def _is_market_closed(now_utc: datetime) -> bool:
    """True when NYSE is closed (overnight or weekend)."""
    et = now_utc + _ET_OFFSET
    if et.weekday() >= 5:
        return True
    t = et.time()
    return t < _NYSE_OPEN or t >= _NYSE_CLOSE


class ClosedMarketOvernightStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        lookback = int(p.get("momentum_lookback", 12))
        breakout_pct = float(p.get("breakout_pct", 0.002))

        if len(data) < lookback + 2:
            return None

        now = datetime.now(timezone.utc)
        if not _is_market_closed(now):
            return None

        current = data.iloc[-1]
        prior = data.iloc[-(lookback + 1):-1]
        price = float(current["close"])
        hi = float(prior["high"].max())
        lo = float(prior["low"].min())

        if price <= 0 or hi <= 0 or lo <= 0:
            return None

        up = (price - hi) / hi
        dn = (lo - price) / lo

        if up >= breakout_pct:
            strength = min(0.5 + up * 50, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"closed-market breakout LONG: +{up*100:.2f}% over {lookback}-bar high",
                metadata={"breakout_pct": up, "window_high": hi},
            )

        if dn >= breakout_pct:
            strength = min(0.5 + dn * 50, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"closed-market breakout SHORT: -{dn*100:.2f}% under {lookback}-bar low",
                metadata={"breakout_pct": dn, "window_low": lo},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        tp_pct = float(p.get("tp_pct", 0.010))
        sl_pct = float(p.get("sl_pct", 0.008))

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = float(position.get("pnl_perc", 0))
        close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT

        if pnl_pct >= tp_pct * 100:
            return Signal(signal_type=close_type, symbol=self.config.symbol,
                          reason=f"closed-market TP: {pnl_pct:.2f}%")
        if pnl_pct <= -sl_pct * 100:
            return Signal(signal_type=close_type, symbol=self.config.symbol,
                          reason=f"closed-market SL: {pnl_pct:.2f}%")
        # Force close at market open — edge is overnight, do not bleed through.
        now = datetime.now(timezone.utc)
        if not _is_market_closed(now):
            return Signal(signal_type=close_type, symbol=self.config.symbol,
                          reason=f"closed-market force-close at NYSE open (pnl {pnl_pct:.2f}%)")

        return None
