"""
Trend Cross Strategy — the validated dual-MA always-in-market edge.

This is the ONLY trend edge proven out-of-sample by scripts/edge_probe.py:
position = sign(MA_fast - MA_slow), always in the market, long/short, 1-bar
shifted (no look-ahead). OOS Sharpe +2..+3 on BTC/ETH/SOL @1h for sma 30/50.

Deliberately pure: no ADX/support-resistance/Bollinger overlays. Stacking
indicators on `close` adds correlation, not alpha (alpha-doctrine-snowball).
The existing `sma_crossover` (SMAStrategy) is a DIFFERENT thing — single SMA
vs price + ADX + S/R — and is NOT this validated edge.

Always-in semantics in the BaseStrategy lifecycle:
- flat  -> enter in the direction of the current cross (re-syncs to trend even
           mid-move, e.g. after a restart) — faithful to "hold sign every bar".
- in pos-> close when the cross flips against the position; the next iteration
           re-enters the other side (1-bar gap == the probe's shift-1).
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class TrendCrossStrategy(BaseStrategy):

    def _ma(self, x: pd.Series, n: int) -> pd.Series:
        ma_type = self.config.params.get("ma_type", "sma")
        if ma_type == "ema":
            return x.ewm(span=n, adjust=False).mean()
        return x.rolling(window=n).mean()

    def _cross_sign(self, data: pd.DataFrame) -> Optional[int]:
        """+1 if fast>slow, -1 if fast<slow, on the last CLOSED bar (no look-ahead).
        None if not enough data or MAs equal/NaN."""
        fast = int(self.config.params.get("fast_period", 30))
        slow = int(self.config.params.get("slow_period", 50))
        if fast >= slow or len(data) < slow + 2:
            return None
        close = data["close"]
        f = self._ma(close, fast)
        s = self._ma(close, slow)
        # iloc[-2] = last fully-closed bar; iloc[-1] may be the forming candle.
        fv, sv = f.iloc[-2], s.iloc[-2]
        if pd.isna(fv) or pd.isna(sv) or fv == sv:
            return None
        return 1 if fv > sv else -1

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        sign = self._cross_sign(data)
        if sign is None:
            return None
        price = float(data["close"].iloc[-1])
        fast = self.config.params.get("fast_period", 30)
        slow = self.config.params.get("slow_period", 50)
        ma_type = self.config.params.get("ma_type", "sma")
        if sign > 0:
            return Signal(
                signal_type=SignalType.LONG, symbol=self.config.symbol, price=price,
                size_usd=self.config.size_usd, strength=0.85,
                reason=f"{ma_type}{fast}>{slow} cross LONG @ {price:.4f}",
            )
        return Signal(
            signal_type=SignalType.SHORT, symbol=self.config.symbol, price=price,
            size_usd=self.config.size_usd, strength=0.85,
            reason=f"{ma_type}{fast}<{slow} cross SHORT @ {price:.4f}",
        )

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        # State-independent risk reflex first (protects across restarts).
        reflex = self._risk_reflex_exit(position)
        if reflex is not None:
            return reflex

        sign = self._cross_sign(data)
        if sign is None:
            return None
        is_long = position.get("is_long", position.get("size", 0) > 0)
        # Close only when the trend flips against the open position.
        if is_long and sign < 0:
            return Signal(signal_type=SignalType.CLOSE_LONG, symbol=self.config.symbol,
                          reason="trend flip -> close long")
        if not is_long and sign > 0:
            return Signal(signal_type=SignalType.CLOSE_SHORT, symbol=self.config.symbol,
                          reason="trend flip -> close short")
        return None


def _demo() -> None:
    """Self-check: the cross sign and signals match sign(fast-slow) on the last
    closed bar, and a flip produces a close. Run: python3 trend_cross_strategy.py"""
    import asyncio
    import numpy as np
    from .base_strategy import StrategyConfig

    cfg = StrategyConfig(name="t", symbol="BTC", params={"fast_period": 3, "slow_period": 5})
    strat = TrendCrossStrategy(cfg)

    def df(prices):
        return pd.DataFrame({"open": prices, "high": prices, "low": prices,
                             "close": prices, "volume": [1.0] * len(prices)})

    up = df([10, 10, 10, 10, 11, 12, 13, 14, 15])  # fast MA above slow -> LONG
    sig = asyncio.run(strat.should_enter(up))
    assert sig is not None and sig.signal_type == SignalType.LONG, sig

    down = df([15, 14, 13, 12, 11, 10, 9, 8, 7])    # fast below slow -> SHORT
    sig = asyncio.run(strat.should_enter(down))
    assert sig is not None and sig.signal_type == SignalType.SHORT, sig

    # in a long, a down-trend flips -> CLOSE_LONG
    sig = asyncio.run(strat.should_exit(down, {"is_long": True, "size": 1.0, "pnl_perc": 0.0}))
    assert sig is not None and sig.signal_type == SignalType.CLOSE_LONG, sig

    # in a long, an up-trend holds -> None
    sig = asyncio.run(strat.should_exit(up, {"is_long": True, "size": 1.0, "pnl_perc": 0.0}))
    assert sig is None, sig

    # too little data -> no signal (no look-ahead crash)
    assert strat._cross_sign(df([1, 2, 3])) is None
    print("trend_cross_strategy demo OK")


if __name__ == "__main__":
    _demo()
