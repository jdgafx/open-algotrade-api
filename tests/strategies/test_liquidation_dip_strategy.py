"""Tests for LiquidationDipStrategy — proximity entry and double-dip logic."""
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType, StrategyConfig
from src.strategies.liquidation_dip_strategy import LiquidationDipStrategy


def _make_strategy(params: dict | None = None) -> LiquidationDipStrategy:
    config = MagicMock(spec=StrategyConfig)
    config.name = "liqdip-btc"
    config.symbol = "BTC"
    config.size_usd = 500.0
    config.target_pct = 3.0
    config.max_loss_pct = -10.0
    config.interval_seconds = 300
    config.params = params or {}
    s = LiquidationDipStrategy(config)
    return s


def _make_ohlcv(prices: list[float], volume: float = 1000.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.002 for p in prices],
        "low": [p * 0.998 for p in prices],
        "close": prices,
        "volume": [volume] * n,
    })


def _seed_liq_state(s: LiquidationDipStrategy, liq_price: float) -> None:
    """Put the strategy into 'initial bounce seen' state at liq_price."""
    s._liq_low_price = liq_price
    s._liq_low_time = datetime.now(timezone.utc)
    s._initial_bounce_seen = True
    s._bounce_high = liq_price * 1.015  # bounced 1.5% above liq level


# ── Proximity Entry (MoonDev Edge A) ─────────────────────────────────────────

class TestProximityEntry:
    """price approaching liq level within 2% → fire LONG without waiting for touch."""

    def test_fires_when_price_within_proximity_zone(self):
        s = _make_strategy({"proximity_pct": 0.02})
        liq_price = 60_000.0
        _seed_liq_state(s, liq_price)

        # Price is 1% above liq level — inside the 2% proximity zone
        approach_price = liq_price * 1.01
        data = _make_ohlcv([liq_price * 1.05] * 18 + [liq_price * 1.03, approach_price])

        signal = _run(s.should_enter(data))

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.metadata.get("mode") == "proximity"

    def test_does_not_fire_when_price_above_proximity_zone(self):
        s = _make_strategy({"proximity_pct": 0.02})
        liq_price = 60_000.0
        _seed_liq_state(s, liq_price)

        # Price is 3% above liq level — outside the 2% zone
        far_price = liq_price * 1.03
        data = _make_ohlcv([liq_price * 1.05] * 19 + [far_price])

        signal = _run(s.should_enter(data))

        assert signal is None

    def test_does_not_fire_before_initial_bounce(self):
        s = _make_strategy({"proximity_pct": 0.02})
        liq_price = 60_000.0
        # State: liq level set but NO initial bounce seen yet
        s._liq_low_price = liq_price
        s._liq_low_time = datetime.now(timezone.utc)
        s._initial_bounce_seen = False
        s._bounce_high = 0.0

        approach_price = liq_price * 1.01
        data = _make_ohlcv([liq_price * 1.05] * 18 + [liq_price * 1.03, approach_price])

        signal = _run(s.should_enter(data))

        # Should return None (still waiting for bounce confirmation)
        assert signal is None

    def test_strength_higher_when_closer_to_liq_level(self):
        s1 = _make_strategy({"proximity_pct": 0.02})
        s2 = _make_strategy({"proximity_pct": 0.02})
        liq_price = 60_000.0
        _seed_liq_state(s1, liq_price)
        _seed_liq_state(s2, liq_price)

        # s1: 0.5% above liq (very close)
        data1 = _make_ohlcv([liq_price * 1.05] * 19 + [liq_price * 1.005])
        # s2: 1.8% above liq (still in zone but farther)
        data2 = _make_ohlcv([liq_price * 1.05] * 19 + [liq_price * 1.018])

        sig1 = _run(s1.should_enter(data1))
        sig2 = _run(s2.should_enter(data2))

        assert sig1 is not None and sig2 is not None
        assert sig1.strength > sig2.strength

    def test_resets_state_after_proximity_entry(self):
        s = _make_strategy({"proximity_pct": 0.02})
        liq_price = 60_000.0
        _seed_liq_state(s, liq_price)

        approach_price = liq_price * 1.01
        data = _make_ohlcv([liq_price * 1.05] * 18 + [liq_price * 1.03, approach_price])

        _run(s.should_enter(data))

        assert s._liq_low_price is None
        assert not s._initial_bounce_seen


# ── Double-Dip (existing logic — regression guard) ───────────────────────────

class TestDoubleDip:
    def test_no_signal_without_liq_level(self):
        s = _make_strategy()
        data = _make_ohlcv([60_000.0] * 25)
        signal = _run(s.should_enter(data))
        assert signal is None

    def test_inject_liquidation_data_sets_events(self):
        s = _make_strategy()
        events = [{"side": "long", "size_usd": 600_000, "price": 59_000.0,
                   "timestamp": "2026-01-01T00:00:00+00:00"}]
        s.inject_liquidation_data(events)
        assert len(s._liq_events) == 1
