"""Tests for DonchianChannelStrategy — classic Donchian breakout trend-follower."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.donchian_strategy import DonchianChannelStrategy

    s = DonchianChannelStrategy.__new__(DonchianChannelStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "channel_period": 20,
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "take_profit_pct": 3.0,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 12,
    }
    s.config.symbol = "BTC"
    s.config.size_usd = 200.0
    s.config.target_pct = 3.0
    s.config.max_loss_pct = -1.5
    s._entry_bar = None
    s._bar_count = 0
    s._entry_price = None
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": True, "size": 1.0, "pnl_perc": pnl_pct}


def _make_flat_df(n: int = 40) -> pd.DataFrame:
    closes = np.ones(n) * 100.0
    return pd.DataFrame({
        "open": closes.copy(), "high": closes * 1.005,
        "low": closes * 0.995, "close": closes,
        "volume": np.ones(n) * 1000.0,
    })


def _make_breakout_df(n: int = 40, direction: str = "up") -> pd.DataFrame:
    """
    N-1 bars of flat price, then last bar breaks above the 20-bar high.
    """
    base_close = 100.0
    closes = np.ones(n) * base_close
    highs = closes * 1.003
    lows = closes * 0.997
    opens = closes.copy()

    if direction == "up":
        # Last bar closes and highs above all previous highs
        closes[-1] = base_close * 1.015  # 1.5% above all prior highs
        highs[-1] = closes[-1] * 1.001
        lows[-1] = base_close * 1.005
        opens[-1] = base_close * 1.008
    else:
        closes[-1] = base_close * 0.985
        lows[-1] = closes[-1] * 0.999
        highs[-1] = base_close * 0.993
        opens[-1] = base_close * 0.992

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.ones(n) * 1000.0,
    })


# ---------------------------------------------------------------------------
# _compute_atr / _channel_bounds
# ---------------------------------------------------------------------------

def test_channel_bounds_returns_tuple():
    s = _make_strategy()
    df = _make_flat_df()
    upper, lower = s._channel_bounds(df, period=20)
    assert isinstance(upper, float)
    assert isinstance(lower, float)


def test_channel_upper_ge_lower():
    s = _make_strategy()
    df = _make_flat_df()
    upper, lower = s._channel_bounds(df, period=20)
    assert upper >= lower


def test_breakout_detected_on_upside():
    s = _make_strategy()
    df = _make_breakout_df(direction="up")
    upper, lower = s._channel_bounds(df[:-1], period=20)  # prior-bar channel
    # Last bar close > prior channel upper
    assert float(df["close"].iloc[-1]) > upper


def test_no_breakout_on_flat():
    s = _make_strategy()
    df = _make_flat_df()
    upper, lower = s._channel_bounds(df[:-1], period=20)
    # Last close == upper (flat) — not strictly above
    assert float(df["close"].iloc[-1]) <= upper


# ---------------------------------------------------------------------------
# should_enter
# ---------------------------------------------------------------------------

def test_no_entry_on_flat_market():
    s = _make_strategy()
    df = _make_flat_df()
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_flat_df(n=5)
    result = _run(s.should_enter(df))
    assert result is None


def test_entry_long_on_upside_breakout():
    s = _make_strategy()
    df = _make_breakout_df(direction="up")
    result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_signal_strength_in_range():
    s = _make_strategy()
    df = _make_breakout_df(direction="up")
    result = _run(s.should_enter(df))
    if result is not None:
        assert 0.0 < result.strength <= 1.0


def test_bar_count_increments():
    s = _make_strategy()
    df = _make_flat_df()
    _run(s.should_enter(df))
    assert s._bar_count == 1
    _run(s.should_enter(df))
    assert s._bar_count == 2


def test_entry_price_set_after_entry():
    s = _make_strategy()
    df = _make_breakout_df(direction="up")
    result = _run(s.should_enter(df))
    if result is not None:
        assert s._entry_price is not None
        assert s._entry_price > 0


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    s._entry_price = 100.0
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=3.1)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    s._entry_price = 100.0
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=-1.6)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_hold_cap():
    s = _make_strategy()
    s._entry_bar = 0
    s._bar_count = 13  # > hold_cap_bars=12
    s._entry_price = 100.0
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_no_exit_within_limits():
    s = _make_strategy()
    s._entry_bar = 3
    s._bar_count = 5
    s._entry_price = 100.0
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=1.0)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    result = _run(s.should_exit(_make_flat_df(), _make_position()))
    assert result is None
