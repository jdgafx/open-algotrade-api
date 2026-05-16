"""Tests for ConsecutiveDownStrategy — N red bars → LONG mean-reversion."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.consecutive_down_strategy import ConsecutiveDownStrategy

    s = ConsecutiveDownStrategy.__new__(ConsecutiveDownStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "n_down_bars": 4,
        "use_vwap_filter": True,
        "rsi_period": 14,
        "rsi_max": 45,
        "take_profit_pct": 2.5,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 8,
    }
    s.config.symbol = "BTC"
    s.config.size_usd = 100.0
    s.config.target_pct = 2.5
    s.config.max_loss_pct = -1.5
    s._entry_bar = None
    s._bar_count = 0
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": True, "size": 1.0, "pnl_perc": pnl_pct}


def _make_flat_df(n: int = 40) -> pd.DataFrame:
    closes = np.ones(n) * 100.0
    vols = np.ones(n) * 1000.0
    return pd.DataFrame({
        "open": closes + 0.05, "high": closes + 0.1,
        "low": closes - 0.1, "close": closes, "volume": vols,
    })


def _make_n_red_bars_df(n: int = 40, red_count: int = 4,
                         price_start: float = 100.0,
                         step: float = 0.5) -> pd.DataFrame:
    """
    DataFrame where the last red_count bars are bearish (close < open).
    Earlier bars are flat/bullish to keep RSI and VWAP sensible.
    """
    closes = np.ones(n) * price_start
    opens = closes.copy()
    highs = closes * 1.002
    lows = closes * 0.998
    vols = np.ones(n) * 1000.0

    # Last red_count bars: each drops by step
    for i in range(red_count):
        idx = n - red_count + i
        opens[idx] = price_start - i * step + step * 0.3
        closes[idx] = price_start - (i + 1) * step
        highs[idx] = opens[idx] + step * 0.1
        lows[idx] = closes[idx] - step * 0.1
        vols[idx] = 800.0  # lower vol on red bars (bears not aggressive)

    # Fix any pricing inconsistencies
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


# ---------------------------------------------------------------------------
# _count_consecutive_down
# ---------------------------------------------------------------------------

def test_count_returns_zero_on_flat():
    s = _make_strategy()
    df = _make_flat_df()
    # All bars are flat (close == open - 0.05, so close < open → bearish)
    # Actually flat bars have close < open by design, so let's make truly flat
    df["open"] = df["close"]
    count = s._count_consecutive_down(df)
    assert count == 0


def test_count_returns_correct_n():
    s = _make_strategy()
    df = _make_n_red_bars_df(red_count=4)
    count = s._count_consecutive_down(df)
    assert count == 4


def test_count_resets_after_green_bar():
    s = _make_strategy()
    df = _make_n_red_bars_df(red_count=6)
    # Insert a green bar 3 bars ago
    df.iloc[-3, df.columns.get_loc("open")] = 99.0
    df.iloc[-3, df.columns.get_loc("close")] = 99.5  # green
    count = s._count_consecutive_down(df)
    assert count == 2  # only last 2 bars are consecutive red


# ---------------------------------------------------------------------------
# _compute_vwap
# ---------------------------------------------------------------------------

def test_vwap_returns_positive():
    s = _make_strategy()
    df = _make_flat_df()
    vwap = s._compute_vwap(df)
    assert vwap > 0.0


# ---------------------------------------------------------------------------
# should_enter
# ---------------------------------------------------------------------------

def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_flat_df(n=5)
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_when_fewer_than_n_red_bars():
    s = _make_strategy()
    df = _make_n_red_bars_df(red_count=3)  # need 4, got 3
    result = _run(s.should_enter(df))
    assert result is None


def test_entry_on_exactly_n_red_bars_no_filter():
    s = _make_strategy(params={
        "n_down_bars": 4,
        "use_vwap_filter": False,
        "rsi_period": 14,
        "rsi_max": 100,  # always passes
        "take_profit_pct": 2.5,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 8,
    })
    df = _make_n_red_bars_df(n=40, red_count=4)
    result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_entry_sets_entry_bar():
    s = _make_strategy(params={
        "n_down_bars": 4, "use_vwap_filter": False,
        "rsi_period": 14, "rsi_max": 100,
        "take_profit_pct": 2.5, "stop_loss_pct": 1.5, "hold_cap_bars": 8,
    })
    df = _make_n_red_bars_df(n=40, red_count=4)
    _run(s.should_enter(df))
    assert s._entry_bar is not None


def test_no_double_entry():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_n_red_bars_df(red_count=4)
    result = _run(s.should_enter(df))
    assert result is None


def test_signal_strength_in_range():
    s = _make_strategy(params={
        "n_down_bars": 4, "use_vwap_filter": False,
        "rsi_period": 14, "rsi_max": 100,
        "take_profit_pct": 2.5, "stop_loss_pct": 1.5, "hold_cap_bars": 8,
    })
    df = _make_n_red_bars_df(n=40, red_count=4)
    result = _run(s.should_enter(df))
    if result is not None:
        assert 0.0 < result.strength <= 1.0


def test_more_red_bars_than_required_still_enters():
    s = _make_strategy(params={
        "n_down_bars": 4, "use_vwap_filter": False,
        "rsi_period": 14, "rsi_max": 100,
        "take_profit_pct": 2.5, "stop_loss_pct": 1.5, "hold_cap_bars": 8,
    })
    df = _make_n_red_bars_df(n=40, red_count=6)
    result = _run(s.should_enter(df))
    assert result is not None


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=2.6)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=-1.6)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_hold_cap():
    s = _make_strategy()
    s._entry_bar = 0
    s._bar_count = 9  # > hold_cap_bars=8
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_no_exit_within_limits():
    s = _make_strategy()
    s._entry_bar = 3
    s._bar_count = 5
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=1.0)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    result = _run(s.should_exit(_make_flat_df(), _make_position()))
    assert result is None
