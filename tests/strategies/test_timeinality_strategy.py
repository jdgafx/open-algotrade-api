"""Tests for TimeinailityStrategy — time-of-day bias, short-only."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.timeinality_strategy import TimeinailityStrategy

    s = TimeinailityStrategy.__new__(TimeinailityStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "bear_hours": [2, 3, 4, 5, 6, 14, 15],
        "bear_days": [0, 6],
        "require_bearish_candle": True,
        "take_profit_pct": 1.5,
        "stop_loss_pct": 0.8,
        "hold_cap_bars": 4,
    }
    s.config.symbol = "SUI"
    s.config.size_usd = 100.0
    s.config.target_pct = 1.5
    s.config.max_loss_pct = -0.8
    s._entry_bar = None
    s._bar_count = 0
    return s


def _make_ohlcv(n: int = 25, bearish_last: bool = True) -> pd.DataFrame:
    closes = np.ones(n) * 100.0
    if bearish_last:
        opens = np.ones(n) * 100.0
        opens[-1] = 101.0  # open > close → bearish candle
    else:
        opens = np.ones(n) * 100.0
        opens[-1] = 99.0  # open < close → bullish candle
    return pd.DataFrame({
        "open": opens,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": np.ones(n) * 1000.0,
    })


def _make_position(is_long: bool, pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": is_long, "size": 1.0 if is_long else -1.0, "pnl_perc": pnl_pct}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Time gate helpers
# ---------------------------------------------------------------------------

def _bear_utc() -> datetime:
    """A UTC datetime that is in the default bear_hours (3) AND bear_days (Monday=0)."""
    # 2026-05-18 is a Monday; hour 3 is in bear_hours
    return datetime(2026, 5, 18, 3, 0, 0, tzinfo=timezone.utc)


def _non_bear_hour_utc() -> datetime:
    """UTC datetime in a non-bear hour on a bear day."""
    return datetime(2026, 5, 18, 10, 0, 0, tzinfo=timezone.utc)  # Monday, hour 10 (not in [2,3,4,5,6,14,15])


def _non_bear_day_utc() -> datetime:
    """UTC datetime in a bear hour but on a non-bear day (Tuesday=1)."""
    return datetime(2026, 5, 19, 3, 0, 0, tzinfo=timezone.utc)  # Tuesday, hour 3


# ---------------------------------------------------------------------------
# _is_bear_time
# ---------------------------------------------------------------------------

def test_is_bear_time_true_in_bear_hour_and_day():
    s = _make_strategy()
    assert s._is_bear_time(_bear_utc()) is True


def test_is_bear_time_false_outside_bear_hour():
    s = _make_strategy()
    assert s._is_bear_time(_non_bear_hour_utc()) is False


def test_is_bear_time_false_outside_bear_day():
    s = _make_strategy()
    assert s._is_bear_time(_non_bear_day_utc()) is False


def test_is_bear_time_false_when_both_wrong():
    s = _make_strategy()
    non = datetime(2026, 5, 19, 10, 0, 0, tzinfo=timezone.utc)  # Tuesday, hour 10
    assert s._is_bear_time(non) is False


# ---------------------------------------------------------------------------
# should_enter — time gate blocks non-bear windows
# ---------------------------------------------------------------------------

def test_no_entry_outside_bear_hours():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _non_bear_hour_utc()
        result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_outside_bear_days():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _non_bear_day_utc()
        result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_on_bullish_candle_when_require_bearish_candle():
    s = _make_strategy({"bear_hours": [3], "bear_days": [0], "require_bearish_candle": True,
                        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "hold_cap_bars": 4})
    df = _make_ohlcv(bearish_last=False)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is None


def test_entry_on_bearish_candle_in_bear_window():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.SHORT


def test_entry_without_candle_filter_when_disabled():
    s = _make_strategy({"bear_hours": [3], "bear_days": [0], "require_bearish_candle": False,
                        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "hold_cap_bars": 4})
    df = _make_ohlcv(bearish_last=False)  # bullish candle, but filter disabled
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.SHORT


def test_signal_type_is_always_short():
    """Timeinality is short-only — never emits LONG."""
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.SHORT


def test_signal_strength_between_zero_and_one():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is not None
    assert 0.0 < result.strength <= 1.0


def test_no_entry_when_insufficient_data():
    s = _make_strategy()
    df = _make_ohlcv(n=2, bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        result = _run(s.should_enter(df))
    assert result is None


def test_bar_count_increments():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _non_bear_hour_utc()
        _run(s.should_enter(df))
    assert s._bar_count == 1


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_ohlcv()
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=1.6)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_ohlcv()
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=-0.9)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_exit_on_hold_cap():
    s = _make_strategy({"bear_hours": [3], "bear_days": [0], "require_bearish_candle": True,
                        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "hold_cap_bars": 4})
    s._entry_bar = 0
    s._bar_count = 5  # 5 - 0 = 5 > hold_cap_bars=4
    df = _make_ohlcv()
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=0.2)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_no_exit_within_hold_cap_no_pnl_trigger():
    s = _make_strategy({"bear_hours": [3], "bear_days": [0], "require_bearish_candle": True,
                        "take_profit_pct": 1.5, "stop_loss_pct": 0.8, "hold_cap_bars": 4})
    s._entry_bar = 3
    s._bar_count = 5  # 5 - 3 = 2 < hold_cap_bars=4
    df = _make_ohlcv()
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=0.3)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    df = _make_ohlcv()
    result = _run(s.should_exit(df, _make_position(is_long=False)))
    assert result is None


def test_entry_sets_entry_bar():
    s = _make_strategy()
    df = _make_ohlcv(bearish_last=True)
    with patch("src.strategies.timeinality_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _bear_utc()
        _run(s.should_enter(df))
    assert s._entry_bar is not None
    assert s._entry_bar == s._bar_count - 1
