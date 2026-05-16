"""Tests for DayOfWeekBiasStrategy — trade only on statistically favorable days."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.day_of_week_strategy import DayOfWeekBiasStrategy

    s = DayOfWeekBiasStrategy.__new__(DayOfWeekBiasStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "trade_days": [2, 3],  # Wednesday=2, Thursday=3
        "rsi_period": 14,
        "rsi_min": 45,
        "sma_period": 20,
        "require_bullish_candle": True,
        "take_profit_pct": 2.0,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 6,
    }
    s.config.symbol = "BTC"
    s.config.size_usd = 200.0
    s.config.target_pct = 2.0
    s.config.max_loss_pct = -1.5
    s._entry_bar = None
    s._bar_count = 0
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": True, "size": 1.0, "pnl_perc": pnl_pct}


# Thursday UTC
_THURSDAY = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
# Wednesday UTC
_WEDNESDAY = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
# Friday UTC (not a trade day)
_FRIDAY = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
# Monday UTC (not a trade day)
_MONDAY = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _make_bullish_df(n: int = 40, rsi_bullish: bool = True) -> pd.DataFrame:
    """Rising price action that produces RSI > 45 and SMA crossover."""
    if rsi_bullish:
        # Steady climb with slight acceleration
        closes = np.array([100.0 + i * 0.3 for i in range(n)])
    else:
        # Flat then declining — RSI neutral/bearish
        closes = np.ones(n) * 100.0
        closes[-5:] = [99.5, 99.0, 98.5, 98.0, 97.5]
    opens = closes * 0.999
    highs = closes * 1.002
    lows = opens * 0.998
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.ones(n) * 1000.0,
    })


def _make_flat_df(n: int = 40) -> pd.DataFrame:
    closes = np.ones(n) * 100.0
    return pd.DataFrame({
        "open": closes.copy(), "high": closes * 1.001,
        "low": closes * 0.999, "close": closes,
        "volume": np.ones(n) * 1000.0,
    })


# ---------------------------------------------------------------------------
# _is_trade_day
# ---------------------------------------------------------------------------

def test_thursday_is_trade_day():
    s = _make_strategy()
    assert s._is_trade_day(_THURSDAY) is True


def test_wednesday_is_trade_day():
    s = _make_strategy()
    assert s._is_trade_day(_WEDNESDAY) is True


def test_friday_is_not_trade_day():
    s = _make_strategy()
    assert s._is_trade_day(_FRIDAY) is False


def test_monday_is_not_trade_day():
    s = _make_strategy()
    assert s._is_trade_day(_MONDAY) is False


def test_custom_trade_days():
    s = _make_strategy({"trade_days": [0, 3], "rsi_period": 14, "rsi_min": 45,
                        "sma_period": 20, "require_bullish_candle": True,
                        "take_profit_pct": 2.0, "stop_loss_pct": 1.5, "hold_cap_bars": 6})
    # Monday=0 and Thursday=3 are trade days; Friday=4 is not
    monday = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    friday = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    assert s._is_trade_day(monday) is True
    assert s._is_trade_day(friday) is False


# ---------------------------------------------------------------------------
# should_enter
# ---------------------------------------------------------------------------

def test_no_entry_on_non_trade_day():
    s = _make_strategy()
    df = _make_bullish_df()
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _FRIDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_bullish_df(n=5)
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _THURSDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    assert result is None


def test_entry_long_on_thursday_bullish():
    s = _make_strategy()
    df = _make_bullish_df(rsi_bullish=True)
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _THURSDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_entry_long_on_wednesday_bullish():
    s = _make_strategy()
    df = _make_bullish_df(rsi_bullish=True)
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _WEDNESDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_no_entry_on_trade_day_when_bearish():
    s = _make_strategy()
    df = _make_bullish_df(rsi_bullish=False)
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _THURSDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    assert result is None


def test_signal_type_always_long():
    s = _make_strategy()
    df = _make_bullish_df()
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _THURSDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    if result is not None:
        assert result.signal_type == SignalType.LONG


def test_signal_strength_in_range():
    s = _make_strategy()
    df = _make_bullish_df()
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _THURSDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = _run(s.should_enter(df))
    if result is not None:
        assert 0.0 < result.strength <= 1.0


def test_bar_count_increments():
    s = _make_strategy()
    df = _make_flat_df()
    with patch("src.strategies.day_of_week_strategy.datetime") as mock_dt:
        mock_dt.now.return_value = _FRIDAY
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        _run(s.should_enter(df))
        assert s._bar_count == 1
        _run(s.should_enter(df))
        assert s._bar_count == 2


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=2.1)))
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
    s._bar_count = 7  # 7 > hold_cap_bars=6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_no_exit_within_limits():
    s = _make_strategy()
    s._entry_bar = 3
    s._bar_count = 5  # 2 bars held < 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    result = _run(s.should_exit(_make_flat_df(), _make_position()))
    assert result is None
