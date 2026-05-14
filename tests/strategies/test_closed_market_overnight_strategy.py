"""Companion tests for ClosedMarketOvernightStrategy — calendar gate + breakout."""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType
from src.strategies.closed_market_overnight_strategy import (
    ClosedMarketOvernightStrategy,
    _is_market_closed,
)


# -- Calendar gate ----------------------------------------------------------

def test_market_closed_on_saturday():
    sat_noon_utc = datetime(2026, 5, 16, 16, 0, tzinfo=timezone.utc)  # noon ET
    assert _is_market_closed(sat_noon_utc) is True


def test_market_closed_on_sunday():
    sun_noon_utc = datetime(2026, 5, 17, 16, 0, tzinfo=timezone.utc)
    assert _is_market_closed(sun_noon_utc) is True


def test_market_open_on_weekday_noon_et():
    # Mon May 18 noon ET = 16:00 UTC (EDT -4)
    mon_noon_utc = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    assert _is_market_closed(mon_noon_utc) is False


def test_market_closed_on_weekday_evening():
    # Mon May 18 22:00 ET = Tue 02:00 UTC. Evening, closed.
    weekday_evening_utc = datetime(2026, 5, 19, 2, 0, tzinfo=timezone.utc)
    assert _is_market_closed(weekday_evening_utc) is True


def test_market_closed_at_exact_close_time():
    # 16:00 ET = 20:00 UTC. Boundary should be closed (>= close time).
    boundary_utc = datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc)
    assert _is_market_closed(boundary_utc) is True


# -- should_enter respects calendar gate -----------------------------------

def _make_bars(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame({
        "open":   prices,
        "high":   [p * 1.001 for p in prices],
        "low":    [p * 0.999 for p in prices],
        "close":  prices,
        "volume": [100.0] * n,
    })


def _make_strategy(**param_overrides):
    s = ClosedMarketOvernightStrategy.__new__(ClosedMarketOvernightStrategy)
    s.config = MagicMock()
    s.config.symbol = "BTC"
    s.config.size_usd = 100
    params = {
        "momentum_lookback": 4,
        "breakout_pct": 0.005,
        "tp_pct": 0.010,
        "sl_pct": 0.008,
    }
    params.update(param_overrides)
    s.config.params = params
    return s


@pytest.mark.asyncio
async def test_no_signal_during_market_open():
    s = _make_strategy()
    # Breakout-worthy data
    df = _make_bars([100, 100, 100, 100, 100, 110])  # +10% above window high
    open_time = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)  # Mon noon ET
    with patch(
        "src.strategies.closed_market_overnight_strategy.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = open_time
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        sig = await s.should_enter(df)
    assert sig is None, "should not signal when NYSE is open"


@pytest.mark.asyncio
async def test_long_signal_on_breakout_when_closed():
    s = _make_strategy()
    df = _make_bars([100, 100, 100, 100, 100, 110])  # +10% breakout
    closed_time = datetime(2026, 5, 19, 2, 0, tzinfo=timezone.utc)  # Mon 22:00 ET
    with patch(
        "src.strategies.closed_market_overnight_strategy.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = closed_time
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        sig = await s.should_enter(df)
    assert sig is not None
    assert sig.signal_type == SignalType.LONG


@pytest.mark.asyncio
async def test_short_signal_on_breakdown_when_closed():
    s = _make_strategy()
    df = _make_bars([100, 100, 100, 100, 100, 90])  # -10% breakdown
    closed_time = datetime(2026, 5, 19, 2, 0, tzinfo=timezone.utc)
    with patch(
        "src.strategies.closed_market_overnight_strategy.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = closed_time
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        sig = await s.should_enter(df)
    assert sig is not None
    assert sig.signal_type == SignalType.SHORT


@pytest.mark.asyncio
async def test_no_signal_when_no_breakout():
    s = _make_strategy()
    df = _make_bars([100, 100.1, 100.2, 100.1, 100.0, 100.0])  # noise only
    closed_time = datetime(2026, 5, 19, 2, 0, tzinfo=timezone.utc)
    with patch(
        "src.strategies.closed_market_overnight_strategy.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = closed_time
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        sig = await s.should_enter(df)
    assert sig is None


# -- should_exit ----------------------------------------------------------

@pytest.mark.asyncio
async def test_exit_on_take_profit():
    s = _make_strategy(tp_pct=0.010)
    pos = {"is_long": True, "pnl_perc": 1.5}  # 1.5% > 1.0% TP
    sig = await s.should_exit(_make_bars([100, 101, 102]), pos)
    assert sig is not None
    assert sig.signal_type == SignalType.CLOSE_LONG


@pytest.mark.asyncio
async def test_exit_on_stop_loss():
    s = _make_strategy(sl_pct=0.008)
    pos = {"is_long": True, "pnl_perc": -1.0}  # -1.0% < -0.8% SL
    sig = await s.should_exit(_make_bars([100, 99, 98]), pos)
    assert sig is not None
    assert sig.signal_type == SignalType.CLOSE_LONG


@pytest.mark.asyncio
async def test_force_close_at_market_open():
    s = _make_strategy()
    pos = {"is_long": True, "pnl_perc": 0.1}  # tiny PnL — would normally hold
    open_time = datetime(2026, 5, 18, 16, 0, tzinfo=timezone.utc)
    with patch(
        "src.strategies.closed_market_overnight_strategy.datetime"
    ) as mock_dt:
        mock_dt.now.return_value = open_time
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        sig = await s.should_exit(_make_bars([100, 100, 100]), pos)
    assert sig is not None
    assert sig.signal_type == SignalType.CLOSE_LONG
    assert "force-close" in sig.reason.lower()
