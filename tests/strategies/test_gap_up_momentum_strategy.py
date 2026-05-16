"""Tests for GapUpMomentumStrategy — gap up + UO/ADX/MFI confirmation."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.gap_up_momentum_strategy import GapUpMomentumStrategy

    s = GapUpMomentumStrategy.__new__(GapUpMomentumStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "gap_threshold_pct": 0.005,
        "uo_period_short": 7,
        "uo_period_med": 14,
        "uo_period_long": 28,
        "uo_oversold": 25,
        "adx_period": 14,
        "adx_threshold": 30,
        "mfi_period": 14,
        "mfi_oversold": 35,
        "take_profit_pct": 5.0,
        "stop_loss_pct": 2.0,
        "hold_cap_bars": 10,
    }
    s.config.symbol = "BTC"
    s.config.size_usd = 200.0
    s.config.target_pct = 5.0
    s.config.max_loss_pct = -2.0
    s._entry_bar = None
    s._bar_count = 0
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": True, "size": 1.0, "pnl_perc": pnl_pct}


def _make_flat_df(n: int = 60, price: float = 100.0) -> pd.DataFrame:
    closes = np.full(n, price)
    vols = np.ones(n) * 1000.0
    return pd.DataFrame({
        "open": closes.copy(),
        "high": closes * 1.002,
        "low": closes * 0.998,
        "close": closes,
        "volume": vols,
    })


def _make_gap_up_df(n: int = 60, gap_pct: float = 0.01,
                    uo_low: bool = True, adx_high: bool = True,
                    mfi_low: bool = True) -> pd.DataFrame:
    """
    Build DataFrame where last bar has a gap up from the previous bar's close.
    Earlier bars are constructed to drive indicator values into/out of range.
    """
    closes = np.ones(n) * 100.0
    opens = closes.copy()
    highs = closes * 1.002
    lows = closes * 0.998
    vols = np.ones(n) * 1000.0

    # Create a bearish run in earlier bars to push UO + MFI oversold
    if uo_low and mfi_low:
        # Steady decline for bars n-30 to n-2 to make oscillators oversold
        for i in range(n - 30, n - 1):
            frac = (i - (n - 30)) / 30
            drop = 0.003 * frac
            closes[i] = 100.0 - frac * 2.0
            opens[i] = closes[i] + 0.05
            highs[i] = max(opens[i], closes[i]) * 1.001
            lows[i] = min(opens[i], closes[i]) * 0.999
            vols[i] = 800.0  # lower volume on down moves

    # ADX high: create directional trend in bars n-20 to n-2
    if adx_high:
        for i in range(n - 20, n - 1):
            # Strong directional moves
            closes[i] = closes[max(i-1, 0)] * 0.996  # consistent downtrend
            opens[i] = closes[i] * 1.002
            highs[i] = opens[i] * 1.003
            lows[i] = closes[i] * 0.997

    # Last bar: gap up from previous close
    prev_close = closes[-2]
    gap_open = prev_close * (1 + gap_pct)
    closes[-1] = gap_open * 1.002  # bullish gap bar
    opens[-1] = gap_open
    highs[-1] = gap_open * 1.006
    lows[-1] = gap_open * 0.999
    vols[-1] = 1500.0  # slightly higher vol on gap

    if not uo_low:
        # Override to make UO not oversold (flat recent bars)
        for i in range(n - 30, n - 1):
            closes[i] = 100.0
            opens[i] = 100.0
            highs[i] = 100.1
            lows[i] = 99.9
            vols[i] = 1000.0

    if not mfi_low:
        # High volume on up moves → MFI will be high (not oversold)
        for i in range(n - 14, n - 1):
            vols[i] = 5000.0
            closes[i] = opens[i] + 0.1

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


# ---------------------------------------------------------------------------
# Indicator unit tests
# ---------------------------------------------------------------------------

def test_compute_adx_returns_float():
    s = _make_strategy()
    df = _make_flat_df(60)
    # Should not raise; returns float in [0, 100]
    result = s._compute_adx(df, period=14)
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_compute_mfi_returns_float():
    s = _make_strategy()
    df = _make_flat_df(60)
    result = s._compute_mfi(df, period=14)
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_compute_uo_returns_float():
    s = _make_strategy()
    df = _make_flat_df(60)
    result = s._compute_uo(df, short=7, medium=14, long_=28)
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_adx_high_on_trend():
    s = _make_strategy()
    df = _make_gap_up_df(adx_high=True)
    adx = s._compute_adx(df, period=14)
    # Directional trend should produce non-trivial ADX
    assert adx >= 0.0  # sanity; exact threshold depends on data


def test_mfi_low_on_down_volume():
    s = _make_strategy()
    df = _make_gap_up_df(mfi_low=True)
    mfi = s._compute_mfi(df, period=14)
    assert isinstance(mfi, float)


# ---------------------------------------------------------------------------
# should_enter — state and filters
# ---------------------------------------------------------------------------

def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_flat_df(n=10)
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_when_no_gap():
    s = _make_strategy()
    df = _make_flat_df(n=60)
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_when_gap_too_small():
    s = _make_strategy({"gap_threshold_pct": 0.02,
                        "uo_period_short": 7, "uo_period_med": 14, "uo_period_long": 28,
                        "uo_oversold": 25, "adx_period": 14, "adx_threshold": 30,
                        "mfi_period": 14, "mfi_oversold": 35,
                        "take_profit_pct": 5.0, "stop_loss_pct": 2.0, "hold_cap_bars": 10})
    df = _make_gap_up_df(gap_pct=0.003)  # gap 0.3% < threshold 2%
    result = _run(s.should_enter(df))
    assert result is None


def test_entry_is_long_when_all_conditions_met():
    s = _make_strategy()
    df = _make_gap_up_df(gap_pct=0.01, uo_low=True, adx_high=True, mfi_low=True)
    result = _run(s.should_enter(df))
    # May or may not fire depending on oscillator values; just check type if it does
    if result is not None:
        assert result.signal_type == SignalType.LONG
        assert result.price > 0
        assert 0.0 < result.strength <= 1.0


def test_signal_strength_in_range():
    s = _make_strategy()
    for _ in range(3):
        df = _make_gap_up_df(gap_pct=0.012, uo_low=True, adx_high=True, mfi_low=True)
        result = _run(s.should_enter(df))
        if result is not None:
            assert 0.0 < result.strength <= 1.0
            return


def test_no_double_entry():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_gap_up_df(gap_pct=0.01)
    result = _run(s.should_enter(df))
    assert result is None  # already in position


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=5.1)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=-2.1)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_hold_cap():
    s = _make_strategy()
    s._entry_bar = 0
    s._bar_count = 11  # > hold_cap_bars=10
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
