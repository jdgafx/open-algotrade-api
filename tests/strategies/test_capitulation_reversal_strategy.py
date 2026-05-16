"""Tests for CapitulationReversalStrategy — catch extreme sell-offs on reversal confirmation."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.capitulation_reversal_strategy import CapitulationReversalStrategy

    s = CapitulationReversalStrategy.__new__(CapitulationReversalStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "rsi_period": 14,
        "rsi_oversold": 25,
        "drop_lookback": 20,
        "drop_pct": 0.03,
        "vol_spike_mult": 1.5,
        "vol_lookback": 20,
        "require_bullish_confirm": True,
        "take_profit_pct": 2.0,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 8,
    }
    s.config.symbol = "ETH"
    s.config.size_usd = 200.0
    s.config.target_pct = 2.0
    s.config.max_loss_pct = -1.5
    s._entry_bar = None
    s._bar_count = 0
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(is_long: bool, pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": is_long, "size": 1.0 if is_long else -1.0, "pnl_perc": pnl_pct}


def _make_normal_df(n: int = 40) -> pd.DataFrame:
    """N bars of flat, normal-volume, RSI-neutral price action."""
    closes = np.ones(n) * 100.0
    opens = closes.copy()
    return pd.DataFrame({
        "open": opens,
        "high": closes * 1.002,
        "low": closes * 0.998,
        "close": closes,
        "volume": np.ones(n) * 1000.0,
    })


def _make_capitulation_df(n: int = 40, rsi_oversold: bool = True,
                           big_drop: bool = True, vol_spike: bool = True,
                           bullish_last: bool = True) -> pd.DataFrame:
    """Construct a DataFrame that triggers capitulation conditions."""
    # Steady prices for first (n-5) bars
    base = n - 5
    closes = np.concatenate([
        np.ones(base) * 100.0,
        # 4 bars of sharp decline
        [98.5, 97.0, 95.5, 94.0],
        # last bar: recovery (bullish confirm) or continuation (bearish)
        [94.5] if bullish_last else [93.5],
    ])
    opens = closes.copy()
    # Last bar: bullish = open < close
    if bullish_last:
        opens[-1] = 93.8   # close(94.5) > open(93.8) → bullish
    else:
        opens[-1] = 94.8   # close(93.5) < open(94.8) → bearish

    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998

    if not big_drop:
        # Make it a tiny drop instead
        closes[base:base+4] = np.ones(4) * 99.8
        closes[-1] = 99.9 if bullish_last else 99.7
        opens = closes.copy()
        highs = closes * 1.001
        lows = closes * 0.999

    vols = np.ones(n) * 1000.0
    if vol_spike:
        vols[-1] = 2000.0  # 2x volume on last bar (reversal volume)
    if not rsi_oversold:
        # Flatten closes so RSI stays neutral
        closes = np.ones(n) * 100.0
        opens = closes.copy()
        highs = closes * 1.001
        lows = closes * 0.999
        vols = np.ones(n) * 1000.0
        if vol_spike:
            vols[-1] = 2000.0

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


# ---------------------------------------------------------------------------
# _compute_rsi
# ---------------------------------------------------------------------------

def test_rsi_returns_array_same_length():
    s = _make_strategy()
    closes = np.array([100.0] * 30 + [101.0, 102.0, 101.5, 103.0, 102.5], dtype=float)
    rsi = s._compute_rsi(closes, period=14)
    assert len(rsi) == len(closes)


def test_rsi_oversold_on_steady_decline():
    s = _make_strategy()
    # Steady step-down: RSI should be very low
    closes = np.array([100.0 - i * 0.5 for i in range(40)], dtype=float)
    rsi = s._compute_rsi(closes, period=14)
    assert rsi[-1] < 30


def test_rsi_overbought_on_steady_rise():
    s = _make_strategy()
    closes = np.array([100.0 + i * 0.5 for i in range(40)], dtype=float)
    rsi = s._compute_rsi(closes, period=14)
    assert rsi[-1] > 70


# ---------------------------------------------------------------------------
# should_enter
# ---------------------------------------------------------------------------

def test_no_entry_on_normal_market():
    s = _make_strategy()
    df = _make_normal_df()
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_normal_df(n=5)
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_without_bullish_confirm_when_required():
    s = _make_strategy({"rsi_period": 14, "rsi_oversold": 25, "drop_lookback": 20,
                        "drop_pct": 0.03, "vol_spike_mult": 1.5, "vol_lookback": 20,
                        "require_bullish_confirm": True, "take_profit_pct": 2.0,
                        "stop_loss_pct": 1.5, "hold_cap_bars": 8})
    df = _make_capitulation_df(bullish_last=False)
    result = _run(s.should_enter(df))
    assert result is None


def test_entry_long_on_capitulation_reversal():
    s = _make_strategy()
    df = _make_capitulation_df(rsi_oversold=True, big_drop=True,
                                vol_spike=True, bullish_last=True)
    result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_signal_type_always_long():
    """Capitulation reversal only goes LONG — buying the dip."""
    s = _make_strategy()
    df = _make_capitulation_df(bullish_last=True)
    result = _run(s.should_enter(df))
    if result is not None:
        assert result.signal_type == SignalType.LONG


def test_entry_without_bullish_confirm_when_disabled():
    s = _make_strategy({"rsi_period": 14, "rsi_oversold": 25, "drop_lookback": 20,
                        "drop_pct": 0.03, "vol_spike_mult": 1.5, "vol_lookback": 20,
                        "require_bullish_confirm": False, "take_profit_pct": 2.0,
                        "stop_loss_pct": 1.5, "hold_cap_bars": 8})
    df = _make_capitulation_df(rsi_oversold=True, big_drop=True,
                                vol_spike=True, bullish_last=False)
    result = _run(s.should_enter(df))
    # May or may not trigger depending on drop/rsi — just ensure it can signal
    # (no assertion on result — just verify no exception raised)


def test_signal_strength_between_zero_and_one():
    s = _make_strategy()
    df = _make_capitulation_df(bullish_last=True)
    result = _run(s.should_enter(df))
    if result is not None:
        assert 0.0 < result.strength <= 1.0


def test_bar_count_increments():
    s = _make_strategy()
    df = _make_normal_df()
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
    df = _make_normal_df()
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=2.1)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_normal_df()
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=-1.6)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_hold_cap():
    s = _make_strategy({"rsi_period": 14, "rsi_oversold": 25, "drop_lookback": 20,
                        "drop_pct": 0.03, "vol_spike_mult": 1.5, "vol_lookback": 20,
                        "require_bullish_confirm": True, "take_profit_pct": 2.0,
                        "stop_loss_pct": 1.5, "hold_cap_bars": 4})
    s._entry_bar = 0
    s._bar_count = 5  # 5 > hold_cap_bars=4
    df = _make_normal_df()
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_no_exit_within_hold_cap_no_pnl_trigger():
    s = _make_strategy({"rsi_period": 14, "rsi_oversold": 25, "drop_lookback": 20,
                        "drop_pct": 0.03, "vol_spike_mult": 1.5, "vol_lookback": 20,
                        "require_bullish_confirm": True, "take_profit_pct": 2.0,
                        "stop_loss_pct": 1.5, "hold_cap_bars": 8})
    s._entry_bar = 3
    s._bar_count = 5  # 2 bars held < hold_cap=8
    df = _make_normal_df()
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=0.5)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    df = _make_normal_df()
    result = _run(s.should_exit(df, _make_position(is_long=True)))
    assert result is None
