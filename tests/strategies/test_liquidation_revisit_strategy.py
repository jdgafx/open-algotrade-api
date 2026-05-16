"""Tests for LiquidationRevisitStrategy — wait for price to revisit liquidation low."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.liquidation_revisit_strategy import LiquidationRevisitStrategy

    s = LiquidationRevisitStrategy.__new__(LiquidationRevisitStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "liq_candle_pct": 0.02,
        "vol_spike_mult": 1.5,
        "vol_lookback": 20,
        "revisit_window_bars": 20,
        "revisit_tolerance_pct": 0.005,
        "rsi_period": 14,
        "rsi_oversold": 35,
        "take_profit_pct": 2.0,
        "stop_loss_pct": 1.5,
        "hold_cap_bars": 8,
    }
    s.config.symbol = "ETH"
    s.config.size_usd = 200.0
    s.config.target_pct = 2.0
    s.config.max_loss_pct = -1.5
    s._liq_low = None
    s._liq_bar = None
    s._entry_bar = None
    s._bar_count = 0
    return s


def _run(coro):
    return asyncio.run(coro)


def _make_position(pnl_pct: float = 0.0) -> Dict[str, Any]:
    return {"is_long": True, "size": 1.0, "pnl_perc": pnl_pct}


def _make_flat_df(n: int = 40) -> pd.DataFrame:
    closes = np.ones(n) * 100.0
    return pd.DataFrame({
        "open": closes.copy(), "high": closes * 1.002,
        "low": closes * 0.998, "close": closes,
        "volume": np.ones(n) * 1000.0,
    })


def _make_liq_then_revisit_df(n: int = 50, revisit: bool = True,
                               rsi_oversold: bool = True) -> pd.DataFrame:
    """
    Build a DataFrame where:
    - bars 0..n-10: steady price 100
    - bar n-5: big red liquidation candle (drop 3%, high vol)
    - bars n-4..n-2: partial recovery to ~98.5
    - bar n-1: revisit the liq low (close near 97) with RSI oversold
    """
    closes = np.ones(n) * 100.0
    opens = closes.copy()
    highs = closes * 1.002
    lows = closes * 0.998
    vols = np.ones(n) * 1000.0

    # Liquidation candle: index n-5
    liq_idx = n - 5
    opens[liq_idx] = 100.0
    closes[liq_idx] = 97.0   # 3% drop
    lows[liq_idx] = 96.8
    highs[liq_idx] = 100.1
    vols[liq_idx] = 2000.0   # 2x volume

    if revisit:
        # Partial recovery then revisit
        closes[liq_idx + 1] = 98.0
        closes[liq_idx + 2] = 98.5
        closes[liq_idx + 3] = 97.8
        # Last bar: close at the liq low (revisit)
        closes[-1] = 97.1
        lows[-1] = 96.9
        opens[-1] = 97.5
        highs[-1] = 97.8
    else:
        # Price recovers and doesn't revisit
        closes[liq_idx + 1] = 99.0
        closes[liq_idx + 2] = 99.5
        closes[liq_idx + 3] = 100.0
        closes[-1] = 100.2

    opens[liq_idx + 1:] = closes[liq_idx + 1:]
    highs[liq_idx + 1:] = np.maximum(opens[liq_idx + 1:], closes[liq_idx + 1:]) * 1.001
    lows[liq_idx + 1:] = np.minimum(opens[liq_idx + 1:], closes[liq_idx + 1:]) * 0.999

    if not rsi_oversold:
        # Override last bar to have neutral RSI (flat price)
        closes[-1] = 100.0
        opens[-1] = 100.0
        highs[-1] = 100.1
        lows[-1] = 99.9

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


# ---------------------------------------------------------------------------
# _is_liquidation_candle
# ---------------------------------------------------------------------------

def test_detects_liquidation_candle():
    s = _make_strategy()
    df = _make_liq_then_revisit_df()
    liq_idx = len(df) - 5
    row = df.iloc[liq_idx]
    avg_vol = float(df["volume"].iloc[:liq_idx].mean())
    assert s._is_liquidation_candle(row, avg_vol) is True


def test_no_liq_on_small_candle():
    s = _make_strategy()
    df = _make_flat_df()
    avg_vol = 1000.0
    row = df.iloc[-1]
    assert s._is_liquidation_candle(row, avg_vol) is False


# ---------------------------------------------------------------------------
# should_enter — state transitions
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


def test_liq_detected_sets_watching_state():
    """After seeing a liquidation bar, strategy enters WATCHING state."""
    s = _make_strategy()
    # Feed bars up to (including) the liq bar
    df = _make_liq_then_revisit_df()
    liq_idx = len(df) - 5
    partial_df = df.iloc[: liq_idx + 1]
    result = _run(s.should_enter(partial_df))
    assert result is None  # No entry yet — watching
    assert s._liq_low is not None


def test_entry_on_revisit_with_rsi_oversold():
    """Full sequence: flat → liq candle detected → revisit → LONG."""
    s = _make_strategy()
    df = _make_liq_then_revisit_df(revisit=True)

    # Feed all bars one by one to simulate streaming
    result = None
    for i in range(len(df.index) - 5, len(df.index)):
        result = _run(s.should_enter(df.iloc[:i + 1]))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_no_entry_when_price_recovers_no_revisit():
    s = _make_strategy()
    df = _make_liq_then_revisit_df(revisit=False)
    result = None
    for i in range(len(df.index) - 5, len(df.index)):
        result = _run(s.should_enter(df.iloc[:i + 1]))
    assert result is None


def test_signal_strength_in_range():
    s = _make_strategy()
    df = _make_liq_then_revisit_df(revisit=True)
    result = None
    for i in range(len(df.index) - 5, len(df.index)):
        r = _run(s.should_enter(df.iloc[:i + 1]))
        if r is not None:
            result = r
    if result is not None:
        assert 0.0 < result.strength <= 1.0


def test_watching_expires_after_window():
    """If price doesn't revisit within revisit_window_bars, reset to IDLE."""
    s = _make_strategy({"liq_candle_pct": 0.02, "vol_spike_mult": 1.5,
                        "vol_lookback": 20, "revisit_window_bars": 3,
                        "revisit_tolerance_pct": 0.005, "rsi_period": 14,
                        "rsi_oversold": 35, "take_profit_pct": 2.0,
                        "stop_loss_pct": 1.5, "hold_cap_bars": 8})
    # Liq detected, then price recovers — no revisit within 3 bars
    df = _make_liq_then_revisit_df(revisit=False, n=50)
    liq_idx = len(df) - 5
    # Feed liq bar
    _run(s.should_enter(df.iloc[:liq_idx + 1]))
    assert s._liq_low is not None
    # Feed 4 more bars (> window=3) without revisit
    for i in range(liq_idx + 1, liq_idx + 5):
        _run(s.should_enter(df.iloc[:i + 1]))
    # Should have reset
    assert s._liq_low is None


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
    s._bar_count = 9  # > hold_cap_bars=8
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_no_exit_within_limits():
    s = _make_strategy()
    s._entry_bar = 3
    s._bar_count = 5
    result = _run(s.should_exit(_make_flat_df(), _make_position(pnl_pct=0.5)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    result = _run(s.should_exit(_make_flat_df(), _make_position()))
    assert result is None
