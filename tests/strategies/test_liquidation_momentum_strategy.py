"""Tests for LiquidationMomentumStrategy — ride the liquidation cascade."""

import asyncio
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType


def _make_strategy(params: dict | None = None):
    from src.strategies.liquidation_momentum_strategy import LiquidationMomentumStrategy
    s = LiquidationMomentumStrategy.__new__(LiquidationMomentumStrategy)
    s.config = MagicMock()
    s.config.params = params or {
        "vol_spike_mult": 2.5,
        "vol_lookback": 10,
        "momentum_candle_pct": 0.003,
        "hold_cap_bars": 4,
        "take_profit_pct": 2.0,
        "stop_loss_pct": 1.2,
    }
    s.config.symbol = "BTC"
    s.config.size_usd = 100.0
    s.config.target_pct = 2.0
    s.config.max_loss_pct = -1.2
    s._entry_bar = None
    s._bar_count = 0
    return s


def _make_ohlcv(closes, opens=None, highs=None, lows=None, volumes=None):
    closes = np.array(closes, dtype=float)
    n = len(closes)
    if opens is None:
        opens = closes
    if highs is None:
        highs = closes * 1.005
    if lows is None:
        lows = closes * 0.995
    if volumes is None:
        volumes = np.ones(n) * 1000.0
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
    })


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _detect_liquidation_spike
# ---------------------------------------------------------------------------

def test_no_spike_on_normal_volume():
    s = _make_strategy()
    vols = [1000.0] * 25
    df = _make_ohlcv([100.0] * 25, volumes=vols)
    assert not s._detect_liquidation_spike(df)


def test_spike_detected_on_high_volume_big_candle():
    s = _make_strategy()
    base_closes = [100.0] * 20
    base_vols = [1000.0] * 20
    # Big bearish candle with 5x volume
    spike_close = [95.0]
    spike_vol = [5500.0]
    opens = base_closes + [101.0]
    closes = base_closes + spike_close
    highs = [c * 1.003 for c in base_closes] + [101.5]
    lows = [c * 0.997 for c in base_closes] + [94.0]
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": base_vols + spike_vol,
    })
    assert s._detect_liquidation_spike(df) is True


def test_no_spike_when_insufficient_data():
    s = _make_strategy()
    df = _make_ohlcv([100.0] * 5)
    assert not s._detect_liquidation_spike(df)


# ---------------------------------------------------------------------------
# _spike_direction
# ---------------------------------------------------------------------------

def test_bearish_spike_returns_short():
    s = _make_strategy()
    # Last candle: open > close = bearish
    closes = [100.0] * 20 + [95.0]
    opens = [100.0] * 20 + [100.5]
    df = _make_ohlcv(closes, opens=opens)
    assert s._spike_direction(df) == "SHORT"


def test_bullish_spike_returns_long():
    s = _make_strategy()
    closes = [100.0] * 20 + [105.0]
    opens = [100.0] * 20 + [99.5]
    df = _make_ohlcv(closes, opens=opens)
    assert s._spike_direction(df) == "LONG"


def test_flat_candle_returns_none():
    s = _make_strategy()
    closes = [100.0] * 21
    opens = [100.0] * 21
    df = _make_ohlcv(closes, opens=opens)
    assert s._spike_direction(df) is None


# ---------------------------------------------------------------------------
# should_enter
# ---------------------------------------------------------------------------

def test_no_entry_on_insufficient_data():
    s = _make_strategy()
    df = _make_ohlcv([100.0] * 5)
    result = _run(s.should_enter(df))
    assert result is None


def test_no_entry_without_volume_spike():
    s = _make_strategy()
    normal_vols = [1000.0] * 25
    closes = list(range(100, 125))
    df = _make_ohlcv(closes, volumes=normal_vols)
    result = _run(s.should_enter(df))
    assert result is None


def test_short_entry_on_bearish_liquidation_spike():
    s = _make_strategy()
    base = [100.0] * 20
    closes = base + [94.0]  # big bearish drop
    opens = base + [100.5]
    highs = [c * 1.003 for c in base] + [101.0]
    lows = [c * 0.997 for c in base] + [93.0]
    vols = [1000.0] * 20 + [5500.0]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
    result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.SHORT


def test_long_entry_on_bullish_liquidation_spike():
    s = _make_strategy()
    base = [100.0] * 20
    closes = base + [106.0]  # big bullish surge (short squeeze)
    opens = base + [99.5]
    highs = [c * 1.003 for c in base] + [107.0]
    lows = [c * 0.997 for c in base] + [99.0]
    vols = [1000.0] * 20 + [5500.0]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
    result = _run(s.should_enter(df))
    assert result is not None
    assert result.signal_type == SignalType.LONG


def test_signal_strength_between_0_and_1():
    s = _make_strategy()
    base = [100.0] * 20
    closes = base + [94.0]
    opens = base + [100.5]
    highs = [c * 1.003 for c in base] + [101.0]
    lows = [c * 0.997 for c in base] + [93.0]
    vols = [1000.0] * 20 + [5500.0]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
    result = _run(s.should_enter(df))
    assert result is not None
    assert 0.0 < result.strength <= 1.0


def test_bar_count_increments_on_each_call():
    s = _make_strategy()
    normal_vols = [1000.0] * 25
    df = _make_ohlcv([100.0] * 25, volumes=normal_vols)
    _run(s.should_enter(df))
    assert s._bar_count == 1
    _run(s.should_enter(df))
    assert s._bar_count == 2


# ---------------------------------------------------------------------------
# should_exit
# ---------------------------------------------------------------------------

def _make_position(is_long: bool, pnl_pct: float = 0.0):
    return {"is_long": is_long, "size": 1.0 if is_long else -1.0, "pnl_perc": pnl_pct}


def test_exit_on_profit_target():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=2.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=-1.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_hold_cap():
    s = _make_strategy({"vol_spike_mult": 2.5, "vol_lookback": 10,
                        "momentum_candle_pct": 0.003, "hold_cap_bars": 4,
                        "take_profit_pct": 2.0, "stop_loss_pct": 1.2})
    s._entry_bar = 0
    s._bar_count = 5  # 5 - 0 = 5 > hold_cap_bars=4 → exit
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=0.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_no_exit_within_hold_cap_and_no_pnl_trigger():
    s = _make_strategy({"vol_spike_mult": 2.5, "vol_lookback": 10,
                        "momentum_candle_pct": 0.003, "hold_cap_bars": 4,
                        "take_profit_pct": 2.0, "stop_loss_pct": 1.2})
    s._entry_bar = 3
    s._bar_count = 5  # 5 - 3 = 2 < hold_cap_bars=4
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=True, pnl_pct=0.5)))
    assert result is None


def test_no_exit_when_not_in_position():
    s = _make_strategy()
    s._entry_bar = None
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=True)))
    assert result is None


def test_short_stop_loss_emits_close_short():
    s = _make_strategy()
    s._entry_bar = 5
    s._bar_count = 6
    df = _make_ohlcv([100.0] * 25)
    result = _run(s.should_exit(df, _make_position(is_long=False, pnl_pct=-1.5)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_entry_sets_entry_bar():
    s = _make_strategy()
    base = [100.0] * 20
    closes = base + [94.0]
    opens = base + [100.5]
    highs = [c * 1.003 for c in base] + [101.0]
    lows = [c * 0.997 for c in base] + [93.0]
    vols = [1000.0] * 20 + [5500.0]
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
    _run(s.should_enter(df))
    assert s._entry_bar is not None
    assert s._entry_bar == s._bar_count - 1
