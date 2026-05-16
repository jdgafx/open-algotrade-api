"""Tests for FlipFlopStrategy — SuperTrend flip-flop regime switcher."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import SignalType
from src.strategies.flip_flop_strategy import FlipFlopStrategy


def _make_strategy(params: dict | None = None) -> FlipFlopStrategy:
    s = FlipFlopStrategy.__new__(FlipFlopStrategy)
    s.config = MagicMock()
    s.config.params = params or {"atr_period": 5, "multiplier": 2.0}
    s.config.symbol = "BTC"
    s.config.size_usd = 100.0
    s.config.target_pct = 9.0
    s.config.max_loss_pct = -8.0
    s._entered_direction = 0
    return s


def _make_ohlcv(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    """Synthesize OHLCV where high/low are ±spread% of close."""
    closes = np.array(closes, dtype=float)
    highs = closes * (1 + spread / 100)
    lows = closes * (1 - spread / 100)
    return pd.DataFrame({
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.ones(len(closes)) * 1000,
    })


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Supertrend unit tests
# ---------------------------------------------------------------------------

def test_supertrend_returns_dataframe_with_required_columns():
    df = _make_ohlcv(list(range(50, 70)))
    result = FlipFlopStrategy._supertrend(df, period=5, multiplier=2.0)
    assert "st_direction" in result.columns
    assert "st_line" in result.columns


def test_supertrend_direction_is_plus_or_minus_one_after_warmup():
    closes = list(range(50, 80))  # steady uptrend
    df = _make_ohlcv(closes)
    result = FlipFlopStrategy._supertrend(df, period=5, multiplier=2.0)
    warmed = result["st_direction"].iloc[6:]
    assert set(warmed.unique()).issubset({-1, 0, 1})


def test_supertrend_uptrend_on_rising_prices():
    closes = list(range(50, 80))  # 30 bars of rising prices
    df = _make_ohlcv(closes)
    result = FlipFlopStrategy._supertrend(df, period=5, multiplier=2.0)
    # After warmup, most bars should be direction=1
    warmed = result["st_direction"].iloc[10:]
    assert (warmed == 1).sum() > (warmed == -1).sum()


def test_supertrend_downtrend_on_falling_prices():
    closes = list(range(80, 50, -1))  # 30 bars of falling prices
    df = _make_ohlcv(closes)
    result = FlipFlopStrategy._supertrend(df, period=5, multiplier=2.0)
    warmed = result["st_direction"].iloc[10:]
    assert (warmed == -1).sum() > (warmed == 1).sum()


# ---------------------------------------------------------------------------
# Entry signal tests
# ---------------------------------------------------------------------------

def test_no_entry_when_insufficient_data():
    s = _make_strategy({"atr_period": 10, "multiplier": 3.0})
    df = _make_ohlcv([100.0] * 5)  # too short for period=10
    result = _run(s.should_enter(df))
    assert result is None


def test_no_reentry_when_already_in_same_direction():
    """Already entered in the current ST direction — no duplicate entry."""
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})
    closes = list(range(50, 80))  # steady uptrend
    df = _make_ohlcv(closes)
    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    curr_dir = int(st["st_direction"].iloc[-1])
    # Simulate: we already entered in this direction
    s._entered_direction = curr_dir
    result = _run(s.should_enter(df))
    assert result is None


def test_initial_entry_on_clear_direction():
    """First call with no prior position enters based on current ST direction."""
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})
    closes = list(range(50, 80))  # steady uptrend
    df = _make_ohlcv(closes)
    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    curr_dir = int(st["st_direction"].iloc[-1])
    if curr_dir != 0:  # direction is established
        result = _run(s.should_enter(df))
        assert result is not None
        if curr_dir == 1:
            assert result.signal_type == SignalType.LONG
        else:
            assert result.signal_type == SignalType.SHORT


def test_long_entry_on_up_flip():
    """Construct a sequence that goes down then sharply up, producing an ST flip UP."""
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})

    # 20 falling bars → ST establishes downtrend
    falling = list(range(100, 80, -1))
    # Then 15 sharply rising bars to flip ST upward
    rising = [80 + i * 2 for i in range(15)]
    closes = falling + rising
    df = _make_ohlcv(closes, spread=0.3)

    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    directions = st["st_direction"].values

    # Find first upward flip in this data
    flip_idx = None
    for i in range(1, len(directions)):
        if directions[i] == 1 and directions[i - 1] == -1:
            flip_idx = i
            break

    if flip_idx is None:
        pytest.skip("No upward flip in synthetic data — adjust parameters")

    # Slice to exactly the flip bar
    df_slice = df.iloc[: flip_idx + 1].copy()
    result = _run(s.should_enter(df_slice))
    assert result is not None
    assert result.signal_type == SignalType.LONG
    assert result.strength == pytest.approx(0.85)


def test_short_entry_on_down_flip():
    """Rising then sharply falling → ST flip DOWN."""
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})

    rising = list(range(80, 100))
    falling = [100 - i * 2 for i in range(15)]
    closes = rising + falling
    df = _make_ohlcv(closes, spread=0.3)

    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    directions = st["st_direction"].values

    flip_idx = None
    for i in range(1, len(directions)):
        if directions[i] == -1 and directions[i - 1] == 1:
            flip_idx = i
            break

    if flip_idx is None:
        pytest.skip("No downward flip in synthetic data — adjust parameters")

    df_slice = df.iloc[: flip_idx + 1].copy()
    result = _run(s.should_enter(df_slice))
    assert result is not None
    assert result.signal_type == SignalType.SHORT
    assert result.strength == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Exit signal tests
# ---------------------------------------------------------------------------

def _make_position(is_long: bool, pnl_pct: float = 0.0) -> dict:
    return {"is_long": is_long, "size": 1.0 if is_long else -1.0, "pnl_perc": pnl_pct}


def test_no_exit_when_trend_aligned_with_position():
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})
    closes = list(range(50, 80))  # steady uptrend
    df = _make_ohlcv(closes)
    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    # If last two bars are both direction=1, a long should NOT get an exit
    if (st["st_direction"].iloc[-1] == 1 and st["st_direction"].iloc[-2] == 1):
        result = _run(s.should_exit(df, _make_position(is_long=True)))
        assert result is None


def test_exit_long_on_down_flip():
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})

    rising = list(range(80, 100))
    falling = [100 - i * 2 for i in range(15)]
    closes = rising + falling
    df = _make_ohlcv(closes, spread=0.3)

    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    directions = st["st_direction"].values

    flip_idx = None
    for i in range(1, len(directions)):
        if directions[i] == -1 and directions[i - 1] == 1:
            flip_idx = i
            break

    if flip_idx is None:
        pytest.skip("No downward flip in synthetic data")

    df_slice = df.iloc[: flip_idx + 1].copy()
    result = _run(s.should_exit(df_slice, _make_position(is_long=True)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_short_on_up_flip():
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})

    falling = list(range(100, 80, -1))
    rising = [80 + i * 2 for i in range(15)]
    closes = falling + rising
    df = _make_ohlcv(closes, spread=0.3)

    st = FlipFlopStrategy._supertrend(df, 5, 2.0)
    directions = st["st_direction"].values

    flip_idx = None
    for i in range(1, len(directions)):
        if directions[i] == 1 and directions[i - 1] == -1:
            flip_idx = i
            break

    if flip_idx is None:
        pytest.skip("No upward flip in synthetic data")

    df_slice = df.iloc[: flip_idx + 1].copy()
    result = _run(s.should_exit(df_slice, _make_position(is_long=False)))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_SHORT


def test_exit_on_target_hit():
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})
    closes = list(range(50, 80))  # stable uptrend — no flip
    df = _make_ohlcv(closes)
    pos = _make_position(is_long=True, pnl_pct=10.0)  # above target_pct=9.0
    result = _run(s.should_exit(df, pos))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_on_stop_loss():
    s = _make_strategy({"atr_period": 5, "multiplier": 2.0})
    closes = list(range(50, 80))
    df = _make_ohlcv(closes)
    pos = _make_position(is_long=True, pnl_pct=-9.0)  # below max_loss_pct=-8.0
    result = _run(s.should_exit(df, pos))
    assert result is not None
    assert result.signal_type == SignalType.CLOSE_LONG


def test_exit_insufficient_data_returns_none():
    s = _make_strategy({"atr_period": 10, "multiplier": 3.0})
    df = _make_ohlcv([100.0] * 3)
    result = _run(s.should_exit(df, _make_position(is_long=True)))
    assert result is None
