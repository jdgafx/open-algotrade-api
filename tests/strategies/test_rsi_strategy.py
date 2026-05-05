"""Companion tests for RSIStrategy — RegimeBiasMixin integration."""
from unittest.mock import MagicMock

import pytest
from src.strategies.base_strategy import Signal, SignalType
from src.strategies.rsi_strategy import RSIStrategy


def _make_strategy(regime_hint: str) -> RSIStrategy:
    s = RSIStrategy.__new__(RSIStrategy)
    s.config = MagicMock()
    s.config.params = {"_regime_hint": regime_hint}
    s.config.symbol = "ETH"
    return s


def test_has_regime_bias_mixin():
    from src.strategies.regime_bias_mixin import RegimeBiasMixin
    assert issubclass(RSIStrategy, RegimeBiasMixin)


def test_suppresses_short_in_uptrend():
    s = _make_strategy("trending_up")
    result = s._apply_regime_bias(Signal(signal_type=SignalType.SHORT, symbol="ETH"))
    assert result is None


def test_suppresses_long_in_downtrend():
    s = _make_strategy("trending_down")
    result = s._apply_regime_bias(Signal(signal_type=SignalType.LONG, symbol="ETH"))
    assert result is None


def test_allows_both_in_ranging():
    s = _make_strategy("ranging")
    assert s._apply_regime_bias(Signal(signal_type=SignalType.LONG, symbol="ETH")) is not None
    assert s._apply_regime_bias(Signal(signal_type=SignalType.SHORT, symbol="ETH")) is not None


def test_close_signals_always_pass():
    s = _make_strategy("trending_up")
    close = Signal(signal_type=SignalType.CLOSE_SHORT, symbol="ETH")
    assert s._apply_regime_bias(close) is close
