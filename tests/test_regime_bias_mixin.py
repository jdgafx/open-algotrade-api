from unittest.mock import MagicMock
import pytest
from src.strategies.base_strategy import Signal, SignalType
from src.strategies.regime_bias_mixin import RegimeBiasMixin


class _FakeStrategy(RegimeBiasMixin):
    def __init__(self, regime_hint: str):
        self.config = MagicMock()
        self.config.params = {"_regime_hint": regime_hint}
        self.config.symbol = "BTC"


def _signal(signal_type: SignalType) -> Signal:
    return Signal(signal_type=signal_type, symbol="BTC")


def test_returns_none_unchanged():
    s = _FakeStrategy("trending_up")
    assert s._apply_regime_bias(None) is None


def test_allows_long_in_trending_up():
    s = _FakeStrategy("trending_up")
    sig = _signal(SignalType.LONG)
    assert s._apply_regime_bias(sig) is sig


def test_suppresses_short_in_trending_up():
    s = _FakeStrategy("trending_up")
    sig = _signal(SignalType.SHORT)
    assert s._apply_regime_bias(sig) is None


def test_allows_short_in_trending_down():
    s = _FakeStrategy("trending_down")
    sig = _signal(SignalType.SHORT)
    assert s._apply_regime_bias(sig) is sig


def test_suppresses_long_in_trending_down():
    s = _FakeStrategy("trending_down")
    sig = _signal(SignalType.LONG)
    assert s._apply_regime_bias(sig) is None


def test_allows_both_in_ranging():
    s = _FakeStrategy("ranging")
    assert s._apply_regime_bias(_signal(SignalType.LONG)) is not None
    assert s._apply_regime_bias(_signal(SignalType.SHORT)) is not None


def test_allows_both_when_hint_unknown():
    s = _FakeStrategy("unknown")
    assert s._apply_regime_bias(_signal(SignalType.LONG)) is not None
    assert s._apply_regime_bias(_signal(SignalType.SHORT)) is not None


def test_allows_close_signals_regardless_of_regime():
    s = _FakeStrategy("trending_up")
    close_long = _signal(SignalType.CLOSE_LONG)
    close_short = _signal(SignalType.CLOSE_SHORT)
    assert s._apply_regime_bias(close_long) is close_long
    assert s._apply_regime_bias(close_short) is close_short
