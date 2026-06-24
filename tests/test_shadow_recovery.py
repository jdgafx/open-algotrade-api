"""U3 — ShadowRecoveryEvaluator: live, no-fill hypothetical PnL for halted strategies."""
import math

import pytest

from src.engine.shadow_recovery import ShadowRecoveryEvaluator
from src.execution.paper_executor import _wilson_interval
from src.strategies.base_strategy import Signal, SignalType


def _sig(stype, size_usd=1000.0, price=None):
    return Signal(signal_type=stype, symbol="BTC", size_usd=size_usd, price=price)


class TestShadowFillAccounting:
    def test_long_win_records_positive_pnl(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        n, wr, wr_lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent > 0          # +10% of $1000 = +$100
        assert wr == 1.0

    def test_long_loss_records_negative_pnl(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=90.0), mid_price=90.0)
        n, wr, wr_lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent < 0
        assert wr == 0.0

    def test_short_sign_flips(self):
        ev = ShadowRecoveryEvaluator()
        # Short, price falls -> profit
        ev.observe("s", _sig(SignalType.SHORT, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_SHORT, price=90.0), mid_price=90.0)
        n, _wr, _lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent > 0          # short into a -10% move is a win

    def test_close_without_open_is_noop(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=90.0), mid_price=90.0)
        n, *_ = ev.edge_stats("s")
        assert n == 0              # nothing to close, no crash, no phantom trade

    def test_double_open_keeps_single_position(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.LONG, price=105.0), mid_price=105.0)  # ignored
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        n, *_ = ev.edge_stats("s")
        assert n == 1              # only the first entry counted

    def test_none_signal_is_noop(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.NONE), mid_price=100.0)
        ev.observe("s", None, mid_price=100.0)
        assert ev.window_count("s") == 0

    def test_window_is_bounded(self):
        ev = ShadowRecoveryEvaluator(window=3)
        for i in range(6):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=101.0), mid_price=101.0)
        assert ev.window_count("s") == 3   # oldest evicted


class TestEdgeStats:
    def test_empty_buffer_no_divide_by_zero(self):
        ev = ShadowRecoveryEvaluator()
        assert ev.edge_stats("nobody") == (0, 0.0, 0.0, 0.0)
        assert ev.is_real_edge("nobody", min_trades=10) is False

    def test_wilson_lower_bound_matches_helper(self):
        """Lock the math: shadow wr_lo equals the canonical _wilson_interval."""
        ev = ShadowRecoveryEvaluator()
        # 7 wins, 3 losses
        for _ in range(7):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        for _ in range(3):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=90.0), mid_price=90.0)
        n, wr, wr_lo, _recent = ev.edge_stats("s")
        assert n == 10
        assert wr == pytest.approx(0.7)
        expected_lo, _ = _wilson_interval(7, 10)
        assert wr_lo == pytest.approx(expected_lo)

    def test_below_min_trades_is_not_real_edge(self):
        ev = ShadowRecoveryEvaluator()
        for _ in range(3):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=120.0), mid_price=120.0)
        # All wins, but only 3 trades < min_trades=10.
        assert ev.is_real_edge("s", min_trades=10) is False

    def test_strong_winrate_clears_edge_bar(self):
        ev = ShadowRecoveryEvaluator()
        for _ in range(15):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        assert ev.is_real_edge("s", min_trades=10) is True

    def test_losing_stream_never_real_edge(self):
        ev = ShadowRecoveryEvaluator()
        for _ in range(15):
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=95.0), mid_price=95.0)
        assert ev.is_real_edge("s", min_trades=10) is False


class TestIsolation:
    def test_clear_drops_state(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        assert ev.window_count("s") == 1
        ev.clear("s")
        assert ev.window_count("s") == 0
        assert ev.has_open_position("s") is False
