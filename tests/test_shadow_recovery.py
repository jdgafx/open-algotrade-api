"""U3 — ShadowRecoveryEvaluator: live, no-fill hypothetical PnL for halted strategies."""
import math

import pytest

from src.engine.shadow_recovery import ShadowRecoveryEvaluator
from src.execution.paper_executor import _wilson_interval
from src.strategies.base_strategy import Signal, SignalType


def _sig(stype, size_usd=1000.0, price=None):
    return Signal(signal_type=stype, symbol="BTC", size_usd=size_usd, price=price)


class TestShadowFillAccounting:
    # Round-trip cost on a $1000 notional at default 0.2% = $2.00, charged per close.
    _COST = 1000.0 * 0.002

    def test_long_win_records_positive_pnl(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=110.0), mid_price=110.0)
        n, wr, wr_lo, recent = ev.edge_stats("s")
        assert n == 1
        # +10% of $1000 = +$100 gross, minus $2 round-trip cost = +$98 net.
        assert recent == pytest.approx(100.0 - self._COST)
        assert wr == 1.0

    def test_long_loss_records_negative_pnl(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, price=90.0), mid_price=90.0)
        n, wr, wr_lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent == pytest.approx(-100.0 - self._COST)   # -$100 gross - $2 cost
        assert wr == 0.0

    def test_short_sign_flips(self):
        ev = ShadowRecoveryEvaluator()
        # Short, price falls -> profit. Exact pin guards the sign-flip math.
        ev.observe("s", _sig(SignalType.SHORT, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_SHORT, price=90.0), mid_price=90.0)
        n, _wr, _lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent == pytest.approx(100.0 - self._COST)   # +$98, not +$50k etc.

    def test_close_all_realizes_open_position(self):
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
        ev.observe("s", _sig(SignalType.CLOSE_ALL, price=110.0), mid_price=110.0)
        n, _wr, _lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent == pytest.approx(100.0 - self._COST)

    def test_size_fallback_used_when_signal_size_missing(self):
        """A size-less signal must use the default size, not yield zero PnL (which
        would silently block recovery forever)."""
        ev = ShadowRecoveryEvaluator()
        ev.observe("s", _sig(SignalType.LONG, size_usd=None, price=100.0),
                   mid_price=100.0, default_size_usd=1000.0)
        ev.observe("s", _sig(SignalType.CLOSE_LONG, size_usd=None, price=110.0),
                   mid_price=110.0, default_size_usd=1000.0)
        n, _wr, _lo, recent = ev.edge_stats("s")
        assert n == 1
        assert recent == pytest.approx(100.0 - self._COST)

    def test_sub_cost_wins_become_losses_and_block_recovery(self):
        """The gaming vector: a stream of tiny gross 'wins' below round-trip cost nets
        negative and must NOT certify edge (closes the frictionless-payoff hole)."""
        ev = ShadowRecoveryEvaluator()
        for _ in range(15):
            # +0.1% gross = +$1 on $1000, minus $2 cost = -$1 net -> a loss.
            ev.observe("s", _sig(SignalType.LONG, price=100.0), mid_price=100.0)
            ev.observe("s", _sig(SignalType.CLOSE_LONG, price=100.1), mid_price=100.1)
        n, wr, _lo, recent = ev.edge_stats("s")
        assert n == 15
        assert wr == 0.0           # every sub-cost "win" is a net loss
        assert recent < 0
        assert ev.is_real_edge("s", min_trades=10) is False

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
