"""Condition-aware circuit-breaker recovery — shadow evaluation (U3/PART B).

A circuit-breaker-halted strategy stops trading, so it produces no new realized
trades and can never demonstrate that its edge returned. Its *pre-halt* trades are
exactly the losses that tripped the breaker, so any recovery gate keyed on real
realized PnL would keep it halted forever.

The only way to gather forward, *live* evidence of renewed edge without un-halting
(which would defeat the breaker) and without a backtest (KTD-C: no backtest->live
bridge) is **shadow evaluation**: while a strategy is halted, keep computing its real
signals against live market data and simulate fills into an isolated per-strategy
ledger. Real market data, simulated fills, zero impact on the real paper balance or
positions. When the shadow track record clears the same statistical edge bar the
system already trusts for a "real edge" (see /paper/edge), the strategy is
auto-re-enabled. Below the bar => stays halted.

This module owns ONLY the shadow bookkeeping + the edge stats. The orchestrator
(U4) owns the loop integration, the recovery decision, and the re-enable.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from src.execution.paper_executor import RECENT_PNL_WINDOW, _wilson_interval
from src.strategies.base_strategy import Signal, SignalType

logger = logging.getLogger(__name__)

# Number of most-recent shadow trades that define the recovery window. Aligned with
# the real RECENT_PNL_WINDOW so recovery judges the same horizon live promotion does.
SHADOW_WINDOW = int(os.getenv("SHADOW_RECOVERY_WINDOW", str(RECENT_PNL_WINDOW)))

# No-loss payoff cap, mirrors paper_executor._live_edge_stats so a clean win streak
# yields a high (not infinite) payoff ratio rather than a raw dollar amount.
_NO_LOSS_PAYOFF = 10.0


@dataclass
class _ShadowPosition:
    """An open hypothetical position in the shadow ledger. No balance, no leverage
    accounting beyond directional PnL — this never touches the real executor."""
    is_long: bool
    entry_price: float
    size_usd: float


class ShadowRecoveryEvaluator:
    """Per-strategy shadow ledger + edge statistics for halted strategies.

    Pure bookkeeping: ``observe`` feeds it live signals + the live mid price; it
    opens/closes a single hypothetical position per strategy and records realized
    shadow PnL into a bounded buffer. ``edge_stats`` / ``is_real_edge`` expose the
    same Wilson-lower-bound edge test the live system uses. Never calls into the
    real executor or mutates any real balance.
    """

    def __init__(self, window: int = SHADOW_WINDOW):
        self._window = max(1, window)
        self._positions: Dict[str, _ShadowPosition] = {}
        self._pnls: Dict[str, Deque[float]] = {}

    # ── feed ────────────────────────────────────────────────────────────────
    def observe(self, name: str, signal: Optional[Signal], mid_price: float) -> None:
        """Advance the shadow ledger for ``name`` by one live signal.

        Entry signals open a hypothetical position (ignored if one is already open).
        Close signals realize PnL against the live ``mid_price`` and append it to the
        recovery window. NONE / None signals and unmatched closes are no-ops.
        """
        if signal is None or mid_price is None or mid_price <= 0:
            return
        st = signal.signal_type
        if st in (SignalType.LONG, SignalType.SHORT):
            self._open(name, is_long=(st == SignalType.LONG),
                       price=signal.price or mid_price, size_usd=signal.size_usd)
        elif st in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT, SignalType.CLOSE_ALL):
            self._close(name, exit_price=signal.price or mid_price)
        # SignalType.NONE -> no-op

    def _open(self, name: str, is_long: bool, price: float, size_usd: Optional[float]) -> None:
        if name in self._positions or price <= 0:
            return  # already in a shadow position; one at a time
        self._positions[name] = _ShadowPosition(
            is_long=is_long, entry_price=price, size_usd=size_usd or 0.0,
        )

    def _close(self, name: str, exit_price: float) -> None:
        pos = self._positions.pop(name, None)
        if pos is None or pos.entry_price <= 0 or exit_price <= 0:
            return  # no open shadow position to close
        move_pct = (exit_price - pos.entry_price) / pos.entry_price
        if not pos.is_long:
            move_pct = -move_pct
        pnl = pos.size_usd * move_pct
        self._pnls.setdefault(name, deque(maxlen=self._window)).append(pnl)

    # ── stats ─────────────────────────────────────────────────────────────--
    def edge_stats(self, name: str) -> Tuple[int, float, float, float]:
        """Return (n, win_rate, wr_lower_90, recent_pnl) over the shadow window.

        n == 0 (and zeroed stats) when no shadow trades have closed yet.
        """
        pnls = list(self._pnls.get(name, ()))
        n = len(pnls)
        if n == 0:
            return 0, 0.0, 0.0, 0.0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n
        wr_lo, _ = _wilson_interval(wins, n)
        return n, wr, wr_lo, round(sum(pnls), 2)

    def _payoff(self, name: str) -> float:
        pnls = list(self._pnls.get(name, ()))
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss > 0:
            return avg_win / avg_loss
        return _NO_LOSS_PAYOFF if avg_win > 0 else 0.0

    def is_real_edge(self, name: str, min_trades: int) -> bool:
        """Renewed positive edge under the SAME bar /paper/edge uses for "real edge":
        enough sample AND Wilson lower-bound win-rate above breakeven for the payoff.
        """
        n, _wr, wr_lo, _recent = self.edge_stats(name)
        if n < min_trades:
            return False
        payoff = self._payoff(name)
        breakeven_wr = (1.0 / (1.0 + payoff)) if payoff > 0 else 1.0
        return wr_lo > breakeven_wr

    # ── lifecycle ───────────────────────────────────────────────────────────
    def clear(self, name: str) -> None:
        """Drop all shadow state for a strategy — on re-enable or manual reset."""
        self._positions.pop(name, None)
        self._pnls.pop(name, None)

    def has_open_position(self, name: str) -> bool:
        return name in self._positions

    def window_count(self, name: str) -> int:
        return len(self._pnls.get(name, ()))
