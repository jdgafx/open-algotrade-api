"""U6 (R6) — correlation-cluster margin BUDGET + total-exposure cap at entry.

A correlated long book compounds losses, so aggregate committed margin within a
cluster is bounded to a share of equity (NOT a one-position gate — the live winner
book is all-BTC, so several small correlated winners must be able to coexist).
Total committed margin across all positions is bounded against equity (not stale
cash, so the buffer does not loosen as open positions bleed).
"""

from datetime import datetime, timezone

import pytest

from src.execution.paper_executor import (
    PaperPosition, PaperTradingExecutor, _MAX_CLUSTER_EXPOSURE_PCT,
)
from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy


def _executor(balance=50000.0):
    ex = PaperTradingExecutor(initial_balance=balance)
    ex._mid_prices = {"BTC": 50000.0, "ETH": 3000.0, "SOL": 150.0,
                      "LINK": 15.0, "UNI": 8.0}
    ex._last_price_fetch = 9999999999.0
    return ex


async def _open(ex, name, symbol):
    config = StrategyConfig(name=name, symbol=symbol, tier=StrategyTier.A,
                            leverage=3, size_usd=500.0)
    strategy = create_strategy("rsi", config)
    signal = Signal(signal_type=SignalType.LONG, symbol=symbol, size_usd=500.0, reason="t")
    return await ex.execute_signal(signal, strategy)


def _inject(ex, key, symbol, size_usd, leverage=1):
    ex._positions[key] = PaperPosition(
        symbol=symbol, side="long", size=1.0, entry_price=ex._mid_prices[symbol],
        entry_time=datetime.now(timezone.utc), leverage=leverage, size_usd=size_usd,
        strategy_name=key.split(":")[0],
    )


@pytest.mark.asyncio
async def test_small_same_cluster_winners_coexist_under_budget():
    """The #2 fix: several small same-cluster (all-BTC) winners open together —
    a binary one-per-cluster rule would have locked all but one out."""
    ex = _executor()
    assert (await _open(ex, "flip-flop-btc", "BTC")).success is True
    assert (await _open(ex, "vwap-btc", "BTC")).success is True
    assert (await _open(ex, "liqdip-btc", "BTC")).success is True
    assert len(ex._positions) == 3   # all three BTC winners hold positions


@pytest.mark.asyncio
async def test_same_cluster_open_blocked_when_budget_exceeded():
    """A new same-cluster open is refused once the cluster's committed margin is
    already at the budget."""
    ex = _executor()
    # Inject a BTC position at the full majors budget (50% of $50k = $25k margin).
    _inject(ex, "whale:BTC", "BTC", size_usd=25000.0, leverage=1)
    result = await _open(ex, "newcomer-btc", "BTC")
    assert result.success is False
    assert "Cluster cap" in (result.error or "")
    assert "majors" in (result.error or "")


@pytest.mark.asyncio
async def test_different_cluster_open_is_allowed():
    ex = _executor()
    assert (await _open(ex, "a-btc", "BTC")).success is True
    assert (await _open(ex, "b-sol", "SOL")).success is True
    assert len(ex._positions) == 2


@pytest.mark.asyncio
async def test_uncorrelated_symbols_have_no_cluster_limit():
    ex = _executor()
    assert (await _open(ex, "a-link", "LINK")).success is True
    assert (await _open(ex, "b-uni", "UNI")).success is True
    assert len(ex._positions) == 2


@pytest.mark.asyncio
async def test_total_exposure_cap_blocks_new_open():
    """Total committed margin near the wallet cap refuses a new (different-cluster)
    open even though its own cluster is empty."""
    ex = _executor()
    _inject(ex, "whale:ETH", "ETH", size_usd=40000.0, leverage=1)  # 80% of $50k
    result = await _open(ex, "newcomer-sol", "SOL")
    assert result.success is False
    assert "exposure" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_exposure_cap_uses_equity_not_stale_cash():
    """The #3 fix: an open unrealized LOSS lowers equity, so the cap tightens (does
    not loosen) during a drawdown. With a big underwater majors position, a new
    different-cluster open is refused against equity though stale cash would allow it."""
    ex = _executor(balance=50000.0)
    # $30k margin position carrying a $20k unrealized loss -> equity = $30k.
    _inject(ex, "whale:ETH", "ETH", size_usd=30000.0, leverage=1)
    ex._positions["whale:ETH"].unrealized_pnl = -20000.0
    # Open margin $30k already exceeds 80% of $30k equity ($24k) -> any new open refused,
    # even though 80% of stale $50k cash ($40k) would still have room.
    result = await _open(ex, "newcomer-sol", "SOL")
    assert result.success is False
    assert "exposure" in (result.error or "").lower()
