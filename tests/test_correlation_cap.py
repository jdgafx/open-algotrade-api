"""U6 (R6) — correlation-cluster + total-exposure caps at entry.

A correlated long book compounds losses, so at most one open position per
cluster, and total committed margin is bounded to a share of the wallet.
"""

from datetime import datetime, timezone

import pytest

from src.execution.paper_executor import PaperPosition, PaperTradingExecutor
from src.strategies.base_strategy import Signal, SignalType, StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy


def _executor():
    ex = PaperTradingExecutor(initial_balance=50000.0)
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


@pytest.mark.asyncio
async def test_same_cluster_second_open_is_blocked():
    """BTC and ETH share the 'majors' cluster — the second open is refused."""
    ex = _executor()
    assert (await _open(ex, "a-btc", "BTC")).success is True
    result = await _open(ex, "b-eth", "ETH")
    assert result.success is False
    assert "Cluster cap" in (result.error or "")
    assert len(ex._positions) == 1


@pytest.mark.asyncio
async def test_different_cluster_open_is_allowed():
    """BTC (majors) and SOL (l1alts) are different clusters — both open."""
    ex = _executor()
    assert (await _open(ex, "a-btc", "BTC")).success is True
    assert (await _open(ex, "b-sol", "SOL")).success is True
    assert len(ex._positions) == 2


@pytest.mark.asyncio
async def test_uncorrelated_symbols_have_no_cluster_limit():
    """Symbols absent from the cluster map are treated as uncorrelated — both open."""
    ex = _executor()
    assert (await _open(ex, "a-link", "LINK")).success is True
    assert (await _open(ex, "b-uni", "UNI")).success is True
    assert len(ex._positions) == 2


@pytest.mark.asyncio
async def test_total_exposure_cap_blocks_new_open():
    """When committed margin already near the wallet cap, a new (different-cluster)
    open is refused even though its own cluster is empty."""
    ex = _executor()
    # Inject a large existing position: $40k notional at 1x = $40k margin, which is
    # 80% of the $50k wallet — at the cap.
    ex._positions["whale:ETH"] = PaperPosition(
        symbol="ETH", side="long", size=1.0, entry_price=3000.0,
        entry_time=datetime.now(timezone.utc), leverage=1, size_usd=40000.0,
        strategy_name="whale",
    )
    result = await _open(ex, "newcomer-sol", "SOL")   # different cluster, but no room
    assert result.success is False
    assert "exposure" in (result.error or "").lower()
