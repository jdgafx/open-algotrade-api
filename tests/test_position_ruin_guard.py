import pytest
from datetime import datetime, timezone
from src.execution.paper_executor import PaperTradingExecutor, PaperPosition
from src.services.liquidation_guard import LiquidationGuard


def _mk_pos(leverage, size_usd=100.0, entry=100.0, side="long"):
    asset = size_usd / entry
    return PaperPosition(symbol="BTC", side=side, size=(asset if side == "long" else -asset),
                         entry_price=entry, entry_time=datetime.now(timezone.utc),
                         leverage=leverage, size_usd=size_usd, strategy_name="x-btc")


@pytest.mark.asyncio
async def test_position_near_liquidation_flagged():
    ex = PaperTradingExecutor(initial_balance=200.0)
    pos = _mk_pos(leverage=40)
    ex._positions["x-btc:BTC"] = pos
    margin = pos.size_usd / pos.leverage
    liq = LiquidationGuard.calculate_liquidation_price(
        entry_price=pos.entry_price, position_size=abs(pos.size), margin=margin,
        leverage=pos.leverage, is_long=True, symbol="BTC")
    ex._fetch_mid_price = lambda s: liq * 1.005  # 0.5% above liq -> inside the 1% buffer
    should_close, dist = await ex.check_position_ruin("BTC", "x-btc", safety_buffer_pct=1.0)
    assert should_close is True
    assert dist <= 1.0


@pytest.mark.asyncio
async def test_safe_position_not_flagged():
    ex = PaperTradingExecutor(initial_balance=200.0)
    pos = _mk_pos(leverage=3)
    ex._positions["x-btc:BTC"] = pos
    ex._fetch_mid_price = lambda s: pos.entry_price  # at entry -> far from liq at 3x
    should_close, dist = await ex.check_position_ruin("BTC", "x-btc", safety_buffer_pct=1.0)
    assert should_close is False


@pytest.mark.asyncio
async def test_no_position_returns_safe():
    ex = PaperTradingExecutor(initial_balance=200.0)
    ex._fetch_mid_price = lambda s: 100.0
    should_close, dist = await ex.check_position_ruin("BTC", "nope")
    assert should_close is False


@pytest.mark.asyncio
async def test_short_position_near_liquidation_flagged():
    # Short liquidates when price rises ABOVE the liq price; exercise the
    # (liq_price - mid)/mid short branch with mid just below the liq price.
    ex = PaperTradingExecutor(initial_balance=200.0)
    pos = _mk_pos(leverage=40, side="short")
    ex._positions["x-btc:BTC"] = pos
    margin = pos.size_usd / pos.leverage
    liq = LiquidationGuard.calculate_liquidation_price(
        entry_price=pos.entry_price, position_size=abs(pos.size), margin=margin,
        leverage=pos.leverage, is_long=False, symbol="BTC")
    ex._fetch_mid_price = lambda s: liq * 0.995  # 0.5% below liq -> inside the 1% buffer
    should_close, dist = await ex.check_position_ruin("BTC", "x-btc", safety_buffer_pct=1.0)
    assert should_close is True
    assert dist <= 1.0


@pytest.mark.asyncio
async def test_mid_param_skips_fetch():
    # When an explicit mid is passed, _fetch_mid_price must NOT be called.
    ex = PaperTradingExecutor(initial_balance=200.0)
    pos = _mk_pos(leverage=3)
    ex._positions["x-btc:BTC"] = pos

    def _boom(s):
        raise AssertionError("_fetch_mid_price should not be called when mid is provided")

    ex._fetch_mid_price = _boom
    should_close, dist = await ex.check_position_ruin(
        "BTC", "x-btc", safety_buffer_pct=1.0, mid=pos.entry_price)
    assert should_close is False
