import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.turtle_strategy import (
    TurtleStrategy,
    TurtleConfig,
    VaultExecutor,
    Position,
)


class MockVaultExecutor(VaultExecutor):
    def __init__(self):
        self.mock_execute = AsyncMock(return_value=True)
        self.mock_position = AsyncMock(return_value=None)
        self.mock_balance = AsyncMock(return_value=Decimal("10000"))

    async def execute_order(
        self, side, size, price=None, reduce_only=False, order_type="limit"
    ):
        return await self.mock_execute(
            side=side,
            size=size,
            price=price,
            reduce_only=reduce_only,
            order_type=order_type,
        )

    async def get_position(self, symbol):
        return await self.mock_position(symbol)

    async def get_balance(self):
        return await self.mock_balance()


@pytest.fixture
def config():
    return TurtleConfig(
        symbol="BTC-USD",
        timeframe="1h",
        size=Decimal("0.1"),
        lookback_period=55,
        atr_period=20,
        trading_hours_only=False,  # Disable time check for testing
        exit_friday=False,
    )


@pytest.fixture
def executor():
    return MockVaultExecutor()


@pytest.fixture
def strategy(executor, config):
    return TurtleStrategy(executor, config)


@pytest.fixture
def ohlcv_data():
    # Create 100 candles
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0] * 100,
            "high": [105.0] * 100,
            "low": [95.0] * 100,
            "close": [100.0] * 100,
            "volume": [1000.0] * 100,
        }
    )
    return df


@pytest.mark.asyncio
async def test_long_entry_signal(strategy, executor, ohlcv_data):
    # Setup: Price breaks above 55-bar high (105.0)
    # Last completed candle (index -2) closed at 100
    # Current candle (index -1) opens at 100, high 110, close 108

    df = ohlcv_data.copy()
    # Set previous highs to 105
    df.loc[:, "high"] = 105.0

    # Current candle (index -1)
    df.iloc[-1, df.columns.get_loc("open")] = 100.0
    df.iloc[-1, df.columns.get_loc("high")] = 110.0
    df.iloc[-1, df.columns.get_loc("close")] = 108.0

    current_price = Decimal("108.0")
    bid = Decimal("107.9")
    ask = Decimal("108.1")

    await strategy.on_candle(df, current_price, bid, ask)

    # Verify execute_order called for LONG
    executor.mock_execute.assert_called_once()
    call_args = executor.mock_execute.call_args
    assert call_args.kwargs["side"] == "buy"
    assert call_args.kwargs["price"] == bid


@pytest.mark.asyncio
async def test_short_entry_signal(strategy, executor, ohlcv_data):
    # Setup: Price breaks below 55-bar low (95.0)

    df = ohlcv_data.copy()
    # Set previous lows to 95
    df.loc[:, "low"] = 95.0

    # Current candle
    df.iloc[-1, df.columns.get_loc("open")] = 100.0
    df.iloc[-1, df.columns.get_loc("low")] = 90.0
    df.iloc[-1, df.columns.get_loc("close")] = 92.0

    current_price = Decimal("92.0")
    bid = Decimal("91.9")
    ask = Decimal("92.1")

    await strategy.on_candle(df, current_price, bid, ask)

    # Verify execute_order called for SHORT
    executor.mock_execute.assert_called_once()
    call_args = executor.mock_execute.call_args
    assert call_args.kwargs["side"] == "sell"
    assert call_args.kwargs["price"] == ask


@pytest.mark.asyncio
async def test_no_signal(strategy, executor, ohlcv_data):
    # Setup: Price inside range [95, 105]

    df = ohlcv_data.copy()
    df.loc[:, "high"] = 105.0
    df.loc[:, "low"] = 95.0

    # Current candle
    df.iloc[-1, df.columns.get_loc("open")] = 100.0
    df.iloc[-1, df.columns.get_loc("high")] = 102.0
    df.iloc[-1, df.columns.get_loc("low")] = 98.0
    df.iloc[-1, df.columns.get_loc("close")] = 100.0

    current_price = Decimal("100.0")
    bid = Decimal("99.9")
    ask = Decimal("100.1")

    await strategy.on_candle(df, current_price, bid, ask)

    # Verify NO order executed
    executor.mock_execute.assert_not_called()


@pytest.mark.asyncio
async def test_stop_loss_exit(strategy, executor, ohlcv_data):
    # Setup: In LONG position, price hits SL

    # Mock existing position
    position = Position(
        symbol="BTC-USD", size=Decimal("0.1"), side="long", entry_price=Decimal("100.0")
    )
    executor.mock_position.return_value = position

    # Set strategy state manually to simulate entry
    strategy.current_position = position
    strategy.stop_loss_price = Decimal("98.0")
    strategy.take_profit_price = Decimal("110.0")

    current_price = Decimal("97.0")  # Below SL
    bid = Decimal("97.0")
    ask = Decimal("97.0")

    await strategy.on_candle(ohlcv_data, current_price, bid, ask)

    # Verify execute_order called for EXIT (sell)
    executor.mock_execute.assert_called_once()
    call_args = executor.mock_execute.call_args
    assert call_args.kwargs["side"] == "sell"
    assert call_args.kwargs["reduce_only"] is True
