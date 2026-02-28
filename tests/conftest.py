"""
Shared test fixtures for the Open Algotrade test suite.

Provides:
- In-memory SQLite database and test client
- OHLCV data generators for strategy testing
- Mock executors and orchestrators
- Strategy config factory
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure the backend src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.database import Base
from src.api.models import StrategyInstance, Trade, User, VaultState
from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyConfig,
    StrategyState,
    StrategyTier,
)


# ──────────────────────────────────────────────
# Database fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestSession()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_engine():
    """Create an in-memory SQLite engine for the test client.

    Uses StaticPool so every connection shares the same in-memory database.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


# ──────────────────────────────────────────────
# FastAPI test client
# ──────────────────────────────────────────────

@pytest.fixture
def client(test_engine):
    """Create a FastAPI test client with an in-memory DB and mocked lifespan.

    We replace the app's lifespan with a no-op so the real startup (which tries
    to connect to Hyperliquid) is skipped.  The test engine provides the DB.
    """
    from contextlib import asynccontextmanager

    from src.api.database import get_db
    from src.api.main import app

    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    # Replace the real lifespan so we don't try to connect to exchanges
    original_router_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _test_lifespan(a):
        # Set minimal app state
        a.state.orchestrator = None
        a.state.risk_controller = None
        a.state.liquidation_tracker = None
        a.state.whale_tracker = None
        a.state.rbi_agent = None
        a.state.regime_detector = None
        a.state.trading_mode = "paper"
        a.state.paper_mode = True
        a.state.executor = None
        a.state.solana_scanner = None
        yield

    app.router.lifespan_context = _test_lifespan

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.router.lifespan_context = original_router_lifespan
    app.dependency_overrides.clear()


# ──────────────────────────────────────────────
# OHLCV data generators
# ──────────────────────────────────────────────

@pytest.fixture
def make_ohlcv():
    """Factory fixture to generate OHLCV DataFrames with configurable patterns."""

    def _make(
        n_bars: int = 200,
        base_price: float = 100.0,
        trend: float = 0.0,
        volatility: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        dates = pd.date_range(
            start="2024-01-01", periods=n_bars, freq="1h", tz="UTC"
        )
        prices = [base_price]
        for _ in range(1, n_bars):
            ret = trend + rng.normal(0, volatility)
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices)
        highs = prices * (1 + rng.uniform(0.001, 0.01, n_bars))
        lows = prices * (1 - rng.uniform(0.001, 0.01, n_bars))
        opens = prices * (1 + rng.normal(0, 0.003, n_bars))
        volumes = rng.uniform(100, 10000, n_bars)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(highs, np.maximum(opens, prices)),
                "low": np.minimum(lows, np.minimum(opens, prices)),
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )
        return df

    return _make


@pytest.fixture
def trending_up_data(make_ohlcv):
    """OHLCV data with a strong uptrend."""
    return make_ohlcv(n_bars=200, base_price=100.0, trend=0.002, volatility=0.01)


@pytest.fixture
def trending_down_data(make_ohlcv):
    """OHLCV data with a strong downtrend."""
    return make_ohlcv(n_bars=200, base_price=100.0, trend=-0.002, volatility=0.01)


@pytest.fixture
def sideways_data(make_ohlcv):
    """OHLCV data with no trend (mean-reverting)."""
    return make_ohlcv(n_bars=200, base_price=100.0, trend=0.0, volatility=0.005)


@pytest.fixture
def volatile_data(make_ohlcv):
    """OHLCV data with high volatility."""
    return make_ohlcv(n_bars=200, base_price=100.0, trend=0.0, volatility=0.05)


@pytest.fixture
def breakout_data():
    """OHLCV data specifically designed for breakout detection.

    - First 100 bars: consolidation around 100
    - Bars 100-120: breakout above (price jumps to 120)
    - Bars 120-150: consolidation around 120
    - Bars 150-170: breakdown below (price drops to 80)
    - Bars 170-200: consolidation around 80
    """
    n = 200
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)

    prices = np.zeros(n)
    # Consolidation phase
    prices[:100] = 100 + rng.normal(0, 0.5, 100)
    # Breakout phase
    for i in range(100, 120):
        prices[i] = prices[i - 1] + rng.uniform(0.5, 1.5)
    # Second consolidation
    for i in range(120, 150):
        prices[i] = prices[119] + rng.normal(0, 0.5)
    # Breakdown phase
    for i in range(150, 170):
        prices[i] = prices[i - 1] - rng.uniform(0.5, 1.5)
    # Third consolidation
    for i in range(170, 200):
        prices[i] = prices[169] + rng.normal(0, 0.5)

    highs = prices * (1 + rng.uniform(0.001, 0.005, n))
    lows = prices * (1 - rng.uniform(0.001, 0.005, n))

    df = pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.001, n)),
            "high": np.maximum(highs, prices),
            "low": np.minimum(lows, prices),
            "close": prices,
            "volume": rng.uniform(500, 5000, n),
        },
        index=dates,
    )
    return df


# ──────────────────────────────────────────────
# Strategy config factory
# ──────────────────────────────────────────────

@pytest.fixture
def make_strategy_config():
    """Factory for creating StrategyConfig instances."""

    def _make(
        name: str = "test-strategy",
        symbol: str = "BTC",
        strategy_type: str = "rsi",
        tier: StrategyTier = StrategyTier.C,
        timeframe: str = "1h",
        leverage: int = 1,
        size_usd: float = 100.0,
        target_pct: float = 5.0,
        max_loss_pct: float = -10.0,
        lookback_days: int = 7,
        interval_seconds: int = 30,
        params: Optional[Dict[str, Any]] = None,
    ) -> StrategyConfig:
        return StrategyConfig(
            name=name,
            symbol=symbol,
            tier=tier,
            timeframe=timeframe,
            leverage=leverage,
            size_usd=size_usd,
            target_pct=target_pct,
            max_loss_pct=max_loss_pct,
            lookback_days=lookback_days,
            interval_seconds=interval_seconds,
            params=params or {},
        )

    return _make


# ──────────────────────────────────────────────
# Position fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def long_position():
    """A mock long position dict."""
    return {
        "size": 0.1,
        "is_long": True,
        "entry_px": 100.0,
        "entry_price": 100.0,
        "pnl_perc": 2.0,
        "unrealized_pnl": 0.2,
        "side": "long",
        "symbol": "BTC",
    }


@pytest.fixture
def short_position():
    """A mock short position dict."""
    return {
        "size": -0.1,
        "is_long": False,
        "entry_px": 100.0,
        "entry_price": 100.0,
        "pnl_perc": 2.0,
        "unrealized_pnl": 0.2,
        "side": "short",
        "symbol": "BTC",
    }


@pytest.fixture
def losing_long_position():
    """A long position at max loss threshold."""
    return {
        "size": 0.1,
        "is_long": True,
        "entry_px": 100.0,
        "entry_price": 100.0,
        "pnl_perc": -11.0,
        "unrealized_pnl": -1.1,
        "side": "long",
        "symbol": "BTC",
    }


@pytest.fixture
def winning_long_position():
    """A long position at take profit threshold."""
    return {
        "size": 0.1,
        "is_long": True,
        "entry_px": 100.0,
        "entry_price": 100.0,
        "pnl_perc": 6.0,
        "unrealized_pnl": 0.6,
        "side": "long",
        "symbol": "BTC",
    }
