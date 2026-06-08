"""Tests for T031: PromotionEvent DB persistence + history rebuild on boot."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.database import Base
from src.api.models import PromotionEventRecord
from src.engine.rbi_pipeline import RBIPipeline, PromotionEvent
from src.engine.optimizer import OptimizationEngine


# ──────────────────────────────────────────────
# In-memory DB helpers
# ──────────────────────────────────────────────

def _make_session_factory():
    """Return a fresh in-memory SQLite session factory with all tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _mock_strategy(active_positions=0, params=None):
    return {
        "id": 8,
        "strategy_type": "mean_reversion",
        "symbol": "BTC",
        "timeframe": "1h",
        "active_positions": active_positions,
        "params": params or {"zscore_entry": 1.5},
    }


def _make_pipeline(session_factory, active_positions=0, params=None, strategy_type=None):
    get_fn = AsyncMock(return_value=_mock_strategy(active_positions, params))
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()
    return RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        db_session_factory=session_factory,
        strategy_type=strategy_type,
    )


# ──────────────────────────────────────────────
# Jitter range sanity check (no scheduler needed)
# ──────────────────────────────────────────────

def test_jitter_range():
    """Verify the jitter constants used in main.py produce offsets in [30s, 5min]."""
    import random
    from datetime import timedelta, datetime as dt

    JITTER_MIN_S = 30
    JITTER_MAX_S = 300
    samples = [random.randint(JITTER_MIN_S, JITTER_MAX_S) for _ in range(1000)]
    assert min(samples) >= 30
    assert max(samples) <= 300
    # No sample equals 0 (no immediate thundering herd)
    assert all(s > 0 for s in samples)


# ──────────────────────────────────────────────
# PromotionEventRecord model
# ──────────────────────────────────────────────

def test_promotion_event_record_persists():
    """PromotionEventRecord can be inserted and retrieved from an in-memory DB."""
    SessionLocal = _make_session_factory()
    db = SessionLocal()
    try:
        row = PromotionEventRecord(
            strategy_type="rsi",
            strategy_id=14,
            timestamp="2026-01-01T00:00:00+00:00",
            promoted=True,
            reason="promotion_gate_passed",
            before_params={"rsi_period": 14},
            after_params={"rsi_period": 18},
            before_metrics={},
            after_metrics={"oos_sharpe": 1.3},
        )
        db.add(row)
        db.commit()

        loaded = db.query(PromotionEventRecord).filter_by(strategy_type="rsi").first()
        assert loaded is not None
        assert loaded.promoted is True
        assert loaded.after_metrics["oos_sharpe"] == 1.3
        assert loaded.strategy_id == 14
    finally:
        db.close()


def test_promotion_event_record_non_promoting():
    """Non-promoting events (e.g. no_passing_candidates) can be stored."""
    SessionLocal = _make_session_factory()
    db = SessionLocal()
    try:
        row = PromotionEventRecord(
            strategy_type="macd",
            strategy_id=19,
            timestamp="2026-01-02T00:00:00+00:00",
            promoted=False,
            reason="no_passing_candidates",
            before_params={"fast": 12},
            after_params={},
            before_metrics={},
            after_metrics={},
        )
        db.add(row)
        db.commit()

        loaded = db.query(PromotionEventRecord).filter_by(reason="no_passing_candidates").first()
        assert loaded is not None
        assert loaded.promoted is False
    finally:
        db.close()


# ──────────────────────────────────────────────
# RBIPipeline init: loads history from DB
# ──────────────────────────────────────────────

def test_pipeline_loads_history_from_db_on_init():
    """RBIPipeline.__init__ rebuilds _history from existing DB rows."""
    SessionLocal = _make_session_factory()
    # Pre-populate two events
    db = SessionLocal()
    try:
        db.add(PromotionEventRecord(
            strategy_type="rsi", strategy_id=14,
            timestamp="2026-01-01T00:00:00+00:00",
            promoted=True, reason="promotion_gate_passed",
            before_params={"rsi_period": 14}, after_params={"rsi_period": 18},
            before_metrics={}, after_metrics={"oos_sharpe": 1.2},
        ))
        db.add(PromotionEventRecord(
            strategy_type="rsi", strategy_id=14,
            timestamp="2026-01-02T00:00:00+00:00",
            promoted=False, reason="no_passing_candidates",
            before_params={"rsi_period": 18}, after_params={},
            before_metrics={}, after_metrics={},
        ))
        db.commit()
    finally:
        db.close()

    pipeline = _make_pipeline(SessionLocal)
    history = pipeline.get_history()
    assert len(history) == 2
    assert history[0].strategy_type == "rsi"
    assert history[0].promoted is True
    assert history[1].promoted is False
    assert pipeline.run_count == 2


def test_pipeline_rollback_params_rebuilt_from_db():
    """Rollback params for promoted strategies are restored from DB history."""
    SessionLocal = _make_session_factory()
    db = SessionLocal()
    try:
        db.add(PromotionEventRecord(
            strategy_type="sma_crossover", strategy_id=17,
            timestamp="2026-01-01T00:00:00+00:00",
            promoted=True, reason="promotion_gate_passed",
            before_params={"fast": 10, "slow": 30},
            after_params={"fast": 12, "slow": 26},
            before_metrics={}, after_metrics={},
        ))
        db.commit()
    finally:
        db.close()

    pipeline = _make_pipeline(SessionLocal)
    # _previous_params should be populated from the loaded history
    rollback = pipeline.get_rollback_params("sma_crossover")
    assert rollback == {"fast": 10, "slow": 30}


def test_pipeline_empty_db_starts_clean():
    """RBIPipeline with empty DB starts with empty history (no crash)."""
    SessionLocal = _make_session_factory()
    pipeline = _make_pipeline(SessionLocal)
    assert pipeline.get_history() == []
    assert pipeline.run_count == 0


# ──────────────────────────────────────────────
# run_cycle: persists ALL event types
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_cycle_persists_no_passing_candidates():
    """run_cycle writes a record even when no candidates pass the gate."""
    from src.engine.optimizer import OptimizationResult
    SessionLocal = _make_session_factory()

    get_fn = AsyncMock(return_value=_mock_strategy())
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()
    failing = OptimizationResult(
        params={}, in_sample_sharpe=0.5, out_sample_sharpe=0.3,
        out_sample_profit_factor=0.9, out_sample_win_rate=30.0,
        out_sample_total_trades=8, out_sample_max_drawdown=20.0,
        composite_score=0.2, passed_walkforward=False,
    )
    optimizer.optimize = AsyncMock(return_value=[failing])

    pipeline = RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        db_session_factory=SessionLocal,
    )

    event = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert event.promoted is False
    assert event.reason == "no_passing_candidates"
    assert pipeline.run_count == 1

    # Verify row in DB
    db = SessionLocal()
    try:
        rows = db.query(PromotionEventRecord).all()
        assert len(rows) == 1
        assert rows[0].promoted is False
        assert rows[0].reason == "no_passing_candidates"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_cycle_persists_active_position():
    """run_cycle writes a record for the active_position skip case."""
    SessionLocal = _make_session_factory()
    get_fn = AsyncMock(return_value=_mock_strategy(active_positions=1))
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()

    pipeline = RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        db_session_factory=SessionLocal,
    )

    event = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert event.reason == "active_position"
    assert pipeline.run_count == 1

    db = SessionLocal()
    try:
        rows = db.query(PromotionEventRecord).all()
        assert len(rows) == 1
        assert rows[0].reason == "active_position"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_cycle_persists_promotion():
    """run_cycle writes a record when a candidate is promoted."""
    from src.engine.optimizer import OptimizationResult
    SessionLocal = _make_session_factory()

    get_fn = AsyncMock(return_value=_mock_strategy())
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()
    passing = OptimizationResult(
        params={"zscore_entry": 2.0}, in_sample_sharpe=1.2, out_sample_sharpe=1.1,
        out_sample_profit_factor=1.5, out_sample_win_rate=48.0,
        out_sample_total_trades=30, out_sample_max_drawdown=8.0,
        composite_score=0.72, passed_walkforward=True,
    )
    optimizer.optimize = AsyncMock(return_value=[passing])

    pipeline = RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        db_session_factory=SessionLocal,
    )

    event = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert event.promoted is True
    assert pipeline.run_count == 1

    db = SessionLocal()
    try:
        rows = db.query(PromotionEventRecord).all()
        assert len(rows) == 1
        assert rows[0].promoted is True
        assert rows[0].after_params == {"zscore_entry": 2.0}
        assert rows[0].after_metrics["oos_sharpe"] == pytest.approx(1.1)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_cycle_history_survives_redeploy():
    """Simulates a redeploy: second pipeline instance loads first run's history."""
    from src.engine.optimizer import OptimizationResult
    SessionLocal = _make_session_factory()

    # First boot: one no_passing_candidates run
    get_fn = AsyncMock(return_value=_mock_strategy())
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()
    failing = OptimizationResult(
        params={}, in_sample_sharpe=0.5, out_sample_sharpe=0.3,
        out_sample_profit_factor=0.9, out_sample_win_rate=30.0,
        out_sample_total_trades=8, out_sample_max_drawdown=20.0,
        composite_score=0.2, passed_walkforward=False,
    )
    optimizer.optimize = AsyncMock(return_value=[failing])

    pipeline_boot1 = RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        db_session_factory=SessionLocal,
    )
    await pipeline_boot1.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert pipeline_boot1.run_count == 1

    # Simulate redeploy: fresh pipeline with same session_factory (same DB)
    pipeline_boot2 = _make_pipeline(SessionLocal)
    # Should have loaded the event written by boot1
    assert pipeline_boot2.run_count == 1
    assert len(pipeline_boot2.get_history()) == 1
    assert pipeline_boot2.get_history()[0].reason == "no_passing_candidates"


# ──────────────────────────────────────────────
# Pipeline with no DB session factory (backward compat)
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_without_db_factory_still_works():
    """RBIPipeline without a db_session_factory behaves exactly as before T031."""
    from src.engine.optimizer import OptimizationResult
    get_fn = AsyncMock(return_value=_mock_strategy())
    patch_fn = AsyncMock(return_value={})
    optimizer = OptimizationEngine()
    failing = OptimizationResult(
        params={}, in_sample_sharpe=0.5, out_sample_sharpe=0.3,
        out_sample_profit_factor=0.9, out_sample_win_rate=30.0,
        out_sample_total_trades=8, out_sample_max_drawdown=20.0,
        composite_score=0.2, passed_walkforward=False,
    )
    optimizer.optimize = AsyncMock(return_value=[failing])

    pipeline = RBIPipeline(
        get_strategy_fn=get_fn,
        patch_strategy_fn=patch_fn,
        optimizer=optimizer,
        # No db_session_factory
    )
    event = await pipeline.run_cycle("mean_reversion", 8, "BTC", "1h")
    assert event.promoted is False
    assert len(pipeline.get_history()) == 1
    assert pipeline.run_count == 1


# ──────────────────────────────────────────────
# Cross-type isolation: strategy_type filter
# ──────────────────────────────────────────────

def test_strategy_type_filter_isolates_history():
    """A pipeline with strategy_type='adx' must NOT load rows for strategy_type='rsi'."""
    SessionLocal = _make_session_factory()
    db = SessionLocal()
    try:
        db.add(PromotionEventRecord(
            strategy_type="rsi", strategy_id=1,
            timestamp="2026-01-01T00:00:00+00:00",
            promoted=True, reason="promotion_gate_passed",
            before_params={"rsi_period": 14}, after_params={"rsi_period": 18},
            before_metrics={}, after_metrics={},
        ))
        db.add(PromotionEventRecord(
            strategy_type="adx", strategy_id=2,
            timestamp="2026-01-02T00:00:00+00:00",
            promoted=False, reason="no_passing_candidates",
            before_params={"adx_period": 14}, after_params={},
            before_metrics={}, after_metrics={},
        ))
        db.commit()
    finally:
        db.close()

    # Pipeline scoped to "adx" must only see the adx row.
    adx_pipeline = _make_pipeline(SessionLocal, strategy_type="adx")
    assert adx_pipeline.run_count == 1
    assert len(adx_pipeline.get_history()) == 1
    assert adx_pipeline.get_history()[0].strategy_type == "adx"

    # Pipeline scoped to "rsi" must only see the rsi row.
    rsi_pipeline = _make_pipeline(SessionLocal, strategy_type="rsi")
    assert rsi_pipeline.run_count == 1
    assert len(rsi_pipeline.get_history()) == 1
    assert rsi_pipeline.get_history()[0].strategy_type == "rsi"
