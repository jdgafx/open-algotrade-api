import asyncio
import logging
import traceback
from dataclasses import asdict
from typing import Any, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from pydantic import BaseModel, Field

from src.api.database import SessionLocal
from src.engine.llm_gate import llm_gate
from src.engine.optimizer import OptimizationEngine
from src.engine.rbi_pipeline import RBIPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimize/rbi", tags=["rbi-optimizer"])

_optimizer = OptimizationEngine(commission_pct=0.14)  # 2x 0.07% research default; promote only on worst-case slippage
_pipelines: dict[str, RBIPipeline] = {}
# In-flight and completed trigger results keyed by strategy_type
_trigger_results: dict[str, dict] = {}
_trigger_running: set[str] = set()

BACKEND_BASE = "http://localhost:8000"


async def _resolve_name(client: httpx.AsyncClient, strategy_id: int) -> str:
    r = await client.get(f"{BACKEND_BASE}/strategies", timeout=10.0)
    r.raise_for_status()
    for s in r.json():
        if s.get("id") == strategy_id:
            return s["name"]
    raise HTTPException(status_code=404, detail=f"strategy_id={strategy_id} not found")


async def _get_strategy(strategy_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        name = await _resolve_name(client, strategy_id)
        r = await client.get(f"{BACKEND_BASE}/strategies/{name}", timeout=10.0)
        r.raise_for_status()
        return r.json()


async def _patch_strategy(strategy_id: int, updates: dict) -> dict:
    async with httpx.AsyncClient() as client:
        name = await _resolve_name(client, strategy_id)
        r = await client.patch(
            f"{BACKEND_BASE}/strategies/{name}",
            json=updates, timeout=10.0,
        )
        r.raise_for_status()
        return r.json()


def _get_or_create_pipeline(strategy_type: str) -> RBIPipeline:
    if strategy_type not in _pipelines:
        _pipelines[strategy_type] = RBIPipeline(
            get_strategy_fn=_get_strategy,
            patch_strategy_fn=_patch_strategy,
            optimizer=_optimizer,
            db_session_factory=SessionLocal,  # enables DB persistence + history rebuild
            strategy_type=strategy_type,
        )
    return _pipelines[strategy_type]


class TriggerRequest(BaseModel):
    strategy_id: int
    symbol: str = "BTC"
    timeframe: str = "1h"
    # Match OptimizationEngine.optimize()'s own 90d default. 14d starves
    # sparse-trigger strategies (liquidation_dip ~1 trade/8d, consolidation_pop):
    # at 14d every Optuna trial takes <5 in-sample trades -> -inf -> no_candidates
    # (proven: 14d 0/30 configs reach >=5 trades, 90d 26/30). ge floor raised to
    # 30 so a caller can't re-create the starvation by hand.
    lookback_days: int = Field(default=90, ge=30, le=365)
    n_trials: int = Field(default=100, ge=10, le=300)


async def _run_trigger_background(strategy_type: str, strategy_id: int, symbol: str,
                                   timeframe: str, lookback_days: int, n_trials: int):
    """Run RBI cycle in background; store result in _trigger_results."""
    _trigger_running.add(strategy_type)
    pipeline = _get_or_create_pipeline(strategy_type)
    try:
        event = await pipeline.run_cycle(
            strategy_type=strategy_type, strategy_id=strategy_id,
            symbol=symbol, timeframe=timeframe,
            lookback_days=lookback_days, n_trials=n_trials,
        )
        result = asdict(event)
        result["status"] = "completed"
        logger.info("RBI background trigger done: %s promoted=%s", strategy_type, event.promoted)
    except Exception as e:
        tb = traceback.format_exc()
        detail = str(e) or repr(e)
        result = {"status": "error", "strategy_type": strategy_type, "detail": detail, "traceback": tb}
        logger.error("RBI background trigger failed for %s: %s\n%s", strategy_type, detail, tb)
    finally:
        _trigger_running.discard(strategy_type)
    _trigger_results[strategy_type] = result


@router.post("/trigger/{strategy_type}", status_code=202)
async def trigger_optimization(strategy_type: str, req: TriggerRequest,
                                background_tasks: BackgroundTasks):
    """Manually trigger an RBI optimize+promote cycle (async, returns 202 immediately).
    Poll GET /optimize/rbi/trigger/{strategy_type}/status for result."""
    if strategy_type in _trigger_running:
        raise HTTPException(status_code=409, detail=f"{strategy_type} optimization already running")
    background_tasks.add_task(
        _run_trigger_background,
        strategy_type, req.strategy_id, req.symbol,
        req.timeframe, req.lookback_days, req.n_trials,
    )
    return {"status": "accepted", "strategy_type": strategy_type,
            "message": f"Optimization queued — poll /optimize/rbi/trigger/{strategy_type}/status"}


@router.get("/trigger/{strategy_type}/status")
async def trigger_status(strategy_type: str):
    """Check status of a background RBI trigger."""
    if strategy_type in _trigger_running:
        return {"status": "running", "strategy_type": strategy_type}
    if strategy_type in _trigger_results:
        return _trigger_results[strategy_type]
    return {"status": "not_started", "strategy_type": strategy_type}


@router.get("/history")
def get_history():
    """Return all past promotion events across all strategies (reads from DB — survives redeploys)."""
    try:
        from src.api.models import PromotionEventRecord
        db = SessionLocal()
        try:
            rows = (
                db.query(PromotionEventRecord)
                .order_by(PromotionEventRecord.timestamp.desc())
                .all()
            )
            return [
                {
                    "strategy_type": r.strategy_type,
                    "strategy_id": r.strategy_id,
                    "timestamp": r.timestamp,
                    "promoted": r.promoted,
                    "reason": r.reason,
                    "before_params": r.before_params or {},
                    "after_params": r.after_params or {},
                    "before_metrics": r.before_metrics or {},
                    "after_metrics": r.after_metrics or {},
                }
                for r in rows
            ]
        finally:
            db.close()
    except Exception as exc:
        logger.error("GET /optimize/rbi/history DB error: %s", exc)
        # Fallback to in-memory if DB unavailable
        all_events = []
        for pipeline in _pipelines.values():
            all_events.extend(asdict(e) for e in pipeline.get_history())
        all_events.sort(key=lambda e: e["timestamp"], reverse=True)
        return all_events


@router.post("/rollback/{strategy_type}")
async def rollback(strategy_type: str, strategy_id: int):
    """Revert a strategy to its params before the last RBI promotion.
    Uses _get_or_create_pipeline (not a raw dict lookup) so a cold boot —
    before this type's jittered scheduler job has fired yet — still rebuilds
    rollback params from the persistent DB instead of false-404ing."""
    pipeline = _get_or_create_pipeline(strategy_type)
    prev_params = pipeline.get_rollback_params(strategy_type)
    if not prev_params:
        raise HTTPException(status_code=404, detail=f"No rollback params stored for {strategy_type}")
    result = await _patch_strategy(strategy_id, {"params": prev_params})
    return {"rolled_back": True, "restored_params": prev_params, "strategy": result}


@router.get("/status")
def get_status():
    """Return per-strategy RBI heartbeat — includes last_run, run_count, promotions, last_outcome.
    Reads from DB so it reflects all historical runs, not just this boot session."""
    try:
        from src.api.models import PromotionEventRecord
        db = SessionLocal()
        try:
            rows = db.query(PromotionEventRecord).all()
        finally:
            db.close()
    except Exception as exc:
        logger.error("GET /optimize/rbi/status DB error: %s", exc)
        rows = []

    # Aggregate by strategy_type from DB rows
    from collections import defaultdict
    agg: dict[str, dict] = defaultdict(lambda: {
        "run_count": 0,
        "promotions": 0,
        "last_run": None,
        "last_outcome": None,
    })
    for r in rows:
        s = agg[r.strategy_type]
        s["run_count"] += 1
        if r.promoted:
            s["promotions"] += 1
        if s["last_run"] is None or r.timestamp > s["last_run"]:
            s["last_run"] = r.timestamp
            s["last_outcome"] = r.reason

    # Merge with in-memory pipelines for any types that fired this session
    # but may not have committed to DB yet (edge case).
    for stype, p in _pipelines.items():
        history = p.get_history()
        if not history:
            continue
        s = agg[stype]
        in_mem_last = history[-1].timestamp
        if s["last_run"] is None or in_mem_last > s["last_run"]:
            s["last_run"] = in_mem_last
            s["last_outcome"] = history[-1].reason
        # run_count and promotions from DB are authoritative; don't double-count.

    return dict(agg)


llm_router = APIRouter(prefix="/optimize/llm-gate", tags=["llm-gate"])


@llm_router.get("/stats")
def get_llm_gate_stats():
    """Return LLM advisory gate accuracy stats."""
    return llm_gate.get_stats()
