import logging
from dataclasses import asdict
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.engine.llm_gate import llm_gate
from src.engine.optimizer import OptimizationEngine
from src.engine.rbi_pipeline import RBIPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimize/rbi", tags=["rbi-optimizer"])

_optimizer = OptimizationEngine()
_pipelines: dict[str, RBIPipeline] = {}

BACKEND_BASE = "http://localhost:8000"


async def _get_strategy(strategy_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BACKEND_BASE}/strategies/{strategy_id}", timeout=10.0)
        r.raise_for_status()
        return r.json()


async def _patch_strategy(strategy_id: int, updates: dict) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{BACKEND_BASE}/strategies/{strategy_id}",
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
        )
    return _pipelines[strategy_type]


class TriggerRequest(BaseModel):
    strategy_id: int
    symbol: str = "BTC"
    timeframe: str = "1h"
    lookback_days: int = Field(default=90, ge=30, le=365)
    n_trials: int = Field(default=100, ge=10, le=300)


@router.post("/trigger/{strategy_type}")
async def trigger_optimization(strategy_type: str, req: TriggerRequest):
    """Manually trigger an RBI optimize+promote cycle for a strategy."""
    pipeline = _get_or_create_pipeline(strategy_type)
    try:
        event = await pipeline.run_cycle(
            strategy_type=strategy_type,
            strategy_id=req.strategy_id,
            symbol=req.symbol,
            timeframe=req.timeframe,
            lookback_days=req.lookback_days,
            n_trials=req.n_trials,
        )
        return asdict(event)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("RBI trigger failed for %s: %s", strategy_type, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def get_history():
    """Return all past promotion events across all strategies."""
    all_events = []
    for pipeline in _pipelines.values():
        all_events.extend(asdict(e) for e in pipeline.get_history())
    all_events.sort(key=lambda e: e["timestamp"], reverse=True)
    return all_events


@router.post("/rollback/{strategy_type}")
async def rollback(strategy_type: str, strategy_id: int):
    """Revert a strategy to its params before the last RBI promotion."""
    pipeline = _pipelines.get(strategy_type)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"No RBI history for {strategy_type}")
    prev_params = pipeline.get_rollback_params(strategy_type)
    if not prev_params:
        raise HTTPException(status_code=404, detail=f"No rollback params stored for {strategy_type}")
    result = await _patch_strategy(strategy_id, {"params": prev_params})
    return {"rolled_back": True, "restored_params": prev_params, "strategy": result}


@router.get("/status")
def get_status():
    """Return per-strategy RBI history summary."""
    return {
        stype: {
            "promotions": sum(1 for e in p.get_history() if e.promoted),
            "last_run": p.get_history()[-1].timestamp if p.get_history() else None,
        }
        for stype, p in _pipelines.items()
    }


llm_router = APIRouter(prefix="/optimize/llm-gate", tags=["llm-gate"])


@llm_router.get("/stats")
def get_llm_gate_stats():
    """Return LLM advisory gate accuracy stats."""
    return llm_gate.get_stats()
