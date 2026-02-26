"""
RBI Agent Routes — Layer 2: The Brain.

Research -> Backtest -> Implement pipeline.
Submit ideas (text/PDF/YouTube) -> AI generates strategy specs ->
auto-backtests -> ranks results -> one-click deploy to live trading.
"""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.services.rbi_models import (
    RBIDeployRequest,
    RBIJob,
    RBIJobCreate,
    RBIJobOut,
    RBIJobResultsOut,
    RBIStrategyResult,
    RBIStrategyResultOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rbi", tags=["rbi"])


# ── Helpers ──────────────────────────────────────

def _get_rbi_agent(request: Request):
    agent = getattr(request.app.state, "rbi_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="RBI Agent not initialized")
    return agent


# ── Endpoints ────────────────────────────────────

@router.post("/jobs", response_model=RBIJobOut)
async def create_rbi_job(
    job_data: RBIJobCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start a new RBI pipeline job.

    Input can be:
    - text: A trading idea or strategy description
    - pdf: Content from a PDF document
    - youtube: A YouTube transcript or URL

    The AI will generate 5-10 strategy specs, code them, backtest across
    BTC/ETH/SOL on multiple timeframes, and rank by Sharpe/profit factor.
    """
    # Create job in DB
    job = RBIJob(
        input_type=job_data.input_type,
        input_text=job_data.input_text,
        ai_provider=job_data.ai_provider,
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Start the pipeline in background
    agent = _get_rbi_agent(request)
    background_tasks.add_task(agent.run_pipeline, job.id, job_data.input_text, job_data.input_type, job_data.ai_provider)

    return job


@router.get("/jobs", response_model=List[RBIJobOut])
def list_rbi_jobs(
    status: Optional[str] = None,
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
):
    """List all RBI jobs, optionally filtered by status."""
    query = db.query(RBIJob)
    if status:
        query = query.filter(RBIJob.status == status)
    return query.order_by(RBIJob.created_at.desc()).limit(limit).all()


@router.get("/jobs/{job_id}", response_model=RBIJobOut)
def get_rbi_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific RBI job."""
    job = db.query(RBIJob).filter(RBIJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"RBI job {job_id} not found")
    return job


@router.get("/jobs/{job_id}/results", response_model=RBIJobResultsOut)
def get_rbi_results(
    job_id: int,
    sort_by: str = Query("sharpe_ratio", description="Sort by: sharpe_ratio, total_return_pct, profit_factor, win_rate"),
    db: Session = Depends(get_db),
):
    """Get sorted backtest results for an RBI job."""
    job = db.query(RBIJob).filter(RBIJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"RBI job {job_id} not found")

    sort_column = getattr(RBIStrategyResult, sort_by, RBIStrategyResult.sharpe_ratio)
    results = (
        db.query(RBIStrategyResult)
        .filter(RBIStrategyResult.job_id == job_id)
        .order_by(sort_column.desc().nullslast())
        .all()
    )

    return RBIJobResultsOut(
        job=RBIJobOut.model_validate(job),
        results=[RBIStrategyResultOut.model_validate(r) for r in results],
    )


@router.post("/jobs/{job_id}/deploy")
async def deploy_rbi_strategy(
    job_id: int,
    deploy: RBIDeployRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Deploy a discovered strategy to live trading (one-click deploy)."""
    result = db.query(RBIStrategyResult).filter(
        RBIStrategyResult.id == deploy.result_id,
        RBIStrategyResult.job_id == job_id,
    ).first()
    if not result:
        raise HTTPException(status_code=404, detail="Strategy result not found")

    agent = _get_rbi_agent(request)
    try:
        instance = await agent.deploy_strategy(
            result=result,
            symbol=deploy.symbol,
            size_usd=deploy.size_usd,
            leverage=deploy.leverage,
        )
        return {
            "status": "deployed",
            "strategy_name": result.strategy_name,
            "symbol": deploy.symbol,
            "instance": instance,
        }
    except Exception as e:
        logger.error("Failed to deploy RBI strategy: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{job_id}")
async def rbi_progress_websocket(websocket: WebSocket, job_id: int):
    """Real-time progress updates for an RBI job."""
    await websocket.accept()
    logger.info("RBI WebSocket client connected for job %d", job_id)

    agent = getattr(websocket.app.state, "rbi_agent", None)
    if agent is None:
        await websocket.send_json({"error": "RBI Agent not initialized"})
        await websocket.close()
        return

    progress_queue: asyncio.Queue = asyncio.Queue()

    def on_progress(update):
        if update.get("job_id") == job_id:
            try:
                progress_queue.put_nowait(update)
            except asyncio.QueueFull:
                pass

    agent.subscribe_progress(on_progress)

    try:
        while True:
            try:
                update = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                await websocket.send_json({"type": "progress", "data": update})
                # Close when job is done
                if update.get("status") in ("completed", "failed"):
                    await websocket.send_json({"type": "done", "data": update})
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("RBI WebSocket client disconnected for job %d", job_id)
    except Exception as e:
        logger.error("RBI WebSocket error: %s", e)
    finally:
        agent.unsubscribe_progress(on_progress)
