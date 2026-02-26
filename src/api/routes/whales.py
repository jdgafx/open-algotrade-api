"""
Whale Tracker API Routes — Layer 1: The Eyes.

Track top depositors on HyperLiquid, their positions,
P&L, proximity to liquidation, and survival statistics.
"""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/whales", tags=["whales"])


# ── Schemas ──────────────────────────────────────

class WhalePosition(BaseModel):
    symbol: str
    side: str
    size_usd: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: Optional[float] = None
    distance_to_liq_pct: Optional[float] = None

class WhaleInfo(BaseModel):
    address: str
    label: Optional[str] = None
    tags: List[str] = []
    total_deposit: float = 0.0
    current_equity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    positions: List[WhalePosition] = []
    is_blown_up: bool = False
    alert_enabled: bool = False

class WhaleStats(BaseModel):
    total_tracked: int
    blown_up_count: int
    blown_up_pct: float
    total_whale_equity: float
    avg_leverage: float
    top_profitable_count: int

class WhaleLabelUpdate(BaseModel):
    label: Optional[str] = None
    tags: Optional[List[str]] = None

class WhaleAlertToggle(BaseModel):
    enabled: bool

class WhaleTradeAlert(BaseModel):
    address: str
    label: Optional[str] = None
    action: str  # opened, closed, increased, decreased
    symbol: str
    side: str
    size_usd: float
    timestamp: str


# ── Helpers ──────────────────────────────────────

def _get_tracker(request: Request):
    tracker = getattr(request.app.state, "whale_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Whale tracker not initialized")
    return tracker


# ── Endpoints ────────────────────────────────────

@router.get("", response_model=List[WhaleInfo])
async def list_whales(
    request: Request,
    sort_by: str = Query("equity", description="Sort by: equity, pnl, deposit, leverage"),
    limit: int = Query(50, le=500),
):
    """Get tracked whales with their current positions and P&L."""
    tracker = _get_tracker(request)
    try:
        whales = await tracker.get_all_whales(sort_by=sort_by, limit=limit)
        return [WhaleInfo(**w) for w in whales]
    except Exception as e:
        logger.error("Error fetching whales: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=WhaleStats)
async def get_whale_stats(request: Request):
    """Get whale survival statistics — '75% of whale traders blow up'."""
    tracker = _get_tracker(request)
    try:
        stats = await tracker.get_survival_stats()
        return WhaleStats(**stats)
    except Exception as e:
        logger.error("Error fetching whale stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{address}", response_model=WhaleInfo)
async def get_whale(request: Request, address: str):
    """Get detailed info for a specific whale."""
    tracker = _get_tracker(request)
    try:
        whale = await tracker.get_whale(address)
        if not whale:
            raise HTTPException(status_code=404, detail=f"Whale {address} not found")
        return WhaleInfo(**whale)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching whale %s: %s", address, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{address}/label")
async def update_whale_label(request: Request, address: str, update: WhaleLabelUpdate):
    """Set label and tags for a whale."""
    tracker = _get_tracker(request)
    try:
        await tracker.update_label(address, label=update.label, tags=update.tags)
        return {"status": "updated", "address": address}
    except Exception as e:
        logger.error("Error updating whale label: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{address}/alert")
async def toggle_whale_alert(request: Request, address: str, toggle: WhaleAlertToggle):
    """Toggle alerts for a whale's trades."""
    tracker = _get_tracker(request)
    try:
        await tracker.set_alert(address, enabled=toggle.enabled)
        return {"status": "updated", "address": address, "alert_enabled": toggle.enabled}
    except Exception as e:
        logger.error("Error toggling whale alert: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def whales_websocket(websocket: WebSocket):
    """Real-time whale trade alerts."""
    await websocket.accept()
    logger.info("Whales WebSocket client connected")

    tracker = getattr(websocket.app.state, "whale_tracker", None)
    if tracker is None:
        await websocket.send_json({"error": "Whale tracker not initialized"})
        await websocket.close()
        return

    alert_queue: asyncio.Queue = asyncio.Queue()

    def on_alert(alert):
        try:
            alert_queue.put_nowait(alert)
        except asyncio.QueueFull:
            pass

    tracker.subscribe(on_alert)

    try:
        while True:
            try:
                alert = await asyncio.wait_for(alert_queue.get(), timeout=10.0)
                await websocket.send_json({"type": "whale_trade", "data": alert})
            except asyncio.TimeoutError:
                try:
                    stats = await tracker.get_survival_stats()
                    await websocket.send_json({"type": "stats", "data": stats})
                except Exception:
                    await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("Whales WebSocket client disconnected")
    except Exception as e:
        logger.error("Whales WebSocket error: %s", e)
    finally:
        tracker.unsubscribe(on_alert)
