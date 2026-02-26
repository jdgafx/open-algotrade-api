"""
Liquidation API Routes — Layer 1: The Eyes.

Endpoints for querying positions near liquidation, heatmap data,
liquidation volume, safety thresholds, and WebSocket real-time events.
"""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/liquidations", tags=["liquidations"])


# ── Schemas ──────────────────────────────────────

class NearLiquidation(BaseModel):
    address: str
    symbol: str
    side: str
    size_usd: float
    leverage: float
    liquidation_price: float
    mark_price: float
    distance_pct: float

class LiquidationHeatmapLevel(BaseModel):
    price_level: float
    total_long_liq_usd: float
    total_short_liq_usd: float
    position_count: int

class LiquidationVolume(BaseModel):
    window: str
    total_liquidated_usd: float
    long_liquidated_usd: float
    short_liquidated_usd: float
    event_count: int

class LiquidationEvent(BaseModel):
    address: str
    symbol: str
    side: str
    size_usd: float
    liquidation_price: float
    timestamp: str

class SafetyStatus(BaseModel):
    is_safe_to_trade: bool
    recent_liquidation_volume: float
    threshold: float
    message: str


# ── Helpers ──────────────────────────────────────

def _get_tracker(request: Request):
    tracker = getattr(request.app.state, "liquidation_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Liquidation tracker not initialized")
    return tracker


# ── Endpoints ────────────────────────────────────

@router.get("/near", response_model=List[NearLiquidation])
async def get_near_liquidations(
    request: Request,
    symbol: Optional[str] = None,
    max_distance_pct: float = Query(5.0, description="Max distance to liquidation (%)"),
    limit: int = Query(50, le=200),
):
    """Get positions closest to liquidation."""
    tracker = _get_tracker(request)
    try:
        positions = await tracker.get_near_liquidations(
            symbol=symbol, max_distance_pct=max_distance_pct, limit=limit
        )
        return [NearLiquidation(**p) for p in positions]
    except Exception as e:
        logger.error("Error fetching near liquidations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap", response_model=List[LiquidationHeatmapLevel])
async def get_liquidation_heatmap(
    request: Request,
    symbol: str = Query("BTC", description="Symbol to get heatmap for"),
    levels: int = Query(20, le=50, description="Number of price levels"),
):
    """Get aggregated liquidation levels by price — where whales will blow up."""
    tracker = _get_tracker(request)
    try:
        heatmap = await tracker.get_heatmap(symbol=symbol, num_levels=levels)
        return [LiquidationHeatmapLevel(**h) for h in heatmap]
    except Exception as e:
        logger.error("Error fetching liquidation heatmap: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volume", response_model=List[LiquidationVolume])
async def get_liquidation_volume(request: Request):
    """Get liquidation volume over time windows (5m, 15m, 30m, 1h, 4h, 24h)."""
    tracker = _get_tracker(request)
    try:
        volumes = await tracker.get_liquidation_volumes()
        return [LiquidationVolume(**v) for v in volumes]
    except Exception as e:
        logger.error("Error fetching liquidation volume: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threshold", response_model=SafetyStatus)
async def get_safety_status(request: Request):
    """Is it safe to trade right now? Checks liquidation volume against threshold."""
    tracker = _get_tracker(request)
    try:
        safe, volume, threshold = await tracker.is_safe_to_trade()
        msg = "Market is calm — safe to trade" if safe else f"Mass liquidation event! ${volume:,.0f} liquidated in last 30min (threshold: ${threshold:,.0f})"
        return SafetyStatus(
            is_safe_to_trade=safe,
            recent_liquidation_volume=volume,
            threshold=threshold,
            message=msg,
        )
    except Exception as e:
        logger.error("Error checking safety: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events", response_model=List[LiquidationEvent])
async def get_recent_events(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = Query(50, le=500),
):
    """Get recent liquidation events."""
    tracker = _get_tracker(request)
    try:
        events = await tracker.get_recent_events(symbol=symbol, limit=limit)
        return [LiquidationEvent(**e) for e in events]
    except Exception as e:
        logger.error("Error fetching liquidation events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def liquidations_websocket(websocket: WebSocket):
    """Real-time liquidation event stream."""
    await websocket.accept()
    logger.info("Liquidations WebSocket client connected")

    tracker = getattr(websocket.app.state, "liquidation_tracker", None)
    if tracker is None:
        await websocket.send_json({"error": "Liquidation tracker not initialized"})
        await websocket.close()
        return

    event_queue: asyncio.Queue = asyncio.Queue()

    def on_event(event):
        try:
            event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    tracker.subscribe(on_event)

    try:
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
                await websocket.send_json({"type": "liquidation", "data": event})
            except asyncio.TimeoutError:
                # Send heartbeat with summary stats
                try:
                    safe, volume, threshold = await tracker.is_safe_to_trade()
                    await websocket.send_json({
                        "type": "heartbeat",
                        "data": {
                            "is_safe": safe,
                            "recent_volume": volume,
                            "threshold": threshold,
                        },
                    })
                except Exception:
                    await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("Liquidations WebSocket client disconnected")
    except Exception as e:
        logger.error("Liquidations WebSocket error: %s", e)
    finally:
        tracker.unsubscribe(on_event)
