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


# -- Schemas --------------------------------------------------

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


# -- Helpers --------------------------------------------------

def _get_tracker(request: Request):
    tracker = getattr(request.app.state, "liquidation_tracker", None)
    if tracker is None:
        raise HTTPException(status_code=503, detail="Liquidation tracker not initialized")
    return tracker


# -- Endpoints ------------------------------------------------

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
        # Tracker method is sync with param name max_proximity_pct (not max_distance_pct)
        positions = tracker.get_near_liquidations(
            symbol=symbol, max_proximity_pct=max_distance_pct, limit=limit
        )
        # Tracker returns NearLiquidation dataclass objects; map to response schema
        return [
            NearLiquidation(
                address=p.address,
                symbol=p.symbol,
                side=p.side,
                size_usd=p.size_usd,
                leverage=p.leverage,
                liquidation_price=p.liquidation_price,
                mark_price=p.mark_price,
                distance_pct=p.proximity_pct,
            )
            for p in positions
        ]
    except Exception as e:
        logger.error("Error fetching near liquidations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/heatmap", response_model=List[LiquidationHeatmapLevel])
async def get_liquidation_heatmap(
    request: Request,
    symbol: str = Query("BTC", description="Symbol to get heatmap for"),
    levels: int = Query(20, le=50, description="Number of price levels"),
):
    """Get aggregated liquidation levels by price -- where whales will blow up."""
    tracker = _get_tracker(request)
    try:
        # Tracker method is sync: get_heatmap(symbol) -- no num_levels param
        heatmap = tracker.get_heatmap(symbol=symbol)
        # Truncate to requested number of levels (sorted by total volume descending)
        heatmap_sorted = sorted(
            heatmap,
            key=lambda h: h.long_liq_usd + h.short_liq_usd,
            reverse=True,
        )[:levels]
        # Re-sort by price_level ascending for display
        heatmap_sorted.sort(key=lambda h: h.price_level)
        return [
            LiquidationHeatmapLevel(
                price_level=h.price_level,
                total_long_liq_usd=h.long_liq_usd,
                total_short_liq_usd=h.short_liq_usd,
                position_count=h.total_positions,
            )
            for h in heatmap_sorted
        ]
    except Exception as e:
        logger.error("Error fetching liquidation heatmap: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volume", response_model=List[LiquidationVolume])
async def get_liquidation_volume(request: Request):
    """Get liquidation volume over time windows (5m, 15m, 30m, 1h, 4h, 24h)."""
    tracker = _get_tracker(request)
    try:
        # Tracker has get_liquidation_volume(minutes, symbol) -- call per window
        windows = [
            ("5m", 5),
            ("15m", 15),
            ("30m", 30),
            ("1h", 60),
            ("4h", 240),
            ("24h", 1440),
        ]
        volumes = []
        for label, minutes in windows:
            total_usd = tracker.get_liquidation_volume(minutes=minutes)
            # Get recent events in this window to split long/short and count
            events = tracker.get_recent_events(minutes=minutes, limit=10000)
            long_usd = sum(e.size_usd for e in events if e.side == "long")
            short_usd = sum(e.size_usd for e in events if e.side == "short")
            volumes.append(
                LiquidationVolume(
                    window=label,
                    total_liquidated_usd=total_usd,
                    long_liquidated_usd=long_usd,
                    short_liquidated_usd=short_usd,
                    event_count=len(events),
                )
            )
        return volumes
    except Exception as e:
        logger.error("Error fetching liquidation volume: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threshold", response_model=SafetyStatus)
async def get_safety_status(request: Request):
    """Is it safe to trade right now? Checks liquidation volume against threshold."""
    tracker = _get_tracker(request)
    try:
        # Use get_threshold_status() which returns a LiquidationThreshold dataclass
        # with safe_to_trade, reason, volume windows, threshold_usd, risk_level
        threshold = tracker.get_threshold_status()
        msg = (
            "Market is calm -- safe to trade"
            if threshold.safe_to_trade
            else (
                f"Mass liquidation event! ${threshold.total_liq_30min:,.0f} liquidated "
                f"in last 30min (threshold: ${threshold.threshold_usd:,.0f})"
            )
        )
        return SafetyStatus(
            is_safe_to_trade=threshold.safe_to_trade,
            recent_liquidation_volume=threshold.total_liq_30min,
            threshold=threshold.threshold_usd,
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
        # Tracker method is sync: get_recent_events(minutes, symbol, limit)
        events = tracker.get_recent_events(minutes=60, symbol=symbol, limit=limit)
        return [
            LiquidationEvent(
                address=e.address,
                symbol=e.symbol,
                side=e.side,
                size_usd=e.size_usd,
                liquidation_price=e.price,
                timestamp=e.timestamp.isoformat(),
            )
            for e in events
        ]
    except Exception as e:
        logger.error("Error fetching liquidation events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def liquidations_websocket(websocket: WebSocket):
    """Real-time liquidation event stream via polling.

    The tracker has no subscribe/unsubscribe API, so we poll
    get_recent_events() and get_threshold_status() periodically and push
    new events to the connected client.
    """
    await websocket.accept()
    logger.info("Liquidations WebSocket client connected")

    tracker = getattr(websocket.app.state, "liquidation_tracker", None)
    if tracker is None:
        await websocket.send_json({"error": "Liquidation tracker not initialized"})
        await websocket.close()
        return

    # Track timestamps of events we have already sent to avoid duplicates
    last_sent_ts = None

    try:
        while True:
            try:
                # Poll for new liquidation events (last 2 minutes)
                events = tracker.get_recent_events(minutes=2, limit=50)

                for event in events:
                    # Only send events newer than the last one we sent
                    if last_sent_ts is None or event.timestamp > last_sent_ts:
                        await websocket.send_json({
                            "type": "liquidation",
                            "data": {
                                "address": event.address,
                                "symbol": event.symbol,
                                "side": event.side,
                                "size_usd": event.size_usd,
                                "price": event.price,
                                "timestamp": event.timestamp.isoformat(),
                                "is_cascade": event.is_cascade,
                            },
                        })

                if events:
                    last_sent_ts = max(e.timestamp for e in events)

                # Send heartbeat with safety status
                threshold = tracker.get_threshold_status()
                await websocket.send_json({
                    "type": "heartbeat",
                    "data": {
                        "is_safe": threshold.safe_to_trade,
                        "recent_volume": threshold.total_liq_30min,
                        "threshold": threshold.threshold_usd,
                        "risk_level": threshold.risk_level,
                    },
                })

                # Wait before next poll (5 seconds)
                await asyncio.sleep(5.0)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                logger.error("Liquidations WebSocket poll error: %s", e)
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
                await asyncio.sleep(5.0)

    except WebSocketDisconnect:
        logger.info("Liquidations WebSocket client disconnected")
    except Exception as e:
        logger.error("Liquidations WebSocket error: %s", e)
