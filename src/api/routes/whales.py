"""
Whale Tracker API Routes — Layer 1: The Eyes.

Track top depositors on HyperLiquid, their positions,
P&L, proximity to liquidation, and survival statistics.
"""

import asyncio
import logging
from dataclasses import asdict
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


def _whale_to_info(whale) -> WhaleInfo:
    """Convert a WhaleProfile dataclass to the WhaleInfo response schema."""
    positions = []
    for p in whale.positions:
        # Calculate distance to liquidation as a percentage
        dist_pct = None
        if p.liquidation_price and p.mark_price and p.mark_price > 0:
            dist_pct = round(
                abs(p.mark_price - p.liquidation_price) / p.mark_price * 100, 2
            )
        positions.append(WhalePosition(
            symbol=p.symbol,
            side=p.side,
            size_usd=p.size_usd,
            entry_price=p.entry_price,
            mark_price=p.mark_price,
            unrealized_pnl=p.unrealized_pnl,
            leverage=p.leverage,
            liquidation_price=p.liquidation_price,
            distance_to_liq_pct=dist_pct,
        ))

    # PnL percentage relative to initial deposit
    pnl_pct = 0.0
    if whale.initial_deposit_estimate > 0:
        pnl_pct = round(
            whale.total_unrealized_pnl / whale.initial_deposit_estimate * 100, 2
        )

    return WhaleInfo(
        address=whale.address,
        label=whale.label,
        tags=whale.tags,
        total_deposit=whale.initial_deposit_estimate,
        current_equity=whale.account_value,
        pnl=whale.total_unrealized_pnl,
        pnl_pct=pnl_pct,
        positions=positions,
        is_blown_up=whale.is_blown_up,
        alert_enabled=whale.alert_enabled,
    )


def _stats_to_schema(stats) -> WhaleStats:
    """Convert a WhaleSurvivalStats dataclass to the WhaleStats response schema."""
    return WhaleStats(
        total_tracked=stats.total_tracked,
        blown_up_count=stats.blown_up,
        blown_up_pct=stats.blow_up_rate,
        total_whale_equity=stats.total_whale_equity,
        avg_leverage=0.0,  # Not tracked in service; placeholder
        top_profitable_count=stats.still_active,
    )


def _alert_to_schema(alert) -> dict:
    """Convert a WhaleTradeAlert dataclass to a JSON-serializable dict."""
    return {
        "address": alert.address,
        "label": alert.label,
        "action": alert.action,
        "symbol": alert.symbol,
        "side": alert.side,
        "size_usd": alert.size_usd,
        "timestamp": alert.timestamp.isoformat(),
    }


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
        # Map frontend sort keys to service sort keys
        sort_map = {
            "equity": "account_value",
            "pnl": "pnl",
            "deposit": "account_value",
            "leverage": "account_value",
            "label": "label",
            "positions": "positions",
        }
        service_sort = sort_map.get(sort_by, "account_value")
        whales = tracker.list_whales(sort_by=service_sort, limit=limit, offset=0)
        return [_whale_to_info(w) for w in whales]
    except Exception as e:
        logger.error("Error fetching whales: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=WhaleStats)
async def get_whale_stats(request: Request):
    """Get whale survival statistics — '75% of whale traders blow up'."""
    tracker = _get_tracker(request)
    try:
        stats = tracker.get_survival_stats()
        return _stats_to_schema(stats)
    except Exception as e:
        logger.error("Error fetching whale stats: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{address}", response_model=WhaleInfo)
async def get_whale(request: Request, address: str):
    """Get detailed info for a specific whale."""
    tracker = _get_tracker(request)
    try:
        whale = tracker.get_whale(address)
        if not whale:
            raise HTTPException(status_code=404, detail=f"Whale {address} not found")
        return _whale_to_info(whale)
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
        # Service method is set_label(address, label, notes).
        # Map the tags list to a notes string for storage.
        notes = ", ".join(update.tags) if update.tags else None
        result = tracker.set_label(address, label=update.label, notes=notes)
        if not result:
            raise HTTPException(status_code=404, detail=f"Whale {address} not found")
        return {"status": "updated", "address": address}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating whale label: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{address}/alert")
async def toggle_whale_alert(request: Request, address: str, toggle: WhaleAlertToggle):
    """Toggle alerts for a whale's trades."""
    tracker = _get_tracker(request)
    try:
        result = tracker.toggle_alert(address, enabled=toggle.enabled)
        if not result:
            raise HTTPException(status_code=404, detail=f"Whale {address} not found")
        return {"status": "updated", "address": address, "alert_enabled": toggle.enabled}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error toggling whale alert: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def whales_websocket(websocket: WebSocket):
    """Real-time whale trade alerts via WebSocket.

    Uses the tracker's on_alert() callback to push alerts in real time.
    Sends periodic stats as a heartbeat when no alerts fire.
    """
    await websocket.accept()
    logger.info("Whales WebSocket client connected")

    tracker = getattr(websocket.app.state, "whale_tracker", None)
    if tracker is None:
        await websocket.send_json({"error": "Whale tracker not initialized"})
        await websocket.close()
        return

    alert_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def _on_alert_callback(alert):
        """Push alert into the async queue (called from sync context)."""
        try:
            alert_queue.put_nowait(alert)
        except asyncio.QueueFull:
            pass

    # Register callback -- tracker.on_alert() appends to its internal list.
    # There is no unsubscribe method, so we use a flag to disable the callback
    # after disconnect instead of trying to remove it.
    active = True

    def guarded_callback(alert):
        if active:
            _on_alert_callback(alert)

    tracker.on_alert(guarded_callback)

    try:
        while True:
            try:
                alert = await asyncio.wait_for(alert_queue.get(), timeout=10.0)
                await websocket.send_json({
                    "type": "whale_trade",
                    "data": _alert_to_schema(alert),
                })
            except asyncio.TimeoutError:
                # No alert in 10s -- send stats as heartbeat
                try:
                    stats = tracker.get_survival_stats()
                    await websocket.send_json({
                        "type": "stats",
                        "data": asdict(stats),
                    })
                except Exception:
                    await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        logger.info("Whales WebSocket client disconnected")
    except Exception as e:
        logger.error("Whales WebSocket error: %s", e)
    finally:
        # Disable the callback so it stops pushing to the queue
        active = False
