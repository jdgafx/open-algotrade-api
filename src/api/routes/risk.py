"""
Risk API Routes — Layer 0: The Seatbelt.

Endpoints for configuring, starting/stopping, and monitoring the risk controller.
Includes a WebSocket endpoint for real-time risk alerts.
"""

import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from src.services.fee_calculator import calculate_fees
from src.services.risk_models import (
    FeeCalculatorInput,
    FeeCalculatorOutput,
    RiskConfig,
    RiskEvent,
    RiskSnapshot,
    RiskStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_risk_controller(request: Request):
    """Get the risk controller from app state."""
    rc = getattr(request.app.state, "risk_controller", None)
    if rc is None:
        raise HTTPException(
            status_code=503,
            detail="Risk controller not initialized. Check API key configuration.",
        )
    return rc


# ──────────────────────────────────────────────
# Risk Configuration
# ──────────────────────────────────────────────


@router.get("/risk/config", response_model=RiskConfig)
def get_risk_config(request: Request):
    """Get current risk configuration."""
    rc = _get_risk_controller(request)
    return rc.config


@router.put("/risk/config", response_model=RiskConfig)
def update_risk_config(config: RiskConfig, request: Request):
    """Update risk configuration. Takes effect immediately."""
    rc = _get_risk_controller(request)
    rc.update_config(config)
    return rc.config


# ──────────────────────────────────────────────
# Risk Control
# ──────────────────────────────────────────────


@router.post("/risk/start")
async def start_risk_monitoring(request: Request):
    """Start the risk monitoring service."""
    rc = _get_risk_controller(request)

    if rc.is_active:
        return {"status": "already_running", "message": "Risk monitoring is already active"}

    await rc.start()
    return {
        "status": "started",
        "message": "Risk monitoring started",
        "config": rc.config.model_dump(),
    }


@router.post("/risk/stop")
async def stop_risk_monitoring(request: Request):
    """Stop the risk monitoring service."""
    rc = _get_risk_controller(request)
    await rc.stop()
    return {"status": "stopped", "message": "Risk monitoring stopped"}


# ──────────────────────────────────────────────
# Risk Status & Events
# ──────────────────────────────────────────────


@router.get("/risk/status", response_model=RiskStatusResponse)
def get_risk_status(request: Request):
    """Get current risk status including config, snapshot, and recent events."""
    rc = _get_risk_controller(request)
    return RiskStatusResponse(
        config=rc.config,
        snapshot=rc.get_snapshot(),
        recent_events=rc.get_events(20),
    )


@router.get("/risk/events", response_model=List[RiskEvent])
def get_risk_events(request: Request, limit: int = 100):
    """Get risk event log."""
    rc = _get_risk_controller(request)
    return rc.get_events(limit)


# ──────────────────────────────────────────────
# WebSocket — Real-time Risk Alerts
# ──────────────────────────────────────────────


@router.websocket("/ws/risk")
async def risk_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time risk alerts.

    Sends:
    - Risk snapshots every check interval
    - Risk events as they happen
    """
    await websocket.accept()
    logger.info("Risk WebSocket client connected")

    rc = getattr(websocket.app.state, "risk_controller", None)
    if rc is None:
        await websocket.send_json({"error": "Risk controller not initialized"})
        await websocket.close()
        return

    event_queue: asyncio.Queue = asyncio.Queue()

    def on_event(event: RiskEvent):
        try:
            event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    rc.subscribe(on_event)

    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": rc.get_snapshot().model_dump(mode="json"),
        })

        while True:
            # Wait for events with timeout for periodic status updates
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=3.0)
                await websocket.send_json({
                    "type": "event",
                    "data": event.model_dump(mode="json"),
                })
            except asyncio.TimeoutError:
                # Send periodic status update
                await websocket.send_json({
                    "type": "status",
                    "data": rc.get_snapshot().model_dump(mode="json"),
                })

    except WebSocketDisconnect:
        logger.info("Risk WebSocket client disconnected")
    except Exception as e:
        logger.error("Risk WebSocket error: %s", e)
    finally:
        rc.unsubscribe(on_event)


# ──────────────────────────────────────────────
# Fee Calculator — "Days Until Death"
# ──────────────────────────────────────────────


@router.post("/tools/fee-calculator", response_model=FeeCalculatorOutput)
def fee_calculator(input: FeeCalculatorInput):
    """
    Calculate projected trading fee costs and 'days until death'.

    Shows how quickly fees will eat your account at your current trading rate.
    """
    return calculate_fees(input)
