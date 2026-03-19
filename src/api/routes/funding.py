"""
Funding Rate API Routes -- Exposes FundingMonitor data to the frontend.

Endpoints:
    GET /funding/rates          -- all current funding rates with severity/bias
    GET /funding/anomalies      -- top N most extreme funding rates
    GET /funding/bias/{symbol}  -- bias classification for a single symbol
    GET /funding/history/{symbol} -- 24h funding rate history for a symbol
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/funding", tags=["funding"])


# -- Schemas --------------------------------------------------

class FundingRate(BaseModel):
    symbol: str
    annualized_rate: float
    raw_8h_rate: float
    severity: str       # extreme | high | elevated | normal
    bias: str           # long_crowded | short_crowded | neutral


class FundingAnomaly(BaseModel):
    symbol: str
    annualized_rate: float
    raw_8h_rate: float
    bias: str
    severity: str


class FundingBias(BaseModel):
    symbol: str
    bias: str
    annualized_rate: Optional[float] = None
    severity: Optional[str] = None


class FundingHistoryEntry(BaseModel):
    timestamp: str
    annualized_rate: float


class FundingRatesSummary(BaseModel):
    last_refresh: Optional[str] = None
    symbol_count: int
    rates: List[FundingRate]


# -- Helpers --------------------------------------------------

def _get_monitor(request: Request):
    monitor = getattr(request.app.state, "funding_monitor", None)
    if monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Funding monitor not initialized",
        )
    return monitor


# -- Endpoints ------------------------------------------------

@router.get("/rates", response_model=FundingRatesSummary)
async def get_funding_rates(request: Request):
    """All current funding rates with severity and bias classification."""
    monitor = _get_monitor(request)
    summary = monitor.get_funding_summary()

    rates = [
        FundingRate(
            symbol=sym,
            annualized_rate=info["annualized_rate"],
            raw_8h_rate=info["raw_8h_rate"],
            severity=info["severity"],
            bias=info["bias"],
        )
        for sym, info in summary["rates"].items()
    ]

    return FundingRatesSummary(
        last_refresh=summary["last_refresh"],
        symbol_count=summary["symbol_count"],
        rates=rates,
    )


@router.get("/anomalies", response_model=List[FundingAnomaly])
async def get_funding_anomalies(
    request: Request,
    n: int = Query(10, ge=1, le=50, description="Number of top anomalies"),
):
    """Top N most extreme funding rates, sorted by absolute magnitude."""
    monitor = _get_monitor(request)
    anomalies = monitor.get_top_anomalies(n)
    return [
        FundingAnomaly(
            symbol=a["symbol"],
            annualized_rate=a["annualized_rate"],
            raw_8h_rate=a["raw_8h_rate"],
            bias=a["bias"],
            severity=a["severity"],
        )
        for a in anomalies
    ]


@router.get("/bias/{symbol}", response_model=FundingBias)
async def get_funding_bias(request: Request, symbol: str):
    """Funding bias classification for a specific symbol."""
    monitor = _get_monitor(request)
    symbol = symbol.strip().upper()

    bias = monitor.get_funding_bias(symbol)
    rate = monitor.get_funding_rate(symbol)
    severity = monitor.classify_rate(rate) if rate is not None else None

    return FundingBias(
        symbol=symbol,
        bias=bias,
        annualized_rate=rate,
        severity=severity,
    )


@router.get("/history/{symbol}", response_model=List[FundingHistoryEntry])
async def get_funding_history(
    request: Request,
    symbol: str,
    limit: int = Query(96, ge=1, le=500, description="Max entries to return"),
):
    """24-hour funding rate history for a symbol (15-min granularity)."""
    monitor = _get_monitor(request)
    symbol = symbol.strip().upper()

    history = monitor.get_history(symbol, limit=limit)
    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"No funding history for {symbol}",
        )

    return [
        FundingHistoryEntry(
            timestamp=entry["timestamp"],
            annualized_rate=entry["annualized_rate"],
        )
        for entry in history
    ]
