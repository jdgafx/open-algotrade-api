"""
Regime Detection API Routes — Layer 2: The Brain.

Hidden Markov Model based market regime detection.
Jim Simons' approach: markets switch between hidden states.
Strategies should only run in their favorable regime.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regime", tags=["regime"])


# ── Schemas ──────────────────────────────────────

class RegimeInfo(BaseModel):
    symbol: str
    current_regime: str  # TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING, HIGH_VOLATILITY, LOW_VOLATILITY
    confidence: float
    duration_hours: float
    recommended_strategies: List[str]
    avoid_strategies: List[str]

class RegimeTransition(BaseModel):
    symbol: str
    from_regime: str
    to_regime: str
    timestamp: str
    probability: float

class VolatilityStatus(BaseModel):
    symbol: str
    is_volatile: bool
    atr_value: float
    atr_percentile: float
    volatility_regime: str  # low, medium, high, extreme
    message: str

class RegimeMatrix(BaseModel):
    symbol: str
    states: List[str]
    transition_matrix: List[List[float]]

class StrategyRegimeMap(BaseModel):
    strategy_type: str
    favorable_regimes: List[str]
    unfavorable_regimes: List[str]
    should_trade_now: Dict[str, bool]  # symbol -> bool


# ── Helpers ──────────────────────────────────────

def _get_detector(request: Request):
    detector = getattr(request.app.state, "regime_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Regime detector not initialized")
    return detector


# ── Endpoints ────────────────────────────────────

@router.get("/current", response_model=List[RegimeInfo])
async def get_current_regimes(
    request: Request,
    symbols: str = Query("BTC,ETH,SOL", description="Comma-separated symbols"),
):
    """Get current market regime for each symbol. Color-coded in the UI."""
    detector = _get_detector(request)
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    results = []
    for symbol in symbol_list:
        try:
            regime = await detector.get_regime(symbol)
            rec = detector.get_strategy_recommendations(regime["regime"])
            results.append(RegimeInfo(
                symbol=symbol,
                current_regime=regime["regime"],
                confidence=regime.get("confidence", 0.0),
                duration_hours=regime.get("duration_hours", 0.0),
                recommended_strategies=rec.get("recommended", []),
                avoid_strategies=rec.get("avoid", []),
            ))
        except Exception as e:
            logger.error("Error getting regime for %s: %s", symbol, e)
            results.append(RegimeInfo(
                symbol=symbol,
                current_regime="UNKNOWN",
                confidence=0.0,
                duration_hours=0.0,
                recommended_strategies=[],
                avoid_strategies=[],
            ))

    return results


@router.get("/history", response_model=List[RegimeTransition])
async def get_regime_history(
    request: Request,
    symbol: str = Query("BTC"),
    limit: int = Query(50, le=200),
):
    """Get historical regime transitions for a symbol."""
    detector = _get_detector(request)
    try:
        history = await detector.get_transition_history(symbol=symbol, limit=limit)
        return [RegimeTransition(**t) for t in history]
    except Exception as e:
        logger.error("Error fetching regime history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility", response_model=List[VolatilityStatus])
async def get_volatility_status(
    request: Request,
    symbols: str = Query("BTC,ETH,SOL"),
):
    """Get ATR volatility status — strategies pause during extreme volatility."""
    detector = _get_detector(request)
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    results = []
    for symbol in symbol_list:
        try:
            vol = await detector.get_volatility(symbol)
            is_vol = vol.get("is_volatile", False)
            results.append(VolatilityStatus(
                symbol=symbol,
                is_volatile=is_vol,
                atr_value=vol.get("atr_value", 0.0),
                atr_percentile=vol.get("atr_percentile", 0.0),
                volatility_regime=vol.get("regime", "medium"),
                message="Extreme volatility — bots should pause" if is_vol else "Normal volatility",
            ))
        except Exception as e:
            logger.error("Error getting volatility for %s: %s", symbol, e)
            results.append(VolatilityStatus(
                symbol=symbol,
                is_volatile=False,
                atr_value=0.0,
                atr_percentile=0.0,
                volatility_regime="unknown",
                message="Could not determine volatility",
            ))

    return results


@router.get("/matrix/{symbol}", response_model=RegimeMatrix)
async def get_transition_matrix(request: Request, symbol: str):
    """Get the HMM transition probability matrix for a symbol."""
    detector = _get_detector(request)
    try:
        matrix = await detector.get_transition_matrix(symbol)
        return RegimeMatrix(**matrix)
    except Exception as e:
        logger.error("Error getting transition matrix for %s: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy-map", response_model=List[StrategyRegimeMap])
async def get_strategy_regime_map(request: Request):
    """Which strategies work in which regimes — the regime-aware strategy selector."""
    detector = _get_detector(request)
    try:
        mapping = await detector.get_full_strategy_map()
        return [StrategyRegimeMap(**m) for m in mapping]
    except Exception as e:
        logger.error("Error getting strategy regime map: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
