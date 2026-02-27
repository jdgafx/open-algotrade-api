"""
Regime Detection API Routes -- Layer 2: The Brain.

Hidden Markov Model based market regime detection.
Jim Simons' approach: markets switch between hidden states.
Strategies should only run in their favorable regime.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from src.services.regime_detector import REGIME_STRATEGY_MAP, MarketRegime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regime", tags=["regime"])


# -- Schemas --------------------------------------------------

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


# -- Helpers --------------------------------------------------

def _get_detector(request: Request):
    detector = getattr(request.app.state, "regime_detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Regime detector not initialized")
    return detector


def _build_strategy_recommendations(regime_str: str) -> dict:
    """Build recommended/avoid strategy lists from REGIME_STRATEGY_MAP."""
    try:
        current = MarketRegime(regime_str)
    except ValueError:
        return {"recommended": [], "avoid": []}

    recommended = REGIME_STRATEGY_MAP.get(current, [])

    # Collect all strategies that appear in any regime
    all_strategies = set()
    for strats in REGIME_STRATEGY_MAP.values():
        all_strategies.update(strats)

    avoid = [s for s in sorted(all_strategies) if s not in recommended]
    return {"recommended": list(recommended), "avoid": avoid}


def _compute_transition_matrix(history: List[dict]) -> dict:
    """Compute transition probability matrix from regime history."""
    regimes = [r.value for r in MarketRegime if r != MarketRegime.UNKNOWN]
    n = len(regimes)
    regime_idx = {r: i for i, r in enumerate(regimes)}

    # Count transitions
    counts = [[0] * n for _ in range(n)]
    for entry in history:
        fr = entry.get("from_regime", "")
        to = entry.get("to_regime", "")
        if fr in regime_idx and to in regime_idx:
            counts[regime_idx[fr]][regime_idx[to]] += 1

    # Normalize rows to probabilities
    matrix = []
    for row in counts:
        total = sum(row)
        if total > 0:
            matrix.append([c / total for c in row])
        else:
            # Uniform distribution for states with no observed transitions
            matrix.append([1.0 / n] * n)

    return {
        "states": regimes,
        "transition_matrix": matrix,
    }


def _classify_volatility_regime(atr_pct: float) -> str:
    """Map ATR percentage to a volatility regime label."""
    if atr_pct < 1.5:
        return "low"
    elif atr_pct < 3.0:
        return "medium"
    elif atr_pct < 5.0:
        return "high"
    else:
        return "extreme"


# -- Endpoints ------------------------------------------------

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
            regime = detector.get_current_regime(symbol)
            if regime is None:
                results.append(RegimeInfo(
                    symbol=symbol,
                    current_regime="UNKNOWN",
                    confidence=0.0,
                    duration_hours=0.0,
                    recommended_strategies=[],
                    avoid_strategies=[],
                ))
                continue

            rec = _build_strategy_recommendations(regime["regime"])
            results.append(RegimeInfo(
                symbol=symbol,
                current_regime=regime["regime"],
                confidence=regime.get("confidence", 0.0),
                duration_hours=regime.get("duration_hours", 0.0),
                recommended_strategies=rec["recommended"],
                avoid_strategies=rec["avoid"],
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
        history = detector.get_regime_history(symbol=symbol, limit=limit)
        # The service returns dicts with from_regime, to_regime, timestamp
        # but the schema also expects probability -- default to 1.0 (observed transition)
        return [
            RegimeTransition(
                symbol=t.get("symbol", symbol),
                from_regime=t["from_regime"],
                to_regime=t["to_regime"],
                timestamp=t["timestamp"],
                probability=t.get("probability", 1.0),
            )
            for t in history
        ]
    except Exception as e:
        logger.error("Error fetching regime history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volatility", response_model=List[VolatilityStatus])
async def get_volatility_status(
    request: Request,
    symbols: str = Query("BTC,ETH,SOL"),
):
    """Get ATR volatility status -- strategies pause during extreme volatility."""
    detector = _get_detector(request)
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # get_volatility_status() returns {symbol: {atr, atr_pct, price, is_volatile, ...}}
    all_vol = detector.get_volatility_status()

    results = []
    for symbol in symbol_list:
        try:
            vol = all_vol.get(symbol)
            if vol is None:
                results.append(VolatilityStatus(
                    symbol=symbol,
                    is_volatile=False,
                    atr_value=0.0,
                    atr_percentile=0.0,
                    volatility_regime="unknown",
                    message="No volatility data available",
                ))
                continue

            is_vol = vol.get("is_volatile", False)
            atr_pct = vol.get("atr_pct", 0.0)
            regime_label = _classify_volatility_regime(atr_pct)

            results.append(VolatilityStatus(
                symbol=symbol,
                is_volatile=is_vol,
                atr_value=vol.get("atr", 0.0),
                atr_percentile=atr_pct,
                volatility_regime=regime_label,
                message="Extreme volatility -- bots should pause" if is_vol else "Normal volatility",
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
        symbol = symbol.strip().upper()

        # First try to get the HMM transition matrix from the regime info
        regime_info = detector.get_current_regime(symbol)
        if regime_info and regime_info.get("transition_matrix") is not None:
            # HMM fitted -- use the real transition matrix
            transmat = regime_info["transition_matrix"]
            regimes = [r.value for r in MarketRegime if r != MarketRegime.UNKNOWN]
            # The HMM may have fewer/more states; pad or truncate to match
            n_expected = len(regimes)
            n_actual = len(transmat)
            if n_actual == n_expected:
                return RegimeMatrix(
                    symbol=symbol,
                    states=regimes,
                    transition_matrix=transmat,
                )

        # Fallback: compute from observed regime history
        history = detector.get_regime_history(symbol=symbol, limit=500)
        result = _compute_transition_matrix(history)
        return RegimeMatrix(symbol=symbol, **result)
    except Exception as e:
        logger.error("Error getting transition matrix for %s: %s", symbol, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy-map", response_model=List[StrategyRegimeMap])
async def get_strategy_regime_map(request: Request):
    """Which strategies work in which regimes -- the regime-aware strategy selector."""
    detector = _get_detector(request)
    try:
        # Invert REGIME_STRATEGY_MAP: strategy -> list of favorable regimes
        strategy_to_regimes: Dict[str, List[str]] = {}
        all_regimes = [r.value for r in MarketRegime if r != MarketRegime.UNKNOWN]

        for regime, strategies in REGIME_STRATEGY_MAP.items():
            for strat in strategies:
                strategy_to_regimes.setdefault(strat, []).append(regime.value)

        # Build the response for each strategy
        current_regimes = detector.get_all_regimes()
        results = []
        for strategy_type, favorable in sorted(strategy_to_regimes.items()):
            unfavorable = [r for r in all_regimes if r not in favorable]

            # Check should_trade_now for each tracked symbol
            should_trade_now = {}
            for symbol in current_regimes:
                should_trade_now[symbol] = detector.should_trade(symbol, strategy_type)

            results.append(StrategyRegimeMap(
                strategy_type=strategy_type,
                favorable_regimes=favorable,
                unfavorable_regimes=unfavorable,
                should_trade_now=should_trade_now,
            ))

        return results
    except Exception as e:
        logger.error("Error getting strategy regime map: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
