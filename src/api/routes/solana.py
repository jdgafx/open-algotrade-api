"""
Solana DEX Scanner API Routes

Provides endpoints for:
- Token scanning (manual trigger and status)
- Discovered tokens list with scores
- Paper positions tracking
- Scanner configuration
- Paper trading actions (buy/sell)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/solana", tags=["solana"])


# ---------------------------------------------------------------------------
# Pydantic models for request/response
# ---------------------------------------------------------------------------

class ScanResponse(BaseModel):
    status: str
    tokens_fetched: int = 0
    security_passed: int = 0
    metrics_passed: int = 0
    tokens_passed: int = 0
    passed_tokens: List[Dict[str, Any]] = []
    scan_time_seconds: float = 0.0
    error: Optional[str] = None
    message: Optional[str] = None


class TokenListResponse(BaseModel):
    total: int
    tokens: List[Dict[str, Any]]


class PositionListResponse(BaseModel):
    total: int
    positions: List[Dict[str, Any]]
    paper_balance: float = 0.0


class TradeHistoryResponse(BaseModel):
    total: int
    trades: List[Dict[str, Any]]


class ScannerStatsResponse(BaseModel):
    scanner_enabled: bool = False
    birdeye_api_key_set: bool = False
    total_scans: int = 0
    last_scan_time: Optional[str] = None
    total_tokens_seen: int = 0
    total_tokens_passed: int = 0
    blacklisted_count: int = 0
    currently_scanning: bool = False
    paper_balance: float = 0.0
    initial_balance: float = 0.0
    open_positions: int = 0
    total_trades: int = 0
    total_closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_realized_pnl: float = 0.0
    total_return_pct: float = 0.0


class ConfigUpdateRequest(BaseModel):
    hours_lookback: Optional[float] = None
    max_top10_holder_pct: Optional[float] = None
    drop_if_mutable_metadata: Optional[bool] = None
    drop_if_token_2022: Optional[bool] = None
    max_market_cap: Optional[float] = None
    min_market_cap: Optional[float] = None
    min_liquidity: Optional[float] = None
    min_trades_1h: Optional[int] = None
    min_unique_wallets_24h: Optional[int] = None
    max_sell_pct: Optional[float] = None
    ohlcv_timeframe: Optional[str] = None
    max_bars_before_drop: Optional[int] = None
    require_above_avg_close: Optional[bool] = None
    sma_fast: Optional[int] = None
    sma_slow: Optional[int] = None
    paper_balance: Optional[float] = None
    position_size_usdc: Optional[float] = None
    sell_at_multiple: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    scan_interval_seconds: Optional[int] = None
    scanner_enabled: Optional[bool] = None


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class PaperActionRequest(BaseModel):
    token_address: str


class PaperActionResponse(BaseModel):
    status: str
    message: Optional[str] = None
    position: Optional[Dict[str, Any]] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    balance: Optional[float] = None


# ---------------------------------------------------------------------------
# Helper to get scanner from app state
# ---------------------------------------------------------------------------

def _get_scanner(request: Request):
    scanner = getattr(request.app.state, "solana_scanner", None)
    if scanner is None:
        raise HTTPException(
            status_code=503,
            detail="Solana scanner is not initialized. Set SOLANA_SCANNER_ENABLED=true to enable.",
        )
    return scanner


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/scan", response_model=ScanResponse)
async def trigger_scan(request: Request):
    """Trigger a full token scan pipeline.

    Fetches new tokens from Jupiter, runs them through security, metrics,
    and OHLCV filters, and returns tokens that passed all checks.
    In paper mode with no BirdEye key, security and metrics checks use
    simulated data — only Jupiter token discovery is real.
    """
    scanner = _get_scanner(request)
    result = await scanner.run_scan()
    return ScanResponse(**result)


@router.post("/scan/auto-buy", response_model=ScanResponse)
async def scan_and_auto_buy(request: Request):
    """Run a scan and automatically paper-buy tokens that pass all filters."""
    scanner = _get_scanner(request)
    result = await scanner.run_scan()

    passed = result.get("passed_tokens", [])
    if passed:
        buy_results = await scanner.auto_buy_passed_tokens(passed)
        result["auto_buy_results"] = buy_results

    return ScanResponse(**{k: v for k, v in result.items() if k in ScanResponse.model_fields})


@router.get("/tokens", response_model=TokenListResponse)
async def list_tokens(
    request: Request,
    status: Optional[str] = None,
    limit: int = 100,
):
    """List all discovered tokens with their scores and filter status.

    Optional query params:
    - status: Filter by status (discovered, security_passed, metrics_passed,
              ohlcv_passed, rejected_security, rejected_metrics, rejected_ohlcv,
              bought, sold_tp, sold_sl, blacklisted)
    - limit: Max number of tokens to return (default 100)
    """
    scanner = _get_scanner(request)
    tokens = scanner.get_all_tokens(status_filter=status, limit=limit)
    return TokenListResponse(total=len(tokens), tokens=tokens)


@router.get("/positions", response_model=PositionListResponse)
async def get_positions(request: Request):
    """Get all open Solana paper trading positions with current prices and PnL."""
    scanner = _get_scanner(request)

    # Update prices before returning
    await scanner.update_positions()

    positions = scanner.get_positions()
    return PositionListResponse(
        total=len(positions),
        positions=positions,
        paper_balance=round(scanner._paper_balance, 2),
    )


@router.get("/trades", response_model=TradeHistoryResponse)
async def get_trades(request: Request, limit: int = 100):
    """Get Solana paper trade history."""
    scanner = _get_scanner(request)
    trades = scanner.get_trade_history(limit=limit)
    return TradeHistoryResponse(total=len(trades), trades=trades)


@router.get("/stats", response_model=ScannerStatsResponse)
async def get_stats(request: Request):
    """Get Solana scanner statistics: scan counts, PnL, win rate, balance."""
    scanner = _get_scanner(request)
    stats = scanner.get_stats()
    return ScannerStatsResponse(**stats)


@router.post("/config", response_model=ConfigResponse)
async def update_config(request: Request, data: ConfigUpdateRequest):
    """Update Solana scanner configuration at runtime.

    Only provided fields are updated; omitted fields keep their current values.
    Changes take effect on the next scan cycle.
    """
    scanner = _get_scanner(request)
    updates = data.model_dump(exclude_unset=True)
    if not updates:
        return ConfigResponse(config=scanner.config.to_dict())

    new_config = scanner.update_config(updates)
    return ConfigResponse(config=new_config)


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """Get current Solana scanner configuration."""
    scanner = _get_scanner(request)
    return ConfigResponse(config=scanner.config.to_dict())


@router.post("/buy", response_model=PaperActionResponse)
async def paper_buy(request: Request, data: PaperActionRequest):
    """Manually paper-buy a specific token by address.

    The token must have been discovered in a previous scan.
    """
    scanner = _get_scanner(request)
    result = await scanner.paper_buy(data.token_address)
    return PaperActionResponse(**result)


@router.post("/sell", response_model=PaperActionResponse)
async def paper_sell(request: Request, data: PaperActionRequest):
    """Manually paper-sell a position by token address."""
    scanner = _get_scanner(request)
    result = await scanner.paper_sell(data.token_address, reason="manual_sell")
    return PaperActionResponse(**result)


@router.post("/reset")
async def reset_scanner(request: Request):
    """Reset the Solana scanner: clear all positions, trades, and token history."""
    scanner = _get_scanner(request)
    scanner.reset()
    return {
        "status": "reset",
        "balance": scanner._paper_balance,
        "message": "Solana scanner reset. All positions, trades, and scanned tokens cleared.",
    }


@router.post("/scanner/start")
async def start_background_scanner(request: Request):
    """Start the background auto-scanning loop."""
    scanner = _get_scanner(request)
    await scanner.start_background_scanner()
    return {
        "status": "started",
        "interval_seconds": scanner.config.scan_interval_seconds,
        "message": "Background Solana scanner started.",
    }


@router.post("/scanner/stop")
async def stop_background_scanner(request: Request):
    """Stop the background auto-scanning loop."""
    scanner = _get_scanner(request)
    await scanner.stop_background_scanner()
    return {"status": "stopped", "message": "Background Solana scanner stopped."}
