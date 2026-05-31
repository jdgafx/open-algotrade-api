from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any


class UserBase(BaseModel):
    username: str
    email: Optional[str] = None


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    balance: float
    shares: float
    created_at: datetime

    class Config:
        from_attributes = True

class AuthSignIn(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    id: str
    email: Optional[str] = None
    name: str
    avatar: Optional[str] = None
    status: str = "ONLINE"
    access_token: Optional[str] = None
    token_type: str = "bearer"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class DepositCreate(BaseModel):
    user_id: int
    amount: float
    tx_hash: Optional[str] = None


class Deposit(BaseModel):
    id: int
    user_id: int
    amount: float
    tx_hash: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class WithdrawCreate(BaseModel):
    user_id: int
    shares_to_redeem: float


class Withdrawal(BaseModel):
    id: int
    user_id: int
    amount: float
    shares_burned: float
    timestamp: datetime

    class Config:
        from_attributes = True


class VaultStatus(BaseModel):
    total_equity: float
    total_shares: float
    nav_per_share: float
    live_equity: Optional[float] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Portfolio(BaseModel):
    user_id: int
    username: str
    shares: float
    nav_per_share: float
    portfolio_value: float
    total_deposited: float
    unrealized_pnl: float
    pnl_percent: float


class TradeOut(BaseModel):
    id: int
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    strategy: str
    opened_at: datetime
    closed_at: Optional[datetime] = None
    is_open: bool

    class Config:
        from_attributes = True


# Legacy single-strategy config (kept for backwards compat)
class StrategyConfig(BaseModel):
    symbol: str
    timeframe: str
    lookback_period: int
    atr_period: int
    atr_multiplier: float
    leverage: int


class StrategyStatusOut(BaseModel):
    name: str
    status: str
    symbol: str
    timeframe: str
    lookback_period: int
    atr_period: int
    atr_multiplier: float
    leverage: int
    last_signal: Optional[str] = None
    last_signal_time: Optional[datetime] = None
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


# Multi-strategy schemas
class StrategyInstanceCreate(BaseModel):
    name: str
    strategy_type: str
    symbol: str = "BTC"
    timeframe: str = "1h"
    leverage: int = 3
    size_usd: float = 100.0
    target_pct: float = 5.0
    max_loss_pct: float = -10.0
    lookback_days: int = 7
    interval_seconds: int = 30
    enabled: bool = True
    params: Dict[str, Any] = {}


class StrategyInstanceUpdate(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    leverage: Optional[int] = None
    size_usd: Optional[float] = None
    target_pct: Optional[float] = None
    max_loss_pct: Optional[float] = None
    lookback_days: Optional[int] = None
    interval_seconds: Optional[int] = None
    enabled: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None


class StrategyInstanceOut(BaseModel):
    id: int
    name: str
    strategy_type: str
    tier: str
    status: str
    symbol: str
    timeframe: str
    leverage: int
    size_usd: float
    target_pct: float
    max_loss_pct: float
    lookback_days: Optional[int] = None
    interval_seconds: int
    enabled: bool
    params: Dict[str, Any]
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    edge_confidence_score: float = 0.0
    iterations: int
    errors: int
    last_signal: Optional[str] = None
    last_signal_time: Optional[datetime] = None
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Populated by PATCH when the change was hot-applied to a running strategy:
    # {applied, deferred, restart_required, running}. None on create/read.
    live_update: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class StrategyTypeInfo(BaseModel):
    strategy_type: str
    tier: str
    description: str
    default_symbol: str
    default_timeframe: str
    default_params: Dict[str, Any]
    category: str = "trend"
    risk_level: str = "medium"


class StrategyRegistryOut(BaseModel):
    available_strategies: List[StrategyTypeInfo]
    total: int


class MarketPrice(BaseModel):
    symbol: str
    price: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


class Position(BaseModel):
    symbol: str
    size: float
    side: str
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int


class DashboardStats(BaseModel):
    total_strategies: int
    running_strategies: int
    total_pnl: float
    total_trades: int
    win_rate: float
    vault_equity: Optional[float] = None
    active_positions: int


# Vault history
class VaultHistoryPoint(BaseModel):
    timestamp: datetime
    equity: float
    nav_per_share: float


class VaultHistoryOut(BaseModel):
    data: List[VaultHistoryPoint]
    total_points: int


# Dashboard performance
class EquityCurvePoint(BaseModel):
    timestamp: datetime
    equity: float


class StrategyPerformanceBreakdown(BaseModel):
    name: str
    strategy_type: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    status: str


class DashboardPerformance(BaseModel):
    total_pnl: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    equity_curve: List[EquityCurvePoint]
    strategy_breakdown: List[StrategyPerformanceBreakdown]


# Extended strategy registry info (includes category and risk_level)
class StrategyTypeInfoFull(BaseModel):
    strategy_type: str
    tier: str
    description: str
    default_symbol: str
    default_timeframe: str
    default_params: Dict[str, Any]
    category: str
    risk_level: str


# ──────────────────────────────────────────────
# Batch Deploy
# ──────────────────────────────────────────────

class BatchStrategyItem(BaseModel):
    """Single strategy config within a batch deploy request."""
    name: str
    strategy_type: str
    symbol: str = "BTC"
    timeframe: str = "1h"
    leverage: int = 3
    size_usd: float = 100.0
    target_pct: float = 5.0
    max_loss_pct: float = -10.0
    lookback_days: int = 7
    interval_seconds: int = 30
    enabled: bool = True
    params: Dict[str, Any] = {}


class BatchDeployRequest(BaseModel):
    strategies: List[BatchStrategyItem]


class BatchDeployResultItem(BaseModel):
    name: str
    strategy_type: str
    symbol: str
    status: str  # "started", "created", "error"
    error: Optional[str] = None


class BatchDeployResponse(BaseModel):
    total: int
    started: int
    failed: int
    results: List[BatchDeployResultItem]


# ──────────────────────────────────────────────
# Paper Trading Stats / Positions / Trades
# ──────────────────────────────────────────────

class PaperPosition(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    leverage: int = 1
    size_usd: float = 0.0
    strategy_name: str = ""
    entry_time: Optional[str] = None


class PaperTrade(BaseModel):
    id: int
    symbol: str
    side: str
    action: str
    price: float
    size: float
    size_usd: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    strategy: str = ""
    timestamp: str


class PaperEquityCurvePoint(BaseModel):
    timestamp: str
    equity: float


class PaperStrategyBreakdown(BaseModel):
    strategy_name: str
    total_trades: int
    realized_pnl: float
    active_position: Optional[PaperPosition] = None


class PaperStatsResponse(BaseModel):
    mode: str
    balance: float
    initial_balance: float
    total_return_pct: float
    total_realized_pnl: float
    peak_balance: float
    max_drawdown_pct: float
    total_executions: int
    total_trades: int
    success_rate: float
    active_positions: int
    positions: List[PaperPosition]
    equity_curve: List[PaperEquityCurvePoint]
    strategy_breakdown: List[PaperStrategyBreakdown]


class PaperPositionsResponse(BaseModel):
    total: int
    positions: List[PaperPosition]


class PaperTradesResponse(BaseModel):
    total: int
    trades: List[PaperTrade]


# ──────────────────────────────────────────────
# Aggregate Strategy Performance
# ──────────────────────────────────────────────

class StrategyPerformanceItem(BaseModel):
    name: str
    strategy_type: str
    symbol: str
    status: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    avg_trade_pnl: float
    profit_factor: float
    edge_confidence_score: float = 0.0


class AggregatePerformanceResponse(BaseModel):
    total_strategies: int
    running_strategies: int
    total_pnl: float
    total_trades: int
    overall_win_rate: float
    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None
    strategies: List[StrategyPerformanceItem]


# ──────────────────────────────────────────────
# Strategy Trades & Signals
# ──────────────────────────────────────────────

class StrategyTradeItem(BaseModel):
    id: int
    symbol: str
    side: str
    action: str
    price: float
    size: float
    size_usd: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    timestamp: str


class StrategyTradesResponse(BaseModel):
    strategy_name: str
    total: int
    trades: List[StrategyTradeItem]


class SignalLogItem(BaseModel):
    signal_type: str
    symbol: str
    strength: float
    price: Optional[float] = None
    reason: str = ""
    timestamp: str


class StrategySignalsResponse(BaseModel):
    strategy_name: str
    total: int
    signals: List[SignalLogItem]


# ──────────────────────────────────────────────
# Strategy Optimization
# ──────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    param_ranges: Dict[str, List[Any]]
    lookback_days: int = 90
    initial_capital: float = 10000.0
    max_combinations: int = 50


class OptimizeResponse(BaseModel):
    strategy_name: str
    total_combinations: int
    tested: int
    best_params: Dict[str, Any]
    best_sharpe: float
    best_return_pct: float
    best_win_rate: float
    top_results: List[Dict[str, Any]]


# ──────────────────────────────────────────────
# Portfolio Summary & Allocation
# ──────────────────────────────────────────────

class PortfolioSummary(BaseModel):
    trading_mode: str
    total_equity: float
    initial_equity: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    total_strategies: int
    running_strategies: int
    total_trades: int
    win_rate: float
    active_positions: int
    total_exposure_usd: float


class AllocationItem(BaseModel):
    strategy_name: str
    strategy_type: str
    symbol: str
    status: str
    size_usd: float
    allocation_pct: float
    current_pnl: float


class PortfolioAllocationResponse(BaseModel):
    total_allocated_usd: float
    strategies: List[AllocationItem]


# ──────────────────────────────────────────────
# Market Overview
# ──────────────────────────────────────────────

class MarketSymbolOverview(BaseModel):
    symbol: str
    price: Optional[float] = None
    regime: str = "unknown"
    regime_confidence: float = 0.0
    volatility_regime: str = "unknown"
    is_volatile: bool = False
    recommended_strategies: List[str] = []


class MarketOverviewResponse(BaseModel):
    timestamp: str
    symbols: List[MarketSymbolOverview]


# ──────────────────────────────────────────────
# Enhanced Health Check
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    trading_mode: str
    registry_strategies: int
    registry_ok: bool
    orchestrator_ok: bool
    execution_stats: Optional[Dict[str, Any]] = None
