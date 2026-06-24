"""
Risk Models — Pydantic models for the Risk Controller (Layer 0: The Seatbelt).

Defines risk configuration, event types, and status models used across
the risk controller service and API routes.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class RiskEventType(str, Enum):
    MAX_LOSS_HIT = "MAX_LOSS_HIT"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    MARGIN_EXCEEDED = "MARGIN_EXCEEDED"
    TILT_LOCKOUT = "TILT_LOCKOUT"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    TRAILING_STOP_HIT = "TRAILING_STOP_HIT"
    LIQUIDATION_PAUSE = "LIQUIDATION_PAUSE"
    POSITION_REDUCED = "POSITION_REDUCED"
    ALL_POSITIONS_CLOSED = "ALL_POSITIONS_CLOSED"
    AUTO_WITHDRAWAL = "AUTO_WITHDRAWAL"
    MONITORING_STARTED = "MONITORING_STARTED"
    MONITORING_STOPPED = "MONITORING_STOPPED"
    CONFIG_UPDATED = "CONFIG_UPDATED"
    # Multi-tier drawdown cascade events
    TIER1_REDUCE = "TIER1_REDUCE"
    TIER2_STOP_NEW = "TIER2_STOP_NEW"
    TIER3_CLOSE_ALL = "TIER3_CLOSE_ALL"
    WEEKLY_SHUTDOWN = "WEEKLY_SHUTDOWN"
    MONTHLY_SHUTDOWN = "MONTHLY_SHUTDOWN"
    TRAILING_DRAWDOWN_HALT = "TRAILING_DRAWDOWN_HALT"
    ABSOLUTE_FLOOR_HALT = "ABSOLUTE_FLOOR_HALT"


class RiskSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskStatus(str, Enum):
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    LOCKED_OUT = "locked_out"
    EMERGENCY = "emergency"


class RiskConfig(BaseModel):
    """Per-user risk parameters. This is what protects people's money."""

    max_daily_loss_pct: float = Field(
        default=3.0,
        ge=0.1,
        le=50.0,
        description="Max daily loss as % of account. Triggers close-all + withdrawal.",
    )
    max_leverage: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Max leverage per position. Auto-reduces if exceeded.",
    )
    max_margin_pct: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Max margin as % of account per position.",
    )
    stop_loss_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Global stop loss % on all positions.",
    )
    take_profit_pct: float = Field(
        default=10.0,
        ge=0.1,
        le=200.0,
        description="Global take profit % on all positions.",
    )
    trailing_stop_pct: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=50.0,
        description="Trailing stop that follows price. None = disabled.",
    )
    anti_tilt_lockout_hours: float = Field(
        default=12.0,
        ge=0.0,
        le=168.0,
        description="Hours to lock trading after hitting daily max loss. 0 = disabled.",
    )
    liquidation_pause_threshold: float = Field(
        default=250000.0,
        ge=0.0,
        description="Pause all trading if >$X liquidated in last 30min. 0 = disabled.",
    )
    auto_withdraw_on_max_loss: bool = Field(
        default=True,
        description="Auto-withdraw to wallet when max daily loss hit.",
    )
    check_interval_seconds: float = Field(
        default=3.0,
        ge=1.0,
        le=30.0,
        description="How often to check positions (seconds).",
    )

    # Multi-tier drawdown cascade (overrides simple max_daily_loss_pct when set)
    drawdown_tier1_pct: float = Field(
        default=1.0,
        ge=0.1,
        le=50.0,
        description="Tier 1: reduce position sizes by 50%",
    )
    drawdown_tier2_pct: float = Field(
        default=2.0,
        ge=0.1,
        le=50.0,
        description="Tier 2: stop opening new positions",
    )
    drawdown_tier3_pct: float = Field(
        default=3.0,
        ge=0.1,
        le=50.0,
        description="Tier 3: close ALL positions + lockout",
    )
    drawdown_weekly_max_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=100.0,
        description="Weekly max drawdown — 24h shutdown",
    )
    drawdown_monthly_max_pct: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Monthly max drawdown — full shutdown, manual review needed",
    )

    trailing_drawdown_from_peak_pct: float = Field(
        default=35.0, ge=0.1, le=100.0,
        description="Account-level halt: flatten + stop new entries if equity falls this % below its running peak.",
    )
    absolute_floor_usd: float = Field(
        default=50.0, ge=0.0,
        description="Hard give-up floor: flatten + lock out if account value drops to/below this $ amount. Set to 0 to disable.",
    )

    ruin_guard_buffer_pct: float = Field(
        default=1.0, ge=0.1, le=10.0,
        description="Per-position Ruin Guard: force-close when distance-to-liquidation drops below this %% (1%% ≈ one wick from liquidation).",
    )


class RiskEvent(BaseModel):
    """A single risk event logged by the controller."""

    id: int = 0
    event_type: RiskEventType
    severity: RiskSeverity
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True


class RiskSnapshot(BaseModel):
    """Current risk state snapshot — sent via WebSocket and REST."""

    status: RiskStatus
    account_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    start_of_day_equity: float = 0.0
    total_margin_used: float = 0.0
    margin_usage_pct: float = 0.0
    max_position_leverage: float = 0.0
    open_positions: int = 0
    lockout_until: Optional[datetime] = None
    trailing_stops: dict = Field(default_factory=dict)
    last_check: Optional[datetime] = None
    # Multi-tier drawdown state
    current_tier: int = Field(default=0, description="0=normal, 1=reduced, 2=no new, 3=closed")
    weekly_pnl_pct: float = 0.0
    monthly_pnl_pct: float = 0.0
    new_entries_blocked: bool = False
    # Account-level ruin guard state
    running_peak_equity: float = 0.0
    trailing_drawdown_from_peak_pct: float = 0.0


class RiskStatusResponse(BaseModel):
    """Full risk status response for the API."""

    config: RiskConfig
    snapshot: RiskSnapshot
    recent_events: List[RiskEvent] = []


class FeeCalculatorInput(BaseModel):
    """Input for the Days Until Death fee calculator."""

    account_balance: float = Field(ge=0, description="Current account balance in USD")
    avg_trades_per_day: float = Field(ge=0, description="Average number of trades per day")
    avg_position_size: float = Field(ge=0, description="Average position size in USD")
    maker_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Ratio of maker orders (0.0-1.0). Remainder is taker.",
    )
    maker_fee_bps: float = Field(
        default=0.2,
        description="Maker fee in basis points (HL default: 0.02%)",
    )
    taker_fee_bps: float = Field(
        default=3.5,
        description="Taker fee in basis points (HL default: 0.035%)",
    )


class FeeCalculatorOutput(BaseModel):
    """Output of the Days Until Death calculator."""

    daily_fee_cost: float
    weekly_fee_cost: float
    monthly_fee_cost: float
    yearly_fee_cost: float
    fee_pct_of_balance_daily: float
    fee_pct_of_balance_monthly: float
    days_until_10pct_eaten: Optional[float]
    days_until_25pct_eaten: Optional[float]
    days_until_50pct_eaten: Optional[float]
    days_until_100pct_eaten: Optional[float]
    avg_fee_per_trade: float
    blended_fee_bps: float
    warning: Optional[str] = None


def calculate_fees(input: "FeeCalculatorInput") -> "FeeCalculatorOutput":
    """HyperLiquid fee burn-rate calculator ('Days Until Death')."""
    taker_ratio = 1.0 - input.maker_ratio
    blended_fee_bps = (
        input.maker_fee_bps * input.maker_ratio
        + input.taker_fee_bps * taker_ratio
    )
    blended_fee_pct = blended_fee_bps / 100.0
    avg_fee_per_trade = input.avg_position_size * (blended_fee_pct / 100.0) * 2
    daily_fee_cost = avg_fee_per_trade * input.avg_trades_per_day
    weekly_fee_cost = daily_fee_cost * 7
    monthly_fee_cost = daily_fee_cost * 30
    yearly_fee_cost = daily_fee_cost * 365
    fee_pct_daily = (daily_fee_cost / input.account_balance * 100) if input.account_balance > 0 else 0.0
    fee_pct_monthly = (monthly_fee_cost / input.account_balance * 100) if input.account_balance > 0 else 0.0

    def _days_until(pct: float) -> Optional[float]:
        if daily_fee_cost <= 0 or input.account_balance <= 0:
            return None
        return round(input.account_balance * (pct / 100.0) / daily_fee_cost, 1)

    days_100 = _days_until(100)
    warning = None
    if days_100 is not None and days_100 < 365:
        if days_100 < 30:
            warning = f"CRITICAL: Fees will consume your entire account in {days_100:.0f} days at this rate."
        elif days_100 < 90:
            warning = f"WARNING: Fees will consume your entire account in {days_100:.0f} days. Consider reducing trade frequency or size."
        else:
            warning = f"CAUTION: Fees will consume your entire account in {days_100:.0f} days. You need to outperform fees by {fee_pct_monthly:.1f}%/month."

    return FeeCalculatorOutput(
        daily_fee_cost=round(daily_fee_cost, 2),
        weekly_fee_cost=round(weekly_fee_cost, 2),
        monthly_fee_cost=round(monthly_fee_cost, 2),
        yearly_fee_cost=round(yearly_fee_cost, 2),
        fee_pct_of_balance_daily=round(fee_pct_daily, 4),
        fee_pct_of_balance_monthly=round(fee_pct_monthly, 2),
        days_until_10pct_eaten=_days_until(10),
        days_until_25pct_eaten=_days_until(25),
        days_until_50pct_eaten=_days_until(50),
        days_until_100pct_eaten=days_100,
        avg_fee_per_trade=round(avg_fee_per_trade, 4),
        blended_fee_bps=round(blended_fee_bps, 2),
        warning=warning,
    )
