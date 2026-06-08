from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
    Boolean,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import enum


class StrategyStatus(str, enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


class TradeSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    balance = Column(Float, default=0.0)
    shares = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    deposits = relationship("Deposit", back_populates="user")
    withdrawals = relationship("Withdrawal", back_populates="user")


class Deposit(Base):
    __tablename__ = "deposits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    tx_hash = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="deposits")


class Withdrawal(Base):
    __tablename__ = "withdrawals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    shares_burned = Column(Float)
    tx_hash = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="withdrawals")


class VaultState(Base):
    __tablename__ = "vault_state"

    id = Column(Integer, primary_key=True, index=True)
    total_equity = Column(Float, default=0.0)
    total_shares = Column(Float, default=0.0)
    nav_per_share = Column(Float, default=1.0)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    size = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)
    strategy = Column(String, default="turtle")
    opened_at = Column(DateTime(timezone=True), server_default=func.now())
    closed_at = Column(DateTime(timezone=True), nullable=True)
    is_open = Column(Boolean, default=True)


class StrategyInstance(Base):
    """Per-strategy state — supports multiple concurrent strategies."""
    __tablename__ = "strategy_instances"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    strategy_type = Column(String, index=True)
    tier = Column(String, default="A")
    status = Column(String, default="stopped")
    symbol = Column(String, default="BTC")
    timeframe = Column(String, default="1h")
    leverage = Column(Integer, default=3)
    size_usd = Column(Float, default=100.0)
    max_positions = Column(Integer, default=1)
    target_pct = Column(Float, default=5.0)
    max_loss_pct = Column(Float, default=-10.0)
    lookback_days = Column(Integer, default=7)
    interval_seconds = Column(Integer, default=30)
    enabled = Column(Boolean, default=True)
    params = Column(JSON, default=dict)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    # MIGRATION: ALTER TABLE strategy_instances ADD COLUMN edge_confidence_score FLOAT DEFAULT 0.0;
    edge_confidence_score = Column(Float, default=0.0)
    iterations = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    last_signal = Column(String, nullable=True)
    last_signal_time = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ──────────────────────────────────────────────
# Liquidation & Whale Tracking
# ──────────────────────────────────────────────

class TrackedWhale(Base):
    """Whale address with user-defined labels and alert preferences."""
    __tablename__ = "tracked_whales"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, unique=True, index=True)
    label = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    alert_enabled = Column(Boolean, default=False)
    account_value = Column(Float, default=0.0)
    peak_account_value = Column(Float, default=0.0)
    is_blown_up = Column(Boolean, default=False)
    blown_up_at = Column(DateTime(timezone=True), nullable=True)
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    last_updated = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LiquidationEventRecord(Base):
    """Historical liquidation event for backtesting."""
    __tablename__ = "liquidation_events"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    size = Column(Float)
    size_usd = Column(Float)
    price = Column(Float)
    is_cascade = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class WhaleAlertRecord(Base):
    """Historical whale trade alert."""
    __tablename__ = "whale_alerts"

    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    label = Column(String, nullable=True)
    action = Column(String)  # opened, closed, increased, decreased
    symbol = Column(String, index=True)
    side = Column(String)
    size = Column(Float)
    size_usd = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class PromotionEventRecord(Base):
    """Durable log of every RBI run_cycle outcome — survives Railway redeploys."""
    __tablename__ = "rbi_promotion_events"

    id = Column(Integer, primary_key=True, index=True)
    strategy_type = Column(String, index=True, nullable=False)
    strategy_id = Column(Integer, nullable=False)
    timestamp = Column(String, nullable=False, index=True)
    promoted = Column(Boolean, nullable=False, default=False)
    reason = Column(String, nullable=False)
    before_params = Column(JSON, default=dict)
    after_params = Column(JSON, default=dict)
    before_metrics = Column(JSON, default=dict)
    after_metrics = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Keep legacy table for migration compatibility
class StrategyState(Base):
    __tablename__ = "strategy_state"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="turtle")
    status = Column(String, default="stopped")
    symbol = Column(String, default="BTC")
    timeframe = Column(String, default="1h")
    lookback_period = Column(Integer, default=55)
    atr_period = Column(Integer, default=20)
    atr_multiplier = Column(Float, default=2.0)
    leverage = Column(Integer, default=5)
    last_signal = Column(String, nullable=True)
    last_signal_time = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(String, nullable=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
