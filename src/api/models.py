from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
    Boolean,
    Enum,
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
