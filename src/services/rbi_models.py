"""
RBI Agent Models — Database models and Pydantic schemas for the
Research → Backtest → Implement pipeline.
"""

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, DateTime, Enum, Float, Integer, JSON, String, Text,
)
from sqlalchemy.sql import func

from src.api.database import Base


# ── Enums ────────────────────────────────────────

class RBIJobStatus(str, enum.Enum):
    PENDING = "pending"
    RESEARCHING = "researching"
    IDEATING = "ideating"
    CODING = "coding"
    DEBUGGING = "debugging"
    TESTING = "testing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"


class RBIInputType(str, enum.Enum):
    TEXT = "text"
    PDF = "pdf"
    YOUTUBE = "youtube"


# ── SQLAlchemy Models ────────────────────────────

class RBIJob(Base):
    __tablename__ = "rbi_jobs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default=RBIJobStatus.PENDING.value, index=True)
    input_type = Column(String, default=RBIInputType.TEXT.value)
    input_text = Column(Text, nullable=False)
    progress_pct = Column(Integer, default=0)
    progress_message = Column(String, default="")
    strategies_found = Column(Integer, default=0)
    error_message = Column(String, nullable=True)
    ai_provider = Column(String, default="anthropic")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class RBIStrategyResult(Base):
    __tablename__ = "rbi_strategy_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, index=True)
    strategy_name = Column(String)
    description = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    symbol = Column(String, default="BTC")
    timeframe = Column(String, default="1h")

    # Backtest metrics
    total_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    expectancy = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    avg_trade_pct = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)

    # Equity curve data (stored as JSON list of {timestamp, equity})
    equity_curve = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    is_optimized = Column(String, default="no")  # "no", "yes"
    rank = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ── Pydantic Schemas ─────────────────────────────

class RBIJobCreate(BaseModel):
    input_type: str = "text"
    input_text: str
    ai_provider: str = "anthropic"


class RBIJobOut(BaseModel):
    id: int
    status: str
    input_type: str
    input_text: str
    progress_pct: int
    progress_message: str
    strategies_found: int
    error_message: Optional[str] = None
    ai_provider: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RBIStrategyResultOut(BaseModel):
    id: int
    job_id: int
    strategy_name: str
    description: Optional[str] = None
    code: Optional[str] = None
    symbol: str
    timeframe: str
    total_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    expectancy: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: Optional[int] = None
    avg_trade_pct: Optional[float] = None
    calmar_ratio: Optional[float] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None
    is_optimized: str = "no"
    rank: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RBIJobResultsOut(BaseModel):
    job: RBIJobOut
    results: List[RBIStrategyResultOut]


class RBIDeployRequest(BaseModel):
    result_id: int
    symbol: str = "BTC"
    size_usd: float = 100.0
    leverage: int = 3


class RBIProgressUpdate(BaseModel):
    job_id: int
    status: str
    progress_pct: int
    message: str
