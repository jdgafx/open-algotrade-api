from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


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
