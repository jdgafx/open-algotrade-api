import os
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import models, schemas
from .database import engine, get_db
from .auth import get_password_hash, verify_password, create_access_token, require_current_user

models.Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

app = FastAPI(title="Open Algotrade API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_or_create_vault_state(db: Session) -> models.VaultState:
    vault = db.query(models.VaultState).first()
    if not vault:
        vault = models.VaultState(total_equity=0.0, total_shares=0.0, nav_per_share=1.0)
        db.add(vault)
        db.commit()
        db.refresh(vault)
    return vault


def _get_or_create_strategy_state(db: Session) -> models.StrategyState:
    strategy = db.query(models.StrategyState).first()
    if not strategy:
        strategy = models.StrategyState(name="turtle", status="stopped")
        db.add(strategy)
        db.commit()
        db.refresh(strategy)
    return strategy


def _get_live_vault_equity() -> Optional[float]:
    try:
        from ..vault.vault_manager import VaultManager

        manager = VaultManager()
        vault_addr = manager.create_vault_if_not_exists()
        return manager.get_vault_equity()
    except Exception as e:
        logger.warning(f"Could not fetch live vault equity: {e}")
        return None


def _get_live_positions() -> List[schemas.Position]:
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        import os

        network = os.getenv("HYPERLIQUID_NETWORK", "testnet").lower()
        vault_address = os.getenv("VAULT_ADDRESS")
        base_url = (
            constants.TESTNET_API_URL
            if network == "testnet"
            else constants.MAINNET_API_URL
        )

        if not vault_address:
            return []

        info = Info(base_url, skip_ws=True)
        user_state = info.user_state(vault_address)
        positions = []

        for pos in user_state.get("assetPositions", []):
            p = pos.get("position", {})
            size = float(p.get("szi", 0))
            if size == 0:
                continue
            positions.append(
                schemas.Position(
                    symbol=p.get("coin", ""),
                    size=abs(size),
                    side="long" if size > 0 else "short",
                    entry_price=float(p.get("entryPx", 0)),
                    mark_price=float(p.get("markPx", 0) if p.get("markPx") else 0),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                    leverage=int(float(p.get("leverage", {}).get("value", 1))),
                )
            )
        return positions
    except Exception as e:
        logger.warning(f"Could not fetch live positions: {e}")
        return []


def _get_market_price(symbol: str) -> Optional[schemas.MarketPrice]:
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        import os

        network = os.getenv("HYPERLIQUID_NETWORK", "testnet").lower()
        base_url = (
            constants.TESTNET_API_URL
            if network == "testnet"
            else constants.MAINNET_API_URL
        )
        info = Info(base_url, skip_ws=True)

        mids = info.all_mids()
        price = float(mids.get(symbol, 0))

        meta = info.meta()
        funding_rate = None
        for asset in meta.get("universe", []):
            if asset.get("name") == symbol:
                funding_rate = float(asset.get("fundingRate", 0))
                break

        return schemas.MarketPrice(
            symbol=symbol, price=price, funding_rate=funding_rate
        )
    except Exception as e:
        logger.warning(f"Could not fetch market price for {symbol}: {e}")
        return None


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "open-algotrade-api", "version": "2.0.0"}


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────

@app.post("/auth/sign-in", response_model=schemas.AuthResponse)
def auth_sign_in(credentials: schemas.AuthSignIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return schemas.AuthResponse(
        id=str(user.id),
        email=user.email,
        name=user.username,
        avatar=None,
        status="ONLINE",
    )


@app.post("/auth/register", response_model=schemas.AuthResponse)
def auth_register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(models.User).filter(models.User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    db_user = models.User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return schemas.AuthResponse(
        id=str(db_user.id),
        email=db_user.email,
        name=db_user.username,
        avatar=None,
        status="ONLINE",
    )


@app.get("/auth/me", response_model=schemas.AuthResponse)
async def auth_me(user: models.User = Depends(require_current_user)):
    return schemas.AuthResponse(
        id=str(user.id),
        email=user.email,
        name=user.username,
        avatar=None,
        status="ONLINE",
    )


# ──────────────────────────────────────────────
# Users
# ──────────────────────────────────────────────

@app.post("/users", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    db_user = models.User(username=user.username, email=user.email, hashed_password=get_password_hash(user.password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/users/{user_id}", response_model=schemas.User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ──────────────────────────────────────────────
# Vault
# ──────────────────────────────────────────────

@app.get("/vault/status", response_model=schemas.VaultStatus)
def vault_status(db: Session = Depends(get_db)):
    vault = _get_or_create_vault_state(db)
    live_equity = _get_live_vault_equity()
    if live_equity is not None and vault.total_shares > 0:
        vault.total_equity = live_equity
        vault.nav_per_share = live_equity / vault.total_shares
        db.commit()
        db.refresh(vault)
    return schemas.VaultStatus(
        total_equity=vault.total_equity,
        total_shares=vault.total_shares,
        nav_per_share=vault.nav_per_share,
        live_equity=live_equity,
        updated_at=vault.updated_at,
    )


@app.post("/deposit", response_model=schemas.User)
def deposit(deposit: schemas.DepositCreate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == deposit.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    vault = _get_or_create_vault_state(db)

    share_price = vault.nav_per_share if vault.nav_per_share > 0 else 1.0
    shares_to_issue = deposit.amount / share_price

    user.balance += deposit.amount
    user.shares += shares_to_issue

    vault.total_equity += deposit.amount
    vault.total_shares += shares_to_issue
    vault.nav_per_share = vault.total_equity / vault.total_shares

    db_deposit = models.Deposit(
        user_id=user.id, amount=deposit.amount, tx_hash=deposit.tx_hash
    )
    db.add(db_deposit)
    db.commit()
    db.refresh(user)
    return user


@app.post("/withdraw", response_model=schemas.Withdrawal)
def withdraw(withdrawal: schemas.WithdrawCreate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == withdrawal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.shares < withdrawal.shares_to_redeem:
        raise HTTPException(status_code=400, detail="Insufficient shares")

    vault = _get_or_create_vault_state(db)
    nav = vault.nav_per_share if vault.nav_per_share > 0 else 1.0
    usd_amount = withdrawal.shares_to_redeem * nav

    user.shares -= withdrawal.shares_to_redeem
    user.balance -= min(usd_amount, user.balance)
    vault.total_shares -= withdrawal.shares_to_redeem
    vault.total_equity -= usd_amount
    if vault.total_shares > 0:
        vault.nav_per_share = vault.total_equity / vault.total_shares

    db_withdrawal = models.Withdrawal(
        user_id=user.id, amount=usd_amount, shares_burned=withdrawal.shares_to_redeem
    )
    db.add(db_withdrawal)
    db.commit()
    db.refresh(db_withdrawal)
    return db_withdrawal


@app.get("/portfolio/{user_id}", response_model=schemas.Portfolio)
def get_portfolio(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    vault = _get_or_create_vault_state(db)
    nav = vault.nav_per_share if vault.nav_per_share > 0 else 1.0
    portfolio_value = user.shares * nav
    unrealized_pnl = portfolio_value - user.balance
    pnl_percent = (unrealized_pnl / user.balance * 100) if user.balance > 0 else 0.0

    return schemas.Portfolio(
        user_id=user.id,
        username=user.username,
        shares=user.shares,
        nav_per_share=nav,
        portfolio_value=portfolio_value,
        total_deposited=user.balance,
        unrealized_pnl=unrealized_pnl,
        pnl_percent=pnl_percent,
    )


# ──────────────────────────────────────────────
# Positions & Trades
# ──────────────────────────────────────────────

@app.get("/positions", response_model=List[schemas.Position])
def get_positions():
    return _get_live_positions()


@app.get("/trades", response_model=List[schemas.TradeOut])
def get_trades(limit: int = 50, open_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(models.Trade)
    if open_only:
        query = query.filter(models.Trade.is_open == True)
    return query.order_by(models.Trade.opened_at.desc()).limit(limit).all()


# ──────────────────────────────────────────────
# Legacy single-strategy endpoints (backwards compat)
# ──────────────────────────────────────────────

@app.get("/strategy/status", response_model=schemas.StrategyStatusOut)
def strategy_status(db: Session = Depends(get_db)):
    return _get_or_create_strategy_state(db)


@app.post("/strategy/start")
def strategy_start(config: schemas.StrategyConfig, db: Session = Depends(get_db)):
    strategy = _get_or_create_strategy_state(db)
    strategy.status = "running"
    strategy.symbol = config.symbol
    strategy.timeframe = config.timeframe
    strategy.lookback_period = config.lookback_period
    strategy.atr_period = config.atr_period
    strategy.atr_multiplier = config.atr_multiplier
    strategy.leverage = config.leverage
    strategy.started_at = datetime.utcnow()
    strategy.error_message = None
    db.commit()
    return {"status": "running", "symbol": config.symbol}


@app.post("/strategy/stop")
def strategy_stop(db: Session = Depends(get_db)):
    strategy = _get_or_create_strategy_state(db)
    strategy.status = "stopped"
    db.commit()
    return {"status": "stopped"}


# ──────────────────────────────────────────────
# Multi-Strategy Endpoints (v2)
# ──────────────────────────────────────────────

@app.get("/strategies/registry", response_model=schemas.StrategyRegistryOut)
def get_strategy_registry():
    """List all available strategy types with their default configs."""
    from src.strategies.registry import list_strategies
    strategies = list_strategies()
    return schemas.StrategyRegistryOut(
        available_strategies=[
            schemas.StrategyTypeInfo(**s) for s in strategies
        ],
        total=len(strategies),
    )


@app.get("/strategies", response_model=List[schemas.StrategyInstanceOut])
def list_strategy_instances(db: Session = Depends(get_db)):
    """List all configured strategy instances."""
    instances = db.query(models.StrategyInstance).all()
    return instances


@app.post("/strategies", response_model=schemas.StrategyInstanceOut)
def create_strategy_instance(
    data: schemas.StrategyInstanceCreate, db: Session = Depends(get_db)
):
    """Create a new strategy instance."""
    from src.strategies.registry import get_strategy_class, list_strategies

    # Validate strategy type
    available = [s["strategy_type"] for s in list_strategies()]
    if data.strategy_type not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy type: {data.strategy_type}. Available: {available}",
        )

    # Check name uniqueness
    existing = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == data.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Strategy name '{data.name}' already exists")

    # Get tier from registry
    registry_info = next(s for s in list_strategies() if s["strategy_type"] == data.strategy_type)

    instance = models.StrategyInstance(
        name=data.name,
        strategy_type=data.strategy_type,
        tier=registry_info["tier"],
        symbol=data.symbol,
        timeframe=data.timeframe,
        leverage=data.leverage,
        size_usd=data.size_usd,
        target_pct=data.target_pct,
        max_loss_pct=data.max_loss_pct,
        lookback_days=data.lookback_days,
        interval_seconds=data.interval_seconds,
        enabled=data.enabled,
        params={**registry_info["default_params"], **data.params},
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


@app.get("/strategies/{name}", response_model=schemas.StrategyInstanceOut)
def get_strategy_instance(name: str, db: Session = Depends(get_db)):
    """Get a specific strategy instance by name."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    return instance


@app.patch("/strategies/{name}", response_model=schemas.StrategyInstanceOut)
def update_strategy_instance(
    name: str, data: schemas.StrategyInstanceUpdate, db: Session = Depends(get_db)
):
    """Update a strategy instance's configuration."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "params" and value is not None:
            current_params = instance.params or {}
            current_params.update(value)
            instance.params = current_params
        else:
            setattr(instance, field, value)

    db.commit()
    db.refresh(instance)
    return instance


@app.delete("/strategies/{name}")
def delete_strategy_instance(name: str, db: Session = Depends(get_db)):
    """Delete a strategy instance."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    if instance.status == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running strategy. Stop it first.")
    db.delete(instance)
    db.commit()
    return {"status": "deleted", "name": name}


@app.post("/strategies/{name}/start")
def start_strategy_instance(name: str, db: Session = Depends(get_db)):
    """Start a strategy instance."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    instance.status = "running"
    instance.started_at = datetime.utcnow()
    instance.error_message = None
    db.commit()
    return {"status": "running", "name": name, "strategy_type": instance.strategy_type, "symbol": instance.symbol}


@app.post("/strategies/{name}/stop")
def stop_strategy_instance(name: str, db: Session = Depends(get_db)):
    """Stop a strategy instance."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    instance.status = "stopped"
    db.commit()
    return {"status": "stopped", "name": name}


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────

@app.get("/dashboard/stats", response_model=schemas.DashboardStats)
def dashboard_stats(db: Session = Depends(get_db)):
    """Get aggregated dashboard statistics."""
    instances = db.query(models.StrategyInstance).all()
    total = len(instances)
    running = sum(1 for i in instances if i.status == "running")
    total_pnl = sum(i.total_pnl for i in instances)
    total_trades = sum(i.total_trades for i in instances)
    winning = sum(i.winning_trades for i in instances)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0

    live_equity = _get_live_vault_equity()
    positions = _get_live_positions()

    return schemas.DashboardStats(
        total_strategies=total,
        running_strategies=running,
        total_pnl=round(total_pnl, 2),
        total_trades=total_trades,
        win_rate=round(win_rate, 1),
        vault_equity=live_equity,
        active_positions=len(positions),
    )


# ──────────────────────────────────────────────
# Market Data
# ──────────────────────────────────────────────

@app.get("/market/price/{symbol}", response_model=schemas.MarketPrice)
def market_price(symbol: str):
    price = _get_market_price(symbol.upper())
    if not price:
        raise HTTPException(
            status_code=503, detail=f"Could not fetch price for {symbol}"
        )
    return price


@app.get("/deposits/{user_id}", response_model=List[schemas.Deposit])
def get_deposits(user_id: int, db: Session = Depends(get_db)):
    """Get deposit history for a user."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.query(models.Deposit).filter(
        models.Deposit.user_id == user_id
    ).order_by(models.Deposit.timestamp.desc()).all()
