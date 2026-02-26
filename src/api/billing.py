"""
Subscription / Tier System — Stripe integration stubs.

Tiers:
  - Free:       3 strategies, backtesting with delays, community support
  - Pro ($9):   Unlimited strategies, instant backtesting, priority execution
  - Enterprise ($49): Custom strategies, API access, dedicated vault

Stripe keys are loaded from environment variables.
"""

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db
from . import models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])

# Stripe configuration (stubs — set these in .env for production)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Tier limits
TIER_LIMITS = {
    SubscriptionTier.FREE: {
        "max_strategies": 3,
        "backtest_delay_seconds": 30,
        "instant_backtest": False,
        "priority_execution": False,
        "custom_strategies": False,
        "api_access": False,
        "dedicated_vault": False,
        "price_monthly": 0,
    },
    SubscriptionTier.PRO: {
        "max_strategies": 999,
        "backtest_delay_seconds": 0,
        "instant_backtest": True,
        "priority_execution": True,
        "custom_strategies": False,
        "api_access": False,
        "dedicated_vault": False,
        "price_monthly": 9,
    },
    SubscriptionTier.ENTERPRISE: {
        "max_strategies": 999,
        "backtest_delay_seconds": 0,
        "instant_backtest": True,
        "priority_execution": True,
        "custom_strategies": True,
        "api_access": True,
        "dedicated_vault": True,
        "price_monthly": 49,
    },
}

# Stripe price IDs (set these when you create products in Stripe dashboard)
STRIPE_PRICE_IDS = {
    SubscriptionTier.PRO: os.getenv("STRIPE_PRO_PRICE_ID", "price_pro_placeholder"),
    SubscriptionTier.ENTERPRISE: os.getenv("STRIPE_ENTERPRISE_PRICE_ID", "price_enterprise_placeholder"),
}


# ── Schemas ──────────────────────────────────────────


class TierInfo(BaseModel):
    tier: str
    max_strategies: int
    backtest_delay_seconds: int
    instant_backtest: bool
    priority_execution: bool
    custom_strategies: bool
    api_access: bool
    dedicated_vault: bool
    price_monthly: int


class TiersResponse(BaseModel):
    tiers: list[TierInfo]


class SubscriptionStatus(BaseModel):
    user_id: int
    tier: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_period_end: Optional[str] = None
    limits: Dict[str, Any]


class CreateCheckoutRequest(BaseModel):
    tier: str
    success_url: str = "https://app.openalgotrade.com/billing/success"
    cancel_url: str = "https://app.openalgotrade.com/billing/cancel"


class CheckoutResponse(BaseModel):
    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    portal_url: str


# ── Helper Functions ──────────────────────────────────


def get_user_tier(user: models.User) -> SubscriptionTier:
    """Get user's current subscription tier."""
    tier_str = getattr(user, "subscription_tier", "free")
    try:
        return SubscriptionTier(tier_str)
    except ValueError:
        return SubscriptionTier.FREE


def check_strategy_limit(user: models.User, current_count: int) -> bool:
    """Check if user can create another strategy."""
    tier = get_user_tier(user)
    limits = TIER_LIMITS[tier]
    return current_count < limits["max_strategies"]


def get_backtest_delay(user: models.User) -> int:
    """Get backtest delay for user's tier (seconds)."""
    tier = get_user_tier(user)
    return TIER_LIMITS[tier]["backtest_delay_seconds"]


# ── Stripe Stub Functions ────────────────────────────


def _get_stripe():
    """Lazy-load stripe module."""
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        return stripe
    except ImportError:
        return None


def _create_stripe_customer(email: str, name: str) -> Optional[str]:
    """Create a Stripe customer. Returns customer ID or None."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        logger.warning("Stripe not configured — returning stub customer ID")
        return f"cus_stub_{name}"

    try:
        customer = stripe.Customer.create(email=email, name=name)
        return customer.id
    except Exception as e:
        logger.error("Stripe customer creation failed: %s", e)
        return None


def _create_checkout_session(
    customer_id: str, price_id: str, success_url: str, cancel_url: str
) -> Optional[Dict[str, str]]:
    """Create a Stripe Checkout session. Returns {url, id} or None."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        logger.warning("Stripe not configured — returning stub checkout URL")
        return {
            "url": f"{success_url}?session_id=stub_session",
            "id": "stub_session",
        }

    try:
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
        )
        return {"url": session.url, "id": session.id}
    except Exception as e:
        logger.error("Stripe checkout creation failed: %s", e)
        return None


def _create_billing_portal(customer_id: str, return_url: str) -> Optional[str]:
    """Create a Stripe Billing Portal session. Returns URL or None."""
    stripe = _get_stripe()
    if not stripe or not STRIPE_SECRET_KEY:
        return f"{return_url}?portal=stub"

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id, return_url=return_url
        )
        return session.url
    except Exception as e:
        logger.error("Stripe portal creation failed: %s", e)
        return None


# ── API Endpoints ────────────────────────────────────


@router.get("/tiers", response_model=TiersResponse)
def list_tiers():
    """List all available subscription tiers and their limits."""
    tiers = []
    for tier, limits in TIER_LIMITS.items():
        tiers.append(TierInfo(tier=tier.value, **limits))
    return TiersResponse(tiers=tiers)


@router.get("/subscription/{user_id}", response_model=SubscriptionStatus)
def get_subscription(user_id: int, db: Session = Depends(get_db)):
    """Get a user's current subscription status."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tier = get_user_tier(user)
    return SubscriptionStatus(
        user_id=user.id,
        tier=tier.value,
        stripe_customer_id=getattr(user, "stripe_customer_id", None),
        stripe_subscription_id=getattr(user, "stripe_subscription_id", None),
        current_period_end=None,
        limits=TIER_LIMITS[tier],
    )


@router.post("/checkout", response_model=CheckoutResponse)
def create_checkout(req: CreateCheckoutRequest, db: Session = Depends(get_db)):
    """Create a Stripe Checkout session for upgrading subscription."""
    try:
        tier = SubscriptionTier(req.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {req.tier}. Must be 'pro' or 'enterprise'")

    if tier == SubscriptionTier.FREE:
        raise HTTPException(status_code=400, detail="Cannot checkout for free tier")

    price_id = STRIPE_PRICE_IDS.get(tier)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"No price configured for tier: {tier.value}")

    # In production, get customer_id from authenticated user
    customer_id = "cus_stub"
    result = _create_checkout_session(customer_id, price_id, req.success_url, req.cancel_url)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

    return CheckoutResponse(checkout_url=result["url"], session_id=result["id"])


@router.post("/portal", response_model=PortalResponse)
def create_portal():
    """Create a Stripe Customer Portal session for managing subscription."""
    customer_id = "cus_stub"  # In production, get from authenticated user
    return_url = "https://app.openalgotrade.com/billing"

    url = _create_billing_portal(customer_id, return_url)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to create portal session")

    return PortalResponse(portal_url=url)


@router.post("/webhook")
async def stripe_webhook():
    """
    Handle Stripe webhook events.

    In production, verify the webhook signature and process:
    - checkout.session.completed -> activate subscription
    - customer.subscription.updated -> update tier
    - customer.subscription.deleted -> downgrade to free
    - invoice.payment_failed -> handle failed payment
    """
    # Stub: In production, parse and verify the webhook payload
    return {"status": "received"}
