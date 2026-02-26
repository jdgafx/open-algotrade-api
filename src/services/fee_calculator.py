"""
Fee Calculator — "Days Until Death"

Calculates how quickly trading fees will eat your account.
Uses HyperLiquid's fee structure (maker/taker) to project fee burn rate.

This is a critical awareness tool — most traders don't realize fees compound.
"""

import logging
from typing import Optional

from src.services.risk_models import FeeCalculatorInput, FeeCalculatorOutput

logger = logging.getLogger(__name__)


def calculate_fees(input: FeeCalculatorInput) -> FeeCalculatorOutput:
    """
    Calculate projected fee costs and 'days until death'.

    HyperLiquid fee structure (default):
    - Maker: 0.02% (0.2 bps)
    - Taker: 0.035% (3.5 bps)

    A trade involves both open and close, so fees are 2x per round-trip.
    """
    # Blended fee rate (weighted average of maker/taker)
    taker_ratio = 1.0 - input.maker_ratio
    blended_fee_bps = (
        input.maker_fee_bps * input.maker_ratio
        + input.taker_fee_bps * taker_ratio
    )
    blended_fee_pct = blended_fee_bps / 100.0  # Convert bps to %

    # Fee per trade (open + close = 2x the fee)
    avg_fee_per_trade = input.avg_position_size * (blended_fee_pct / 100.0) * 2

    # Daily/weekly/monthly/yearly costs
    daily_fee_cost = avg_fee_per_trade * input.avg_trades_per_day
    weekly_fee_cost = daily_fee_cost * 7
    monthly_fee_cost = daily_fee_cost * 30
    yearly_fee_cost = daily_fee_cost * 365

    # Fee as % of balance
    fee_pct_daily = (
        (daily_fee_cost / input.account_balance * 100)
        if input.account_balance > 0
        else 0.0
    )
    fee_pct_monthly = (
        (monthly_fee_cost / input.account_balance * 100)
        if input.account_balance > 0
        else 0.0
    )

    # Days until fees eat X% of account
    def days_until_pct(pct: float) -> Optional[float]:
        if daily_fee_cost <= 0 or input.account_balance <= 0:
            return None
        target = input.account_balance * (pct / 100.0)
        return round(target / daily_fee_cost, 1)

    days_10 = days_until_pct(10)
    days_25 = days_until_pct(25)
    days_50 = days_until_pct(50)
    days_100 = days_until_pct(100)

    # Warning generation
    warning = None
    if days_100 is not None and days_100 < 365:
        if days_100 < 30:
            warning = f"CRITICAL: Fees will consume your entire account in {days_100:.0f} days at this rate."
        elif days_100 < 90:
            warning = f"WARNING: Fees will consume your entire account in {days_100:.0f} days. Consider reducing trade frequency or size."
        elif days_100 < 365:
            warning = f"CAUTION: Fees will consume your entire account in {days_100:.0f} days. You need to outperform fees by {fee_pct_monthly:.1f}%/month."

    return FeeCalculatorOutput(
        daily_fee_cost=round(daily_fee_cost, 2),
        weekly_fee_cost=round(weekly_fee_cost, 2),
        monthly_fee_cost=round(monthly_fee_cost, 2),
        yearly_fee_cost=round(yearly_fee_cost, 2),
        fee_pct_of_balance_daily=round(fee_pct_daily, 4),
        fee_pct_of_balance_monthly=round(fee_pct_monthly, 2),
        days_until_10pct_eaten=days_10,
        days_until_25pct_eaten=days_25,
        days_until_50pct_eaten=days_50,
        days_until_100pct_eaten=days_100,
        avg_fee_per_trade=round(avg_fee_per_trade, 4),
        blended_fee_bps=round(blended_fee_bps, 2),
        warning=warning,
    )
