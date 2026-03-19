"""
Liquidation Guard — Pre-checks entries to reject trades where the
liquidation price is dangerously close to the current entry price.

Hyperliquid liquidation formula (simplified):
    liq_price = entry_price - side * margin / position_size / (1 - maint_ratio * side)

Where:
    side = +1 for long, -1 for short
    margin = position_notional / leverage
    maint_ratio = maintenance_margin_ratio (half of initial margin at max leverage)

Maintenance margin tiers (mainnet):
    BTC  40x max → 1.25%
    ETH  25x max → 2.00%
    SOL  20x max → 2.50%
    Mid  10x max → 5.00%
    Low   3x max → 16.7%
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Maintenance margin ratios by symbol.
# Derived from Hyperliquid max-leverage tiers: maint = 1 / (2 * max_leverage).
MAINTENANCE_MARGIN_MAP: Dict[str, float] = {
    # Tier 1 — 40x max
    "BTC": 0.0125,
    "BTC-PERP": 0.0125,
    # Tier 2 — 25x max
    "ETH": 0.02,
    "ETH-PERP": 0.02,
    # Tier 3 — 20x max
    "SOL": 0.025,
    "SOL-PERP": 0.025,
    "AVAX": 0.025,
    "AVAX-PERP": 0.025,
    "MATIC": 0.025,
    "MATIC-PERP": 0.025,
    "BNB": 0.025,
    "BNB-PERP": 0.025,
    "ARB": 0.025,
    "ARB-PERP": 0.025,
    "OP": 0.025,
    "OP-PERP": 0.025,
    "DOGE": 0.025,
    "DOGE-PERP": 0.025,
    "LINK": 0.025,
    "LINK-PERP": 0.025,
    # Tier 4 — 10x max (mid-cap)
    "ATOM": 0.05,
    "ATOM-PERP": 0.05,
    "DOT": 0.05,
    "DOT-PERP": 0.05,
    "FTM": 0.05,
    "FTM-PERP": 0.05,
    "NEAR": 0.05,
    "NEAR-PERP": 0.05,
    "APE": 0.05,
    "APE-PERP": 0.05,
    "UNI": 0.05,
    "UNI-PERP": 0.05,
    "AAVE": 0.05,
    "AAVE-PERP": 0.05,
    "LTC": 0.05,
    "LTC-PERP": 0.05,
    "CRV": 0.05,
    "CRV-PERP": 0.05,
    "MKR": 0.05,
    "MKR-PERP": 0.05,
    "TIA": 0.05,
    "TIA-PERP": 0.05,
    "INJ": 0.05,
    "INJ-PERP": 0.05,
    "SEI": 0.05,
    "SEI-PERP": 0.05,
    "SUI": 0.05,
    "SUI-PERP": 0.05,
    "JUP": 0.05,
    "JUP-PERP": 0.05,
    "WIF": 0.05,
    "WIF-PERP": 0.05,
    "PYTH": 0.05,
    "PYTH-PERP": 0.05,
    "ONDO": 0.05,
    "ONDO-PERP": 0.05,
    "RENDER": 0.05,
    "RENDER-PERP": 0.05,
    "FET": 0.05,
    "FET-PERP": 0.05,
    "PEPE": 0.05,
    "PEPE-PERP": 0.05,
    "WLD": 0.05,
    "WLD-PERP": 0.05,
    "PENDLE": 0.05,
    "PENDLE-PERP": 0.05,
    "STX": 0.05,
    "STX-PERP": 0.05,
}

# Fallback for unknown symbols — conservative 3x (low-tier)
DEFAULT_MAINTENANCE_MARGIN = 0.167


@dataclass
class LiquidationCheckResult:
    """Result of a pre-entry liquidation safety check."""
    is_safe: bool
    liq_price: float
    distance_pct: float
    reason: str


class LiquidationGuard:
    """
    Pre-entry liquidation safety gate.

    Calculates the estimated liquidation price for a prospective trade and
    rejects entries where the liq price is too close to the entry price.

    "Too close" means:
      - Within ``min_distance_atr_multiple`` * ATR (if ATR is available)
      - Within ``min_distance_pct`` of entry price (fallback when no ATR)
    """

    def __init__(
        self,
        min_distance_pct: float = 5.0,
        min_distance_atr_multiple: float = 2.0,
    ):
        self.min_distance_pct = min_distance_pct
        self.min_distance_atr_multiple = min_distance_atr_multiple
        self._blocks: int = 0

        logger.info(
            "LiquidationGuard initialized | min_dist_pct=%.1f%% | min_dist_atr_mult=%.1f",
            min_distance_pct,
            min_distance_atr_multiple,
        )

    @staticmethod
    def get_maintenance_margin(symbol: str) -> float:
        """Look up the maintenance margin ratio for a symbol."""
        # Normalize: strip common suffixes so "BTC-PERP" and "BTC" both match
        base = symbol.upper().replace("-PERP", "").replace("/USD", "").replace("USDT", "")
        # Try exact match first, then base symbol
        if symbol.upper() in MAINTENANCE_MARGIN_MAP:
            return MAINTENANCE_MARGIN_MAP[symbol.upper()]
        if base in MAINTENANCE_MARGIN_MAP:
            return MAINTENANCE_MARGIN_MAP[base]
        return DEFAULT_MAINTENANCE_MARGIN

    @staticmethod
    def calculate_liquidation_price(
        entry_price: float,
        position_size: float,
        margin: float,
        leverage: float,
        is_long: bool,
        symbol: str = "",
    ) -> float:
        """
        Estimate the liquidation price using the Hyperliquid formula.

        Parameters
        ----------
        entry_price : float
            Price at which the position is opened.
        position_size : float
            Size of the position in base asset units (absolute value).
        margin : float
            Collateral backing the position (USD).
        leverage : float
            Leverage multiplier used for the position.
        is_long : bool
            True for long, False for short.
        symbol : str
            Trading symbol, used to look up maintenance margin tier.

        Returns
        -------
        float
            Estimated liquidation price.
        """
        if position_size == 0 or entry_price <= 0:
            return 0.0

        maint_ratio = LiquidationGuard.get_maintenance_margin(symbol)
        side = 1.0 if is_long else -1.0
        position_size = abs(position_size)

        # liq_price = entry - side * margin / size / (1 - maint_ratio * side)
        denominator = 1.0 - maint_ratio * side
        if abs(denominator) < 1e-12:
            # Degenerate case — treat as extremely close liquidation
            return entry_price

        liq_price = entry_price - side * (margin / position_size) / denominator

        # Liquidation price can never go below 0 for longs
        return max(liq_price, 0.0)

    def is_entry_safe(
        self,
        entry_price: float,
        leverage: float,
        is_long: bool,
        symbol: str,
        atr_value: Optional[float] = None,
        min_distance_atr_multiple: Optional[float] = None,
    ) -> Tuple[bool, float, float, str]:
        """
        Check whether a proposed entry is safe from a liquidation standpoint.

        Builds a hypothetical position of 1 unit at ``entry_price`` with
        ``leverage`` and calculates where liquidation would occur.  Then
        compares the distance to ATR-based or percentage-based thresholds.

        Parameters
        ----------
        entry_price : float
            Proposed entry price.
        leverage : float
            Leverage to be used.
        is_long : bool
            Direction of the trade.
        symbol : str
            Symbol being traded.
        atr_value : float, optional
            Current ATR value for the symbol (preferred threshold).
        min_distance_atr_multiple : float, optional
            Override the instance-level ATR multiple for this check.

        Returns
        -------
        tuple of (is_safe, liq_price, distance_pct, reason)
        """
        if entry_price <= 0 or leverage <= 0:
            return (False, 0.0, 0.0, "invalid entry_price or leverage")

        # Build a hypothetical 1-unit position
        position_size = 1.0
        notional = entry_price * position_size
        margin = notional / leverage

        liq_price = self.calculate_liquidation_price(
            entry_price=entry_price,
            position_size=position_size,
            margin=margin,
            leverage=leverage,
            is_long=is_long,
            symbol=symbol,
        )

        distance = abs(entry_price - liq_price)
        distance_pct = (distance / entry_price) * 100.0 if entry_price > 0 else 0.0

        # Determine the safety threshold
        atr_mult = min_distance_atr_multiple or self.min_distance_atr_multiple

        if atr_value is not None and atr_value > 0:
            required_distance = atr_mult * atr_value
            required_distance_pct = (required_distance / entry_price) * 100.0
            threshold_label = f"{atr_mult:.1f}x ATR (${atr_value:.2f})"
        else:
            required_distance = entry_price * (self.min_distance_pct / 100.0)
            required_distance_pct = self.min_distance_pct
            threshold_label = f"{self.min_distance_pct:.1f}% of entry"

        is_safe = distance >= required_distance

        if is_safe:
            reason = (
                f"OK | liq distance {distance_pct:.2f}% >= threshold {required_distance_pct:.2f}% "
                f"({threshold_label})"
            )
        else:
            self._blocks += 1
            reason = (
                f"TOO CLOSE | liq distance {distance_pct:.2f}% < threshold {required_distance_pct:.2f}% "
                f"({threshold_label}) | leverage={leverage}x | total_blocks={self._blocks}"
            )

        return (is_safe, liq_price, distance_pct, reason)

    @property
    def total_blocks(self) -> int:
        return self._blocks

    def get_stats(self) -> Dict:
        return {
            "total_blocks": self._blocks,
            "min_distance_pct": self.min_distance_pct,
            "min_distance_atr_multiple": self.min_distance_atr_multiple,
        }
