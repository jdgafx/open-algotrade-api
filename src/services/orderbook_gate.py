"""
Order-Book Imbalance Gate — Gate 4.7 in the trade entry pipeline.

Estimates +3-5% WR improvement by blocking entries where the order book is
stacked against the trade direction.

Gate logic:
- Fetch L2 order book (top 20 levels) via HyperliquidClient.get_l2_book()
- Compute dollar-weighted imbalance ratio = bid_depth / ask_depth
- Block long entries if imbalance_ratio < 0.8 (asks outweigh bids by 20%+)
- Block short entries if imbalance_ratio > 1.2 (bids outweigh asks by 20%+)

SHADOW_MODE = True (default): always allows the entry but logs what would have
been blocked — accumulates attribution data without affecting live trading.
Flip SHADOW_MODE = False to activate hard blocking.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────────────
SHADOW_MODE = True       # flip to False to activate hard blocking

LONG_BLOCK_RATIO = 0.8   # block long if bid_depth / ask_depth < this
SHORT_BLOCK_RATIO = 1.2  # block short if bid_depth / ask_depth > this
TOP_N_LEVELS = 20        # only consider the top N levels per side


@dataclass
class GateResult:
    allowed: bool
    imbalance_ratio: float  # bid_dollar_depth / ask_dollar_depth
    reason: str


@dataclass
class _SymbolStats:
    total_checked: int = 0
    would_block_count: int = 0


class OrderBookImbalanceGate:
    """Shadow-mode order-book imbalance gate (Gate 4.7)."""

    def __init__(self, client, shadow_mode: bool = SHADOW_MODE):
        self.client = client
        self.shadow_mode = shadow_mode
        # Per-symbol attribution counters
        self._stats: Dict[str, _SymbolStats] = defaultdict(_SymbolStats)
        self._total_checked: int = 0
        self._total_would_block: int = 0

    async def check(self, symbol: str, side: str) -> GateResult:
        """
        Fetch order book, compute dollar-weighted imbalance ratio.

        imbalance_ratio = sum(bid_qty * bid_price) / sum(ask_qty * ask_price)

        Block long entries if imbalance_ratio < LONG_BLOCK_RATIO (0.8)
        Block short entries if imbalance_ratio > SHORT_BLOCK_RATIO (1.2)

        In SHADOW_MODE: always return allowed=True but set reason to what
        would happen. In active mode: return allowed=False when blocked.

        On any fetch error: return GateResult(allowed=True, ratio=1.0,
        reason="fetch_error_allow") — never block on errors.
        """
        try:
            book = await asyncio.to_thread(self.client.get_l2_book, symbol)
        except Exception as exc:
            logger.warning("[OB-GATE] fetch error for %s (fail-open): %s", symbol, exc)
            return GateResult(allowed=True, imbalance_ratio=1.0, reason="fetch_error_allow")

        try:
            levels = book.get("levels", [])
            # levels[0] = bid side, levels[1] = ask side (HL convention)
            if len(levels) < 2:
                return GateResult(allowed=True, imbalance_ratio=1.0, reason="fetch_error_allow")

            bid_levels = levels[0][:TOP_N_LEVELS]
            ask_levels = levels[1][:TOP_N_LEVELS]

            bid_depth = sum(float(l["px"]) * float(l["sz"]) for l in bid_levels)
            ask_depth = sum(float(l["px"]) * float(l["sz"]) for l in ask_levels)

            if ask_depth <= 0:
                return GateResult(allowed=True, imbalance_ratio=1.0, reason="fetch_error_allow")

            ratio = bid_depth / ask_depth
        except Exception as exc:
            logger.warning("[OB-GATE] parse error for %s (fail-open): %s", symbol, exc)
            return GateResult(allowed=True, imbalance_ratio=1.0, reason="fetch_error_allow")

        is_long = side.lower() == "long"
        would_block = (is_long and ratio < LONG_BLOCK_RATIO) or (
            not is_long and ratio > SHORT_BLOCK_RATIO
        )

        # Update attribution counters
        self._total_checked += 1
        self._stats[symbol].total_checked += 1
        if would_block:
            self._total_would_block += 1
            self._stats[symbol].would_block_count += 1

        if would_block:
            reason = (
                f"ob_imbalance: ratio={ratio:.3f} "
                f"({'asks dominate' if is_long else 'bids dominate'}) "
                f"{'(SHADOW — not blocking)' if self.shadow_mode else '(BLOCKED)'}"
            )
            if self.shadow_mode:
                return GateResult(allowed=True, imbalance_ratio=ratio, reason=reason)
            else:
                return GateResult(allowed=False, imbalance_ratio=ratio, reason=reason)

        reason = f"ob_imbalance: ratio={ratio:.3f} allow"
        return GateResult(allowed=True, imbalance_ratio=ratio, reason=reason)

    def get_stats(self) -> dict:
        """Return attribution data: total_checked, would_block_count, block_pct_by_symbol."""
        by_symbol = {}
        for sym, s in self._stats.items():
            block_pct = (
                round(s.would_block_count / s.total_checked * 100.0, 1)
                if s.total_checked > 0 else 0.0
            )
            by_symbol[sym] = {
                "total_checked": s.total_checked,
                "would_block_count": s.would_block_count,
                "block_pct": block_pct,
            }
        overall_block_pct = (
            round(self._total_would_block / self._total_checked * 100.0, 1)
            if self._total_checked > 0 else 0.0
        )
        return {
            "shadow_mode": self.shadow_mode,
            "long_block_ratio": LONG_BLOCK_RATIO,
            "short_block_ratio": SHORT_BLOCK_RATIO,
            "top_n_levels": TOP_N_LEVELS,
            "total_checked": self._total_checked,
            "would_block_count": self._total_would_block,
            "overall_block_pct": overall_block_pct,
            "by_symbol": by_symbol,
        }
