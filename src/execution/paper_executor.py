"""
PaperTradingExecutor — Simulated execution against live Hyperliquid prices.

Drop-in replacement for HyperliquidVaultExecutor. Uses the public HL Info API
(no wallet/account/deposit needed) to get real-time prices, then simulates
order fills, position tracking, and PnL locally.

Usage:
    # In main.py lifespan, swap executor based on PAPER_TRADING env var:
    if os.getenv("PAPER_TRADING", "true").lower() == "true":
        executor = PaperTradingExecutor(base_url=constants.MAINNET_API_URL)
    else:
        executor = HyperliquidVaultExecutor(client=client)
"""

import asyncio
import json
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.services.confidence_ladder import STRONG_EDGE_HK, ladder_leverage, HL_MAX_LEVERAGE
from src.services.kelly import half_kelly_fraction
from src.services.liquidation_guard import LiquidationGuard
from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    SignalType,
)

logger = logging.getLogger(__name__)


# Skeptical prior variance for a (strategy×symbol) with NO same-regime siblings:
# N(0, _PRIOR_EDGE_VAR) on per-trade edge (~±0.8/trade at 90%). Anchors a thin,
# uncorroborated, HIGH-variance sample toward "no edge" so a lucky spike can't rank
# #1; a thin LOW-variance (consistent) sample still overrides it via its own precision.
# ponytail: single global knob — make it regime/strategy-specific only if it bites.
_PRIOR_EDGE_VAR = 0.25


def posterior_edge(own_pnls: List[float], pool_pnls: List[float], kappa: float = 10.0) -> Dict[str, float]:
    """Empirical-Bayes posterior on a (strategy×symbol)'s per-trade edge (T009).

    Normal-Normal conjugate update. The PRIOR is the sibling pool — recent
    per-trade PnLs of same-strategy-type instances on symbols currently in the
    SAME regime — treated as ~kappa pseudo-trades at the pool mean. The
    LIKELIHOOD is the instance's OWN recent trades. So a thin winner is RESOLVED:
    a 4-trade win with a corroborating family posts a high prob_edge; the same win
    with a flat/negative family stays near 0.5 with a wide CI — never blindly
    shrunk to 0 (which would kill a real edge) nor trusted at face value (luck).

    Returns prob_edge = P(per-trade edge > 0), posterior_mean_edge, a 90% credible
    interval (ci_low/ci_high), and input sample sizes. No data + no family ->
    prob_edge 0.5, mean 0.

    ponytail: shared known variance (pooled), own-variance floored at half the
    family variance so a tiny zero-variance sample isn't over-trusted; kappa is the
    prior strength in pseudo-trades. Upgrade to a full hierarchical fit only if
    this proves too coarse.
    """
    own = list(own_pnls)
    pool = list(pool_pnls)
    n = len(own)
    sigma2_pool = statistics.pvariance(pool) if len(pool) >= 2 else 0.0
    sigma2_own = statistics.pvariance(own) if n >= 2 else (sigma2_pool or 1.0)
    sigma2_eff = max(sigma2_own, 0.5 * sigma2_pool, 1e-3)  # over-confidence floor
    if pool:
        mu0 = sum(pool) / len(pool)            # prior centred on the regime-family edge
        tau2 = max(sigma2_pool, 1e-3) / kappa  # ~kappa pseudo-trades of family evidence
    else:
        # No same-regime siblings -> SKEPTICAL prior N(0, _PRIOR_EDGE_VAR): assume no
        # edge until the instance's own trades prove it. A thin HIGH-variance sample
        # (one lucky spike) then shrinks toward 0; a thin CONSISTENT sample (low
        # variance -> high own precision) still posts a high posterior. Without this
        # a no-family fluke trusts its raw mean and ranks #1.
        mu0 = 0.0
        tau2 = _PRIOR_EDGE_VAR
    xbar = (sum(own) / n) if n else mu0
    prec_prior = 1.0 / tau2
    prec_data = (n / sigma2_eff) if n else 0.0
    post_var = 1.0 / (prec_prior + prec_data)
    post_mean = post_var * (mu0 * prec_prior + xbar * prec_data)
    post_sd = math.sqrt(post_var)
    prob = (0.5 * (1.0 + math.erf(post_mean / (post_sd * math.sqrt(2.0))))
            if post_sd > 0 else (1.0 if post_mean > 0 else 0.0))
    z = 1.645  # 90% one-sided credible bound
    return {
        "prob_edge": prob,
        "posterior_mean_edge": post_mean,
        "ci_low": post_mean - z * post_sd,
        "ci_high": post_mean + z * post_sd,
        "n_own": n,
        "n_pool": len(pool),
    }


_OBSERVATION_FLOOR = 0.10  # fresh/edgeless strategies size at 10% until confidence accrues

# U6 (R6) — correlation-cluster + total-exposure caps. A book of 5 crypto longs
# behaves like ~1.5 independent bets, so losses compound; bound the correlation.
# Symbols in the same cluster share a committed-margin BUDGET (not a one-position
# gate — the live winner book is entirely BTC, so a one-per-cluster rule would lock
# 3 of 4 winners out of the market; a margin budget bounds correlated exposure while
# letting several small correlated winners coexist). Symbols absent from the map are
# uncorrelated (no cluster limit).
_CORRELATION_CLUSTERS = {
    "BTC": "majors", "ETH": "majors",
    "SOL": "l1alts", "AVAX": "l1alts",
    "DOGE": "memes",  "WIF": "memes",
}
# Aggregate committed margin within a single correlation cluster may not exceed
# this share of the wallet (correlated-exposure budget).
_MAX_CLUSTER_EXPOSURE_PCT = float(os.getenv("MAX_CLUSTER_EXPOSURE_PCT", "0.50"))
# Total committed margin across ALL open positions may not exceed this share of
# CURRENT EQUITY (balance + unrealized) — measured against equity, not stale cash,
# so the buffer does not loosen as open positions bleed.
_MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.80"))

# Last N closed trades that define a strategy's "recent" realized performance —
# the window F1 champion-challenger promotion decides on (not lifetime PnL).
RECENT_PNL_WINDOW = 20

# Minimum closed trades before a strategy's edge is called "real" (vs still-building
# sample). Single source for the bar used by /paper/edge AND circuit-breaker
# auto-recovery, so raising it via env moves both in lockstep.
EDGE_MIN_TRADES_REAL = int(os.getenv("EDGE_MIN_TRADES_REAL", "10"))

# Wilson score interval constants
_WILSON_Z_90 = 1.645  # z-score for 90% confidence interval


def _wilson_interval(wins: int, n: int, z: float = _WILSON_Z_90) -> tuple[float, float]:
    """Wilson score 90% CI for win-rate. stdlib math only.

    Returns (lower, upper) bounds in [0, 1]. When n == 0 returns (0.0, 0.0).
    """
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half_width = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


# No losses observed -> capped high payoff ratio so Kelly ~ win_rate (not a raw $ amount).
_NO_LOSS_PAYOFF = 10.0


def breakeven_wr(payoff: float) -> float:
    """Break-even win-rate for a given payoff ratio: a strategy with win-rate above
    this is net-positive. Single source for the edge bar used by /paper/edge and the
    circuit-breaker auto-recovery gate. Payoff <= 0 means no upside -> unreachable 1.0."""
    return (1.0 / (1.0 + payoff)) if payoff > 0 else 1.0

DEFAULT_RUIN_GUARD_BUFFER_PCT = 1.0  # force-close ~1 wick before liquidation; overridable via RiskConfig.ruin_guard_buffer_pct


@dataclass
class PaperPosition:
    symbol: str
    side: str  # "long" or "short"
    size: float  # asset units (signed: positive=long, negative=short)
    entry_price: float
    entry_time: datetime
    leverage: int = 1
    size_usd: float = 0.0
    strategy_name: str = ""
    unrealized_pnl: float = 0.0
    regime: Optional[str] = None  # market regime at entry (T010 regime-conditional history)


@dataclass
class PaperTrade:
    id: int
    symbol: str
    side: str
    action: str  # "entry" or "exit"
    price: float
    size: float
    size_usd: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime: Optional[str] = None  # market regime at entry (carried to the exit trade)


@dataclass
class PaperOrderResult:
    """Mimics OrderResult from nice_funcs for compatibility."""
    success: bool
    oid: Optional[int] = None
    status: str = ""
    raw: Optional[Dict] = None
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Same interface as hl_executor.ExecutionResult."""
    success: bool
    order_result: Optional[PaperOrderResult] = None
    realized_pnl: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PaperTradingExecutor:
    """
    Simulated executor using live Hyperliquid prices.

    No wallet, no account, no deposit needed. Uses the public Info API
    for real-time mid prices and simulates order fills with configurable slippage.
    """

    def __init__(
        self,
        base_url: str = "https://api.hyperliquid.xyz",
        default_slippage: float = 0.0005,  # Reduced: assume maker orders (MoonDev: "maker only")
        max_position_usd: float = 5000.0,  # Reduced: cap per-position exposure
        initial_balance: float = 10000.0,
        commission_pct: float = 0.0002,  # HL maker fee: 0.02% expressed as decimal fraction
    ):
        self.base_url = base_url
        self.default_slippage = default_slippage
        self.max_position_usd = max_position_usd
        self.commission_pct = commission_pct  # already a decimal fraction (0.0002 = 0.02%)

        # Paper account state
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self._positions: Dict[str, PaperPosition] = {}  # symbol -> position
        self._trades: List[PaperTrade] = []
        self._trade_counter = 0
        self._execution_history: List[ExecutionResult] = []
        # Optional symbol->regime lookup, set by the app (lifespan) to the live
        # regime_detector. Stamps each trade with the regime at entry so
        # regime-conditional edge history accrues durably (T010). None -> untagged.
        self._regime_fn = None

        # Price cache (refreshed on each call)
        self._mid_prices: Dict[str, float] = {}
        self._last_price_fetch: float = 0

        # For compatibility with orchestrator
        self.vault_address = "paper-trading"

        logger.info(
            "PaperTradingExecutor initialized | balance=$%.2f | slippage=%.2f%% | commission=%.4f%%",
            initial_balance,
            default_slippage * 100,
            commission_pct,
        )

    def reset(self) -> None:
        """Reset paper trading to initial state: restore balance, clear positions and trades."""
        logger.info(
            "[PAPER] Resetting | balance=$%.2f -> $%.2f | closing %d positions | clearing %d trades",
            self.balance,
            self.initial_balance,
            len(self._positions),
            len(self._trades),
        )
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self._positions.clear()
        self._trades.clear()
        self._trade_counter = 0
        self._execution_history.clear()

    # ── Persistence ──

    def _state_path(self) -> Path:
        data_dir = Path(os.getenv("DATA_DIR", "/data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "paper_state.json"

    def _ledger_path(self) -> Path:
        """Append-only durable trade ledger on the persistent volume.

        The paper_state.json snapshot keeps only the last 500 trades (a fast-boot
        hot cache); this JSONL ledger is the uncapped, redeploy-proof source of
        truth for the realized track record that feeds every live-edge / Wilson-CI
        / champion-challenger / de-freeze decision. F5 (R11).
        """
        return self._state_path().parent / "paper_trades.jsonl"

    @staticmethod
    def _trade_to_dict(t: "PaperTrade") -> dict:
        return {
            "id": t.id, "symbol": t.symbol, "side": t.side,
            "action": t.action, "price": t.price, "size": t.size,
            "size_usd": t.size_usd, "pnl": t.pnl,
            "pnl_pct": t.pnl_pct, "reason": t.reason,
            "strategy_name": t.strategy_name,
            "timestamp": t.timestamp.isoformat(),
            "regime": t.regime,
        }

    @staticmethod
    def _trade_from_dict(t: dict) -> "PaperTrade":
        return PaperTrade(
            id=t["id"], symbol=t["symbol"], side=t["side"],
            action=t["action"], price=t["price"], size=t["size"],
            size_usd=t["size_usd"], pnl=t.get("pnl", 0),
            pnl_pct=t.get("pnl_pct", 0), reason=t.get("reason", ""),
            strategy_name=t.get("strategy_name", ""),
            timestamp=datetime.fromisoformat(t["timestamp"]),
            regime=t.get("regime"),
        )

    def _append_ledger(self, trade: "PaperTrade") -> None:
        """Append one realized trade to the durable JSONL ledger. Best-effort:
        a ledger write must never break trade execution."""
        try:
            line = json.dumps(self._trade_to_dict(trade))
            with self._ledger_path().open("a") as fh:
                fh.write(line + "\n")
        except Exception as e:
            logger.warning("[PAPER] Failed to append trade to ledger: %s", e)

    def _seed_ledger_from(self, trades: List["PaperTrade"]) -> None:
        """One-time migration: when no ledger exists yet (first boot after F5
        ships) but the snapshot carries trades, write them so the ledger becomes
        the durable base that future exits append to. Cannot recover trades the
        500-cap already truncated historically; prevents all FUTURE truncation."""
        if not trades:
            return
        try:
            with self._ledger_path().open("w") as fh:
                for t in trades:
                    fh.write(json.dumps(self._trade_to_dict(t)) + "\n")
            logger.info("[PAPER] Seeded trade ledger with %d snapshot trades", len(trades))
        except Exception as e:
            logger.warning("[PAPER] Failed to seed ledger: %s", e)

    def _load_ledger(self) -> List["PaperTrade"]:
        """Read the full (uncapped) trade ledger. Tolerant of a malformed or
        truncated final line — skip-and-log, never throw."""
        path = self._ledger_path()
        if not path.exists():
            return []
        trades: List[PaperTrade] = []
        skipped = 0
        for raw in path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                trades.append(self._trade_from_dict(json.loads(raw)))
            except Exception:
                skipped += 1
        if skipped:
            logger.warning("[PAPER] Ledger load skipped %d malformed line(s)", skipped)
        return trades

    def save_state(self) -> None:
        """Persist paper trading state to disk so it survives redeploys."""
        try:
            state = {
                "balance": self.balance,
                "initial_balance": self.initial_balance,
                "peak_balance": self.peak_balance,
                "trade_counter": self._trade_counter,
                "positions": {
                    k: {
                        "symbol": p.symbol, "side": p.side, "size": p.size,
                        "entry_price": p.entry_price,
                        "entry_time": p.entry_time.isoformat(),
                        "leverage": p.leverage, "size_usd": p.size_usd,
                        "strategy_name": p.strategy_name,
                        "unrealized_pnl": p.unrealized_pnl,
                        "regime": p.regime,
                    }
                    for k, p in self._positions.items()
                },
                "trades": [self._trade_to_dict(t) for t in self._trades[-500:]],  # last 500; carries regime
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state_path().write_text(json.dumps(state, indent=2))
            logger.info("[PAPER] State saved | %d trades | balance=$%.2f", len(self._trades), self.balance)
        except Exception as e:
            logger.warning("[PAPER] Failed to save state: %s", e)

    def load_state(self) -> bool:
        """Restore paper trading state from disk. Returns True if state was loaded."""
        path = self._state_path()
        if not path.exists():
            logger.info("[PAPER] No saved state found at %s", path)
            return False
        try:
            state = json.loads(path.read_text())
            self.balance = state.get("balance", self.initial_balance)
            self.peak_balance = state.get("peak_balance", self.balance)
            self._trade_counter = state.get("trade_counter", 0)

            self._positions.clear()
            for k, p in state.get("positions", {}).items():
                self._positions[k] = PaperPosition(
                    symbol=p["symbol"], side=p["side"], size=p["size"],
                    entry_price=p["entry_price"],
                    entry_time=datetime.fromisoformat(p["entry_time"]),
                    leverage=p.get("leverage", 1),
                    size_usd=p.get("size_usd", 0),
                    strategy_name=p.get("strategy_name", ""),
                    unrealized_pnl=p.get("unrealized_pnl", 0),
                    regime=p.get("regime"),
                )

            self._trades.clear()
            for t in state.get("trades", []):
                self._trades.append(self._trade_from_dict(t))

            # ── Durable ledger rehydration (F5/R11) ──
            # The snapshot above is capped at the last 500 trades. If a durable
            # ledger exists, it is the uncapped source of truth — replace the
            # snapshot trades with the full history so live-edge / Wilson-CI /
            # champion-challenger / de-freeze decisions read the complete record.
            # If no ledger exists yet (first boot after F5 ships), seed it from
            # the snapshot so future exits append to a durable base.
            ledger_trades = self._load_ledger()
            if ledger_trades:
                self._trades = ledger_trades
            else:
                self._seed_ledger_from(self._trades)

            # Keep the trade counter ahead of every known id so new trades never
            # collide with a rehydrated one.
            if self._trades:
                self._trade_counter = max(
                    self._trade_counter, max(t.id for t in self._trades)
                )

            logger.info(
                "[PAPER] State restored | %d trades | %d positions | balance=$%.2f (saved %s)",
                len(self._trades), len(self._positions), self.balance,
                state.get("saved_at", "unknown"),
            )
            return True
        except Exception as e:
            logger.warning("[PAPER] Failed to load state: %s", e)
            return False

    def replay_strategy_state(self, strategy) -> int:
        """Rebuild a freshly-constructed strategy's durable StrategyState from the
        executor's realized-trade ledger so per-strategy circuit-breaker state
        (consecutive losses, max drawdown) survives a redeploy instead of resetting
        to zero. Reads the in-memory `self._trades` (already rehydrated from the
        uncapped ledger in load_state). Returns the number of trades replayed. F5/R11/R8.
        """
        # Idempotency / race guard: only rehydrate a freshly-constructed strategy.
        # restore_from_pnls is additive, so replaying onto a strategy that already
        # carries live state (a second call, or a trade recorded between construction
        # and this call) would double its counters. Skip when state is already non-empty.
        if strategy.state.total_trades:
            return 0
        name = strategy.config.name
        # Only REALIZED exits carry pnl and belong in the track record. Entry rows
        # (action="entry", pnl=0) live in self._trades too and would each be
        # miscounted as a loss — so filter to exits, mirroring _live_edge_stats /
        # _recent_realized. Trades are append-ordered, i.e. chronological.
        pnls = [t.pnl for t in self._trades
                if t.strategy_name == name and t.action == "exit"]
        if pnls:
            strategy.restore_from_pnls(pnls)
            if strategy.state.circuit_breaker_triggered:
                logger.warning(
                    "[PAPER] Restored TRIPPED circuit breaker for %s after replay of %d trades | %s",
                    name, len(pnls), strategy.state.circuit_breaker_reason,
                )
        return len(pnls)

    def _fetch_mid_price(self, symbol: str) -> float:
        """Fetch current mid price from HL public API."""
        now = time.time()
        # Cache prices for 2 seconds to avoid hammering the API
        if now - self._last_price_fetch > 2:
            try:
                resp = requests.post(
                    f"{self.base_url}/info",
                    headers={"Content-Type": "application/json"},
                    json={"type": "allMids"},
                    timeout=5,
                )
                resp.raise_for_status()
                self._mid_prices = {k: float(v) for k, v in resp.json().items()}
                self._last_price_fetch = now
            except Exception as e:
                logger.warning("Failed to fetch mid prices: %s", e)

        price = self._mid_prices.get(symbol)
        if price is None:
            raise ValueError(f"No price data for {symbol}")
        return price

    def _get_fill_price(self, symbol: str, is_buy: bool) -> float:
        """Get simulated fill price with slippage."""
        mid = self._fetch_mid_price(symbol)
        if is_buy:
            return mid * (1 + self.default_slippage)
        else:
            return mid * (1 - self.default_slippage)

    async def execute_signal(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Execute a strategy signal in paper mode."""
        if signal.signal_type == SignalType.NONE:
            return ExecutionResult(success=True)

        if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
            return await self._execute_entry(signal, strategy)
        elif signal.signal_type in (
            SignalType.CLOSE_LONG,
            SignalType.CLOSE_SHORT,
            SignalType.CLOSE_ALL,
        ):
            return await self._execute_exit(signal, strategy)
        else:
            return ExecutionResult(
                success=False, error=f"Unknown signal type: {signal.signal_type}"
            )

    def _live_edge_stats(self, strategy_name: str) -> tuple[float, float, int, float, float]:
        """Return (win_rate, payoff_ratio, n_trades, wr_lower_90, wr_upper_90) from CLOSED trades.

        wr_lower_90 / wr_upper_90 are the Wilson score 90% CI bounds on the win-rate.
        """
        pnls = [t.pnl for t in self._trades
                if t.strategy_name == strategy_name and t.action == "exit"]
        n = len(pnls)
        if n == 0:
            return 0.0, 0.0, 0, 0.0, 0.0
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        win_rate = len(wins) / n
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else (_NO_LOSS_PAYOFF if avg_win > 0 else 0.0)
        wr_lo, wr_hi = _wilson_interval(len(wins), n)
        return win_rate, payoff, n, wr_lo, wr_hi

    def recent_realized_pnl(
        self, strategy_name: str, window: int = RECENT_PNL_WINDOW
    ) -> tuple[float, int]:
        """Realized PnL summed over the most-recent `window` CLOSED trades for a
        strategy, plus the trade count in that window.

        Returns (recent_pnl, n) where n <= window; n == 0 when the strategy has
        no closed trades. This is the RECENT-window live signal F1 promotion
        keys off — it tracks current performance, unlike lifetime cumulative PnL
        which lags a stale historical buffer. self._trades is append-ordered so
        the last matching slice is the most recent."""
        pnls = [
            t.pnl for t in self._trades
            if t.strategy_name == strategy_name and t.action == "exit"
        ]
        recent = pnls[-window:]
        return round(sum(recent), 2), len(recent)

    def open_position_count(self, strategy_name: str) -> int:
        """Number of currently-open positions belonging to a strategy. Used to
        skip RBI re-optimization/promotion while a position is open (so params
        are never swapped mid-trade)."""
        return sum(
            1 for p in self._positions.values()
            if p.strategy_name == strategy_name
        )

    async def _execute_entry(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Simulate an entry order."""
        config = strategy.config
        symbol = signal.symbol or config.symbol
        is_buy = signal.signal_type == SignalType.LONG

        try:
            fill_price = await asyncio.to_thread(
                self._get_fill_price, symbol, is_buy
            )

            size_usd = signal.size_usd or config.size_usd
            # Compound sizing: grow proportionally on profit, cap at 3×; never shrink below base
            if self.initial_balance > 0:
                compound_mult = min(max(self.balance / self.initial_balance, 1.0), 3.0)
                size_usd = size_usd * compound_mult

            # Confidence-scaled leverage + half-Kelly sizing from live edge (docs/adr/0001).
            # Fresh/edgeless strategies trade at observation size (10%) and 1x leverage;
            # leverage + size ramp up only as live edge confidence accrues.
            # F2 (R9): size on the Wilson LOWER-BOUND win-rate, not the point
            # estimate. A lucky streak (high point WR but a wide CI on thin data)
            # must not ramp Kelly/leverage before the edge's confidence interval
            # clears breakeven. The lower bound converges to the point estimate as
            # the sample grows, so a genuine sustained edge still ramps fully — only
            # under-proven luck is held back. payoff stays the point estimate (its
            # tail noise is already bounded by the _NO_LOSS_PAYOFF cap).
            win_rate, payoff, n_trades, wr_lower_90, _ = self._live_edge_stats(config.name)
            sizing_wr = wr_lower_90
            effective_leverage = ladder_leverage(symbol, n_trades, sizing_wr, payoff)

            # Realtime adaptation (ADR-0002): scale ladder leverage by the per-entry multiplier
            # the orchestrator computed from live regime/vol/funding. Default 1.0 if absent —
            # so existing callers that don't set it are unaffected.
            _meta = getattr(signal, "metadata", None) or {}
            _adapt = float(_meta.get("adaptation_multiplier", 1.0))
            _lev_cap = HL_MAX_LEVERAGE.get(symbol.upper(), 5)
            effective_leverage = max(1, min(int(round(effective_leverage * _adapt)), _lev_cap))

            hk = half_kelly_fraction(sizing_wr, payoff)
            kelly_mult = max(_OBSERVATION_FLOOR, min(hk / STRONG_EDGE_HK, 1.0)) if hk > 0 else _OBSERVATION_FLOOR
            size_usd = size_usd * kelly_mult
            logger.debug(
                "[PAPER] Sizing %s | point_wr=%.2f lower_wr=%.2f payoff=%.2f n=%d "
                "-> lev=%dx kelly_mult=%.2f",
                config.name, win_rate, wr_lower_90, payoff, n_trades,
                effective_leverage, kelly_mult,
            )

            # Leverage scales the notional (how leverage amplifies exposure).
            # Apply AFTER compound+kelly so the cap bounds TRUE leveraged exposure.
            size_usd = size_usd * effective_leverage
            size_usd = min(size_usd, self.max_position_usd)  # cap TRUE notional exposure

            # Margin = notional / leverage; this is what we need in the account
            margin_required = size_usd / effective_leverage if effective_leverage > 1 else size_usd
            if margin_required > self.balance:
                return ExecutionResult(
                    success=False,
                    error=f"Insufficient balance: need ${margin_required:.2f}, have ${self.balance:.2f}",
                )

            asset_size = size_usd / fill_price
            commission = size_usd * self.commission_pct
            signed_size = asset_size if is_buy else -asset_size

            # Record position — keyed by strategy:symbol for isolation
            pos_key = f"{config.name}:{symbol}"
            if pos_key in self._positions:
                return ExecutionResult(
                    success=False,
                    error=f"Already in position for {config.name}:{symbol}",
                )

            # Current equity (cash + unrealized) is the base for BOTH exposure
            # budgets, so neither loosens as open positions bleed (a stale-cash base
            # would permit MORE margin exactly when the account is underwater).
            _equity = self.balance + sum(p.unrealized_pnl for p in self._positions.values())

            # U6 (R6): correlation-cluster margin BUDGET — bound aggregate committed
            # margin WITHIN a cluster to a share of equity. Not a one-position gate:
            # the live winner book is all-BTC, so a binary one-per-cluster rule would
            # lock 3 of 4 winners out of the market; a margin budget lets several
            # small correlated winners coexist while still capping correlated exposure.
            _cluster = _CORRELATION_CLUSTERS.get(symbol.upper())
            if _cluster is not None:
                _cluster_margin = sum(
                    p.size_usd / max(p.leverage, 1)
                    for p in self._positions.values()
                    if _CORRELATION_CLUSTERS.get(p.symbol.upper()) == _cluster
                )
                if _cluster_margin + margin_required > _equity * _MAX_CLUSTER_EXPOSURE_PCT:
                    return ExecutionResult(
                        success=False,
                        error=(f"Cluster cap: {_cluster} margin "
                               f"${_cluster_margin + margin_required:.0f} > "
                               f"{_MAX_CLUSTER_EXPOSURE_PCT:.0%} of ${_equity:.0f} equity"),
                    )

            # U6 (R6): total-exposure cap — committed margin (notional / leverage =
            # capital actually at risk) across ALL open positions bounded to a share
            # of equity, leaving a buffer against a broad drawdown.
            _open_margin = sum(
                p.size_usd / max(p.leverage, 1) for p in self._positions.values()
            )
            if _open_margin + margin_required > _equity * _MAX_TOTAL_EXPOSURE_PCT:
                return ExecutionResult(
                    success=False,
                    error=(f"Total-exposure cap: ${_open_margin + margin_required:.0f} margin "
                           f"> {_MAX_TOTAL_EXPOSURE_PCT:.0%} of ${_equity:.0f} equity"),
                )

            # Regime at entry (T010): stamped on the position and carried to the
            # exit trade, so realized PnL can later be conditioned on regime.
            _entry_regime = None
            if self._regime_fn is not None:
                try:
                    _entry_regime = self._regime_fn(symbol)
                except Exception:  # noqa: BLE001 - never let a tag lookup block a fill
                    _entry_regime = None

            self._positions[pos_key] = PaperPosition(
                symbol=symbol,
                side="long" if is_buy else "short",
                size=signed_size,
                entry_price=fill_price,
                entry_time=datetime.now(timezone.utc),
                leverage=effective_leverage,
                size_usd=size_usd,
                strategy_name=config.name,
                regime=_entry_regime,
            )

            self.balance -= commission
            self._trade_counter += 1

            trade = PaperTrade(
                id=self._trade_counter,
                symbol=symbol,
                side="long" if is_buy else "short",
                action="entry",
                price=fill_price,
                size=asset_size,
                size_usd=size_usd,
                reason=signal.reason,
                strategy_name=config.name,
                regime=_entry_regime,
            )
            self._trades.append(trade)
            self._append_ledger(trade)  # durable, uncapped (F5/R11) — ledger mirrors self._trades

            order_result = PaperOrderResult(
                success=True,
                oid=self._trade_counter,
                status="filled",
            )
            result = ExecutionResult(success=True, order_result=order_result)
            self._execution_history.append(result)

            logger.info(
                "[PAPER] Entry | %s | %s %s %.6f @ $%.2f ($%.0f) | commission=$%.2f | balance=$%.2f",
                config.name,
                "LONG" if is_buy else "SHORT",
                symbol,
                asset_size,
                fill_price,
                size_usd,
                commission,
                self.balance,
            )
            self.save_state()
            return result

        except Exception as e:
            logger.error("[PAPER] Entry error | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def _execute_exit(
        self, signal: Signal, strategy: BaseStrategy
    ) -> ExecutionResult:
        """Simulate an exit order."""
        config = strategy.config
        symbol = signal.symbol or config.symbol

        try:
            pos_key = f"{config.name}:{symbol}"
            pos = self._positions.get(pos_key)
            if not pos:
                return ExecutionResult(success=True, error="No position to close")

            is_buy = pos.size < 0  # closing a short = buy
            fill_price = await asyncio.to_thread(
                self._get_fill_price, symbol, is_buy
            )

            abs_size = abs(pos.size)
            exit_usd = abs_size * fill_price
            commission = exit_usd * self.commission_pct

            # Calculate PnL — leverage is already reflected in abs_size (via leveraged notional
            # at entry), so do NOT multiply again. One count only.
            if pos.side == "long":
                realized_pnl = (fill_price - pos.entry_price) * abs_size
            else:
                realized_pnl = (pos.entry_price - fill_price) * abs_size

            pnl_pct = (realized_pnl / (pos.entry_price * abs_size)) * 100

            self.balance += realized_pnl - commission

            # Track peak for drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

            # Record trade
            strategy.record_trade(realized_pnl)

            self._trade_counter += 1
            trade = PaperTrade(
                id=self._trade_counter,
                symbol=symbol,
                side=pos.side,
                action="exit",
                price=fill_price,
                size=abs_size,
                size_usd=exit_usd,
                pnl=realized_pnl,
                pnl_pct=pnl_pct,
                reason=signal.reason,
                strategy_name=config.name,
                regime=getattr(pos, "regime", None),  # regime at entry -> conditions this realized PnL
            )
            self._trades.append(trade)
            self._append_ledger(trade)  # durable, uncapped (F5/R11)

            # Remove position
            del self._positions[pos_key]

            # Fire-and-forget: persist trade outcome to Supermemory for self-tuning
            try:
                import asyncio as _aio
                from src.services.trade_memory import store_trade as _store
                _aio.create_task(_store(
                    strategy=config.name,
                    symbol=symbol,
                    signal=pos.side.upper(),
                    pnl=realized_pnl,
                    regime="unknown",
                    params=config.params,
                    price=fill_price,
                ))
            except Exception:
                pass

            order_result = PaperOrderResult(
                success=True,
                oid=self._trade_counter,
                status="filled",
            )
            result = ExecutionResult(
                success=True,
                order_result=order_result,
                realized_pnl=realized_pnl,
            )
            self._execution_history.append(result)

            logger.info(
                "[PAPER] Exit | %s | %s %s %.6f @ $%.2f | pnl=$%.2f (%.2f%%) | balance=$%.2f",
                config.name,
                pos.side,
                symbol,
                abs_size,
                fill_price,
                realized_pnl,
                pnl_pct,
                self.balance,
            )

            try:
                self.save_state()
            except Exception as e:
                logger.warning(f"Failed to save state after exit: {e}")

            return result

        except Exception as e:
            logger.error("[PAPER] Exit error | %s | %s | %s", config.name, symbol, e)
            result = ExecutionResult(success=False, error=str(e))
            self._execution_history.append(result)
            return result

    async def get_position(self, symbol: str, strategy_name: str = "") -> Optional[Dict[str, Any]]:
        """Get current paper position for a strategy:symbol pair."""
        pos_key = f"{strategy_name}:{symbol}" if strategy_name else symbol
        pos = self._positions.get(pos_key)
        if not pos:
            return None

        try:
            mid = await asyncio.to_thread(self._fetch_mid_price, symbol)
            abs_size = abs(pos.size)

            if pos.side == "long":
                unrealized_pnl = (mid - pos.entry_price) * abs_size
            else:
                unrealized_pnl = (pos.entry_price - mid) * abs_size

            pnl_pct = (unrealized_pnl / (pos.entry_price * abs_size)) * 100

            return {
                "symbol": symbol,
                "size": pos.size,
                "entry_px": pos.entry_price,
                "mark_price": mid,  # current mid — reused by the Ruin Guard to avoid a 2nd fetch
                "pnl_perc": pnl_pct,
                "unrealized_pnl": unrealized_pnl,
                "is_long": pos.side == "long",
                "side": pos.side,
            }
        except Exception:
            return {
                "symbol": symbol,
                "size": pos.size,
                "entry_px": pos.entry_price,
                "mark_price": None,  # price fetch failed; Ruin Guard will fetch on its own
                "pnl_perc": 0,
                "unrealized_pnl": 0,
                "is_long": pos.side == "long",
                "side": pos.side,
            }

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all open paper positions."""
        positions = []
        for pos_key, pos in self._positions.items():
            try:
                mid = await asyncio.to_thread(self._fetch_mid_price, pos.symbol)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    unrealized_pnl = (mid - pos.entry_price) * abs_size
                else:
                    unrealized_pnl = (pos.entry_price - mid) * abs_size
                pnl_pct = (unrealized_pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                positions.append({
                    "symbol": pos.symbol,
                    "strategy_name": pos.strategy_name,
                    "size_usd": pos.size_usd,
                    "size": pos.size,
                    "entry_px": pos.entry_price,
                    "pnl_perc": pnl_pct,
                    "unrealized_pnl": unrealized_pnl,
                    "is_long": pos.side == "long",
                    "side": pos.side,
                })
            except Exception:
                positions.append({
                    "symbol": pos.symbol,
                    "strategy_name": pos.strategy_name,
                    "size_usd": pos.size_usd,
                    "size": pos.size,
                    "entry_px": pos.entry_price,
                    "pnl_perc": 0,
                    "unrealized_pnl": 0,
                    "is_long": pos.side == "long",
                    "side": pos.side,
                })
        return positions

    async def get_account_value(self) -> float:
        """Get total paper account value (balance + unrealized PnL)."""
        total_unrealized = 0
        for pos_key, pos in self._positions.items():
            try:
                mid = await asyncio.to_thread(self._fetch_mid_price, pos.symbol)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    total_unrealized += (mid - pos.entry_price) * abs_size
                else:
                    total_unrealized += (pos.entry_price - mid) * abs_size
            except Exception:
                pass
        return self.balance + total_unrealized

    async def close_by_symbol(self, symbol: str) -> List[ExecutionResult]:
        """Close all paper positions for a given symbol — called by risk controller."""
        results = []
        keys_to_close = [k for k, p in self._positions.items() if p.symbol == symbol]
        for pos_key in keys_to_close:
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    realized_pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    realized_pnl = (pos.entry_price - fill_price) * abs_size
                # leverage already reflected in abs_size; do NOT multiply again
                commission = abs_size * fill_price * self.commission_pct
                self.balance += realized_pnl - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                pnl_pct = (realized_pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=realized_pnl,
                    pnl_pct=pnl_pct, reason="risk_stop",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                self._append_ledger(trade)  # durable realized exit (F5/R11)
                del self._positions[pos_key]
                results.append(ExecutionResult(success=True, realized_pnl=realized_pnl))
                logger.info("[PAPER] Risk close | %s | pnl=$%.2f (%.2f%%)", symbol, realized_pnl, pnl_pct)
            except Exception as e:
                logger.error("[PAPER] close_by_symbol error | %s | %s", symbol, e)
                results.append(ExecutionResult(success=False, error=str(e)))
        self.save_state()
        return results

    async def close_by_strategy(self, strategy_name: str) -> List[ExecutionResult]:
        """Close every open paper position whose strategy_name matches.

        Used when a strategy is being disabled — without this, the orchestrator
        stops calling should_exit on the disabled strategy and its positions
        drift indefinitely. Mirrors close_by_symbol; emits exit trades with
        reason='strategy_disabled' so they're attributable in the trade log.
        """
        results: List[ExecutionResult] = []
        keys_to_close = [
            k for k, p in self._positions.items() if p.strategy_name == strategy_name
        ]
        for pos_key in keys_to_close:
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)
                if pos.side == "long":
                    realized_pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    realized_pnl = (pos.entry_price - fill_price) * abs_size
                # leverage already reflected in abs_size; do NOT multiply again
                commission = abs_size * fill_price * self.commission_pct
                self.balance += realized_pnl - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                pnl_pct = (realized_pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=pos.symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=realized_pnl,
                    pnl_pct=pnl_pct, reason="strategy_disabled",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                self._append_ledger(trade)  # durable realized exit (F5/R11)
                del self._positions[pos_key]
                results.append(ExecutionResult(success=True, realized_pnl=realized_pnl))
                logger.info(
                    "[PAPER] Orphan close | %s | %s | pnl=$%.2f (%.2f%%)",
                    strategy_name, pos.symbol, realized_pnl, pnl_pct,
                )
            except Exception as e:
                logger.error("[PAPER] close_by_strategy error | %s | %s | %s", strategy_name, pos.symbol, e)
                results.append(ExecutionResult(success=False, error=str(e)))
        if results:
            self.save_state()
        return results

    async def emergency_close_all(self) -> List[ExecutionResult]:
        """Close all paper positions."""
        logger.critical("[PAPER] EMERGENCY CLOSE ALL")
        results = []
        for pos_key in list(self._positions.keys()):
            pos = self._positions[pos_key]
            is_buy = pos.size < 0
            try:
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    realized_pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    realized_pnl = (pos.entry_price - fill_price) * abs_size

                # leverage already reflected in abs_size; do NOT multiply again
                commission = abs_size * fill_price * self.commission_pct
                pnl_pct = (realized_pnl / (pos.entry_price * abs_size)) * 100 if abs_size > 0 else 0
                self.balance += realized_pnl - commission
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance

                self._trade_counter += 1
                trade = PaperTrade(
                    id=self._trade_counter, symbol=pos.symbol, side=pos.side,
                    action="exit", price=fill_price, size=abs_size,
                    size_usd=abs_size * fill_price, pnl=realized_pnl,
                    pnl_pct=pnl_pct, reason="emergency_close",
                    strategy_name=pos.strategy_name,
                )
                self._trades.append(trade)
                self._append_ledger(trade)  # durable realized exit (F5/R11)
                del self._positions[pos_key]

                results.append(ExecutionResult(success=True, realized_pnl=realized_pnl))
                logger.info("[PAPER] Emergency close | %s | pnl=$%.2f (%.2f%%)", pos.symbol, realized_pnl, pnl_pct)
            except Exception as e:
                results.append(ExecutionResult(success=False, error=str(e)))
        return results

    async def check_pnl_guard(
        self,
        symbol: str,
        target_pct: float = 5.0,
        max_loss_pct: float = -10.0,
        strategy_name: str = "",
    ) -> Optional[ExecutionResult]:
        """Check PnL guard and close if triggered."""
        pos_data = await self.get_position(symbol, strategy_name=strategy_name)
        if not pos_data:
            return None

        pnl_pct = pos_data.get("pnl_perc", 0)
        if pnl_pct >= target_pct or pnl_pct <= max_loss_pct:
            logger.info(
                "[PAPER] PnL guard triggered | %s | pnl=%.2f%% | target=%.1f%% | max_loss=%.1f%%",
                symbol, pnl_pct, target_pct, max_loss_pct,
            )
            pos_key = f"{strategy_name}:{symbol}" if strategy_name else symbol
            pos = self._positions.get(pos_key)
            if pos:
                is_buy = pos.size < 0
                fill_price = await asyncio.to_thread(self._get_fill_price, pos.symbol, is_buy)
                abs_size = abs(pos.size)

                if pos.side == "long":
                    pnl = (fill_price - pos.entry_price) * abs_size
                else:
                    pnl = (pos.entry_price - fill_price) * abs_size

                self.balance += pnl - (abs_size * fill_price * self.commission_pct)
                del self._positions[pos_key]

                return ExecutionResult(success=True, realized_pnl=pnl)
        return None

    async def check_position_ruin(self, symbol: str, strategy_name: str,
                                  safety_buffer_pct: float = DEFAULT_RUIN_GUARD_BUFFER_PCT,
                                  mid: Optional[float] = None) -> tuple[bool, float]:
        """Reflex-layer guard: return (should_force_close, distance_to_liquidation_pct).

        Runs every monitoring iteration. If the current mid is within safety_buffer_pct
        of the estimated liquidation price, the position must be flattened BEFORE a faster
        wick liquidates it. Decoupled from (and faster than) the tuning loop.

        ``mid`` may be supplied by the caller (e.g. the orchestrator already fetched it
        via get_position) to avoid a redundant price fetch; if None it is fetched here."""
        pos = self._positions.get(f"{strategy_name}:{symbol}")
        if pos is None:
            return False, 100.0
        if mid is None:
            mid = await asyncio.to_thread(self._fetch_mid_price, symbol)
        if not mid or mid <= 0:
            return False, 100.0
        margin = pos.size_usd / pos.leverage if pos.leverage > 1 else pos.size_usd
        liq_price = LiquidationGuard.calculate_liquidation_price(
            entry_price=pos.entry_price, position_size=abs(pos.size), margin=margin,
            leverage=pos.leverage, is_long=(pos.side == "long"), symbol=symbol,
        )
        if pos.side == "long":
            distance_pct = (mid - liq_price) / mid * 100.0
        else:
            distance_pct = (liq_price - mid) / mid * 100.0
        return (distance_pct <= safety_buffer_pct), distance_pct

    def get_active_positions(self) -> Dict[str, Dict]:
        """Get active paper positions as dict keyed by symbol."""
        return {
            v.symbol: {
                "strategy": v.strategy_name,
                "symbol": v.symbol,
                "side": v.side,
                "size": v.size,
                "entry_time": v.entry_time.isoformat(),
                "entry_price": v.entry_price,
            }
            for k, v in self._positions.items()
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get paper trading execution statistics.

        ponytail: realized PnL / trades / positions come from DURABLE state —
        self._trades is rehydrated from the ledger on boot (load_state:367-369)
        and self._positions is restored too, so /health survives redeploys.
        _execution_history is session-local and only drives success_rate (it
        resets each boot, which is fine). Previously the whole dict was derived
        from _execution_history, so total_realized_pnl reset to 0 every redeploy.
        """
        exits = [t for t in self._trades if t.action == "exit"]
        total_realized_pnl = sum(t.pnl for t in exits)

        drawdown = 0
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100

        n_exec = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)

        return {
            "mode": "paper",
            "total_executions": n_exec,
            "successful": successes,
            "failed": n_exec - successes,
            "success_rate": round(successes / n_exec * 100, 1) if n_exec else 0,
            "total_realized_pnl": round(total_realized_pnl, 2),
            "total_pnl": round(total_realized_pnl, 2),  # alias: legacy consumers of the old degraded shape
            "active_positions": len(self._positions),
            "balance": round(self.balance, 2),
            "initial_balance": self.initial_balance,
            "total_return_pct": round(
                ((self.balance - self.initial_balance) / self.initial_balance) * 100, 2
            ),
            "peak_balance": round(self.peak_balance, 2),
            "max_drawdown_pct": round(drawdown, 2),
            "total_trades": len(self._trades),
            "vault_address": "paper-trading",
        }

    def get_trade_history(self) -> List[Dict]:
        """Get all paper trades as dicts."""
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "action": t.action,
                "price": t.price,
                "size": t.size,
                "size_usd": t.size_usd,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "reason": t.reason,
                "strategy": t.strategy_name,
                "timestamp": t.timestamp.isoformat(),
                "regime": t.regime,
            }
            for t in self._trades
        ]

    def get_equity_curve(self) -> List[Dict]:
        """Build equity curve from trade history."""
        curve = [{"timestamp": self._trades[0].timestamp.isoformat() if self._trades else datetime.now(timezone.utc).isoformat(), "equity": self.initial_balance}]
        running = self.initial_balance

        for t in self._trades:
            if t.action == "exit":
                running += t.pnl
                curve.append({
                    "timestamp": t.timestamp.isoformat(),
                    "equity": round(running, 2),
                })

        return curve


if __name__ == "__main__":
    # posterior_edge: a thin winner is RESOLVED by its family, not blindly shrunk.
    _noisy4 = [0.0, 0.0, 19.0, 0.0]  # gridfib-style: 4 trades, one big lucky win
    flat = posterior_edge(_noisy4, [0.1, -0.2, 0.0, 0.3, -0.1, 0.2])   # family flat
    good = posterior_edge(_noisy4, [1.0, 0.8, 1.2, 0.9, 1.1, 0.7])     # family positive
    # corroborated > uncorroborated, and uncorroborated thin win is NOT auto-#1, NOT auto-0
    assert good["prob_edge"] > flat["prob_edge"], "family corroboration must lift prob_edge"
    assert good["posterior_mean_edge"] > flat["posterior_mean_edge"]
    assert 0.0 < flat["posterior_mean_edge"] < 1.0, "thin uncorroborated win: small +mean, not the raw 4.75"
    assert flat["ci_low"] < 0.0 < flat["ci_high"], "thin sample must have a wide CI straddling 0"
    # a steady high-sample winner posts high confidence; a high-sample loser, low.
    assert posterior_edge([0.5] * 30, [0.1, 0.2])["prob_edge"] > 0.9
    assert posterior_edge([-0.5] * 40, [0.0, 0.1])["prob_edge"] < 0.1
    # no data + no family -> neutral.
    _none = posterior_edge([], [])
    assert abs(_none["prob_edge"] - 0.5) < 1e-9 and abs(_none["posterior_mean_edge"]) < 1e-9
    # NO-FAMILY skeptical prior: a thin HIGH-variance lucky spike must shrink toward
    # 0 (not rank #1), while a thin CONSISTENT edge still posts high confidence.
    _spike = posterior_edge(_noisy4, [])          # gridfib with no same-regime siblings
    assert _spike["posterior_mean_edge"] < 1.0 and _spike["prob_edge"] < 0.75, \
        "uncorroborated noisy spike must not post a confident edge"
    _steady4 = posterior_edge([1.85, 1.9, 1.8, 1.86], [])  # funding-arb-like: 4 consistent
    assert _steady4["prob_edge"] > 0.9, "a consistent thin sample is still trustworthy"
    assert _steady4["prob_edge"] > _spike["prob_edge"]
    print("paper_executor self-checks OK")
