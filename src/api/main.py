import asyncio
import json
import os
import re
import logging
import random
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import models, schemas
from .database import engine, get_db
from .auth import get_password_hash, verify_password, create_access_token, require_current_user

# Ensure all SQLAlchemy models register with Base.metadata BEFORE create_all.
# Without this import, rbi_jobs / rbi_strategy_results tables never get created
# and /rbi/jobs returns 500 with "no such table". Importing is enough — the
# module just needs to be loaded so its Column definitions execute. Adding
# `# noqa: F401` to keep linters quiet.
from src.services import rbi_models  # noqa: F401
from src.services.confidence_ladder import edge_confidence
from src.engine.rbi_pipeline import build_rbi_job_specs
from src.engine.param_spaces import PARAM_SPACES

models.Base.metadata.create_all(bind=engine)

# Inline migration: add columns introduced after the initial schema was created.
# create_all() skips existing tables, so new Column()s on old tables never land
# without an explicit ALTER TABLE.
from sqlalchemy import text as _sql_text  # noqa: E402
with engine.connect() as _conn:
    _cols = {row[1] for row in _conn.execute(_sql_text("PRAGMA table_info(strategy_instances)"))}
    if "edge_confidence_score" not in _cols:
        _conn.execute(_sql_text("ALTER TABLE strategy_instances ADD COLUMN edge_confidence_score FLOAT DEFAULT 0.0"))
        _conn.commit()

with engine.connect() as _conn:
    _pnl_cols = {row[1] for row in _conn.execute(_sql_text("PRAGMA table_info(pnl_snapshots)"))}
    if "balance_pnl" not in _pnl_cols:
        _conn.execute(_sql_text("ALTER TABLE pnl_snapshots ADD COLUMN balance_pnl FLOAT"))
        _conn.commit()

logger = logging.getLogger(__name__)


# ── Bleeder-cull controller config (env-overridable) ──
# Autonomously stops chronic loser strategies so they stop bleeding the paper
# balance. Tuned (2026-06-05 operator directive) to stop a clear bleeder within
# ~a day instead of ~a week: a strategy still must clear a trade-count floor and
# be genuinely losing before a PnL or win-rate trigger fires, confirmed winners
# are never culled, and a running-book floor (CULL_MIN_RUNNING) caps how much the
# more-aggressive thresholds can remove per book.
CULL_MIN_PNL = float(os.getenv("CULL_MIN_PNL", "-10.0"))        # cull if cumulative PnL <= this (2026-06-05: -15 -> -10)
CULL_MIN_TRADES = int(os.getenv("CULL_MIN_TRADES", "4"))         # need this many closed trades first (2026-06-05: 6 -> 4)
CULL_MIN_WINRATE = float(os.getenv("CULL_MIN_WINRATE", "0.25"))  # cull if win-rate below this (guarded by pnl<0)
CULL_MAX_PER_RUN = int(os.getenv("CULL_MAX_PER_RUN", "6"))       # cap culls per run (2026-06-05: 3 -> 6, clear backlog faster)
CULL_INTERVAL_MIN = int(os.getenv("CULL_INTERVAL_MIN", "10"))    # scheduler cadence (2026-06-05: 15 -> 10 min)
CULL_MIN_RUNNING = int(os.getenv("CULL_MIN_RUNNING", "12"))      # floor: never cull the running book below this (safety rail)

# T028 — zombie killer: a strategy can sit just above the pnl/win-rate floors
# (flat/breakeven) for a long sample with genuinely NO live edge — neither a
# winner nor a clear loser. Scored the same way the compounder scores
# confidence (half-Kelly edge_confidence on the live Wilson-lower-bound win
# rate); a long, edgeless sample is culled even though pnl > CULL_MIN_PNL.
CULL_ZOMBIE_MIN_TRADES = int(os.getenv("CULL_ZOMBIE_MIN_TRADES", "20"))  # need a real sample before declaring "no edge" (not early variance)
CULL_ZOMBIE_MAX_CONF = float(os.getenv("CULL_ZOMBIE_MAX_CONF", "0.03"))  # edge_confidence_score <= this after the trade floor = zombie

# U6 — confidence-governed allocation + winner-set de-freeze (T027 + F3 + F4).
WINNER_OBS_PCT = float(os.getenv("WINNER_OBS_PCT", "0.10"))      # per-winner share at ZERO earned confidence (observation-small)
WINNER_MAX_PCT = float(os.getenv("WINNER_MAX_PCT", "0.75"))      # per-winner ceiling, reached only near FULL confidence (~30 live trades)
DEFREEZE_MIN_RECENT_TRADES = int(os.getenv("DEFREEZE_MIN_RECENT_TRADES", "10"))  # recent closed trades before a static winner can be demoted
PROVEN_REALIZED_CONF = float(os.getenv("PROVEN_REALIZED_CONF", "0.5"))  # confidence floor for a strategy with sustained POSITIVE realized PnL — so an asymmetric fat-tail winner (low win-rate, real money) is not pinned at the observation floor by its hit-rate

# T010 conditional-edge allocation (slice 2): the compounder tilts size by the
# leaderboard POSTERIOR. Non-winners get a Thompson exploration stake ∝ upside
# (ci_high) — a promising wide-CI maybe-edge is probed, a dead loser shrinks to the
# floor (never 0, so a real-but-young edge is never starved). Winners' confidence
# cap is scaled by prob_edge. The executor's exposure/cluster/max-position caps
# (paper_executor) remain the hard safety net on the resulting notionals.
EXPLORE_MIN_USD = float(os.getenv("EXPLORE_MIN_USD", "25.0"))        # exploration floor — never starve a maybe-edge to 0
EXPLORE_UPSIDE_REF = float(os.getenv("EXPLORE_UPSIDE_REF", "0.8"))   # ci_high ($/trade) at which exploration reaches full non-winner size
ALLOC_RAMP_UP_MAX = float(os.getenv("ALLOC_RAMP_UP_MAX", "1.5"))     # max size INCREASE per compound cycle (no 10x jumps); cuts apply immediately
POSTERIOR_WINNER_PROB = float(os.getenv("POSTERIOR_WINNER_PROB", "0.9"))  # prob_edge >= this (and +mean) promotes an instance into the winner tier — the posterior is a better thin-sample filter than trades>=6 (funding-arb 4t prob 0.998 grows; gridfib 4t prob 0.59 does not). Ramp keeps the grow gradual; $500-unproven cap still bounds it.
POSTERIOR_CULL_PROB = float(os.getenv("POSTERIOR_CULL_PROB", "0.3"))  # prob_edge < this AND negative posterior mean (with >= CULL_MIN_TRADES) is a confident loser — cull via the existing path to shed the tail faster than the pnl-only gate. gridfib (prob 0.59) is NOT culled — it's a probe, not a loser.
POSTERIOR_DEMOTE_PROB = float(os.getenv("POSTERIOR_DEMOTE_PROB", "0.5"))  # hysteresis: an instance is winner-SIZED only while prob_edge holds >= this. A decayed winner the signal now rejects (vwap-btc prob 0.0) falls below it and shrinks to the exploration floor — even a static/de-frozen winner — stopping the $500-on-losers leak. Promote at POSTERIOR_WINNER_PROB(0.9), demote here(0.5) -> band avoids flapping.

# T028 — REVIVE (cull-or-revive is never terminal). A stopped strategy is
# continuously re-evaluated: REVIVE-BY-RETUNE re-optimizes its params against
# recent data via the existing RBI pipeline/gate (deterministic Optuna only —
# zero LLM/Opus in this path); REVIVE-BY-REGIME re-enables it unchanged if its
# existing params now fit the CURRENT regime. Either path re-enters at
# OBSERVATION tier (1x leverage, $REVIVE_BASE_USD, fresh live-evidence window)
# — LIVE evidence decides whether it earns size, never the reviving backtest
# (ADR-0001). Anti-flap cooldown reuses T031's PromotionEventRecord table
# (reason prefixed "lifecycle_") as the persisted cull<->revive hysteresis —
# no second persistence layer.
REVIVE_INTERVAL_MIN = int(os.getenv("REVIVE_INTERVAL_MIN", "60"))        # scheduler cadence — retune is expensive (Optuna n_trials), so slower than the 10-min cull
REVIVE_MAX_PER_RUN = int(os.getenv("REVIVE_MAX_PER_RUN", "3"))           # cap stopped strategies examined per tick (bounds optimizer compute + revive churn)
REVIVE_COOLDOWN_HOURS = float(os.getenv("REVIVE_COOLDOWN_HOURS", "24"))  # anti-flap: no cull<->revive (or repeat retune attempt) for a strategy within this window
REVIVE_BASE_USD = float(os.getenv("REVIVE_BASE_USD", "100.0"))           # OBSERVATION-tier re-entry size
REVIVE_LOOKBACK_DAYS = int(os.getenv("REVIVE_LOOKBACK_DAYS", "90"))
REVIVE_N_TRIALS = int(os.getenv("REVIVE_N_TRIALS", "100"))


def _winner_cap_usd(balance: float, confidence: float,
                    obs_pct: float = None, max_pct: float = None) -> float:
    """U6 (R5): per-winner allocation ceiling (USD) that scales with EARNED live
    confidence. A thin-evidence winner (low edge_confidence_score, ~6 trades) is
    held to an observation-small share; the WINNER_MAX_PCT ceiling is reached only
    as confidence approaches 1 (~30 live trades). Confidence is computed from
    realized fills only — no backtest→live bridge (ADR-0003 / KTD-5)."""
    obs_pct = WINNER_OBS_PCT if obs_pct is None else obs_pct
    max_pct = WINNER_MAX_PCT if max_pct is None else max_pct
    conf = max(0.0, min(confidence, 1.0))
    pct = obs_pct + (max_pct - obs_pct) * conf
    return round(balance * pct, 2)


def _defrosted_winner_set(static_winners: set, dynamic_winners: set,
                          recent_pnl_fn, min_recent_trades: int = None) -> set:
    """U6 (R10 / F3): de-freeze the static winner set. A static name stays
    protected UNLESS it shows a SUSTAINED recent live loss — enough recent closed
    trades (>= min_recent_trades) AND negative recent realized PnL — in which case
    it is DROPPED from protection so the cull can demote it and the compounder
    stops over-allocating it. Thin recent data (fresh strategy or the brief
    post-redeploy window) keeps it protected (safe degradation, mirrors F1).

    `recent_pnl_fn(name) -> (recent_pnl, recent_trades)` reads the redeploy-proof
    recent realized window. Dynamic winners (live-proven THIS window) are always
    unioned in — that is the promotion-INTO-capital side of de-freezing.
    """
    min_recent_trades = DEFREEZE_MIN_RECENT_TRADES if min_recent_trades is None else min_recent_trades
    protected = set()
    for name in static_winners:
        try:
            recent_pnl, recent_n = recent_pnl_fn(name)
        except Exception:
            protected.add(name)  # cannot assess -> keep protected (safe)
            continue
        sustained_loser = recent_n >= min_recent_trades and recent_pnl < 0
        if not sustained_loser:
            protected.add(name)
    return set(dynamic_winners) | protected


def _select_cull_candidates(
    stats: dict,
    winners: set,
    *,
    min_pnl: float,
    min_trades: int,
    min_winrate: float,
    max_per_run: int,
) -> list[str]:
    """Pure selection of which RUNNING strategies should be culled.

    `stats` maps strategy_name -> {"pnl": float, "trades": int, "win_rate": float}.
    A name is selected when it is NOT a confirmed winner AND it has cleared the
    trade-count floor AND it is actually losing money via one of two triggers:

        (pnl <= min_pnl AND trades >= min_trades)
        OR (win_rate < min_winrate AND pnl < 0 AND trades >= min_trades)

    The `pnl < 0` guard on the win-rate trigger protects asymmetric net-positive
    strategies (few wins, big trend wins — e.g. flip-flop-btc at ~30% win-rate)
    from being culled just for a low hit-rate: a bleeder by definition loses money.

    Result is sorted by pnl ascending (worst bleeders first) and truncated to
    `max_per_run`. No I/O — this is the unit-tested core of the cull job.
    """
    candidates: list[tuple[float, str]] = []
    for name, s in stats.items():
        if name in winners:
            continue
        pnl = float(s.get("pnl", 0.0))
        trades = int(s.get("trades", 0))
        win_rate = float(s.get("win_rate", 0.0))
        if trades < min_trades:
            continue
        if (pnl <= min_pnl) or (win_rate < min_winrate and pnl < 0.0):
            candidates.append((pnl, name))
    candidates.sort(key=lambda c: c[0])  # worst PnL first
    return [name for _pnl, name in candidates[:max_per_run]]


def _record_lifecycle_event(
    db, *, strategy_type: str, strategy_id: int, reason: str, promoted: bool,
    before_params: dict | None = None, after_params: dict | None = None,
    before_metrics: dict | None = None, after_metrics: dict | None = None,
) -> None:
    """T028 — persist a cull/revive state-transition as a PromotionEventRecord
    row (reason prefixed "lifecycle_"). Reuses T031's existing DB persistence
    layer instead of a second one: GET /optimize/rbi/history already surfaces
    every row in this table, so lifecycle events are auditable for free, and
    `_lifecycle_in_cooldown` reads the SAME table for anti-flap hysteresis.
    """
    from .models import PromotionEventRecord
    row = PromotionEventRecord(
        strategy_type=strategy_type, strategy_id=strategy_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        promoted=promoted, reason=reason,
        before_params=before_params or {}, after_params=after_params or {},
        before_metrics=before_metrics or {}, after_metrics=after_metrics or {},
    )
    db.add(row)
    db.commit()


def _lifecycle_in_cooldown(db, strategy_id: int, hours: float) -> bool:
    """T028 ANTI-FLAP: true if this strategy (by DB row id — stable across a
    revive rename) had a lifecycle event within the last `hours`. Blocks
    cull<->revive churn AND throttles repeat optimizer attempts on a strategy
    that just failed revive-by-retune. DB-backed (PromotionEventRecord) so the
    cooldown survives a Railway redeploy, unlike an in-memory timer.
    """
    from .models import PromotionEventRecord
    last = (
        db.query(PromotionEventRecord)
        .filter(
            PromotionEventRecord.strategy_id == strategy_id,
            PromotionEventRecord.reason.like("lifecycle_%"),
        )
        .order_by(PromotionEventRecord.timestamp.desc())
        .first()
    )
    if last is None:
        return False
    try:
        last_ts = datetime.fromisoformat(last.timestamp)
    except (TypeError, ValueError):
        return False
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last_ts) < timedelta(hours=hours)


def _next_revive_name(current_name: str) -> str:
    """T028 — generate a fresh instance NAME for a revived strategy.

    executor.get_trade_history()/recent_realized_pnl/_live_edge_stats are all
    keyed by strategy NAME (paper_executor.py — not in this task's allowed
    files). Renaming the SAME DB row (same `id`, so PromotionEventRecord audit
    history via strategy_id stays unbroken) is how the live-edge-stats RESET
    is achieved with zero executor changes: a fresh name has zero trade
    history, so total_trades/total_pnl/recent_pnl/edge_confidence_score all
    read back as zero — genuinely OBSERVATION tier, not a relabeled veteran.
    Pre-revive performance stays queryable under the old name for audit.
    Idempotent suffix counter (vwap-btc -> vwap-btc-r1 -> vwap-btc-r2 ...).
    """
    m = re.search(r"-r(\d+)$", current_name)
    if m:
        base = current_name[: m.start()]
        n = int(m.group(1)) + 1
    else:
        base = current_name
        n = 1
    return f"{base}-r{n}"


def _auto_deploy_winners(db, orchestrator):
    """Deploy default winner strategies into an empty DB so they auto-start.

    2026-06-07 SURVIVOR-ONLY: portfolio is -$419 across 76 strategies.
    Only these 4 have proven live edge — all others are disabled.
    To re-enable a strategy: add it back here AND remove from purge list below.
    """
    winners = [
        # ── DE-FROZEN 2026-06-29 (T006): flip-flop-btc removed — the durable ledger
        #    shows -82.25 total / -20.67 recent / 4-of-47 wins (8.5%). A frozen
        #    "winner" that has decayed into the book's worst bleeder is exactly the
        #    stale-winner problem; it is culled + disabled and must NOT reseed here.
        #    flip-flop-btc-v2 (improved R:R + ADX filter) remains as the flip_flop bet.
        # ── SURVIVOR: vwap-btc — VWAP probability bias, confirmed live winner ──
        {"name": "vwap-btc", "strategy_type": "vwap_bot", "symbol": "BTC", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"vwap_bias_long": 0.7, "vwap_bias_short": 0.3, "min_vwap_distance": 0.0008, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        # ── SURVIVOR: closed-mkt-btc — overnight/weekend breakout, NYSE-close gate ──
        {"name": "closed-mkt-btc", "strategy_type": "closed_market_overnight", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"momentum_lookback": 12, "breakout_pct": 0.002, "tp_pct": 0.010, "sl_pct": 0.008, "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4, "min_signal_strength": 0.65}},
        # ── SURVIVOR: liqdip-btc — liquidation double-dip, MoonDev edge ──
        {"name": "liqdip-btc", "strategy_type": "liquidation_dip", "symbol": "BTC", "timeframe": "5m", "size_usd": 100, "leverage": 3, "params": {"liq_threshold_usd": 500000, "bounce_pct": 0.005, "dip_tolerance_pct": 0.002, "max_liq_age_hours": 4, "mode": "double_dip", "take_profit_pct": 1.5, "stop_loss_pct": 10.0, "time_limit_hours": 24, "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5, "min_signal_strength": 0.60}},
        # ── PAPER TEST: flip-flop-btc-v2 — improved R:R (12%/5%) + ADX>25 filter ──
        {"name": "flip-flop-btc-v2", "strategy_type": "flip_flop", "symbol": "BTC", "timeframe": "1h",
         "size_usd": 100, "leverage": 4,
         "params": {"atr_period": 10, "multiplier": 3.0, "cooldown_seconds": 0,
                    "max_trades_per_hour": 24, "min_signal_strength": 0.80,
                    "adx_period": 14, "adx_threshold": 25.0},
         "target_pct": 12.0, "max_loss_pct": -5.0},
        # NOTE: conspop-btc is INTENTIONALLY EXCLUDED — paper data is stale, do not re-add.
        # ── TREND BASKET: the validated dual-MA always-in edge (sma 30/50), diversified
        #    across liquid perps × {1h,4h}. Each (coin,tf) cleared edge_probe OOS Sharpe
        #    >0.5 walk-forward (scripts/edge_probe.py, 2026-06-28). The snowball lever:
        #    many semi-independent trend bets -> smoother equity. SOL/BNB/DOGE 4h and
        #    AVAX dropped (OOS Sharpe <=0.5). Paper-first; live champion-challenger +
        #    oos_trades>=30 gate decides promotion to real capital. ──
        #    T038 (2026-06-30): added a 30m tranche (shorter standard candle = more
        #    qualified trades for the Optuna tuner to learn from, NOT HFT). Each 30m
        #    (coin,tf) cleared the SAME walk-forward bar the original basket used AND
        #    the exact live sma30/50 param cleared OOS Sharpe>0.5 (BTC 2.42 / SOL 2.36
        #    / BNB 3.69 / XRP 4.21, ~31 OOS trades each; edge_probe methodology,
        #    HL 30m ~104d). 15m was REJECTED: best-combo passed but sma30/50 backtests
        #    NEGATIVE OOS at 15m for every coin (-4.7..-6.2) -> leverage-on-noise, ADR-0001.
        #    ETH/DOGE/LINK/ARB/SUI 30m dropped (live param or best-combo <=0.5 OOS).
        *[
            {"name": f"trend-{coin.lower()}-{tf}", "strategy_type": "trend_cross",
             "symbol": coin, "timeframe": tf, "size_usd": 50, "leverage": 3,
             "target_pct": 100.0, "max_loss_pct": -20.0,
             "params": {"ma_type": "sma", "fast_period": 30, "slow_period": 50,
                        "cooldown_seconds": 0, "max_trades_per_hour": 24,
                        "min_hold_bars": 1, "min_signal_strength": 0.80}}
            for coin, tf in [
                ("BTC", "1h"), ("ETH", "1h"), ("SOL", "1h"), ("BNB", "1h"),
                ("XRP", "1h"), ("DOGE", "1h"), ("LINK", "1h"), ("SUI", "1h"), ("ARB", "1h"),
                ("BTC", "4h"), ("ETH", "4h"), ("SOL", "4h"), ("XRP", "4h"),
                ("SUI", "4h"), ("ARB", "4h"),
                ("BTC", "30m"), ("SOL", "30m"), ("BNB", "30m"), ("XRP", "30m"),
            ]
        ],
        # Formerly-running strategies (mm-eth, mm-sol, arb-eth, nw-eth, nw-sol, adx-eth,
        # mean-rev-eth, turtle-btc, bollinger-btc, conspop-btc, sdz-btc, rsi-btc, pivot-btc,
        # rsivwap-btc, sma-btc, vwma-btc, macd-btc, ichimoku-btc, emabb-btc, combo-btc,
        # corr-sol, gridfib-btc) are disabled: all had PnL < 0 with N > 10 trades.
        # Re-enable only after fresh live validation shows positive edge.
    ]
    for w in winners:
        try:
            # lookback_days default of 14 covers all strategies' min-bar requirements:
            # 4h × 14d = 84 bars (ichimoku needs 57, grid_fib needs 50),
            # 1h × 14d = 336 bars, 15m × 14d = 1344 bars, 5m × 14d = 4032 bars.
            # sdz uses zone_lookback_days=30 (param) so its own data needs 30 days.
            # Without this, the SQLAlchemy default of 7 was failing to apply for
            # reasons not yet root-caused, leaving lookback_days=None → orchestrator
            # called get_ohlcv(..., None) → undersized data window → strategies on
            # long timeframes never accumulated enough bars to generate signals.
            inst = models.StrategyInstance(
                name=w["name"], strategy_type=w["strategy_type"], symbol=w["symbol"],
                timeframe=w["timeframe"], leverage=w.get("leverage", 3),
                size_usd=w.get("size_usd", 100),
                target_pct=w.get("target_pct", 9.0),
                max_loss_pct=w.get("max_loss_pct", -8.0),
                lookback_days=w.get("lookback_days", 14),
                interval_seconds=30, enabled=True, params=w.get("params", {}),
                tier="bonus_algos", status="running",
            )
            db.add(inst)
            db.commit()
            logger.info("Auto-deploy: created %s (%s on %s)", w["name"], w["strategy_type"], w["symbol"])
        except Exception as e:
            db.rollback()
            logger.warning("Auto-deploy: failed to create %s — %s", w["name"], e)



import concurrent.futures as _cf


def _await_xsec_task(task):
    """Return an asyncio-awaitable for an xsec engine handle. Boot-restart yields an
    asyncio.Task; the sync POST endpoint schedules via run_coroutine_threadsafe and
    yields a concurrent.futures.Future, which asyncio.gather can't await directly."""
    if isinstance(task, _cf.Future):
        return asyncio.wrap_future(task)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
    # ponytail: optimizer CPCV sims (backtest-<type>-<uuid>, backtester.py:99) trip
    # their circuit breaker on simulated losing streaks and flood Railway at CRITICAL
    # under logger "strategy.backtest-*" (~20/s), drowning ALL real observability.
    # Drop them at the HANDLER level — child-logger records skip ancestor-logger
    # filters, so a filter on the "strategy"/root logger alone wouldn't catch them.
    # Real-strategy CB logs (strategy.<name>) still pass; does NOT throttle RBI.
    class _DropBacktestSimLogs(logging.Filter):
        def filter(self, record):
            return not record.name.startswith("strategy.backtest-")
    # Own the "strategy" logger's output: route ALL strategy.* records through a
    # single handler that drops backtest sims, and stop propagation so they don't
    # also reach root/lastResort unfiltered. Robust to root-handler timing — every
    # strategy.<name> logger propagates to this "strategy" parent. Real-strategy CB
    # logs still emit here; only strategy.backtest-* is dropped.
    _strat_logger = logging.getLogger("strategy")
    _already = any(
        f.__class__.__name__ == "_DropBacktestSimLogs"
        for h in _strat_logger.handlers for f in h.filters
    )
    if not _already:
        _bt_handler = logging.StreamHandler()
        _bt_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        _bt_handler.addFilter(_DropBacktestSimLogs())
        _strat_logger.addHandler(_bt_handler)
        _strat_logger.propagate = False

    orchestrator = None
    risk_controller = None
    liquidation_tracker = None
    whale_tracker = None
    rbi_agent = None
    regime_detector = None

    # ── 1. Core Trading Engine ──
    # TRADING_MODE: "paper" (no wallet needed), "testnet", or "mainnet"
    trading_mode = os.getenv("TRADING_MODE", "paper").lower()
    paper_mode = trading_mode == "paper"
    client = None
    executor = None

    if paper_mode:
        # Paper trading: no wallet needed, simulated fills against live prices
        try:
            from src.execution.paper_executor import PaperTradingExecutor
            from src.lib.nice_funcs import HyperliquidDataClient

            initial_balance = float(os.getenv("PAPER_BALANCE", "10000"))
            data_network = os.getenv("PAPER_PRICE_SOURCE", "mainnet")
            client = HyperliquidDataClient(network=data_network)
            executor = PaperTradingExecutor(
                base_url=client.base_url,
                initial_balance=initial_balance,
            )
            logger.info(
                "PAPER MODE | balance=$%.0f | prices from %s",
                initial_balance, data_network,
            )
        except Exception as e:
            logger.warning("Could not initialize paper trading: %s", e)
    else:
        # Testnet or mainnet: requires wallet/private key
        try:
            from src.lib.nice_funcs import HyperliquidClient
            from src.execution.hl_executor import HyperliquidVaultExecutor

            network = "mainnet" if trading_mode == "mainnet" else "testnet"
            client = HyperliquidClient(network=network)
            executor = HyperliquidVaultExecutor(client=client)
            logger.info("%s MODE | account=%s", network.upper(), client.account.address)
        except Exception as e:
            logger.warning("Could not initialize %s executor: %s", trading_mode, e)

    # ── 2. Regime Detector (moved before orchestrator so it can be injected) ──
    try:
        from src.services.regime_detector import RegimeDetector
        regime_detector = RegimeDetector()
        logger.info("RegimeDetector initialized")
    except Exception as e:
        logger.warning("Could not initialize RegimeDetector: %s", e)

    # ── 3. Strategy Orchestrator (with MoonDev profitability controls) ──
    try:
        from src.engine.orchestrator import StrategyOrchestrator
        from src.services.liquidation_guard import LiquidationGuard
        from src.services.hlp_gate import HLPSentimentGate
        if client and executor:
            liquidation_guard = LiquidationGuard()
            hlp_gate = HLPSentimentGate()
            orchestrator = StrategyOrchestrator(
                client=client,
                executor=executor,
                regime_detector=regime_detector,
                liquidation_guard=liquidation_guard,
                hlp_gate=hlp_gate,
                max_global_trades_per_hour=100,
                daily_loss_limit_pct=2.0,
                max_portfolio_exposure_pct=80.0,
            )
            logger.info("StrategyOrchestrator initialized | mode=%s | regime_gate=ON | liq_guard=ON | hlp_gate=ON", trading_mode.upper())
    except Exception as e:
        logger.warning("Could not initialize StrategyOrchestrator: %s", e)

    # ── 4. Risk Controller (Layer 0: The Seatbelt) ──
    try:
        from src.services.risk_controller import RiskController
        if executor:
            risk_controller = RiskController(client=client, executor=executor)
            logger.info("RiskController initialized | mode=%s", trading_mode.upper())
            # Inject into orchestrator now that it's available
            if orchestrator is not None:
                orchestrator.risk_controller = risk_controller
        else:
            logger.warning("RiskController skipped — no executor available")
    except Exception as e:
        logger.warning("Could not initialize RiskController: %s", e)

    # ── 5. Liquidation Tracker (Layer 1: The Eyes) ──
    try:
        from src.services.liquidation_tracker import LiquidationTracker
        hl_base_url = client.base_url if client else "https://api.hyperliquid.xyz"
        liquidation_tracker = LiquidationTracker(base_url=hl_base_url)
        await liquidation_tracker.start()
        logger.info("LiquidationTracker initialized + started | url=%s", hl_base_url)
    except Exception as e:
        logger.warning("Could not initialize LiquidationTracker: %s", e)

    # ── 6. Whale Tracker (Layer 1: The Eyes) ──
    try:
        from src.services.whale_tracker import WhaleTracker
        hl_base_url = client.base_url if client else "https://api.hyperliquid.xyz"
        whale_tracker = WhaleTracker(base_url=hl_base_url)
        await whale_tracker.start()
        logger.info("WhaleTracker initialized + started | url=%s", hl_base_url)
    except Exception as e:
        logger.warning("Could not initialize WhaleTracker: %s", e)

    # ── 7. RBI Agent (Layer 2: The Brain) ──
    try:
        from src.services.rbi_agent import RBIAgentManager
        rbi_agent = RBIAgentManager()
        logger.info("RBIAgentManager initialized")
    except Exception as e:
        logger.warning("Could not initialize RBIAgentManager: %s", e)

    # ── 8. Solana DEX Scanner ──
    solana_scanner = None
    try:
        solana_enabled = os.getenv("SOLANA_SCANNER_ENABLED", "true").lower() == "true"
        if solana_enabled:
            from src.services.solana_scanner import SolanaScanner, ScannerConfig
            solana_config = ScannerConfig()
            solana_scanner = SolanaScanner(config=solana_config)
            logger.info(
                "SolanaScanner initialized | birdeye_key=%s | balance=$%.0f",
                "SET" if solana_config.birdeye_api_key else "NOT SET",
                solana_config.paper_balance,
            )
    except Exception as e:
        logger.warning("Could not initialize SolanaScanner: %s", e)

    # ── 9. Funding Rate Monitor ──
    funding_monitor = None
    try:
        from src.services.funding_monitor import FundingMonitor
        hl_info_url = (
            f"{client.base_url}/info" if client and hasattr(client, "base_url")
            else "https://api.hyperliquid.xyz/info"
        )
        funding_monitor = FundingMonitor(base_url=hl_info_url, auto_start=True)
        logger.info("FundingMonitor initialized | url=%s", hl_info_url)
        if orchestrator is not None:
            orchestrator.funding_monitor = funding_monitor
    except Exception as e:
        logger.warning("Could not initialize FundingMonitor: %s", e)

    if orchestrator is not None and liquidation_tracker is not None:
        orchestrator.liquidation_tracker = liquidation_tracker
        logger.info("LiquidationTracker injected into orchestrator (chaos gate active)")

    # Attach all to app.state
    app.state.orchestrator = orchestrator
    app.state.risk_controller = risk_controller
    app.state.liquidation_tracker = liquidation_tracker
    app.state.whale_tracker = whale_tracker
    app.state.rbi_agent = rbi_agent
    app.state.regime_detector = regime_detector
    app.state.trading_mode = trading_mode
    app.state.paper_mode = paper_mode
    app.state.executor = executor
    app.state.client = client   # needed by POST /xsec/instances to start engines at runtime

    # T010: stamp each paper trade with the current regime at entry, so
    # regime-conditional edge history accrues durably (sharpens the leaderboard
    # posterior over time). Fail-safe: a None/lookup miss just leaves it untagged.
    if executor is not None and regime_detector is not None and hasattr(executor, "_regime_fn"):
        executor._regime_fn = lambda _sym: (regime_detector.get_current_regime(_sym) or {}).get("regime")
    app.state.solana_scanner = solana_scanner
    app.state.funding_monitor = funding_monitor

    # ── Restore paper trading state from disk ──
    if paper_mode and executor is not None:
        try:
            executor.load_state()
        except Exception as e:
            logger.warning("Paper state restore failed: %s", e)

    # ── Rehydrate PnL stats + daily_starting_balance from DB ──
    # Runs after load_state() so DB values only fill gaps not covered by the JSON snapshot.
    # The trade history loaded from paper_state.json already carries per-trade PnL, so the
    # executor's in-memory stats are authoritative once load_state() succeeds.  We only
    # restore _daily_starting_balance (orchestrator circuit-breaker) from the DB here,
    # because that lives on the orchestrator, not on the executor JSON snapshot.
    try:
        from .database import SessionLocal as _RH_SL
        _rh_db = _RH_SL()
        try:
            _snap_rows = _rh_db.query(models.PnlSnapshot).all()
            if _snap_rows:
                # Restore daily_starting_balance for the circuit breaker
                _port_snap = next((r for r in _snap_rows if r.strategy_name == "_portfolio"), None)
                if _port_snap and _port_snap.daily_starting_balance and orchestrator is not None:
                    if orchestrator._daily_starting_balance is None:
                        orchestrator._daily_starting_balance = _port_snap.daily_starting_balance
                        logger.info(
                            "PnL rehydrate: daily_starting_balance=$%.2f (from DB)",
                            _port_snap.daily_starting_balance,
                        )

                # Restore weekly circuit-breaker state.
                # Also set _weekly_start so _check_weekly_drawdown() doesn't immediately
                # overwrite the restored balance on the first tick (reset branch fires
                # when _weekly_start is None).
                _weekly_snap = next((r for r in _snap_rows if r.strategy_name == "_weekly"), None)
                if _weekly_snap and _weekly_snap.daily_starting_balance and orchestrator is not None:
                    if orchestrator._weekly_starting_balance is None:
                        from datetime import timedelta as _td
                        _now_utc = datetime.now(timezone.utc)
                        _monday = _now_utc - _td(days=_now_utc.weekday())
                        orchestrator._weekly_starting_balance = _weekly_snap.daily_starting_balance
                        orchestrator._weekly_start = _monday.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )
                        logger.info(
                            "PnL rehydrate: weekly_starting_balance=$%.2f week_start=%s (from DB)",
                            _weekly_snap.daily_starting_balance, orchestrator._weekly_start,
                        )

                # Restore monthly circuit-breaker state.
                # Same pattern: set _monthly_start to prevent immediate overwrite.
                _monthly_snap = next((r for r in _snap_rows if r.strategy_name == "_monthly"), None)
                if _monthly_snap and _monthly_snap.daily_starting_balance and orchestrator is not None:
                    if orchestrator._monthly_starting_balance is None:
                        _now_utc2 = datetime.now(timezone.utc)
                        orchestrator._monthly_starting_balance = _monthly_snap.daily_starting_balance
                        orchestrator._monthly_start = _now_utc2.replace(
                            day=1, hour=0, minute=0, second=0, microsecond=0
                        )
                        logger.info(
                            "PnL rehydrate: monthly_starting_balance=$%.2f month_start=%s (from DB)",
                            _monthly_snap.daily_starting_balance, orchestrator._monthly_start,
                        )

                _strat_snaps = [r for r in _snap_rows if r.strategy_name not in ("_portfolio", "_weekly", "_monthly")]
                logger.info(
                    "PnL rehydrate: found %d strategy snapshot(s) in DB (trade history from paper_state.json is authoritative)",
                    len(_strat_snaps),
                )
            else:
                logger.info("PnL rehydrate: no snapshots in DB (fresh deploy)")
        finally:
            _rh_db.close()
    except Exception as _rh_e:
        logger.warning("PnL rehydrate failed (non-fatal): %s", _rh_e)

    # ── Auto-start strategies that were running before shutdown ──
    if orchestrator is not None:
        try:
            from .database import SessionLocal
            from src.strategies.base_strategy import StrategyConfig, StrategyTier
            from src.strategies.registry import list_strategies

            db = SessionLocal()
            try:
                # If DB is empty (fresh deploy), auto-deploy winner strategies
                all_instances = db.query(models.StrategyInstance).count()
                if all_instances == 0:
                    logger.info("Auto-deploy: empty DB detected — deploying winner strategies")
                    _auto_deploy_winners(db, orchestrator)
                instances_to_start = db.query(models.StrategyInstance).filter(
                    models.StrategyInstance.enabled == True,
                    models.StrategyInstance.status != "stopped",
                ).all()

                if instances_to_start:
                    logger.info("Auto-start: found %d enabled, non-stopped strategies", len(instances_to_start))
                    tier_map = {"A": StrategyTier.A, "B": StrategyTier.B, "C": StrategyTier.C, "D": StrategyTier.D}
                    available = [s["strategy_type"] for s in list_strategies()]

                    for inst in instances_to_start:
                        try:
                            if inst.strategy_type not in available:
                                logger.warning(
                                    "Auto-start: skipping %s — unknown strategy_type %s",
                                    inst.name, inst.strategy_type,
                                )
                                continue

                            config = StrategyConfig(
                                name=inst.name,
                                symbol=inst.symbol,
                                tier=tier_map.get(inst.tier, StrategyTier.A),
                                timeframe=inst.timeframe,
                                leverage=inst.leverage,
                                size_usd=inst.size_usd,
                                target_pct=inst.target_pct,
                                max_loss_pct=inst.max_loss_pct,
                                lookback_days=inst.lookback_days,
                                interval_seconds=inst.interval_seconds,
                                enabled=True,
                                params=inst.params or {},
                            )
                            orchestrator.add_strategy(inst.name, inst.strategy_type, config)
                            await orchestrator.start_strategy(inst.name)
                            inst.status = "running"          # persist so cull/compounder see the real running book (≥CULL_MIN_RUNNING)
                            inst.started_at = datetime.utcnow()
                            inst.error_message = None
                            logger.info("Auto-start: started %s (%s on %s)", inst.name, inst.strategy_type, inst.symbol)
                        except Exception as e:
                            logger.warning("Auto-start: failed to start %s — %s", inst.name, e)
                            inst.status = "error"
                            inst.error_message = f"Auto-start failed: {e}"
                            db.commit()
                    db.commit()
                else:
                    logger.info("Auto-start: no enabled strategies found")

                logger.info("Auto-start: started %d enabled strategies", len(instances_to_start))
            finally:
                db.close()
        except Exception as e:
            logger.warning("Auto-start: could not restore strategies — %s", e)

    # ── Restore per-strategy circuit-breaker state from the durable ledger (F5/R11/R8) ──
    # Runs after load_state() (ledger rehydrated) AND auto-start (strategies built).
    # Without this, a strategy that tripped its circuit breaker (e.g. 5 consecutive
    # losses) silently re-enables on every redeploy because StrategyState is in-memory.
    if paper_mode and executor is not None and orchestrator is not None:
        try:
            for _strat in orchestrator._strategies.values():
                _n = executor.replay_strategy_state(_strat)
                if _n:
                    logger.info(
                        "Strategy-state rehydrate: %s replayed %d trades | cb=%s",
                        _strat.config.name, _n, _strat.state.circuit_breaker_triggered,
                    )
        except Exception as e:
            logger.warning("Strategy-state rehydrate failed: %s", e)

    # ── Periodic paper state saver (every 5 minutes) ──
    async def _paper_state_saver():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            if paper_mode and executor is not None:
                try:
                    executor.save_state()
                except Exception as e:
                    logger.warning("Paper state save failed: %s", e)

    if paper_mode and executor is not None:
        asyncio.create_task(_paper_state_saver())

    # ── PnL flush loop (every 5 minutes) — persist track record across redeploys ──
    async def _pnl_flush_loop():
        """Every 300 s, upsert per-strategy PnL stats + daily_starting_balance to SQLite.

        Uses strategy_name="_portfolio" as a reserved row for the orchestrator's
        daily circuit-breaker balance so it survives a Railway redeploy.
        """
        from .database import SessionLocal as _PFLSL
        while True:
            await asyncio.sleep(300)
            try:
                if executor is None:
                    continue

                # Build per-strategy stats from live executor trade history
                _pfl_stats: dict = {}
                if hasattr(executor, "get_trade_history"):
                    for _t in executor.get_trade_history():
                        if _t.get("action") != "exit":
                            continue
                        _sname = _t.get("strategy", "")
                        if not _sname:
                            continue
                        if _sname not in _pfl_stats:
                            _pfl_stats[_sname] = {"total_pnl": 0.0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0}
                        _pfl_stats[_sname]["total_trades"] += 1
                        _pfl_stats[_sname]["total_pnl"] += _t.get("pnl", 0.0)
                        if (_t.get("pnl") or 0.0) > 0:
                            _pfl_stats[_sname]["winning_trades"] += 1
                        else:
                            _pfl_stats[_sname]["losing_trades"] += 1

                _dsb = orchestrator._daily_starting_balance if orchestrator is not None else None

                _pfl_db = _PFLSL()
                try:
                    for _sname, _s in _pfl_stats.items():
                        _row = _pfl_db.query(models.PnlSnapshot).filter(
                            models.PnlSnapshot.strategy_name == _sname
                        ).first()
                        if _row is None:
                            _row = models.PnlSnapshot(strategy_name=_sname)
                            _pfl_db.add(_row)
                        _row.total_pnl = round(_s["total_pnl"], 4)
                        _row.total_trades = _s["total_trades"]
                        _row.winning_trades = _s["winning_trades"]
                        _row.losing_trades = _s["losing_trades"]

                    # Portfolio row: stores daily_starting_balance for the circuit breaker
                    _port_row = _pfl_db.query(models.PnlSnapshot).filter(
                        models.PnlSnapshot.strategy_name == "_portfolio"
                    ).first()
                    if _port_row is None:
                        _port_row = models.PnlSnapshot(strategy_name="_portfolio")
                        _pfl_db.add(_port_row)
                    if _dsb is not None:
                        _port_row.daily_starting_balance = _dsb
                    # Canonical portfolio PnL = balance − initial_balance (survives redeploy via paper_state.json)
                    if executor is not None and hasattr(executor, "balance") and hasattr(executor, "initial_balance"):
                        _port_row.balance_pnl = round(executor.balance - executor.initial_balance, 2)

                    # Weekly row: daily_starting_balance=weekly_start_balance,
                    # total_trades encodes _weekly_halved flag (1=halved, 0=not)
                    _wsb = orchestrator._weekly_starting_balance if orchestrator is not None else None
                    _whalved = int(orchestrator._weekly_halved) if orchestrator is not None else 0
                    _weekly_row = _pfl_db.query(models.PnlSnapshot).filter(
                        models.PnlSnapshot.strategy_name == "_weekly"
                    ).first()
                    if _weekly_row is None:
                        _weekly_row = models.PnlSnapshot(strategy_name="_weekly")
                        _pfl_db.add(_weekly_row)
                    if _wsb is not None:
                        _weekly_row.daily_starting_balance = _wsb
                    _weekly_row.total_trades = _whalved

                    # Monthly row: daily_starting_balance=monthly_start_balance,
                    # total_trades encodes _monthly_halt_triggered (1=halted, 0=not)
                    _msb = orchestrator._monthly_starting_balance if orchestrator is not None else None
                    _mhalt = int(orchestrator._monthly_halt_triggered) if orchestrator is not None else 0
                    _monthly_row = _pfl_db.query(models.PnlSnapshot).filter(
                        models.PnlSnapshot.strategy_name == "_monthly"
                    ).first()
                    if _monthly_row is None:
                        _monthly_row = models.PnlSnapshot(strategy_name="_monthly")
                        _pfl_db.add(_monthly_row)
                    if _msb is not None:
                        _monthly_row.daily_starting_balance = _msb
                    _monthly_row.total_trades = _mhalt

                    _pfl_db.commit()
                    logger.info(
                        "PnL flush: %d strategies | dsb=$%s",
                        len(_pfl_stats),
                        f"{_dsb:.2f}" if _dsb is not None else "None",
                    )
                finally:
                    _pfl_db.close()
            except Exception as _pfl_e:
                logger.warning("PnL flush error (non-fatal): %s", _pfl_e)

    asyncio.create_task(_pnl_flush_loop())

    # ── _BTC_WINNERS static boost REMOVED (U6 / R5) ──
    # The hardcoded per-name leverage/size boost (flip-flop-btc 4x, etc., a frozen
    # 2026-06-07 snapshot) re-leveraged named strategies on every boot regardless
    # of CURRENT live edge — re-boosting strategies that had since decayed into
    # losers. Allocation is now confidence-governed in _compound_job
    # (_winner_cap_usd scales each winner's share by its earned edge_confidence_score),
    # and the winner set de-freezes (_defrosted_winner_set) so a sustained live
    # loser is demoted rather than perpetually boosted. No static leverage boost.

    # ── Auto-start risk controller (Layer 0 seatbelt — always on) ──
    if risk_controller is not None:
        try:
            await risk_controller.start()
            logger.info("RiskController auto-started | monitoring active")
        except Exception as e:
            logger.warning("RiskController auto-start failed: %s", e)

    # ── RBI autonomous optimizer scheduler ──
    from src.api.routes.rbi_optimize import _get_or_create_pipeline
    _scheduler = AsyncIOScheduler()
    _RBI_SCHEDULE = [
        ("nadaraya_watson", 4,  "ETH", "1h", 4),
        ("sma_crossover",   17, "BTC", "1h", 4),
        ("rsi",             14, "BTC", "1h", 4),
        ("adx",             6,  "ETH", "1h", 4),
        ("macd",            19, "BTC", "1h", 4),
        ("correlation",     23, "SOL", "1h", 4),
        ("market_maker",    1,  "ETH", "1h", 4),
        ("closed_market_overnight", 27, "BTC", "1h", 4),
        ("bollinger",       10, "BTC", "1h", 8),
        ("ichimoku",        20, "BTC", "4h", 8),
        ("grid_fibonacci",  26, "BTC", "4h", 8),
        ("vwma",            18, "BTC", "1h", 8),
        ("mean_reversion",  8,  "ETH", "1h", 8),
        ("consolidation_pop", 11, "BTC", "15m", 24),
        ("vwap_bot",          7,  "BTC", "1h",  24),
        ("pivot_lines",       15, "BTC", "1h",  24),
        ("flip_flop",         39, "BTC", "1h",  4),
        ("liquidation_dip",   28, "BTC", "5m",  8),
        ("turtle",             9, "BTC", "1h",  8),
    ]

    # Hard concurrency cap on top of start-time staggering. Staggering alone
    # only spreads *when jobs start* -- once running, each optimize() call
    # (100 Optuna trials, CPU-bound work offloaded to a thread) takes real
    # wall time, so by minute ~5 of the boot window most of the ~21 staggered
    # jobs are running *concurrently* regardless of their start offsets,
    # which is exactly the cold-start storm stop_if warns about (proven live
    # 2026-06-30: with no cap, self-loopback httpx calls timed out even with
    # a 20s timeout + retry). A semaphore bounds true concurrent CPU load.
    _RBI_BOOT_CONCURRENCY = 3
    _rbi_semaphore = asyncio.Semaphore(_RBI_BOOT_CONCURRENCY)

    async def _rbi_job(strategy_type: str, strategy_id: int, symbol: str, timeframe: str):
        async with _rbi_semaphore:
            pipeline = _get_or_create_pipeline(strategy_type)
            try:
                event = await pipeline.run_cycle(
                    strategy_type=strategy_type, strategy_id=strategy_id,
                    symbol=symbol, timeframe=timeframe, lookback_days=90, n_trials=100,
                )
                if event.promoted:
                    logger.info("Scheduler RBI promoted %s: %s", strategy_type, event.after_metrics)
            except Exception as e:
                # repr(), not str(): httpx timeout/connect exceptions stringify
                # to "" (proven live 2026-06-30 -- "failed for X: " with no
                # detail, making the boot-storm root cause undiagnosable from
                # logs alone).
                logger.error("Scheduled RBI cycle failed for %s: %r", strategy_type, e, exc_info=True)

    # Build DB-derived schedule, filtered to optimizer-supported strategy types.
    # Falls back to the hardcoded _RBI_SCHEDULE if no running instances exist.
    _supported_types: set[str] = set(PARAM_SPACES.keys())
    from .database import SessionLocal as _RBI_SL
    try:
        _rbi_db = _RBI_SL()
        try:
            _running_instances = _rbi_db.query(models.StrategyInstance).filter(
                models.StrategyInstance.enabled == True
            ).all()
        finally:
            _rbi_db.close()
    except Exception as _dbe:
        logger.warning("RBI scheduler: DB query failed (%s); using hardcoded schedule", _dbe)
        _running_instances = []

    # Per-job stagger: spread boot-run offsets evenly across 30s-10min so the
    # ~19-21 RBI jobs don't thunder-herd on startup. Pure random.randint draws
    # (the old approach) can land several jobs within 1-2s of each other by
    # chance (birthday-paradox clustering) — proven live 2026-06-30: 4 jobs
    # landed within a 4s window, each one's optimize() Optuna-CPU burst then
    # starved the others' self-loopback httpx calls (GET /strategies) past
    # their timeout, so the boot-run *fired* but mostly *failed* and those
    # jobs silently fell back to waiting their full 4-24h interval — defeating
    # the whole point of a boot-run. Deterministic even-spacing (small random
    # jitter only to avoid identical-tick collisions across restarts)
    # guarantees a real minimum gap between any two jobs' boot-fire times.
    _JITTER_MIN_S = 30
    _JITTER_MAX_S = 600
    _JITTER_SPAN_S = _JITTER_MAX_S - _JITTER_MIN_S

    def _stagger_offset_s(index: int, total: int) -> int:
        step = _JITTER_SPAN_S / max(total, 1)
        base = _JITTER_MIN_S + index * step
        return int(base + random.randint(0, 5))

    if _running_instances:
        _job_specs = build_rbi_job_specs(_running_instances, _supported_types)
        _skipped = [i.strategy_type for i in _running_instances if i.strategy_type not in _supported_types]
        if _skipped:
            logger.warning("RBI scheduler: skipping unsupported strategy types (not in param_spaces): %s", sorted(set(_skipped)))
        for _i, _spec in enumerate(_job_specs):
            _jitter = timedelta(seconds=_stagger_offset_s(_i, len(_job_specs)))
            _scheduler.add_job(
                _rbi_job, "interval", hours=_spec["hours"],
                args=[_spec["strategy_type"], _spec["strategy_id"], _spec["symbol"], _spec["timeframe"]],
                id=f"rbi_{_spec['strategy_type']}_{_spec['strategy_id']}",
                replace_existing=True,
                next_run_time=datetime.now(timezone.utc) + _jitter,
            )
        logger.info("RBI scheduler: %d DB-derived jobs scheduled (supported types: %s)", len(_job_specs), sorted(_supported_types))
    else:
        # Fallback: no running instances — use hardcoded schedule so behaviour
        # doesn't regress on an empty DB.
        _fallback_skipped = [stype for stype, *_ in _RBI_SCHEDULE if stype not in _supported_types]
        if _fallback_skipped:
            logger.warning("RBI scheduler (fallback): skipping unsupported types: %s", sorted(set(_fallback_skipped)))
        _fallback_specs = [s for s in _RBI_SCHEDULE if s[0] in _supported_types]
        for _i, (stype, sid, sym, tf, hours) in enumerate(_fallback_specs):
            _jitter = timedelta(seconds=_stagger_offset_s(_i, len(_fallback_specs)))
            _scheduler.add_job(
                _rbi_job, "interval", hours=hours,
                args=[stype, sid, sym, tf],
                id=f"rbi_{stype}",
                replace_existing=True,
                next_run_time=datetime.now(timezone.utc) + _jitter,
            )
        logger.info(
            "RBI scheduler: empty DB — fell back to %d hardcoded jobs",
            len(_fallback_specs),
        )

    # ── Compounding controller: reinvest 90% of profits every 30 min ──
    _COMPOUND_BASE = 100.0          # treat $100 as starting capital — everything above is investable
    _COMPOUND_RESERVE_PCT = 0.10    # keep 10% of total balance as reserve
    _INITIAL_BALANCE = float(os.getenv("PAPER_BALANCE", "10000"))

    async def _compound_job():
        try:
            if not paper_mode or executor is None:
                return
            balance = executor.balance
            investable = max(_COMPOUND_BASE, balance * (1.0 - _COMPOUND_RESERVE_PCT))
            from .database import SessionLocal as _CSL
            _cdb = _CSL()
            try:
                running = _cdb.query(models.StrategyInstance).filter(
                    models.StrategyInstance.status == "running"
                ).all()
                n = len(running)
                if n == 0:
                    return
                # Build real per-strategy stats from live executor memory (DB is always stale —
                # the GET /strategies endpoint merges paper stats in-memory but never commits)
                _live_stats: dict = {}
                for _t in executor.get_trade_history():
                    _sname = _t.get("strategy", "")
                    if not _sname:
                        continue
                    if _sname not in _live_stats:
                        _live_stats[_sname] = {"pnl": 0.0, "trades": 0, "wins": 0}
                    if _t.get("action") == "exit":
                        _live_stats[_sname]["trades"] += 1
                        _live_stats[_sname]["pnl"] += _t.get("pnl", 0.0)
                        if (_t.get("pnl") or 0.0) > 0:
                            _live_stats[_sname]["wins"] += 1

                # Flush live stats to DB so other queries (and next compound run) see real numbers
                for _cinst in running:
                    if _cinst.name in _live_stats:
                        _s = _live_stats[_cinst.name]
                        _cinst.total_trades = _s["trades"]
                        _cinst.winning_trades = _s["wins"]
                        _cinst.total_pnl = round(_s["pnl"], 4)
                        if hasattr(executor, "_live_edge_stats"):
                            # U6: confidence (which governs allocation) is computed on the
                            # Wilson LOWER-BOUND win-rate, consistent with F2 entry sizing —
                            # thin-evidence luck does not inflate the allocation cap.
                            _wr, _payoff, _n, _wr_lo, _ = executor._live_edge_stats(_cinst.name)
                            _cinst.edge_confidence_score = edge_confidence(_n, _wr_lo, _payoff)

                # Dynamic winner detection from live stats.
                # Static fallback covers the 4 confirmed survivors (2026-06-07 purge)
                # plus flip-flop-btc-v2 (paper test — do not cull during trial period).
                # Only these run at all — the purge block already stopped everything else.
                _STATIC_WINNERS = {
                    'vwap-btc',           # confirmed live winner (VWAP probability bias)
                    'closed-mkt-btc',     # survivor: overnight/weekend breakout
                    'liqdip-btc',         # survivor: liquidation double-dip
                    'flip-flop-btc-v2',   # paper test: 12%/5% R:R + ADX>25 filter
                }
                _WINNER_MIN_TRADES = 6  # require a real sample — 2-4 trade noise no longer qualifies as a winner
                dynamic_winners = {
                    r.name for r in running
                    if _live_stats.get(r.name, {}).get("pnl", r.total_pnl) > 0
                    and _live_stats.get(r.name, {}).get("trades", r.total_trades) >= _WINNER_MIN_TRADES
                }
                # U6 (R10/F3): de-freeze the static winner set. Dynamic winners
                # (live-proven this window) are unioned in; a static winner stays
                # protected UNLESS it shows a sustained recent live loss, in which
                # case it drops out (no longer winner-allocated, and cullable).
                # recent_realized_pnl is redeploy-proof (JSON on the volume), so the
                # old "union always, to survive the post-redeploy reset window"
                # workaround is no longer needed — protection is earned, not frozen.
                _recent_fn = (executor.recent_realized_pnl
                              if hasattr(executor, "recent_realized_pnl")
                              else lambda _n: (0.0, 0))
                _WINNER_SET = _defrosted_winner_set(_STATIC_WINNERS, dynamic_winners, _recent_fn)

                # T010: conditional-edge posterior — drives BOTH winner promotion and
                # per-instance sizing. A high-confidence edge the crude trades>=6 gate
                # misses (funding-arb 4t, prob 0.998) is promoted into the winner tier;
                # a noisy thin sample (gridfib 4t, prob 0.59) is not.
                try:
                    _post = _posterior_scores(executor, regime_detector, running, window=30)
                except Exception as _pe:  # noqa: BLE001 - allocation must fall back, never crash the job
                    logger.warning("Compounder: posterior scores unavailable (%s); flat sizing", _pe)
                    _post = {}
                # T012: skip sizing while regimes are COLD (e.g. the boot run right after a
                # redeploy, before the HMM fits) — the posterior would mis-size off "unknown"
                # pooling and could cold-promote a loser to $500 for a cycle. Keep last-good
                # DB sizes until the detector warms; the next 15-min cycle sizes for real.
                if _post and not any(v.get("current_regime", "unknown") != "unknown" for v in _post.values()):
                    logger.info("Compounder: regimes cold (0 fitted) — keeping last-good sizes this cycle")
                    return
                _posterior_winners = {
                    r.name for r in running
                    if _post.get(r.name, {}).get("prob_edge", 0.0) >= POSTERIOR_WINNER_PROB
                    and _post.get(r.name, {}).get("posterior_mean_edge", 0.0) > 0
                }
                _WINNER_SET = _WINNER_SET | _posterior_winners

                # Winners-first: concentrate investable capital on proven winners,
                # but the per-winner share is now CONFIDENCE-GOVERNED (U6 / R5), not
                # flat. Equal-split is the upper bound (so few winners don't each
                # grab the whole ceiling); the confidence cap holds a thin-evidence
                # winner to an observation-small share and only reaches WINNER_MAX_PCT
                # near ~30 live trades. Non-winners get $100 for signal discovery only.
                # Symmetric demote (hysteresis): winner-SIZED only while the posterior
                # prob_edge holds >= POSTERIOR_DEMOTE_PROB. A decayed winner (vwap-btc
                # prob 0.0) falls out and shrinks to the exploration floor — the missing
                # half of promotion. Missing posterior (failed) -> keep (fail-safe).
                def _winner_sized(_nm):
                    return (_nm in _WINNER_SET
                            and _post.get(_nm, {}).get("prob_edge", 1.0) >= POSTERIOR_DEMOTE_PROB)

                _winners_running = [r for r in running if _winner_sized(r.name)]
                _equal_split = (
                    round(investable / max(len(_winners_running), 1), 2)
                ) if _winners_running else 100.0
                _OTHER_SIZE = 100.0

                # Kelly guard: do not compound a strategy beyond $500 until it has
                # N>=100 completed trades. A negative Kelly fraction (e.g. flip-flop-btc
                # at -16.7% on 12 trades) means there is no proven edge and uncapped
                # compounding is ruin-seeking.
                _COMPOUND_MIN_TRADES = 100
                _COMPOUND_MAX_UNPROVEN = 500.0

                for _cinst in running:
                    _p = _post.get(_cinst.name, {})
                    _prob = float(_p.get("prob_edge", 0.5))
                    _ci_high = float(_p.get("ci_high", 0.0))
                    if _winner_sized(_cinst.name):
                        # confidence-scaled per-winner cap (U6 / R5): observation-small
                        # at thin evidence, ~WINNER_MAX_PCT only near full confidence.
                        _conf = float(_cinst.edge_confidence_score or 0.0)
                        # A strategy with SUSTAINED positive realized PnL is proven by
                        # real money; floor its allocation confidence so an asymmetric
                        # fat-tail winner (low win-rate, real edge) is not pinned near
                        # the observation floor purely by its hit-rate. The per-entry
                        # half-Kelly still sizes each bet proportional to edge thickness.
                        if hasattr(executor, "recent_realized_pnl"):
                            _rp, _rn = executor.recent_realized_pnl(_cinst.name)
                            if _rn >= DEFREEZE_MIN_RECENT_TRADES and _rp > 0:
                                _conf = max(_conf, PROVEN_REALIZED_CONF)
                        # T010: drive the winner cap by the POSTERIOR confidence too —
                        # max(Wilson-LB, posterior). A high-prob thin edge (funding-arb,
                        # prob 0.998 -> conf ~1.0) grows even when its Wilson-LB win-rate
                        # confidence is still low on few trades; the ramp keeps it gradual
                        # and the $500-unproven cap bounds it.
                        _conf = max(_conf, (_prob - 0.5) / 0.5)
                        new_size = min(_equal_split, _winner_cap_usd(balance, _conf))
                        # T012 continuous demote: scale the winner size by where prob_edge
                        # sits in the demote..promote band, so a band winner (conspop 0.749)
                        # lands mid-size and full $500 is reserved for prob>=0.9 (funding-arb).
                        # Size is now a continuous function of the posterior across the range.
                        _pf = (_prob - POSTERIOR_DEMOTE_PROB) / (POSTERIOR_WINNER_PROB - POSTERIOR_DEMOTE_PROB)
                        _pf = min(max(_pf, 0.0), 1.0)
                        new_size = max(EXPLORE_MIN_USD, new_size * _pf)
                    else:
                        # T010 Thompson exploration: non-winner stake ∝ UPSIDE (ci_high),
                        # floored at EXPLORE_MIN_USD (never 0 — a real-but-young edge keeps
                        # probing), capped at _OTHER_SIZE. A dead loser (ci_high<=0) shrinks
                        # to the floor, freeing the loser-tail capital the flat $100 wasted.
                        _upside = min(max(_ci_high, 0.0) / EXPLORE_UPSIDE_REF, 1.0)
                        new_size = EXPLORE_MIN_USD + (_OTHER_SIZE - EXPLORE_MIN_USD) * _upside
                    _trade_count = _live_stats.get(_cinst.name, {}).get("trades", 0)
                    if new_size > _COMPOUND_MAX_UNPROVEN and _trade_count < _COMPOUND_MIN_TRADES:
                        logger.warning(
                            "compounder suppressed for %s: only %d trades, need %d before scaling above $%.0f (capping at $%.0f)",
                            _cinst.name, _trade_count, _COMPOUND_MIN_TRADES, _COMPOUND_MAX_UNPROVEN, _COMPOUND_MAX_UNPROVEN,
                        )
                        new_size = _COMPOUND_MAX_UNPROVEN
                    # T010 ramp: grow gradually (no >ALLOC_RAMP_UP_MAX× jump per cycle);
                    # cuts apply immediately so the bleed stops fast.
                    _prev = float(_cinst.size_usd or _OTHER_SIZE)
                    if new_size > _prev * ALLOC_RAMP_UP_MAX:
                        new_size = round(_prev * ALLOC_RAMP_UP_MAX, 2)
                    new_size = max(new_size, EXPLORE_MIN_USD)  # exploration floor — never 0
                    _cinst.size_usd = new_size
                    if orchestrator:
                        _cstrat = orchestrator.get_strategy(_cinst.name)
                        if _cstrat:
                            _cstrat.config.size_usd = new_size
                _cdb.commit()
                logger.info(
                    "Compounder: balance=$%.2f | investable=$%.2f | equal_split_cap=$%.2f | "
                    "others=$%.2f | n_winners=%d/%d | winner_set=%s (sizes confidence-scaled)",
                    balance, investable, _equal_split, _OTHER_SIZE,
                    len(_winners_running), n, sorted(_WINNER_SET),
                )
            finally:
                _cdb.close()
        except Exception as _ce:
            logger.error("Compound job error: %s", _ce)

    async def _cull_job():
        """Autonomously stop chronic loser strategies so they stop bleeding.

        Mirrors the compounder: rebuilds live per-strategy stats from
        executor.get_trade_history(), computes the same winner set (so a winner
        is never culled), then uses the pure _select_cull_candidates() helper to
        pick bleeders and stops each via the exact reversible sequence the manual
        /strategies/{name}/stop route uses.
        """
        try:
            if not (paper_mode and executor and orchestrator):
                return
            from .database import SessionLocal as _KSL
            _kdb = _KSL()
            try:
                running = _kdb.query(models.StrategyInstance).filter(
                    models.StrategyInstance.status == "running"
                ).all()
                if not running:
                    return

                # Build real per-strategy stats from live executor memory (same
                # pattern as the compounder — DB is always stale).
                _live: dict = {}
                for _t in executor.get_trade_history():
                    _sname = _t.get("strategy", "")
                    if not _sname:
                        continue
                    if _sname not in _live:
                        _live[_sname] = {"pnl": 0.0, "trades": 0, "wins": 0}
                    if _t.get("action") == "exit":
                        _live[_sname]["trades"] += 1
                        _live[_sname]["pnl"] += _t.get("pnl", 0.0)
                        if (_t.get("pnl") or 0.0) > 0:
                            _live[_sname]["wins"] += 1

                # Winner set, computed exactly as the compounder does, so a
                # confirmed winner can never be culled.
                # 2026-06-07: all survivors are static winners — the purge block
                # already stopped everything else; the cull is a safety net only.
                # flip-flop-btc-v2 is included so the cull never stops the paper test.
                _STATIC_WINNERS = {
                    'vwap-btc',           # confirmed live winner (VWAP probability bias)
                    'closed-mkt-btc',     # survivor: overnight/weekend breakout
                    'liqdip-btc',         # survivor: liquidation double-dip
                    'flip-flop-btc-v2',   # paper test: 12%/5% R:R + ADX>25 filter
                }
                _WINNER_MIN_TRADES = 1
                dynamic_winners = {
                    r.name for r in running
                    if _live.get(r.name, {}).get("pnl", r.total_pnl) > 0
                    and _live.get(r.name, {}).get("trades", r.total_trades) >= _WINNER_MIN_TRADES
                }
                # U6 (R10/F3): de-freeze. A static winner is protected from the
                # cull UNLESS it shows a sustained recent live loss (>= N recent
                # closed trades AND negative recent realized PnL) — then it becomes
                # cullable, so a frozen 2026-06-07 survivor that has since decayed
                # into a bleeder is finally demotable. Thin recent data keeps it
                # protected (safe degradation). recent_realized_pnl is redeploy-proof.
                _recent_fn = (executor.recent_realized_pnl
                              if hasattr(executor, "recent_realized_pnl")
                              else lambda _n: (0.0, 0))
                _WINNER_SET = _defrosted_winner_set(_STATIC_WINNERS, dynamic_winners, _recent_fn)

                # Stats dict in the shape _select_cull_candidates expects, scoped
                # to RUNNING strategies only.
                _stats: dict = {}
                for r in running:
                    _s = _live.get(r.name, {})
                    _trades = int(_s.get("trades", r.total_trades or 0))
                    _wins = int(_s.get("wins", r.winning_trades or 0))
                    _stats[r.name] = {
                        "pnl": float(_s.get("pnl", r.total_pnl or 0.0)),
                        "trades": _trades,
                        "win_rate": (_wins / _trades) if _trades else 0.0,
                    }

                _to_cull = _select_cull_candidates(
                    _stats, _WINNER_SET,
                    min_pnl=CULL_MIN_PNL, min_trades=CULL_MIN_TRADES,
                    min_winrate=CULL_MIN_WINRATE, max_per_run=CULL_MAX_PER_RUN,
                )

                # T010: posterior cull — a confident loser (prob_edge < POSTERIOR_CULL_PROB
                # AND negative posterior mean, with enough trades) the pnl-gate is slow to
                # catch is unioned in, to shed the loser tail faster. Winners excluded;
                # a wide-CI probe (gridfib, prob 0.59) is NOT a confident loser -> kept.
                _cull_reason_tag: dict = {}  # name -> tag, for accurate lifecycle logging below
                try:
                    _cpost = _posterior_scores(executor, regime_detector, running, window=30)
                except Exception:  # noqa: BLE001 - cull must fall back to the pnl gate, never crash
                    _cpost = {}
                for r in running:
                    _pp = _cpost.get(r.name, {})
                    if (r.name not in _WINNER_SET and r.name not in _to_cull
                            and _pp.get("prob_edge", 1.0) < POSTERIOR_CULL_PROB
                            and _pp.get("posterior_mean_edge", 0.0) < 0
                            and _pp.get("n_own", 0) >= CULL_MIN_TRADES):
                        _to_cull.append(r.name)
                        _cull_reason_tag[r.name] = "posterior_loser"

                # T028 zombie killer: conf~0 after a real sample (>=N trades) = no
                # discernible live edge either way — cull even if pnl is still
                # above CULL_MIN_PNL (a flat/breakeven bleeder of opportunity cost).
                # Same confidence math as the compounder (edge_confidence on the
                # live Wilson-lower-bound win rate) so it agrees with sizing.
                if hasattr(executor, "_live_edge_stats"):
                    for r in running:
                        if r.name in _WINNER_SET or r.name in _to_cull:
                            continue
                        if _stats.get(r.name, {}).get("trades", 0) < CULL_ZOMBIE_MIN_TRADES:
                            continue
                        try:
                            _zwr, _zpayoff, _zn, _zwr_lo, _ = executor._live_edge_stats(r.name)
                        except Exception:
                            continue
                        if edge_confidence(_zn, _zwr_lo, _zpayoff) <= CULL_ZOMBIE_MAX_CONF:
                            _to_cull.append(r.name)
                            _cull_reason_tag[r.name] = "zombie_no_edge"

                if not _to_cull:
                    return

                # T028 ANTI-FLAP: a strategy that was just revived (or culled) stays
                # untouched within the cooldown window — never re-cull a fresh
                # OBSERVATION-tier revival on its first noisy trades.
                _by_name_all = {r.name: r for r in running}
                _to_cull = [
                    _n for _n in _to_cull
                    if (_by_name_all.get(_n) is None
                        or not _lifecycle_in_cooldown(_kdb, _by_name_all[_n].id, REVIVE_COOLDOWN_HOURS))
                ]
                if not _to_cull:
                    return

                # Floor guard (2026-06-05 operator directive): the faster/looser cull
                # thresholds must never strip the running book below CULL_MIN_RUNNING.
                # _to_cull is already worst-PnL-first, so slicing keeps the worst bleeders.
                _max_cullable = max(0, len(running) - CULL_MIN_RUNNING)
                if _max_cullable <= 0:
                    logger.info(
                        "Bleeder-cull: %d running at/below floor %d — skipping cull this tick",
                        len(running), CULL_MIN_RUNNING,
                    )
                    return
                if len(_to_cull) > _max_cullable:
                    _to_cull = _to_cull[:_max_cullable]

                _by_name = {r.name: r for r in running}
                _culled = 0
                for _name in _to_cull:
                    _inst = _by_name.get(_name)
                    if _inst is None:
                        continue
                    _m = _stats.get(_name, {})
                    _pnl = _m.get("pnl", 0.0)
                    _trd = _m.get("trades", 0)
                    _wr = _m.get("win_rate", 0.0)
                    _reason = _cull_reason_tag.get(_name) or (
                        "pnl<=min" if _pnl <= CULL_MIN_PNL else "winrate<min"
                    )
                    # Exact reversible stop sequence from /strategies/{name}/stop.
                    try:
                        await orchestrator.stop_strategy(_name)
                        orchestrator.remove_strategy(_name)
                    except Exception as _oe:
                        logger.warning(
                            "Cull: orchestrator stop error for %s (continuing): %s", _name, _oe
                        )
                    # xsec engines live in app.state, not the orchestrator — without
                    # this a culled xsec instance keeps trading behind status=stopped
                    # (live zombie: 112 re-opened legs post-cull 2026-07-01 21:14Z).
                    _xmap = getattr(app.state, "xsec_driver_engines", {})
                    if _name in _xmap:
                        _xeng, _xtask = _xmap.pop(_name)
                        try:
                            _xeng.stop()
                            _xtask.cancel()
                            logger.info("Cull: stopped xsec engine %s", _name)
                        except Exception as _xse:
                            logger.warning("Cull: xsec engine stop error for %s: %s", _name, _xse)
                    if hasattr(executor, "close_by_strategy"):
                        try:
                            _results = await executor.close_by_strategy(_name)
                            _closed = sum(1 for _r in _results if _r.success)
                            if _closed:
                                logger.info("Cull: closed %d orphan position(s) for %s", _closed, _name)
                        except Exception as _fe:
                            logger.warning(
                                "Cull: orphan flush error for %s (continuing): %s", _name, _fe
                            )
                    _inst.status = "stopped"
                    _kdb.commit()
                    try:
                        _record_lifecycle_event(
                            _kdb, strategy_type=_inst.strategy_type, strategy_id=_inst.id,
                            reason=f"lifecycle_culled_{_reason}", promoted=False,
                            before_params=dict(_inst.params or {}), after_params={},
                            before_metrics={"pnl": _pnl, "trades": _trd, "win_rate": _wr},
                            after_metrics={},
                        )
                    except Exception as _le:
                        logger.warning("Cull: lifecycle event persist failed for %s: %s", _name, _le)
                    _culled += 1
                    logger.warning(
                        "Bleeder-cull: stopped %s | pnl=$%.2f | trades=%d | win_rate=%.2f | reason=%s",
                        _name, _pnl, _trd, _wr, _reason,
                    )
                logger.info(
                    "Bleeder-cull run: culled %d/%d running (winners protected: %s)",
                    _culled, len(running), sorted(_WINNER_SET),
                )
            finally:
                _kdb.close()
        except Exception as _ke:
            logger.error("Cull job error: %s", _ke)

    async def _revive_job():
        """T028 — a running/stopped Strategy state is never terminal.

        Continuously re-evaluates the STOPPED set (this IS the graveyard sweep
        — it naturally walks the whole stopped pool over successive ticks
        since REVIVE_MAX_PER_RUN bounds compute per tick):
          1) REVIVE-BY-REGIME (cheap, no optimizer): existing params now fit
             the CURRENT regime (regime_detector.should_trade) -> re-enable
             unchanged.
          2) REVIVE-BY-RETUNE (deterministic Optuna only, PARAM_SPACES types
             only): re-optimize against recent data via the EXISTING RBI
             pipeline/gate (lookback_days=90, n_trials=100) -> re-enable ONLY
             if it clears the SAME walk-forward promotion gate live promotions
             use (0.14 commission, held-out OOS split).
        Either path re-enters at OBSERVATION tier ($REVIVE_BASE_USD, 1x
        leverage) with a FRESH live-evidence window (see _next_revive_name) —
        never sized off the reviving backtest (ADR-0001). Anti-flap cooldown
        (DB-persisted) blocks repeat attempts on a strategy that just
        culled/revived/failed-retune within REVIVE_COOLDOWN_HOURS.
        """
        try:
            if not (paper_mode and executor and orchestrator):
                return
            from .database import SessionLocal as _RSL
            from src.api.routes.rbi_optimize import _get_or_create_pipeline
            from src.strategies.base_strategy import StrategyConfig, StrategyTier
            from src.strategies.registry import list_strategies

            async def _do_revive(db, inst, *, new_params: dict, reason: str, metrics: dict) -> bool:
                old_name = inst.name
                if hasattr(executor, "open_position_count") and executor.open_position_count(old_name) > 0:
                    logger.warning("Revive: skipping %s — still shows an open position (should be 0 when stopped)", old_name)
                    return False
                before_params = dict(inst.params or {})
                new_name = _next_revive_name(old_name)
                inst.name = new_name
                inst.params = new_params
                inst.size_usd = REVIVE_BASE_USD
                inst.leverage = 1  # OBSERVATION tier (confidence_ladder.py: <10 live trades -> 1x)
                inst.total_trades = 0
                inst.winning_trades = 0
                inst.losing_trades = 0
                inst.total_pnl = 0.0
                inst.max_drawdown = 0.0
                inst.edge_confidence_score = 0.0
                inst.error_message = None
                inst.status = "running"
                inst.enabled = True
                inst.started_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    config = StrategyConfig(
                        name=new_name, symbol=inst.symbol, tier=StrategyTier.A,
                        timeframe=inst.timeframe, leverage=1, size_usd=REVIVE_BASE_USD,
                        target_pct=inst.target_pct, max_loss_pct=inst.max_loss_pct,
                        lookback_days=inst.lookback_days, interval_seconds=inst.interval_seconds,
                        enabled=True, params=new_params,
                    )
                    orchestrator.add_strategy(new_name, inst.strategy_type, config)
                    await orchestrator.start_strategy(new_name)
                except Exception as _se:
                    logger.error("Revive: orchestrator start failed for %s (DB still updated): %s", new_name, _se)
                _record_lifecycle_event(
                    db, strategy_type=inst.strategy_type, strategy_id=inst.id,
                    reason=reason, promoted=True,
                    before_params=before_params, after_params=new_params,
                    before_metrics={"old_name": old_name}, after_metrics=metrics,
                )
                return True

            db = _RSL()
            try:
                stopped = db.query(models.StrategyInstance).filter(
                    models.StrategyInstance.status == "stopped"
                ).all()
                if not stopped:
                    return

                available_types = {s["strategy_type"] for s in list_strategies()}
                processed = 0
                revived = 0
                for inst in stopped:
                    if processed >= REVIVE_MAX_PER_RUN:
                        break
                    if inst.strategy_type not in available_types:
                        continue
                    if _lifecycle_in_cooldown(db, inst.id, REVIVE_COOLDOWN_HOURS):
                        continue  # anti-flap: recently culled/revived/rejected — wait out the window
                    processed += 1

                    # ---- REVIVE-BY-REGIME first (no optimizer cost) ----
                    # should_trade() degrades to True when there's NO regime data yet
                    # (safe default for live ENTRY gating — don't block trading on a
                    # cold detector). That default is too permissive for a one-way
                    # revive decision, so require an ACTUAL current regime reading
                    # before trusting "fits the regime" (mirrors the T012 cold-regime
                    # guard already used elsewhere in this file).
                    regime_fit = False
                    _regime_info = {}
                    if regime_detector is not None:
                        try:
                            _regime_info = regime_detector.get_current_regime(inst.symbol) or {}
                            regime_fit = bool(_regime_info) and regime_detector.should_trade(inst.symbol, inst.strategy_type)
                        except Exception:
                            regime_fit = False
                    if regime_fit:
                        if await _do_revive(
                            db, inst, new_params=dict(inst.params or {}),
                            reason="lifecycle_revived_regime",
                            metrics={"regime": _regime_info.get("regime", "unknown")},
                        ):
                            revived += 1
                            logger.warning(
                                "Revive-by-regime: re-enabled %s -> %s (%s/%s) — existing params fit current regime %s",
                                inst.id, inst.name, inst.strategy_type, inst.symbol, _regime_info.get("regime", "unknown"),
                            )
                        continue

                    # ---- REVIVE-BY-RETUNE (deterministic Optuna; PARAM_SPACES types only) ----
                    if inst.strategy_type not in PARAM_SPACES:
                        logger.info(
                            "Revive: skipping retune for %s — %s has no PARAM_SPACES entry (non-optimizable type)",
                            inst.name, inst.strategy_type,
                        )
                        continue

                    pipeline = _get_or_create_pipeline(inst.strategy_type)
                    try:
                        result = await pipeline.run_revive_eligibility(
                            strategy_type=inst.strategy_type, symbol=inst.symbol,
                            timeframe=inst.timeframe, lookback_days=REVIVE_LOOKBACK_DAYS,
                            n_trials=REVIVE_N_TRIALS,
                        )
                    except Exception as _re_exc:
                        logger.warning("Revive-by-retune: optimizer error for %s: %s", inst.name, _re_exc)
                        continue

                    if result["eligible"]:
                        if await _do_revive(
                            db, inst, new_params=result["params"],
                            reason="lifecycle_revived_retune", metrics=result["metrics"],
                        ):
                            revived += 1
                            logger.warning(
                                "Revive-by-retune: re-enabled %s -> %s (%s/%s) — re-optimized params cleared the walk-forward gate",
                                inst.id, inst.name, inst.strategy_type, inst.symbol,
                            )
                    else:
                        _record_lifecycle_event(
                            db, strategy_type=inst.strategy_type, strategy_id=inst.id,
                            reason="lifecycle_revive_retune_rejected", promoted=False,
                            before_params=dict(inst.params or {}), after_params={},
                            before_metrics={}, after_metrics=result["metrics"],
                        )
                        logger.info(
                            "Revive-by-retune: %s stays stopped — retune failed gate (%s)",
                            inst.name, result["metrics"].get("failing_criterion", "unknown"),
                        )

                if processed:
                    logger.info(
                        "Revive job: examined %d/%d stopped strategies, revived %d",
                        processed, len(stopped), revived,
                    )
            finally:
                db.close()
        except Exception as _rve:
            logger.error("Revive job error: %s", _rve)

    from datetime import datetime as _dt
    _scheduler.add_job(
        _compound_job, "interval", minutes=15,
        id="compounder",
        replace_existing=True,
        next_run_time=_dt.now(),
    )
    # Stagger the cull a few minutes after the compounder so the two jobs don't
    # contend on the same DB session / executor state on the first tick.
    _scheduler.add_job(
        _cull_job, "interval", minutes=CULL_INTERVAL_MIN,
        id="bleeder_cull",
        replace_existing=True,
        next_run_time=_dt.now() + timedelta(minutes=3),
    )
    # T028 revive: staggered after compounder + cull (slower cadence — retune
    # is expensive) so the first tick doesn't contend with both on cold boot.
    _scheduler.add_job(
        _revive_job, "interval", minutes=REVIVE_INTERVAL_MIN,
        id="strategy_revive",
        replace_existing=True,
        next_run_time=_dt.now() + timedelta(minutes=6),
    )

    _scheduler.start()
    app.state.rbi_scheduler = _scheduler
    app.state.compound_initial_balance = _INITIAL_BALANCE
    app.state.compound_reserve_pct = _COMPOUND_RESERVE_PCT
    logger.info("RBI scheduler started with %d jobs + compounder + bleeder-cull + revive", len(_scheduler.get_jobs()) - 3)

    # ── XsecCarryEngine (cross-sectional funding carry — standalone, regime-agnostic) ──
    xsec_engine = None
    xsec_task = None
    try:
        if executor is not None and client is not None:
            from src.engine.xsec_engine import XsecCarryEngine
            xsec_engine = XsecCarryEngine(executor=executor, client=client)
            xsec_task = asyncio.create_task(xsec_engine.run())
            app.state.xsec_engine = xsec_engine
            logger.info("XsecCarryEngine started")
        else:
            logger.warning("XsecCarryEngine skipped — no executor/client")
    except Exception as _xe:
        logger.warning("XsecCarryEngine failed to start (non-fatal): %s", _xe)

    # ── XsecDriverEngine: boot-restart persisted enabled instances ──────────
    app.state.xsec_driver_engines = {}  # name -> (engine, task)
    # Capture the main event loop so the SYNC POST /xsec/instances endpoint (which
    # FastAPI runs in a threadpool thread with no running loop) can schedule engine
    # tasks back onto it via run_coroutine_threadsafe.
    app.state.loop = asyncio.get_running_loop()
    if executor is not None and client is not None:
        try:
            from .database import SessionLocal as _XDSL
            from src.engine.xsec_driver_engine import XsecDriverEngine as _XsecDE
            _xd_db = _XDSL()
            try:
                _xd_rows = _xd_db.query(models.StrategyInstance).filter(
                    models.StrategyInstance.strategy_type == "xsec_driver",
                    models.StrategyInstance.enabled == True,
                    models.StrategyInstance.status == "running",
                ).all()
            finally:
                _xd_db.close()
            for _xdr in _xd_rows:
                try:
                    _p = _xdr.params or {}
                    _eng = _XsecDE(
                        executor=executor, client=client,
                        name=_xdr.name,
                        driver=_p.get("driver", "realized_vol_carry"),
                        lookback=int(_p.get("lookback", 24)),
                        q=float(_p.get("q", 0.30)),
                        sign=int(_p.get("sign", -1)),
                        coins=_p.get("coins") or None,
                        per_leg_usd=float(_p.get("per_leg_usd", 50.0)),
                        rebalance_secs=int(_p.get("rebalance_secs", 3600)),
                        timeframe=_xdr.timeframe or "1h",
                        initial_legs=_xsec_open_legs(executor, _xdr.name),
                        members=_p.get("members") or None,
                        trail_days=int(_p.get("trail_days", 14)),
                    )
                    _task = asyncio.create_task(_eng.run())
                    app.state.xsec_driver_engines[_xdr.name] = (_eng, _task)
                    logger.info("xsec_driver: resumed %s (%s)", _xdr.name, _p.get("driver"))
                except Exception as _xe2:
                    logger.warning("xsec_driver: failed to resume %s — %s", _xdr.name, _xe2)
        except Exception as _xde:
            logger.warning("xsec_driver: boot-restart failed (non-fatal) — %s", _xde)

    yield

    _scheduler.shutdown(wait=False)

    # Shutdown: save paper state before stopping
    if paper_mode and executor is not None:
        try:
            executor.save_state()
            logger.info("Paper state saved on shutdown")
        except Exception as e:
            logger.warning("Paper state save on shutdown failed: %s", e)

    # Shutdown: stop all services
    if orchestrator is not None:
        try:
            await orchestrator.stop_all()
            logger.info("StrategyOrchestrator shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down orchestrator: %s", e)

    if risk_controller is not None:
        try:
            await risk_controller.stop()
            logger.info("RiskController shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down risk controller: %s", e)

    if liquidation_tracker is not None:
        try:
            await liquidation_tracker.stop()
            logger.info("LiquidationTracker shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down liquidation tracker: %s", e)

    if whale_tracker is not None:
        try:
            await whale_tracker.stop()
            logger.info("WhaleTracker shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down whale tracker: %s", e)

    if solana_scanner is not None:
        try:
            await solana_scanner.close()
            logger.info("SolanaScanner shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down Solana scanner: %s", e)

    if funding_monitor is not None:
        try:
            funding_monitor.stop()
            logger.info("FundingMonitor shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down funding monitor: %s", e)

    if xsec_task is not None and not xsec_task.done():
        try:
            if xsec_engine is not None:
                xsec_engine.stop()
            xsec_task.cancel()
            await asyncio.gather(xsec_task, return_exceptions=True)
            logger.info("XsecCarryEngine shut down cleanly")
        except Exception as e:
            logger.error("Error shutting down XsecCarryEngine: %s", e)

    # Shutdown all xsec_driver engine tasks
    _xd_running = getattr(app.state, "xsec_driver_engines", {})
    for _xd_name, (_xd_eng, _xd_task) in list(_xd_running.items()):
        if not _xd_task.done():
            try:
                _xd_eng.stop()
                _xd_task.cancel()
                await asyncio.gather(_await_xsec_task(_xd_task), return_exceptions=True)
                logger.info("xsec_driver(%s) shut down cleanly", _xd_name)
            except Exception as _xde2:
                logger.error("xsec_driver(%s) shutdown error — %s", _xd_name, _xde2)


app = FastAPI(title="Open Algotrade API", version="2.0.0", lifespan=lifespan)

# CORS: use CORS_ORIGINS env var in production, fallback to permissive for dev
_cors_env = os.getenv("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [
        "https://open-algotrade-v3.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount all sub-routers ──────────────────────────
from .routes import (
    risk_router,
    liquidations_router,
    whales_router,
    rbi_router,
    backtest_router,
    regime_router,
    solana_router,
    funding_router,
)
from .billing import router as billing_router
from src.api.routes.rbi_optimize import router as rbi_optimize_router, llm_router as llm_gate_router

app.include_router(risk_router, tags=["risk"])
app.include_router(liquidations_router)
app.include_router(whales_router)
app.include_router(rbi_router)
app.include_router(backtest_router)
app.include_router(regime_router)
app.include_router(billing_router)
app.include_router(solana_router)
app.include_router(funding_router)
app.include_router(rbi_optimize_router)
app.include_router(llm_gate_router)


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


def _paper_positions_out(executor) -> List[schemas.Position]:
    """Map the paper executor's live PaperPositions onto schemas.Position.
    Kept sync (no per-position mid fetch) so /positions stays cheap."""
    out = []
    for pos in executor._positions.values():
        abs_size = abs(pos.size)
        # PaperPosition has no live mark field; derive it from the executor's last
        # unrealized_pnl. ponytail: falls back to entry_price when pnl is 0/stale.
        if abs_size > 0 and pos.unrealized_pnl:
            delta = pos.unrealized_pnl / abs_size
            mark = pos.entry_price + delta if pos.side == "long" else pos.entry_price - delta
        else:
            mark = pos.entry_price
        out.append(schemas.Position(
            symbol=pos.symbol,
            size=abs_size,
            side=pos.side,
            entry_price=pos.entry_price,
            mark_price=mark,
            unrealized_pnl=pos.unrealized_pnl,
            leverage=pos.leverage,
        ))
    return out


def _xsec_open_legs(executor, name: str) -> dict:
    """{symbol: side} of the executor's live open positions for one xsec instance.
    Seeds XsecDriverEngine._open_legs across restarts so rotation can close
    pre-restart legs instead of orphaning them."""
    try:
        return {p.symbol: p.side for p in getattr(executor, "_positions", {}).values()
                if getattr(p, "strategy_name", None) == name}
    except Exception:
        return {}


def _paper_trades_out(executor, limit: int, open_only: bool) -> List[schemas.TradeOut]:
    """Reconstruct entry/exit-paired TradeOut rows from the executor's per-action
    trade log (get_trade_history returns one row per entry AND per exit). Entries
    are FIFO-matched to exits by (symbol, strategy, side)."""
    from collections import defaultdict, deque

    open_lots = defaultdict(deque)  # (symbol, strategy, side) -> unmatched entry rows
    closed = []
    for r in executor.get_trade_history():  # chronological
        key = (r["symbol"], r["strategy"], r["side"])
        if r["action"] == "entry":
            open_lots[key].append(r)
            continue
        # exit -> pair with oldest open entry; ponytail: if none (e.g. partial
        # ledger rehydrate) fall back to the exit row for entry fields.
        entry = open_lots[key].popleft() if open_lots[key] else r
        closed.append(schemas.TradeOut(
            id=r["id"], symbol=r["symbol"], side=r["side"], size=r["size"],
            entry_price=entry["price"], exit_price=r["price"],
            pnl=r.get("pnl"), exit_reason=r.get("reason"), strategy=r["strategy"],
            opened_at=entry["timestamp"], closed_at=r["timestamp"], is_open=False,
        ))

    open_trades = [
        schemas.TradeOut(
            id=e["id"], symbol=e["symbol"], side=e["side"], size=e["size"],
            entry_price=e["price"], exit_price=None, pnl=None,
            exit_reason=None, strategy=e["strategy"],
            opened_at=e["timestamp"], closed_at=None, is_open=True,
        )
        for q in open_lots.values() for e in q
    ]

    trades = open_trades if open_only else open_trades + closed
    trades.sort(key=lambda t: t.opened_at, reverse=True)
    return trades[:limit]


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

@app.get("/health", response_model=schemas.HealthResponse)
def health(request: Request):
    trading_mode = getattr(request.app.state, "trading_mode", "unknown")
    executor = getattr(request.app.state, "executor", None)
    orchestrator = getattr(request.app.state, "orchestrator", None)

    # Verify strategy registry
    registry_ok = False
    registry_count = 0
    try:
        from src.strategies.registry import list_strategies
        strategies = list_strategies()
        registry_count = len(strategies)
        registry_ok = registry_count > 0
    except Exception:
        pass

    result = schemas.HealthResponse(
        status="ok" if registry_ok else "degraded",
        service="open-algotrade-api",
        version="2.0.0",
        trading_mode=trading_mode,
        registry_strategies=registry_count,
        registry_ok=registry_ok,
        orchestrator_ok=orchestrator is not None,
        execution_stats=(
            executor.get_execution_stats()
            if executor and hasattr(executor, "get_execution_stats")
            else None
        ),
    )
    return result


# ──────────────────────────────────────────────
# Trading Mode
# ──────────────────────────────────────────────

@app.get("/trading-mode")
def get_trading_mode(request: Request):
    """Get current trading mode and execution stats."""
    trading_mode = getattr(request.app.state, "trading_mode", "unknown")
    executor = getattr(request.app.state, "executor", None)
    result = {
        "mode": trading_mode,
        "available_modes": ["paper", "testnet", "mainnet"],
    }
    if executor and hasattr(executor, "get_execution_stats"):
        result["stats"] = executor.get_execution_stats()
    return result


@app.post("/trading-mode")
async def set_trading_mode(request: Request):
    """
    Switch trading mode at runtime.

    Stops all running strategies, swaps the executor and data client,
    then lets you restart strategies in the new mode.
    """
    body = await request.json()
    new_mode = body.get("mode", "").lower()
    if new_mode not in ("paper", "testnet", "mainnet"):
        raise HTTPException(status_code=400, detail=f"Invalid mode: {new_mode}. Use paper, testnet, or mainnet.")

    old_mode = getattr(request.app.state, "trading_mode", "unknown")
    if new_mode == old_mode:
        return {"status": "unchanged", "mode": old_mode}

    # Persist paper executor state before teardown so open positions / balance survive the switch
    old_executor = getattr(request.app.state, "executor", None)
    if hasattr(old_executor, 'save_state'):
        try:
            old_executor.save_state()
        except Exception as e:
            logger.warning("Failed to save executor state before mode switch: %s", e)

    # Stop all running strategies first
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator:
        try:
            await orchestrator.stop_all()
        except Exception as e:
            logger.warning("Error stopping strategies during mode switch: %s", e)

    client = None
    executor = None

    if new_mode == "paper":
        from src.execution.paper_executor import PaperTradingExecutor
        from src.lib.nice_funcs import HyperliquidDataClient

        initial_balance = float(os.getenv("PAPER_BALANCE", "10000"))
        price_source = body.get("price_source", os.getenv("PAPER_PRICE_SOURCE", "mainnet"))
        client = HyperliquidDataClient(network=price_source)
        executor = PaperTradingExecutor(
            base_url=client.base_url,
            initial_balance=initial_balance,
        )
        logger.info("Switched to PAPER mode | balance=$%.0f | prices=%s", initial_balance, price_source)
    else:
        from src.lib.nice_funcs import HyperliquidClient
        from src.execution.hl_executor import HyperliquidVaultExecutor

        network = "mainnet" if new_mode == "mainnet" else "testnet"
        try:
            client = HyperliquidClient(network=network)
            executor = HyperliquidVaultExecutor(client=client)
            logger.info("Switched to %s mode | account=%s", network.upper(), client.account.address)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize {network} mode: {str(e)}. Check HYPERLIQUID_PRIVATE_KEY.",
            )

    # Rebuild orchestrator with new client/executor
    if client and executor:
        from src.engine.orchestrator import StrategyOrchestrator
        from src.services.liquidation_guard import LiquidationGuard
        regime_detector = getattr(request.app.state, "regime_detector", None)
        existing_risk_controller = getattr(request.app.state, "risk_controller", None)
        request.app.state.orchestrator = StrategyOrchestrator(
            client=client,
            executor=executor,
            regime_detector=regime_detector,
            liquidation_guard=LiquidationGuard(),
            risk_controller=existing_risk_controller,
            max_global_trades_per_hour=20,
            daily_loss_limit_pct=2.0,
            max_portfolio_exposure_pct=80.0,
        )

    request.app.state.executor = executor
    request.app.state.trading_mode = new_mode
    request.app.state.paper_mode = new_mode == "paper"

    return {
        "status": "switched",
        "mode": new_mode,
        "previous_mode": old_mode,
        "message": f"Trading mode switched to {new_mode}. All strategies stopped — restart them to trade in {new_mode} mode.",
    }


# ──────────────────────────────────────────────
# Compounding Controller Status
# ──────────────────────────────────────────────

@app.get("/compound/status")
def get_compound_status(request: Request):
    """Current compounding state: equity, profits, reinvestment rate, next rebalance size."""
    _exec = getattr(request.app.state, "executor", None)
    _paper = getattr(request.app.state, "paper_mode", False)
    initial = getattr(request.app.state, "compound_initial_balance", float(os.getenv("PAPER_BALANCE", "10000")))
    reserve_pct = getattr(request.app.state, "compound_reserve_pct", 0.10)
    compound_base = 100.0

    balance = _exec.balance if (_paper and _exec is not None) else initial
    investable = max(compound_base, balance * (1.0 - reserve_pct))

    return {
        "balance": round(balance, 2),
        "initial_balance": initial,
        "compound_base": compound_base,
        "reserve_pct": round(reserve_pct * 100, 1),
        "investable": round(investable, 2),
        "compound_active": True,
        "rebalance_interval_minutes": 15,
    }


@app.get("/ws/status")
def get_ws_status(request: Request):
    """Hyperliquid native-WS candle feed status (T020/T023).

    Lets ws activation be VERIFIED in prod: the FastAPI app's logger.info does not reach
    Railway stdout, so this endpoint is the way to confirm the ws feed is genuinely serving
    (ws_ready + buffers filling + low fallback_count) rather than silently falling back to
    REST. Returns {enabled:false} when HL_WS_CANDLES is off (orchestrator client unwrapped).
    """
    orch = getattr(request.app.state, "orchestrator", None)
    client = getattr(orch, "client", None) if orch is not None else None
    status_fn = getattr(client, "status", None)
    if callable(status_fn):
        try:
            return {"enabled": True, **status_fn()}
        except Exception as e:  # noqa: BLE001
            return {"enabled": True, "error": str(e)}
    return {
        "enabled": False,
        "reason": "HL_WS_CANDLES off or orchestrator client not ws-wrapped",
        "client_type": type(client).__name__ if client is not None else None,
    }


@app.get("/orchestrator/debug/{name}")
async def get_orchestrator_debug(name: str, request: Request):
    """Ground-truth view of one strategy instance inside the LIVE orchestrator.

    Railway stdout is 100% drowned by backtest CB spam, so logs can't answer
    "why is this instance flat?". This read-only surface exposes the
    orchestrator's actual in-memory view: is the loop registered/running, what
    strategy_type is it mapped to (the regime gate + diagnostic logs key off
    this), its StrategyState, and the global halt/block counters. Defensive by
    construction — returns whatever is available, never raises.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        return {"error": "no live orchestrator on app.state"}

    out = {"name": name}

    # Registration / loop liveness
    try:
        task = getattr(orch, "_tasks", {}).get(name)
        out["in_tasks"] = name in getattr(orch, "_tasks", {})
        out["task_done"] = task.done() if task is not None else None
        out["task_cancelled"] = task.cancelled() if task is not None else None
        out["strategy_type_mapped"] = getattr(orch, "_strategy_types", {}).get(name)
        out["has_strategy_obj"] = name in getattr(orch, "_strategies", {})
        out["registered_names_sample"] = list(getattr(orch, "_strategies", {}).keys())[:30]
    except Exception as e:  # noqa: BLE001
        out["registration_error"] = str(e)

    # In-memory StrategyState
    try:
        strat = getattr(orch, "_strategies", {}).get(name)
        if strat is not None:
            st = strat.state
            sig = getattr(st, "last_signal", None)
            out["state"] = {
                "iterations": getattr(st, "iterations", None),
                "last_iteration": str(getattr(st, "last_iteration", None)),
                "last_signal": (sig.signal_type.value if sig is not None else None),
                "last_signal_reason": (getattr(sig, "reason", None) if sig is not None else None),
                "last_signal_time": (str(getattr(sig, "timestamp", None)) if sig is not None else None),
                "circuit_breaker_triggered": getattr(st, "circuit_breaker_triggered", None),
                "circuit_breaker_reason": getattr(st, "circuit_breaker_reason", None),
                "consecutive_losses": getattr(st, "consecutive_losses", None),
                "consecutive_errors": getattr(st, "consecutive_errors", None),
                "entry_bar_count": getattr(st, "entry_bar_count", None),
                "trades_this_hour": getattr(st, "trades_this_hour", None),
                "last_trade_close_time": str(getattr(st, "last_trade_close_time", None)),
            }
            cfg = strat.config
            out["config"] = {
                "symbol": getattr(cfg, "symbol", None),
                "params": getattr(cfg, "params", None),
            }
    except Exception as e:  # noqa: BLE001
        out["state_error"] = str(e)

    # Global halt / block counters + balances
    try:
        out["global"] = {
            "daily_loss_triggered": getattr(orch, "_daily_loss_triggered", None),
            "daily_loss_flattened": getattr(orch, "_daily_loss_flattened", None),
            "monthly_halt_triggered": getattr(orch, "_monthly_halt_triggered", None),
            "weekly_halved": getattr(orch, "_weekly_halved", None),
            "daily_loss_blocks": getattr(orch, "_daily_loss_blocks", None),
            "regime_blocks": getattr(orch, "_regime_blocks", None),
            "rate_limit_blocks": getattr(orch, "_rate_limit_blocks", None),
            "global_trades_this_hour": getattr(orch, "_global_trades_this_hour", None),
            "daily_starting_balance": getattr(orch, "_daily_starting_balance", None),
            "monthly_starting_balance": getattr(orch, "_monthly_starting_balance", None),
        }
        try:
            out["global"]["current_balance"] = orch._get_current_balance()
        except Exception as be:  # noqa: BLE001
            out["global"]["current_balance_error"] = str(be)
    except Exception as e:  # noqa: BLE001
        out["global_error"] = str(e)

    # Executor position truth — the orchestrator's run_iteration takes the
    # should_exit branch whenever get_position() is non-empty, so a position the
    # executor reports but the active_positions count doesn't (keying desync)
    # would freeze an always-in strategy in should_exit forever (never enters).
    try:
        ex = getattr(orch, "executor", None)
        strat = getattr(orch, "_strategies", {}).get(name)
        sym = getattr(getattr(strat, "config", None), "symbol", None) if strat else None
        ex_out = {"symbol_used": sym}
        if ex is not None and sym is not None and hasattr(ex, "get_position"):
            pos = await ex.get_position(sym, strategy_name=name)
            ex_out["get_position"] = pos
            ex_out["has_position_truth"] = bool(pos and pos.get("size", 0) != 0)
        if ex is not None and hasattr(ex, "open_position_count"):
            ex_out["open_position_count"] = ex.open_position_count(name)
        # Raw _positions table: key -> stored .strategy_name (exposes a key/attr
        # mismatch where get_position finds by key but the count filters by attr).
        raw = getattr(ex, "_positions", None)
        if isinstance(raw, dict):
            ex_out["positions_keys"] = {
                k: {"strategy_name": getattr(v, "strategy_name", None),
                    "size": getattr(v, "size", None),
                    "symbol": getattr(v, "symbol", None)}
                for k, v in raw.items()
                if name in str(k) or getattr(v, "strategy_name", None) == name
                or (sym and getattr(v, "symbol", None) == sym)
            }
            ex_out["positions_total"] = len(raw)
        out["executor"] = ex_out
    except Exception as e:  # noqa: BLE001
        out["executor_error"] = str(e)

    return out


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
    token = create_access_token(data={"sub": str(user.id)})
    return schemas.AuthResponse(
        id=str(user.id),
        email=user.email,
        name=user.username,
        avatar=None,
        status="ONLINE",
        access_token=token,
        token_type="bearer",
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
    token = create_access_token(data={"sub": str(db_user.id)})
    return schemas.AuthResponse(
        id=str(db_user.id),
        email=db_user.email,
        name=db_user.username,
        avatar=None,
        status="ONLINE",
        access_token=token,
        token_type="bearer",
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


# ──────────────────────────────────────────────
# Portfolio Summary & Allocation
# (must be registered BEFORE /portfolio/{user_id} to avoid path collision)
# ──────────────────────────────────────────────

@app.get("/portfolio/summary", response_model=schemas.PortfolioSummary)
def portfolio_summary(request: Request, db: Session = Depends(get_db)):
    """Portfolio-level stats: total PnL, exposure, mode, strategy counts."""
    trading_mode = getattr(request.app.state, "trading_mode", "unknown")
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    instances = db.query(models.StrategyInstance).all()
    running = sum(1 for i in instances if i.status == "running")
    total_trades = sum(i.total_trades for i in instances)
    winning = sum(i.winning_trades for i in instances)
    agg_pnl = sum(i.total_pnl for i in instances)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0

    total_equity = 0.0
    initial_equity = 0.0
    max_dd_pct = 0.0
    active_positions = 0

    if paper_mode and executor and hasattr(executor, "get_execution_stats"):
        stats = executor.get_execution_stats()
        total_equity = stats.get("balance", 0)
        initial_equity = stats.get("initial_balance", 0)
        max_dd_pct = stats.get("max_drawdown_pct", 0)
        active_positions = stats.get("active_positions", 0)
        # Use balance-delta as canonical PnL (survives redeploy; avoids reset-prone _execution_history)
        agg_pnl = round(stats.get("balance", 0) - stats.get("initial_balance", 0), 2)
    else:
        vault = _get_or_create_vault_state(db)
        total_equity = vault.total_equity
        initial_equity = vault.total_equity
        positions = _get_live_positions()
        active_positions = len(positions)

    total_return_pct = (
        ((total_equity - initial_equity) / initial_equity * 100)
        if initial_equity > 0
        else 0.0
    )

    total_exposure = sum(i.size_usd for i in instances if i.status == "running")

    return schemas.PortfolioSummary(
        trading_mode=trading_mode,
        total_equity=round(total_equity, 2),
        initial_equity=round(initial_equity, 2),
        total_pnl=round(agg_pnl, 2),
        total_return_pct=round(total_return_pct, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        total_strategies=len(instances),
        running_strategies=running,
        total_trades=total_trades,
        win_rate=round(win_rate, 1),
        active_positions=active_positions,
        total_exposure_usd=round(total_exposure, 2),
    )


@app.get("/portfolio/allocation", response_model=schemas.PortfolioAllocationResponse)
def portfolio_allocation(request: Request, db: Session = Depends(get_db)):
    """Current strategy allocation showing size and percentage of total."""
    instances = db.query(models.StrategyInstance).all()
    orchestrator = getattr(request.app.state, "orchestrator", None)

    total_allocated = sum(i.size_usd for i in instances)

    items = []
    for inst in instances:
        current_pnl = inst.total_pnl
        if orchestrator and inst.status == "running":
            strategy = orchestrator.get_strategy(inst.name)
            if strategy:
                current_pnl = strategy.state.total_pnl

        alloc_pct = (inst.size_usd / total_allocated * 100) if total_allocated > 0 else 0.0
        items.append(schemas.AllocationItem(
            strategy_name=inst.name,
            strategy_type=inst.strategy_type,
            symbol=inst.symbol,
            status=inst.status,
            size_usd=inst.size_usd,
            allocation_pct=round(alloc_pct, 1),
            current_pnl=round(current_pnl, 2),
        ))

    return schemas.PortfolioAllocationResponse(
        total_allocated_usd=round(total_allocated, 2),
        strategies=items,
    )


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
def get_positions(request: Request):
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)
    if paper_mode and executor is not None and hasattr(executor, "_positions"):
        return _paper_positions_out(executor)
    return _get_live_positions()


@app.get("/trades", response_model=List[schemas.TradeOut])
def get_trades(request: Request, limit: int = 50, open_only: bool = False, db: Session = Depends(get_db)):
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)
    if paper_mode and executor is not None and hasattr(executor, "get_trade_history"):
        return _paper_trades_out(executor, limit, open_only)
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
    # xsec_driver is a standalone engine (not a BaseStrategy), so append it explicitly
    strategies.append({
        "strategy_type": "xsec_driver",
        "tier": "A",
        "description": (
            "Generic runtime-instantiable cross-sectional driver "
            "(realized_vol/dollar_volume) — dollar-neutral basket"
        ),
        "default_symbol": "BTC",
        "default_timeframe": "1h",
        "default_params": {
            "driver": "realized_vol_carry",
            "lookback": 24,
            "q": 0.30,
            "sign": -1,
            "per_leg_usd": 50.0,
            "rebalance_secs": 3600,
        },
        "category": "cross_sectional",
        "risk_level": "medium",
    })
    return schemas.StrategyRegistryOut(
        available_strategies=[
            schemas.StrategyTypeInfo(**s) for s in strategies
        ],
        total=len(strategies),
    )


@app.get("/strategies", response_model=List[schemas.StrategyInstanceOut])
def list_strategy_instances(request: Request, db: Session = Depends(get_db)):
    """List all configured strategy instances, merged with live paper executor data."""
    instances = db.query(models.StrategyInstance).all()

    # Merge paper executor stats into DB records
    executor = getattr(request.app.state, "executor", None)
    if executor and hasattr(executor, "get_execution_stats"):
        try:
            trades = executor.get_trade_history() if hasattr(executor, "get_trade_history") else []
            active_pos = executor.get_active_positions() if hasattr(executor, "get_active_positions") else {}

            # Build per-strategy breakdown from trade history (exits only — entries have pnl=0)
            strat_trades: dict = {}
            for t in trades:
                if t.get("action") != "exit":
                    continue
                sname = t.get("strategy", "unknown")
                if sname not in strat_trades:
                    strat_trades[sname] = {"total": 0, "wins": 0, "pnl": 0.0}
                strat_trades[sname]["total"] += 1
                pnl = t.get("pnl", 0.0)
                strat_trades[sname]["pnl"] += pnl
                if pnl > 0:
                    strat_trades[sname]["wins"] += 1

            # Match active positions to strategies
            strat_positions: dict = {}
            for key, pos_info in active_pos.items():
                sname = pos_info.get("strategy", "")
                if sname:
                    strat_positions[sname] = pos_info

            for inst in instances:
                if inst.name in strat_trades:
                    sd = strat_trades[inst.name]
                    inst.total_trades = sd["total"]
                    inst.winning_trades = sd["wins"]
                    inst.losing_trades = sd["total"] - sd["wins"]
                    inst.total_pnl = round(sd["pnl"], 4)
                if inst.name in strat_positions:
                    inst.last_signal = strat_positions[inst.name].get("side", inst.last_signal)
                # Live open-position count + recent realized PnL from the executor's
                # in-memory state — the SAME source the single GET /strategies/{name}
                # uses (main.py get_strategy). Without this the LIST served the stale
                # active_positions DB column (0) while instances were actually holding.
                # ponytail: in-memory dict/list scans only — no per-instance network fetch.
                if hasattr(executor, "open_position_count"):
                    inst.active_positions = executor.open_position_count(inst.name)
                if hasattr(executor, "recent_realized_pnl"):
                    inst.recent_pnl, inst.recent_trades = executor.recent_realized_pnl(inst.name)
        except Exception as e:
            logger.warning("Failed to merge paper stats: %s", e)

    # Also merge orchestrator iteration data
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator:
        try:
            for inst in instances:
                strat = orchestrator.get_strategy(inst.name)
                if strat and hasattr(strat, "state"):
                    inst.iterations = getattr(strat.state, "iterations", inst.iterations)
                    inst.errors = getattr(strat.state, "errors", inst.errors)
                    if hasattr(strat.state, "last_signal") and strat.state.last_signal:
                        inst.last_signal = str(strat.state.last_signal)
                    if hasattr(strat.state, "last_signal_time") and strat.state.last_signal_time:
                        inst.last_signal_time = strat.state.last_signal_time
        except Exception as e:
            logger.warning("Failed to merge orchestrator stats: %s", e)

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


# ──────────────────────────────────────────────
# Deploy Winners (hardcoded proven strategies)
# (must be registered BEFORE /strategies/{name} to avoid path collision)
# ──────────────────────────────────────────────

@app.post("/strategies/deploy-winners")
async def deploy_winner_strategies(request: Request):
    """
    Deploy the proven winning strategies across multiple assets with MoonDev-tuned parameters.
    This creates strategy instances in the DB and starts them.

    Winners based on paper trading data:
    1. market_maker -- range-based MM (100% WR)
    2. funding_arb -- momentum divergence arb (100% WR)
    3. nadaraya_watson -- kernel regression + StochRSI (60% WR)
    4. adx -- trend filter (36% WR but profitable R:R)
    5. bollinger -- mean reversion (40% WR but profitable)
    """
    orchestrator = request.app.state.orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    db = next(get_db())

    # Winner strategies with MoonDev-tuned params per asset
    deployments = [
        # Market Maker -- works on liquid assets with tight ranges
        {
            "name": "mm-eth", "strategy_type": "market_maker", "symbol": "ETH",
            "timeframe": "5m", "size_usd": 100, "leverage": 3,
            "params": {
                "num_bars": 180, "quartile": 0.33, "max_l2h": 0.05,
                "max_tr_pct": 0.02, "exit_pct": 0.004, "mm_stop_pct": 0.01,
                "time_limit_minutes": 120, "last_n_bars": 17,
                "cooldown_seconds": 300, "max_trades_per_hour": 3,
                "min_signal_strength": 0.5,
            },
        },
        {
            "name": "mm-sol", "strategy_type": "market_maker", "symbol": "SOL",
            "timeframe": "5m", "size_usd": 100, "leverage": 3,
            "params": {
                "num_bars": 180, "quartile": 0.33, "max_l2h": 0.06,
                "max_tr_pct": 0.025, "exit_pct": 0.005, "mm_stop_pct": 0.012,
                "time_limit_minutes": 90, "last_n_bars": 17,
                "cooldown_seconds": 300, "max_trades_per_hour": 3,
                "min_signal_strength": 0.5,
            },
        },
        # Funding Arb -- structural edge, run on BTC and ETH
        {
            "name": "arb-eth", "strategy_type": "funding_arb", "symbol": "ETH",
            "timeframe": "1h", "size_usd": 150, "leverage": 3,
            "params": {
                "momentum_threshold": 0.015, "combined_target_pct": 0.8,
                "arb_max_loss_pct": -1.5, "min_hold_bars": 3,
                "cooldown_seconds": 300, "max_trades_per_hour": 3,
                "min_signal_strength": 0.4,
            },
        },
        # Nadaraya-Watson -- kernel regression, tuned to MoonDev StochRSI 10/90
        {
            "name": "nw-eth", "strategy_type": "nadaraya_watson", "symbol": "ETH",
            "timeframe": "1h", "size_usd": 100, "leverage": 3,
            "params": {
                "kernel_bandwidth": 8.0, "kernel_lookback": 200,
                "overbought": 90, "oversold": 10,
                "stoch_exit_window": 14, "exit_confirmation_times": 2,
                "cooldown_seconds": 600, "max_trades_per_hour": 2,
                "min_signal_strength": 0.5,
            },
        },
        {
            "name": "nw-sol", "strategy_type": "nadaraya_watson", "symbol": "SOL",
            "timeframe": "1h", "size_usd": 100, "leverage": 3,
            "params": {
                "kernel_bandwidth": 8.0, "kernel_lookback": 200,
                "overbought": 90, "oversold": 10,
                "stoch_exit_window": 14, "exit_confirmation_times": 2,
                "cooldown_seconds": 600, "max_trades_per_hour": 2,
                "min_signal_strength": 0.5,
            },
        },
        # ADX -- trend filter, works on BTC and ETH
        {
            "name": "adx-eth", "strategy_type": "adx", "symbol": "ETH",
            "timeframe": "1h", "size_usd": 100, "leverage": 3,
            "params": {
                "adx_period": 14, "adx_threshold": 25,
                "cooldown_seconds": 600, "max_trades_per_hour": 2,
                "min_signal_strength": 0.5,
            },
        },
    ]

    created = []
    skipped = []

    try:
        for dep in deployments:
            # Check if strategy already exists
            existing = db.query(models.StrategyInstance).filter(
                models.StrategyInstance.name == dep["name"]
            ).first()
            if existing:
                skipped.append(dep["name"])
                continue

            # Create new strategy instance
            instance = models.StrategyInstance(
                name=dep["name"],
                strategy_type=dep["strategy_type"],
                symbol=dep["symbol"],
                timeframe=dep["timeframe"],
                leverage=dep.get("leverage", 3),
                size_usd=dep.get("size_usd", 100),
                target_pct=9.0,       # MoonDev global target
                max_loss_pct=-8.0,    # MoonDev global stop
                interval_seconds=30,
                enabled=True,
                params=dep.get("params", {}),
                tier=dep.get("tier", "bonus_algos"),
                status="created",
            )
            db.add(instance)
            db.commit()
            db.refresh(instance)
            created.append(dep["name"])

        return {
            "created": created,
            "skipped": skipped,
            "message": f"Created {len(created)} strategies, skipped {len(skipped)} (already exist). Use /strategies/{{name}}/start to start them.",
            "next_steps": [
                "Start each strategy: POST /strategies/{name}/start",
                "Monitor: GET /strategies/leaderboard",
                "Circuit breaker will auto-disable losers after 5 consecutive losses or $25 drawdown",
            ],
        }
    finally:
        db.close()


# ──────────────────────────────────────────────
# Aggregate Strategy Performance
# (must be registered BEFORE /strategies/{name} to avoid path collision)
# ──────────────────────────────────────────────

@app.get("/strategies/performance", response_model=schemas.AggregatePerformanceResponse)
def get_strategies_performance(request: Request, db: Session = Depends(get_db)):
    """Aggregate performance across all strategy instances."""
    instances = db.query(models.StrategyInstance).all()
    orchestrator = getattr(request.app.state, "orchestrator", None)

    items = []
    for inst in instances:
        total_trades = inst.total_trades
        winning_trades = inst.winning_trades
        losing_trades = inst.losing_trades
        total_pnl = inst.total_pnl
        max_drawdown = inst.max_drawdown

        if orchestrator and inst.status == "running":
            strategy = orchestrator.get_strategy(inst.name)
            if strategy:
                stats = strategy.get_stats()
                total_trades = stats.get("total_trades", total_trades)
                winning_trades = stats.get("winning_trades", winning_trades)
                losing_trades = stats.get("losing_trades", losing_trades)
                total_pnl = stats.get("total_pnl", total_pnl)
                max_drawdown = stats.get("max_drawdown", max_drawdown)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0.0

        gross_profit = total_pnl if total_pnl > 0 else 0.0
        gross_loss = abs(total_pnl) if total_pnl < 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        items.append(schemas.StrategyPerformanceItem(
            name=inst.name,
            strategy_type=inst.strategy_type,
            symbol=inst.symbol,
            status=inst.status,
            total_pnl=round(total_pnl, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 1),
            max_drawdown=round(max_drawdown, 2),
            avg_trade_pnl=round(avg_trade, 2),
            profit_factor=round(profit_factor, 3),
        ))

    agg_trades = sum(i.total_trades for i in items)
    agg_winning = sum(i.winning_trades for i in items)
    agg_pnl = sum(i.total_pnl for i in items)
    overall_wr = (agg_winning / agg_trades * 100) if agg_trades > 0 else 0.0

    best = max(items, key=lambda x: x.total_pnl).name if items else None
    worst = min(items, key=lambda x: x.total_pnl).name if items else None

    return schemas.AggregatePerformanceResponse(
        total_strategies=len(instances),
        running_strategies=sum(1 for i in instances if i.status == "running"),
        total_pnl=round(agg_pnl, 2),
        total_trades=agg_trades,
        overall_win_rate=round(overall_wr, 1),
        best_strategy=best,
        worst_strategy=worst,
        strategies=items,
    )


def _posterior_scores(executor, detector, running, window: int = 30) -> dict:
    """Posterior P(edge) per running (strategy×symbol) — the T009 conditional-edge
    signal, reused by GET /strategies/leaderboard AND the T010 allocator (compounder).

    Returns name -> {current_regime, prob_edge, posterior_mean_edge, ci_low, ci_high,
    n_own, n_pool, pool_mean}. PRIOR = sibling pool (same strategy_type on symbols
    currently in the same regime); LIKELIHOOD = the instance's own recent trades.
    """
    from src.execution.paper_executor import posterior_edge

    by_strat: dict = {}
    if executor is not None and hasattr(executor, "get_trade_history"):
        for t in executor.get_trade_history():
            if t.get("action") == "exit":
                by_strat.setdefault(t.get("strategy", ""), []).append(t.get("pnl", 0.0))

    regime_cache: dict = {}
    def _regime(sym):
        if sym not in regime_cache:
            r = detector.get_current_regime(sym) if detector is not None else None
            regime_cache[sym] = (r or {}).get("regime", "unknown")
        return regime_cache[sym]

    own_map: dict = {}
    pool_map: dict = {}  # (strategy_type, regime) -> {name: recent_pnls}
    for s in running:
        recent = by_strat.get(s.name, [])[-window:]
        own_map[s.name] = recent
        pool_map.setdefault((s.strategy_type, _regime(s.symbol)), {})[s.name] = recent

    out: dict = {}
    for s in running:
        regime = _regime(s.symbol)
        own = own_map[s.name]
        siblings = pool_map.get((s.strategy_type, regime), {})
        pool = [p for nm, pnls in siblings.items() if nm != s.name for p in pnls]
        post = posterior_edge(own, pool)
        post["current_regime"] = regime
        post["pool_mean"] = (sum(pool) / len(pool)) if pool else 0.0
        post["raw_recent_pnl"] = sum(own)
        out[s.name] = post
    return out


@app.get("/strategies/leaderboard")
async def strategy_leaderboard(request: Request, window: int = 30):
    """Conditional-edge ranker (T009 slice 1, READ-ONLY — drives nothing yet).

    Scores each RUNNING (strategy×symbol) by a POSTERIOR P(edge) instead of raw or
    blindly-shrunk PnL, so a thin winner is RESOLVED, not discarded. Empirical-Bayes
    (Normal-Normal, paper_executor.posterior_edge):
      - PRIOR  = the sibling pool — recent per-trade PnLs of same-strategy-type
        instances on symbols CURRENTLY in the same regime (hierarchical pooling:
        if the whole family lifts in this regime it's regime edge; if one instance
        wins alone it's idiosyncratic/luck).
      - LIKELIHOOD = the instance's own recent trades.
    Output per row: current_regime, trades, raw_recent_pnl, prob_edge (P(per-trade
    edge>0)), posterior_mean_edge, ci_low/ci_high (90% credible), family_corroboration.
    Ranked by prob_edge × posterior_mean_edge (expected, confidence-weighted edge),
    tie-broken by posterior_mean_edge so losers sort below untested instances.

    Slice 2 (NOT here) is the intended consumer: size allocation BY prob_edge
    (Thompson-style — small-and-growing for high-uncertainty maybe-edges + an
    exploration floor), so wide-CI thin winners get probed, not dropped.
    """
    executor = getattr(request.app.state, "executor", None)
    detector = getattr(request.app.state, "regime_detector", None)

    db = next(get_db())
    try:
        running = db.query(models.StrategyInstance).filter(
            models.StrategyInstance.status == "running"
        ).all()

        scores = _posterior_scores(executor, detector, running, window=window)

        rows = []
        for s in running:
            post = scores.get(s.name, {})
            pool_mean = post.get("pool_mean", 0.0)
            rows.append({
                "name": s.name,
                "type": s.strategy_type,
                "symbol": s.symbol,
                "current_regime": post.get("current_regime", "unknown"),
                "trades": post.get("n_own", 0),
                "raw_recent_pnl": round(post.get("raw_recent_pnl", 0.0), 2),
                "prob_edge": round(post.get("prob_edge", 0.5), 3),
                "posterior_mean_edge": round(post.get("posterior_mean_edge", 0.0), 4),
                "ci_low": round(post.get("ci_low", 0.0), 4),
                "ci_high": round(post.get("ci_high", 0.0), 4),
                "family_corroboration": {
                    "n_pool": post.get("n_pool", 0),
                    "pool_mean_pnl": round(pool_mean, 4),
                    "positive": pool_mean > 0,
                },
                "rank_score": round(post.get("prob_edge", 0.5) * post.get("posterior_mean_edge", 0.0), 4),
            })

        # Primary: confidence-weighted expected edge. Tie-break: raw posterior mean,
        # so high-sample losers (prob~0 -> rank_score~0) still sort below untested 0s.
        rows.sort(key=lambda r: (r["rank_score"], r["posterior_mean_edge"]), reverse=True)
        for i, r in enumerate(rows):
            r["blended_rank"] = i + 1

        return {
            "window": window,
            "count": len(rows),
            "rows": rows,
            "_schema_gap": (
                "The regime-conditional PRIOR here pools SIBLINGS' RECENT trades on symbols "
                "currently in the same regime (a live proxy). The STRONGER source-1 signal — this "
                "instance's OWN realized PnL in PAST windows that matched the current regime — needs "
                "a per-trade `regime` tag PaperTrade does not record (regime history also resets per "
                "redeploy). Minimal add: stamp PaperTrade.regime from the live detector at exit; "
                "forward trades then accumulate true per-trade-regime history for a sharper prior."
            ),
        }
    finally:
        db.close()


@app.get("/strategies/{name}", response_model=schemas.StrategyInstanceOut)
def get_strategy_instance(name: str, request: Request, db: Session = Depends(get_db)):
    """Get a specific strategy instance by name, enriched with live orchestrator stats."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    # Merge live stats from orchestrator if strategy is running
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator and instance.status == "running":
        strategy = orchestrator.get_strategy(name)
        if strategy:
            stats = strategy.get_stats()
            instance.iterations = stats.get("iterations", instance.iterations)
            instance.total_trades = stats.get("total_trades", instance.total_trades)
            instance.winning_trades = stats.get("winning_trades", instance.winning_trades)
            instance.losing_trades = stats.get("losing_trades", instance.losing_trades)
            instance.total_pnl = stats.get("total_pnl", instance.total_pnl)
            instance.max_drawdown = stats.get("max_drawdown", instance.max_drawdown)
            instance.errors = stats.get("errors", instance.errors)
            last_sig = stats.get("last_signal")
            if last_sig:
                instance.last_signal = str(last_sig)

    # Enrich with executor live-loop signals (transient attrs, not DB columns):
    # recent-window realized PnL + open-position count drive the RBI
    # champion-challenger gate (F1) and its open-position skip guard.
    executor = getattr(request.app.state, "executor", None)
    if executor is not None and hasattr(executor, "recent_realized_pnl"):
        instance.recent_pnl, instance.recent_trades = executor.recent_realized_pnl(name)
        instance.active_positions = executor.open_position_count(name)

    return instance


@app.patch("/strategies/{name}", response_model=schemas.StrategyInstanceOut)
def update_strategy_instance(
    request: Request,
    name: str,
    data: schemas.StrategyInstanceUpdate,
    db: Session = Depends(get_db),
):
    """Update a strategy instance's configuration.

    The DB row is the source of truth, but a running strategy holds its config
    in-memory, so a DB-only write never reaches the live process. After persisting
    we hot-apply the same change to the orchestrator's live strategy object via
    ``update_live_params`` (open-position safety: exit-threshold tightening is
    deferred while a position is open). The applied/deferred/restart_required
    breakdown is returned in ``live_update``. Manual edits and RBI promotion both
    flow through this endpoint, so both reach the live process.
    """
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    from sqlalchemy.orm.attributes import flag_modified

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "params" and value is not None:
            instance.params = {**(instance.params or {}), **value}
            flag_modified(instance, "params")
        else:
            setattr(instance, field, value)

    db.commit()
    db.refresh(instance)

    # Hot-apply to the live running strategy so the change reaches the process,
    # not just the DB row. Never let a live-apply failure undo the persisted write.
    orchestrator = getattr(request.app.state, "orchestrator", None)
    live_update = None
    if orchestrator is not None:
        try:
            live_update = orchestrator.update_live_params(name, update_data)
        except Exception as e:
            logger.warning("Live-apply of PATCH to %s failed (DB write kept): %s", name, e)
    instance.live_update = live_update
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
async def start_strategy_instance(request: Request, name: str, db: Session = Depends(get_db)):
    """Start a strategy instance via the orchestrator."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        try:
            from src.strategies.base_strategy import StrategyConfig, StrategyTier

            tier_map = {"A": StrategyTier.A, "B": StrategyTier.B, "C": StrategyTier.C, "D": StrategyTier.D}
            config = StrategyConfig(
                name=instance.name,
                symbol=instance.symbol,
                tier=tier_map.get(instance.tier, StrategyTier.A),
                timeframe=instance.timeframe,
                leverage=instance.leverage,
                size_usd=instance.size_usd,
                target_pct=instance.target_pct,
                max_loss_pct=instance.max_loss_pct,
                lookback_days=instance.lookback_days,
                interval_seconds=instance.interval_seconds,
                enabled=True,
                params=instance.params or {},
            )
            # Add to orchestrator if not already there, then start
            if not orchestrator.get_strategy(name):
                orchestrator.add_strategy(name, instance.strategy_type, config)
            await orchestrator.start_strategy(name)
        except Exception as e:
            logger.error("Failed to start strategy %s in orchestrator: %s", name, e)
            instance.status = "error"
            instance.error_message = str(e)
            db.commit()
            raise HTTPException(status_code=500, detail=f"Orchestrator error: {e}")

    instance.status = "running"
    instance.started_at = datetime.utcnow()
    instance.error_message = None
    db.commit()
    return {"status": "running", "name": name, "strategy_type": instance.strategy_type, "symbol": instance.symbol}


@app.post("/strategies/{name}/stop")
async def stop_strategy_instance(request: Request, name: str, db: Session = Depends(get_db)):
    """Stop a strategy instance via the orchestrator and flush any open positions.

    Without the position flush, disabling a strategy leaves its open positions
    orphaned (orchestrator stops calling should_exit on them, so they drift
    indefinitely). close_by_strategy makes the disable atomic.
    """
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        try:
            await orchestrator.stop_strategy(name)
            orchestrator.remove_strategy(name)
        except Exception as e:
            logger.warning("Orchestrator stop error for %s (continuing DB update): %s", name, e)

    # Flush orphan positions held by this strategy
    closed_count = 0
    executor = getattr(request.app.state, "executor", None)
    if executor is not None and hasattr(executor, "close_by_strategy"):
        try:
            results = await executor.close_by_strategy(name)
            closed_count = sum(1 for r in results if r.success)
            if closed_count:
                logger.info("Closed %d orphan position(s) for %s", closed_count, name)
        except Exception as e:
            logger.warning("Orphan flush error for %s (continuing DB update): %s", name, e)

    instance.status = "stopped"
    db.commit()
    return {"status": "stopped", "name": name, "closed_positions": closed_count}


@app.post("/strategies/{name}/circuit-breaker/reset")
async def reset_circuit_breaker(name: str, request: Request):
    """Reset a strategy's circuit breaker and optionally restart it."""
    orchestrator = request.app.state.orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    strategy = orchestrator.get_strategy(name)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Strategy {name} not found")

    was_triggered = strategy.state.circuit_breaker_triggered if hasattr(strategy.state, 'circuit_breaker_triggered') else False

    # Reset the FULL circuit-breaker state — not just consecutive_losses. A
    # drawdown-tripped strategy must also have its drawdown inputs (max_drawdown,
    # peak_pnl) cleared, or it re-trips on the first losing trade. The helper is the
    # single source of truth shared with auto-recovery so the two never drift.
    if hasattr(strategy.state, 'reset_circuit_breaker'):
        strategy.state.reset_circuit_breaker()

    return {
        "name": name,
        "circuit_breaker_was_triggered": was_triggered,
        "circuit_breaker_reset": True,
        "message": f"Circuit breaker fully reset for {name} (halt, losses, and drawdown cleared). Use /strategies/{name}/start to restart.",
    }


# ──────────────────────────────────────────────
# Batch Deploy
# ──────────────────────────────────────────────

@app.post("/strategies/deploy-batch", response_model=schemas.BatchDeployResponse)
async def deploy_batch(
    request: Request,
    data: schemas.BatchDeployRequest,
    db: Session = Depends(get_db),
):
    """Create and start multiple strategies in one call."""
    from src.strategies.registry import list_strategies
    from src.strategies.base_strategy import StrategyConfig, StrategyTier

    available = {s["strategy_type"]: s for s in list_strategies()}
    tier_map = {"A": StrategyTier.A, "B": StrategyTier.B, "C": StrategyTier.C, "D": StrategyTier.D}
    orchestrator = getattr(request.app.state, "orchestrator", None)

    results: List[schemas.BatchDeployResultItem] = []
    started = 0
    failed = 0

    for item in data.strategies:
        # Validate strategy type
        if item.strategy_type not in available:
            results.append(schemas.BatchDeployResultItem(
                name=item.name, strategy_type=item.strategy_type, symbol=item.symbol,
                status="error", error=f"Unknown strategy type: {item.strategy_type}",
            ))
            failed += 1
            continue

        # Check name uniqueness
        existing = db.query(models.StrategyInstance).filter(
            models.StrategyInstance.name == item.name
        ).first()
        if existing:
            results.append(schemas.BatchDeployResultItem(
                name=item.name, strategy_type=item.strategy_type, symbol=item.symbol,
                status="error", error=f"Name '{item.name}' already exists",
            ))
            failed += 1
            continue

        registry_info = available[item.strategy_type]

        # Create DB instance
        instance = models.StrategyInstance(
            name=item.name,
            strategy_type=item.strategy_type,
            tier=registry_info["tier"],
            symbol=item.symbol,
            timeframe=item.timeframe,
            leverage=item.leverage,
            size_usd=item.size_usd,
            target_pct=item.target_pct,
            max_loss_pct=item.max_loss_pct,
            lookback_days=item.lookback_days,
            interval_seconds=item.interval_seconds,
            enabled=item.enabled,
            params={**registry_info["default_params"], **item.params},
        )
        db.add(instance)
        db.flush()  # get the ID without committing yet

        # Start via orchestrator
        if orchestrator is not None and item.enabled:
            try:
                config = StrategyConfig(
                    name=item.name,
                    symbol=item.symbol,
                    tier=tier_map.get(registry_info["tier"], StrategyTier.A),
                    timeframe=item.timeframe,
                    leverage=item.leverage,
                    size_usd=item.size_usd,
                    target_pct=item.target_pct,
                    max_loss_pct=item.max_loss_pct,
                    lookback_days=item.lookback_days,
                    interval_seconds=item.interval_seconds,
                    enabled=True,
                    params={**registry_info["default_params"], **item.params},
                )
                orchestrator.add_strategy(item.name, item.strategy_type, config)
                await orchestrator.start_strategy(item.name)
                instance.status = "running"
                instance.started_at = datetime.utcnow()
                started += 1
                results.append(schemas.BatchDeployResultItem(
                    name=item.name, strategy_type=item.strategy_type, symbol=item.symbol,
                    status="started",
                ))
            except Exception as e:
                logger.error("Batch deploy: failed to start %s — %s", item.name, e)
                instance.status = "error"
                instance.error_message = str(e)
                failed += 1
                results.append(schemas.BatchDeployResultItem(
                    name=item.name, strategy_type=item.strategy_type, symbol=item.symbol,
                    status="error", error=str(e),
                ))
        else:
            # No orchestrator or disabled — just create
            results.append(schemas.BatchDeployResultItem(
                name=item.name, strategy_type=item.strategy_type, symbol=item.symbol,
                status="created",
            ))

    db.commit()

    return schemas.BatchDeployResponse(
        total=len(data.strategies),
        started=started,
        failed=failed,
        results=results,
    )


# ──────────────────────────────────────────────
# Paper Trading Monitoring
# ──────────────────────────────────────────────

@app.get("/paper/stats", response_model=schemas.PaperStatsResponse)
async def paper_stats(request: Request):
    """Detailed paper trading stats: balance, PnL, equity curve, per-strategy breakdown."""
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    if not paper_mode or executor is None or not hasattr(executor, "get_execution_stats"):
        raise HTTPException(status_code=400, detail="Paper trading is not active")

    stats = executor.get_execution_stats()
    trades = executor.get_trade_history()
    equity_curve = executor.get_equity_curve()
    positions_raw = await executor.get_all_positions()
    active_pos = executor.get_active_positions()

    # Build per-strategy breakdown from trade history (exits only — entries have pnl=0)
    strat_map: dict = {}
    for t in trades:
        if t.get("action") != "exit":
            continue
        sname = t.get("strategy", "unknown")
        if sname not in strat_map:
            strat_map[sname] = {"total_trades": 0, "realized_pnl": 0.0}
        strat_map[sname]["total_trades"] += 1
        strat_map[sname]["realized_pnl"] += t.get("pnl", 0.0)

    # Enrich positions with mark prices
    positions_out = []
    for p in positions_raw:
        positions_out.append(schemas.PaperPosition(
            symbol=p["symbol"],
            side=p["side"],
            size=abs(p["size"]),
            entry_price=p["entry_px"],
            mark_price=p.get("entry_px", 0) + (p.get("unrealized_pnl", 0) / abs(p["size"])) if abs(p["size"]) > 0 else 0,
            unrealized_pnl=p.get("unrealized_pnl", 0),
            pnl_pct=p.get("pnl_perc", 0),
        ))

    # Build breakdown with active positions per strategy
    breakdown = []
    for sname, sdata in strat_map.items():
        # Find active position for this strategy
        active_position = None
        for key, pos_info in active_pos.items():
            if pos_info.get("strategy") == sname:
                active_position = schemas.PaperPosition(
                    symbol=pos_info["symbol"],
                    side=pos_info["side"],
                    size=abs(pos_info["size"]),
                    entry_price=pos_info["entry_price"],
                    strategy_name=sname,
                    entry_time=pos_info.get("entry_time"),
                )
                break

        breakdown.append(schemas.PaperStrategyBreakdown(
            strategy_name=sname,
            total_trades=sdata["total_trades"],
            realized_pnl=round(sdata["realized_pnl"], 2),
            active_position=active_position,
        ))

    return schemas.PaperStatsResponse(
        mode="paper",
        balance=stats.get("balance", 0),
        initial_balance=stats.get("initial_balance", 0),
        total_return_pct=stats.get("total_return_pct", 0),
        total_realized_pnl=stats.get("total_realized_pnl", 0),
        peak_balance=stats.get("peak_balance", 0),
        max_drawdown_pct=stats.get("max_drawdown_pct", 0),
        total_executions=stats.get("total_executions", 0),
        total_trades=stats.get("total_trades", 0),
        success_rate=stats.get("success_rate", 0),
        active_positions=stats.get("active_positions", 0),
        positions=positions_out,
        equity_curve=[
            schemas.PaperEquityCurvePoint(timestamp=pt["timestamp"], equity=pt["equity"])
            for pt in equity_curve
        ],
        strategy_breakdown=breakdown,
    )


@app.get("/paper/positions", response_model=schemas.PaperPositionsResponse)
async def paper_positions(request: Request):
    """All current paper positions with live unrealized PnL."""
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    if not paper_mode or executor is None or not hasattr(executor, "get_all_positions"):
        raise HTTPException(status_code=400, detail="Paper trading is not active")

    positions_raw = await executor.get_all_positions()

    positions_out = []
    for p in positions_raw:
        abs_size = abs(p["size"])
        positions_out.append(schemas.PaperPosition(
            symbol=p["symbol"],
            side=p["side"],
            size=abs_size,
            size_usd=p.get("size_usd", 0.0),
            entry_price=p["entry_px"],
            mark_price=p.get("entry_px", 0) + (p.get("unrealized_pnl", 0) / abs_size) if abs_size > 0 else 0,
            unrealized_pnl=p.get("unrealized_pnl", 0),
            pnl_pct=p.get("pnl_perc", 0),
            strategy_name=p.get("strategy_name", ""),
            entry_time=None,
        ))

    return schemas.PaperPositionsResponse(total=len(positions_out), positions=positions_out)


@app.get("/paper/trades", response_model=schemas.PaperTradesResponse)
def paper_trades(request: Request, limit: int = 100):
    """Paper trade history with timestamps and PnL."""
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    if not paper_mode or executor is None or not hasattr(executor, "get_trade_history"):
        raise HTTPException(status_code=400, detail="Paper trading is not active")

    trades = executor.get_trade_history()

    # Most recent first, limited
    trades_reversed = list(reversed(trades))[:limit]

    trades_out = [
        schemas.PaperTrade(
            id=t["id"],
            symbol=t["symbol"],
            side=t["side"],
            action=t["action"],
            price=t["price"],
            size=t["size"],
            size_usd=t["size_usd"],
            pnl=t.get("pnl", 0),
            pnl_pct=t.get("pnl_pct", 0),
            reason=t.get("reason", ""),
            strategy=t.get("strategy", ""),
            timestamp=t["timestamp"],
        )
        for t in trades_reversed
    ]

    return schemas.PaperTradesResponse(total=len(trades_out), trades=trades_out)


@app.post("/paper/reset")
async def reset_paper_trading(request: Request):
    """Reset paper trading balance to initial value and close all positions."""
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    if not paper_mode or executor is None or not hasattr(executor, "reset"):
        raise HTTPException(status_code=400, detail="Paper trading is not active")

    # Also reset strategy-level anti-overtrading state. The orchestrator exposes
    # its strategies as `_strategies` — the old `orchestrator.strategies` reference
    # did not exist, so the guard was always False and this loop never ran.
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        for strategy in orchestrator._strategies.values():
            strategy.state.total_trades = 0
            strategy.state.winning_trades = 0
            strategy.state.losing_trades = 0
            strategy.state.total_pnl = 0.0
            strategy.state.max_drawdown = 0.0
            strategy.state.peak_pnl = 0.0
            strategy.state.last_trade_close_time = None
            strategy.state.trades_this_hour = 0
            strategy.state.hour_start = None
            strategy.state.entry_bar_count = 0

    executor.reset()
    return {
        "status": "reset",
        "balance": executor.balance,
        "initial_balance": executor.initial_balance,
    }


# ──────────────────────────────────────────────
# Paper observability endpoints (Track 1)
# ──────────────────────────────────────────────

# Threshold constants — never hardcode in logic below
_DIVERGENCE_ALERT_THRESHOLD = 5.0   # USD: alert when live vs DB portfolio PnL diverge more than this
# Single source shared with circuit-breaker auto-recovery (src/engine/shadow_recovery.py).
from src.execution.paper_executor import EDGE_MIN_TRADES_REAL as _EDGE_MIN_TRADES_REAL
_EDGE_BREAKEVEN_PRECISION = 6       # decimal places for breakeven_wr display


@app.get("/paper/pnl")
async def paper_pnl(request: Request, db: Session = Depends(get_db)):
    """Canonical portfolio PnL — balance-delta method, survives Railway redeploys.

    live_pnl  = executor.balance - executor.initial_balance (in-memory, most accurate when present)
    db_pnl    = pnl_snapshots._portfolio.balance_pnl (persisted every 5 min by _pnl_flush_loop)
    canonical = live_pnl if executor is available, else db_pnl
    """
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    live_pnl = None
    balance = None
    initial_balance = None
    if paper_mode and executor is not None:
        balance = round(float(getattr(executor, "balance", 0)), 2)
        initial_balance = round(float(getattr(executor, "initial_balance", 0)), 2)
        live_pnl = round(balance - initial_balance, 2)

    db_pnl = None
    port_row = db.query(models.PnlSnapshot).filter(
        models.PnlSnapshot.strategy_name == "_portfolio"
    ).first()
    if port_row is not None and port_row.balance_pnl is not None:
        db_pnl = round(float(port_row.balance_pnl), 2)

    if live_pnl is not None:
        canonical_pnl = live_pnl
        source = "live_executor"
    elif db_pnl is not None:
        canonical_pnl = db_pnl
        source = "db_snapshot"
    else:
        canonical_pnl = None
        source = "unavailable"

    return {
        "canonical_pnl": canonical_pnl,
        "source": source,
        "live_pnl": live_pnl,
        "db_pnl": db_pnl,
        "balance": balance,
        "initial_balance": initial_balance,
    }


@app.get("/paper/edge")
def paper_edge(request: Request):
    """Per-strategy edge stats with Wilson 90% CI and real-edge flag.

    real_edge = True when n >= _EDGE_MIN_TRADES_REAL AND wr_lower_90 > breakeven_wr.
    size_flag:
        'observation' — n < _EDGE_MIN_TRADES_REAL (still building sample)
        'real'        — real_edge confirmed
        'noise'       — enough trades but lower CI doesn't clear breakeven
    """
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    if not paper_mode or executor is None or not hasattr(executor, "_live_edge_stats"):
        raise HTTPException(status_code=400, detail="Paper trading is not active")

    # Build strategy name set from full trade history
    trade_history = executor.get_trade_history() if hasattr(executor, "get_trade_history") else []
    strategy_names = {t.get("strategy", "") for t in trade_history if t.get("strategy")}

    from src.execution.paper_executor import breakeven_wr as _breakeven_wr

    result = []
    for name in sorted(strategy_names):
        wr, payoff, n, wr_lo, wr_hi = executor._live_edge_stats(name)
        breakeven_wr = round(_breakeven_wr(payoff), _EDGE_BREAKEVEN_PRECISION)
        is_real_edge = (n >= _EDGE_MIN_TRADES_REAL) and (wr_lo > breakeven_wr)

        if n < _EDGE_MIN_TRADES_REAL:
            size_flag = "observation"
        elif is_real_edge:
            size_flag = "real"
        else:
            size_flag = "noise"

        result.append({
            "strategy": name,
            "n_trades": n,
            "win_rate": round(wr, 4),
            "payoff_ratio": round(payoff, 4),
            "wr_lower_90": round(wr_lo, 4),
            "wr_upper_90": round(wr_hi, 4),
            "breakeven_wr": breakeven_wr,
            "real_edge": is_real_edge,
            "size_flag": size_flag,
        })

    return {"strategies": result, "min_trades_for_real_edge": _EDGE_MIN_TRADES_REAL}


@app.get("/paper/reconcile")
async def paper_reconcile(request: Request, db: Session = Depends(get_db)):
    """Silent-divergence alarm: compares live executor balance-delta vs DB snapshot.

    divergence_alert fires when |executor_balance_pnl - db_portfolio_pnl| > _DIVERGENCE_ALERT_THRESHOLD.
    Typical causes: Railway redeploy without paper_state.json, or _pnl_flush_loop lag.
    """
    executor = getattr(request.app.state, "executor", None)
    paper_mode = getattr(request.app.state, "paper_mode", False)

    executor_balance_pnl = None
    executor_trade_sum_pnl = None
    if paper_mode and executor is not None:
        executor_balance_pnl = round(
            float(getattr(executor, "balance", 0)) - float(getattr(executor, "initial_balance", 0)), 2
        )
        trade_history = executor.get_trade_history() if hasattr(executor, "get_trade_history") else []
        executor_trade_sum_pnl = round(
            sum(t.get("pnl", 0) for t in trade_history if t.get("action") == "exit"), 2
        )

    db_portfolio_pnl = None
    port_row = db.query(models.PnlSnapshot).filter(
        models.PnlSnapshot.strategy_name == "_portfolio"
    ).first()
    if port_row is not None and port_row.balance_pnl is not None:
        db_portfolio_pnl = round(float(port_row.balance_pnl), 2)

    divergence_alert = False
    if executor_balance_pnl is not None and db_portfolio_pnl is not None:
        divergence_alert = abs(executor_balance_pnl - db_portfolio_pnl) > _DIVERGENCE_ALERT_THRESHOLD

    ok = (executor_balance_pnl is not None) and not divergence_alert

    return {
        "executor_balance_pnl": executor_balance_pnl,
        "executor_trade_sum_pnl": executor_trade_sum_pnl,
        "db_portfolio_pnl": db_portfolio_pnl,
        "divergence_alert": divergence_alert,
        "divergence_threshold_usd": _DIVERGENCE_ALERT_THRESHOLD,
        "ok": ok,
    }


# ──────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────

@app.get("/dashboard/stats", response_model=schemas.DashboardStats)
def dashboard_stats(request: Request, db: Session = Depends(get_db)):
    """Get aggregated dashboard statistics, with live paper executor data."""
    instances = db.query(models.StrategyInstance).all()
    total = len(instances)
    running = sum(1 for i in instances if i.status == "running")

    # Try to get live stats from paper executor
    executor = getattr(request.app.state, "executor", None)
    total_pnl = 0.0
    total_trades = 0
    winning = 0
    active_positions = 0
    balance = None

    if executor and hasattr(executor, "get_execution_stats"):
        try:
            stats = executor.get_execution_stats()
            # Use balance-delta as canonical PnL (survives redeploy; avoids reset-prone _execution_history)
            total_pnl = round(stats.get("balance", 0.0) - stats.get("initial_balance", 0.0), 2)
            total_trades = stats.get("total_trades", 0)
            active_positions = stats.get("active_positions", 0)
            balance = stats.get("balance", None)
            # Count wins from trade history
            if hasattr(executor, "get_trade_history"):
                for t in executor.get_trade_history():
                    if t.get("pnl", 0) > 0:
                        winning += 1
        except Exception:
            pass

    if total_trades == 0:
        # Fallback to DB records
        total_pnl = sum(i.total_pnl for i in instances)
        total_trades = sum(i.total_trades for i in instances)
        winning = sum(i.winning_trades for i in instances)

    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0

    live_equity = balance or _get_live_vault_equity()
    if active_positions == 0:
        positions = _get_live_positions()
        active_positions = len(positions)

    return schemas.DashboardStats(
        total_strategies=total,
        running_strategies=running,
        total_pnl=round(total_pnl, 2),
        total_trades=total_trades,
        win_rate=round(win_rate, 1),
        vault_equity=live_equity,
        active_positions=active_positions,
    )


# ──────────────────────────────────────────────
# Vault History
# ──────────────────────────────────────────────

@app.get("/vault/history", response_model=schemas.VaultHistoryOut)
def vault_history(days: int = 30, db: Session = Depends(get_db)):
    """Return time-series vault equity and NAV data.

    Generates simulated data points based on the current vault state.
    In production, this will be replaced with actual historical snapshots.
    """
    vault = _get_or_create_vault_state(db)
    base_equity = vault.total_equity if vault.total_equity > 0 else 10000.0
    base_nav = vault.nav_per_share if vault.nav_per_share > 0 else 1.0

    now = datetime.now(timezone.utc)
    data_points = []

    # Seed the random generator deterministically per day so data is stable across requests
    rng = random.Random(42)

    equity = base_equity * 0.85  # Start 15% lower to show growth
    nav = base_nav * 0.85

    for i in range(days):
        ts = now - timedelta(days=days - i)
        # Simulated daily return: slight upward bias with noise
        daily_return = 1.0 + rng.gauss(0.003, 0.015)
        equity *= daily_return
        nav *= daily_return

        data_points.append(
            schemas.VaultHistoryPoint(
                timestamp=ts.replace(hour=0, minute=0, second=0, microsecond=0),
                equity=round(equity, 2),
                nav_per_share=round(nav, 6),
            )
        )

    # Last point = current actual values
    data_points[-1].equity = round(base_equity, 2) if base_equity > 0 else data_points[-1].equity
    data_points[-1].nav_per_share = round(base_nav, 6) if base_nav > 0 else data_points[-1].nav_per_share

    return schemas.VaultHistoryOut(data=data_points, total_points=len(data_points))


# ──────────────────────────────────────────────
# Dashboard Performance
# ──────────────────────────────────────────────

@app.get("/dashboard/performance", response_model=schemas.DashboardPerformance)
def dashboard_performance(db: Session = Depends(get_db)):
    """Aggregated performance: PnL, win rate, max drawdown, equity curve, per-strategy breakdown."""
    instances = db.query(models.StrategyInstance).all()

    total_pnl = sum(i.total_pnl for i in instances)
    total_trades = sum(i.total_trades for i in instances)
    winning = sum(i.winning_trades for i in instances)
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
    max_drawdown = max((i.max_drawdown for i in instances), default=0.0)

    # Strategy-level breakdown
    breakdown = []
    for inst in instances:
        inst_trades = inst.total_trades
        inst_wr = (inst.winning_trades / inst_trades * 100) if inst_trades > 0 else 0.0
        breakdown.append(
            schemas.StrategyPerformanceBreakdown(
                name=inst.name,
                strategy_type=inst.strategy_type,
                total_pnl=round(inst.total_pnl, 2),
                total_trades=inst_trades,
                winning_trades=inst.winning_trades,
                losing_trades=inst.losing_trades,
                win_rate=round(inst_wr, 1),
                max_drawdown=round(inst.max_drawdown, 2),
                status=inst.status,
            )
        )

    # Equity curve: 30 simulated daily points based on vault state
    vault = _get_or_create_vault_state(db)
    base_equity = vault.total_equity if vault.total_equity > 0 else 10000.0
    now = datetime.now(timezone.utc)
    rng = random.Random(42)
    equity = base_equity * 0.85
    equity_curve = []
    for i in range(30):
        ts = now - timedelta(days=30 - i)
        daily_return = 1.0 + rng.gauss(0.003, 0.015)
        equity *= daily_return
        equity_curve.append(
            schemas.EquityCurvePoint(
                timestamp=ts.replace(hour=0, minute=0, second=0, microsecond=0),
                equity=round(equity, 2),
            )
        )
    equity_curve[-1].equity = round(base_equity, 2) if base_equity > 0 else equity_curve[-1].equity

    return schemas.DashboardPerformance(
        total_pnl=round(total_pnl, 2),
        win_rate=round(win_rate, 1),
        total_trades=total_trades,
        max_drawdown=round(max_drawdown, 2),
        equity_curve=equity_curve,
        strategy_breakdown=breakdown,
    )


# ──────────────────────────────────────────────
# Profitability Controls (MoonDev Intel)
# ──────────────────────────────────────────────

@app.get("/profitability/controls")
def profitability_controls(request: Request):
    """Get MoonDev profitability control stats: regime gate, rate limits, daily loss guard."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    regime_detector = getattr(request.app.state, "regime_detector", None)

    result = {
        "orchestrator_active": orchestrator is not None,
        "regime_detector_active": regime_detector is not None,
    }

    if orchestrator and hasattr(orchestrator, "get_profitability_stats"):
        result["profitability"] = orchestrator.get_profitability_stats()

    if regime_detector:
        result["regimes"] = regime_detector.get_all_regimes()
        result["volatility"] = regime_detector.get_volatility_status()

    return result


@app.get("/gate-stats")
def gate_stats(request: Request):
    """Attribution data from the order-book imbalance gate (Gate 4.7).

    Returns counts and percentages of entries that would have been blocked if
    the gate were in active (non-shadow) mode.  Shadow mode is the default —
    no entries are actually blocked until SHADOW_MODE is flipped to False in
    src/services/orderbook_gate.py.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None or not hasattr(orchestrator, "_ob_gate"):
        return {"error": "orchestrator not initialized", "stats": None}
    return orchestrator._ob_gate.get_stats()


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


# ──────────────────────────────────────────────
# Strategy Trades
# ──────────────────────────────────────────────

@app.get("/strategies/{name}/trades", response_model=schemas.StrategyTradesResponse)
def get_strategy_trades(
    name: str, request: Request, limit: int = 100, db: Session = Depends(get_db)
):
    """Get trade history for a specific strategy."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    paper_mode = getattr(request.app.state, "paper_mode", False)
    executor = getattr(request.app.state, "executor", None)

    trades_out = []
    if paper_mode and executor and hasattr(executor, "get_trade_history"):
        all_trades = executor.get_trade_history()
        strat_trades = [t for t in all_trades if t.get("strategy") == name]
        strat_trades = list(reversed(strat_trades))[:limit]
        for t in strat_trades:
            trades_out.append(schemas.StrategyTradeItem(
                id=t["id"],
                symbol=t["symbol"],
                side=t["side"],
                action=t["action"],
                price=t["price"],
                size=t["size"],
                size_usd=t["size_usd"],
                pnl=t.get("pnl", 0),
                pnl_pct=t.get("pnl_pct", 0),
                reason=t.get("reason", ""),
                timestamp=t["timestamp"],
            ))
    else:
        db_trades = db.query(models.Trade).filter(
            models.Trade.strategy == name
        ).order_by(models.Trade.opened_at.desc()).limit(limit).all()
        for t in db_trades:
            trades_out.append(schemas.StrategyTradeItem(
                id=t.id,
                symbol=t.symbol,
                side=t.side,
                action="exit" if t.closed_at else "entry",
                price=t.exit_price if t.exit_price else t.entry_price,
                size=t.size,
                size_usd=t.size * t.entry_price,
                pnl=t.pnl or 0.0,
                pnl_pct=0.0,
                reason=t.exit_reason or "",
                timestamp=(t.closed_at or t.opened_at).isoformat(),
            ))

    return schemas.StrategyTradesResponse(
        strategy_name=name,
        total=len(trades_out),
        trades=trades_out,
    )


# ──────────────────────────────────────────────
# Strategy Signals
# ──────────────────────────────────────────────

@app.get("/strategies/{name}/signals", response_model=schemas.StrategySignalsResponse)
def get_strategy_signals(name: str, request: Request, db: Session = Depends(get_db)):
    """Get recent signal log for a strategy from the orchestrator."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    orchestrator = getattr(request.app.state, "orchestrator", None)
    signals_out = []

    if orchestrator:
        strategy = orchestrator.get_strategy(name)
        if strategy and strategy.state.last_signal:
            sig = strategy.state.last_signal
            signals_out.append(schemas.SignalLogItem(
                signal_type=sig.signal_type.value,
                symbol=sig.symbol,
                strength=sig.strength,
                price=sig.price,
                reason=sig.reason,
                timestamp=sig.timestamp.isoformat(),
            ))

    if instance.last_signal and instance.last_signal_time:
        if not signals_out or signals_out[0].timestamp != instance.last_signal_time.isoformat():
            signals_out.append(schemas.SignalLogItem(
                signal_type=instance.last_signal,
                symbol=instance.symbol,
                strength=1.0,
                price=None,
                reason="",
                timestamp=instance.last_signal_time.isoformat() if instance.last_signal_time else "",
            ))

    return schemas.StrategySignalsResponse(
        strategy_name=name,
        total=len(signals_out),
        signals=signals_out,
    )


# ──────────────────────────────────────────────
# Strategy Optimization
# ──────────────────────────────────────────────

@app.post("/strategies/{name}/optimize", response_model=schemas.OptimizeResponse)
async def optimize_strategy(name: str, data: schemas.OptimizeRequest, db: Session = Depends(get_db)):
    """Run parameter optimization for a specific strategy instance."""
    instance = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name
    ).first()
    if not instance:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    if not data.param_ranges:
        raise HTTPException(status_code=400, detail="param_ranges must not be empty")

    import itertools
    from src.strategies.registry import get_strategy_class
    from src.engine.backtester import Backtester

    try:
        get_strategy_class(instance.strategy_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    param_names = list(data.param_ranges.keys())
    param_values = list(data.param_ranges.values())
    all_combos = list(itertools.product(*param_values))
    total_combos = len(all_combos)

    if total_combos > data.max_combinations:
        step = total_combos / data.max_combinations
        indices = [int(i * step) for i in range(data.max_combinations)]
        combos_to_test = [all_combos[i] for i in indices]
    else:
        combos_to_test = all_combos

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=data.lookback_days)
    start_date = start_dt.isoformat()
    end_date = end_dt.isoformat()

    results = []
    for combo in combos_to_test:
        params = {**instance.params, **dict(zip(param_names, combo))}
        try:
            bt = Backtester(initial_capital=data.initial_capital, commission_pct=0.07)
            engine_result = await bt.run(
                strategy_type=instance.strategy_type,
                symbol=instance.symbol,
                timeframe=instance.timeframe,
                start_date=start_date,
                end_date=end_date,
                params=params,
            )
            results.append({
                "params": dict(zip(param_names, combo)),
                "sharpe_ratio": engine_result.sharpe_ratio,
                "total_return_pct": engine_result.total_pnl_pct,
                "win_rate": engine_result.win_rate,
                "profit_factor": (
                    engine_result.profit_factor
                    if engine_result.profit_factor != float("inf")
                    else 999.0
                ),
                "total_trades": engine_result.total_trades,
                "max_drawdown_pct": engine_result.max_drawdown_pct,
            })
        except Exception as e:
            logger.warning("Optimize combo %s failed: %s", combo, e)
            continue

    if not results:
        raise HTTPException(status_code=500, detail="All parameter combinations failed")

    results.sort(key=lambda r: r["sharpe_ratio"], reverse=True)
    top_results = results[:10]
    best = top_results[0]

    return schemas.OptimizeResponse(
        strategy_name=name,
        total_combinations=total_combos,
        tested=len(results),
        best_params=best["params"],
        best_sharpe=round(best["sharpe_ratio"], 3),
        best_return_pct=round(best["total_return_pct"], 2),
        best_win_rate=round(best["win_rate"], 2),
        top_results=top_results,
    )


# ──────────────────────────────────────────────
# Market Overview
# ──────────────────────────────────────────────

@app.get("/market/overview", response_model=schemas.MarketOverviewResponse)
def market_overview(
    request: Request,
    symbols: str = "BTC,ETH,SOL",
):
    """Market conditions summary: regime, volatility, recommended strategies per symbol."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    regime_detector = getattr(request.app.state, "regime_detector", None)

    results = []
    for symbol in symbol_list:
        item = schemas.MarketSymbolOverview(symbol=symbol)

        price_data = _get_market_price(symbol)
        if price_data:
            item.price = price_data.price

        if regime_detector:
            try:
                regime = regime_detector.get_current_regime(symbol)
                if regime:
                    item.regime = regime.get("regime", "unknown")
                    item.regime_confidence = regime.get("confidence", 0.0)
                    try:
                        from src.services.regime_detector import REGIME_STRATEGY_MAP, MarketRegime
                        current = MarketRegime(item.regime)
                        item.recommended_strategies = REGIME_STRATEGY_MAP.get(current, [])
                    except (ValueError, ImportError):
                        pass

                vol_status = regime_detector.get_volatility_status()
                vol = vol_status.get(symbol)
                if vol:
                    item.is_volatile = vol.get("is_volatile", False)
                    atr_pct = vol.get("atr_pct", 0)
                    if atr_pct < 1.5:
                        item.volatility_regime = "low"
                    elif atr_pct < 3.0:
                        item.volatility_regime = "medium"
                    elif atr_pct < 5.0:
                        item.volatility_regime = "high"
                    else:
                        item.volatility_regime = "extreme"
            except Exception as e:
                logger.warning("Could not get regime for %s: %s", symbol, e)

        results.append(item)

    return schemas.MarketOverviewResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbols=results,
    )


# ──────────────────────────────────────────────
# XsecDriverEngine runtime management
# ──────────────────────────────────────────────

# Single source of truth lives in the engine module — a duplicated literal here
# silently 400-rejected new drivers the engine already supported (amihud_illiq,
# 2026-07-01 observe funnel).
from src.engine.xsec_driver_engine import SUPPORTED_DRIVERS as _XSEC_DRIVER_SUPPORTED
_XSEC_DRIVER_DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "SUI", "ARB"
]


@app.post("/xsec/instances", response_model=schemas.StrategyInstanceOut)
def create_xsec_driver_instance(
    data: schemas.XsecDriverCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    """Create, persist, and immediately start a new XsecDriverEngine instance.

    The instance appears in GET /strategies, survives Railway redeploys (boot-restart),
    and is controllable via POST /xsec/instances/{name}/stop.
    """
    if data.driver not in _XSEC_DRIVER_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown driver '{data.driver}'. Supported: {sorted(_XSEC_DRIVER_SUPPORTED)}",
        )
    if data.driver == "ensemble":
        from src.engine.xsec_driver_engine import BASE_DRIVERS as _XSEC_BASE
        bad = [m.get("driver") for m in (data.members or [])
               if m.get("driver") not in _XSEC_BASE]
        if not data.members or bad:
            raise HTTPException(
                status_code=400,
                detail=f"ensemble requires members with drivers in {sorted(_XSEC_BASE)}; got {data.members}",
            )

    existing = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == data.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Name '{data.name}' already exists")

    coins = list(data.coins) if data.coins else _XSEC_DRIVER_DEFAULT_COINS

    instance = models.StrategyInstance(
        name=data.name,
        strategy_type="xsec_driver",
        tier="A",
        status="running",
        symbol=coins[0] if coins else "BTC",
        timeframe=data.timeframe,
        leverage=1,
        size_usd=data.per_leg_usd,
        enabled=data.enabled,
        params={
            "driver": data.driver,
            "lookback": data.lookback,
            "q": data.q,
            "sign": data.sign,
            "coins": coins,
            "per_leg_usd": data.per_leg_usd,
            "rebalance_secs": data.rebalance_secs,
            "members": data.members,
            "trail_days": data.trail_days,
        },
    )
    db.add(instance)
    db.commit()
    db.refresh(instance)

    # Start the engine task immediately (non-fatal if executor not available)
    executor = getattr(request.app.state, "executor", None)
    client = getattr(request.app.state, "client", None)
    if executor is not None and client is not None:
        try:
            from src.engine.xsec_driver_engine import XsecDriverEngine
            eng = XsecDriverEngine(
                executor=executor, client=client,
                name=data.name,
                driver=data.driver,
                lookback=data.lookback,
                q=data.q,
                sign=data.sign,
                coins=coins,
                per_leg_usd=data.per_leg_usd,
                rebalance_secs=data.rebalance_secs,
                timeframe=data.timeframe,
                members=data.members,
                trail_days=data.trail_days,
            )
            import asyncio as _aio
            loop = getattr(request.app.state, "loop", None)
            if loop is not None:
                # sync endpoint runs in a threadpool thread — schedule onto the main loop
                task = _aio.run_coroutine_threadsafe(eng.run(), loop)
            else:
                task = _aio.get_event_loop().create_task(eng.run())
            request.app.state.xsec_driver_engines[data.name] = (eng, task)
            logger.info("xsec_driver: started %s (%s)", data.name, data.driver)
        except Exception as exc:
            logger.warning("xsec_driver: failed to start engine for %s — %s", data.name, exc)
    else:
        logger.warning("xsec_driver: created %s but no executor/client — engine not started", data.name)

    return instance


@app.get("/xsec/instances", response_model=List[schemas.XsecDriverInstanceOut])
def list_xsec_driver_instances(request: Request, db: Session = Depends(get_db)):
    """List all xsec_driver instances with live running status."""
    rows = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.strategy_type == "xsec_driver"
    ).all()
    running_map = getattr(request.app.state, "xsec_driver_engines", {})
    result = []
    for row in rows:
        p = row.params or {}
        is_running = row.name in running_map and not running_map[row.name][1].done()
        result.append(schemas.XsecDriverInstanceOut(
            name=row.name,
            driver=p.get("driver", ""),
            lookback=int(p.get("lookback", 24)),
            q=float(p.get("q", 0.30)),
            sign=int(p.get("sign", -1)),
            coins=p.get("coins") or _XSEC_DRIVER_DEFAULT_COINS,
            per_leg_usd=float(p.get("per_leg_usd", 50.0)),
            rebalance_secs=int(p.get("rebalance_secs", 3600)),
            timeframe=row.timeframe or "1h",
            running=is_running,
            open_legs=dict(running_map[row.name][0]._open_legs) if is_running else None,
        ))
    return result


@app.post("/xsec/instances/{name}/stop")
async def stop_xsec_driver_instance(
    name: str, request: Request, db: Session = Depends(get_db)
):
    """Stop a running XsecDriverEngine and mark it stopped in the DB."""
    row = db.query(models.StrategyInstance).filter(
        models.StrategyInstance.name == name,
        models.StrategyInstance.strategy_type == "xsec_driver",
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"xsec_driver instance '{name}' not found")

    running_map = getattr(request.app.state, "xsec_driver_engines", {})
    if name in running_map:
        eng, task = running_map.pop(name)
        eng.stop()
        task.cancel()
        try:
            await asyncio.gather(_await_xsec_task(task), return_exceptions=True)
        except Exception:
            pass
        logger.info("xsec_driver(%s): stopped via API", name)
    else:
        logger.info("xsec_driver(%s): stop requested but was not in running map", name)

    # Flush the instance's open legs — a stopped engine can never close them,
    # so without this they orphan as drifting positions.
    executor = getattr(request.app.state, "executor", None)
    if executor is not None and hasattr(executor, "close_by_strategy"):
        try:
            _res = await executor.close_by_strategy(name)
            _n = sum(1 for r in _res if r.success)
            if _n:
                logger.info("xsec_driver(%s): flushed %d open leg(s)", name, _n)
        except Exception as _fe:
            logger.warning("xsec_driver(%s): leg flush error: %s", name, _fe)

    row.status = "stopped"
    db.commit()
    return {"status": "stopped", "name": name}


# ──────────────────────────────────────────────
# WebSocket: Real-time Signals
# ──────────────────────────────────────────────

_signal_subscribers: List[WebSocket] = []


@app.websocket("/ws/signals")
async def ws_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time strategy signal streaming."""
    await websocket.accept()
    _signal_subscribers.append(websocket)
    logger.info("Signal WebSocket client connected (%d total)", len(_signal_subscribers))

    try:
        orchestrator = getattr(websocket.app.state, "orchestrator", None)
        while True:
            if orchestrator:
                all_stats = orchestrator.get_all_stats()
                signals_data = []
                for s in all_stats:
                    last_sig = s.get("last_signal")
                    if last_sig:
                        signals_data.append({
                            "strategy": s["name"],
                            "symbol": s["symbol"],
                            "signal": last_sig,
                            "is_running": s["is_running"],
                            "iterations": s["iterations"],
                            "total_pnl": s["total_pnl"],
                        })
                if signals_data:
                    await websocket.send_json({
                        "type": "signals",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": signals_data,
                    })
            else:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("Signal WebSocket client disconnected")
    except Exception as e:
        logger.error("Signal WebSocket error: %s", e)
    finally:
        if websocket in _signal_subscribers:
            _signal_subscribers.remove(websocket)


# ──────────────────────────────────────────────
# WebSocket: Real-time Portfolio Updates
# ──────────────────────────────────────────────

@app.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket):
    """WebSocket endpoint for real-time portfolio updates."""
    await websocket.accept()
    logger.info("Portfolio WebSocket client connected")

    try:
        while True:
            executor = getattr(websocket.app.state, "executor", None)
            paper_mode = getattr(websocket.app.state, "paper_mode", False)
            orchestrator = getattr(websocket.app.state, "orchestrator", None)

            portfolio_data = {
                "type": "portfolio_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trading_mode": getattr(websocket.app.state, "trading_mode", "unknown"),
            }

            if paper_mode and executor and hasattr(executor, "get_execution_stats"):
                stats = executor.get_execution_stats()
                portfolio_data["balance"] = stats.get("balance", 0)
                portfolio_data["total_pnl"] = stats.get("total_realized_pnl", 0)
                portfolio_data["total_return_pct"] = stats.get("total_return_pct", 0)
                portfolio_data["active_positions"] = stats.get("active_positions", 0)
                portfolio_data["max_drawdown_pct"] = stats.get("max_drawdown_pct", 0)

            if orchestrator:
                portfolio_data["running_strategies"] = orchestrator.get_running_count()
                portfolio_data["orchestrator_pnl"] = round(orchestrator.get_total_pnl(), 2)

            await websocket.send_json(portfolio_data)

            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=3.0)
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        logger.info("Portfolio WebSocket client disconnected")
    except Exception as e:
        logger.error("Portfolio WebSocket error: %s", e)
