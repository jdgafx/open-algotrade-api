"""
Strategy Orchestrator — Runs multiple strategies concurrently via the executor.

Lifecycle:
1. Load strategy configs from DB (StrategyInstance table)
2. Instantiate each enabled strategy via the registry
3. Run each on its own asyncio loop interval
4. Route signals through HyperliquidVaultExecutor
5. Track PnL, errors, and health per strategy

MoonDev Profitability Controls (integrated from 200-video analysis):
- Regime detection gate: only trade when regime matches strategy type
- Global trade rate limiter: max trades/hour across ALL strategies
- Daily loss guard: hard stop when daily P&L breaches threshold
- Portfolio exposure cap: max total position value
- Dynamic position sizing: scale down in high-vol regimes
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.engine.llm_gate import TradeContext, llm_gate
from src.services.funding_monitor import FUNDING_LONG_SUPPRESS_THRESHOLD
from src.services.adaptation import adaptation_multiplier
from src.execution.hl_executor import HyperliquidVaultExecutor
from src.execution.paper_executor import DEFAULT_RUIN_GUARD_BUFFER_PCT
from src.lib.nice_funcs import HyperliquidClient
from src.services.hlp_gate import HLPSentimentGate
from src.services.liquidation_guard import LiquidationGuard
from src.services.orderbook_gate import OrderBookImbalanceGate
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyTier
from src.strategies.registry import create_strategy, get_strategy_class

logger = logging.getLogger(__name__)


# ── Live-tuning field classes (which config changes can hot-apply to a running strategy) ──
# Forward-looking fields are re-read from config on every iteration / at the next entry,
# so they affect only the NEXT entry's sizing/leverage, never an open position — always
# safe to apply live.
HOT_FORWARD_FIELDS = frozenset({"leverage", "size_usd", "lookback_days"})
# Exit-threshold fields are read by should_exit() on the next bar, so tightening one
# while a position is open can instantly force-close it — defer until flat (or force).
EXIT_THRESHOLD_FIELDS = frozenset({"target_pct", "max_loss_pct"})
# Fields the running loop does not honor live: symbol/timeframe/interval_seconds are
# captured loop-local at strategy-loop start, and `enabled` is read only by the start
# gate (start_all), never re-checked inside _run_strategy_loop. A live mutation has no
# effect until restart, so report it rather than claim a (false) live apply. Urgent
# halt of a running strategy is POST /strategies/{name}/stop, not enabled=false.
RESTART_REQUIRED_FIELDS = frozenset({"symbol", "timeframe", "interval_seconds", "enabled"})


class StrategyOrchestrator:
    """
    Manages the lifecycle of multiple concurrent strategies.

    Each strategy runs on its own interval, fetching data and emitting signals.
    Signals are executed through a shared HyperliquidVaultExecutor.

    MoonDev profitability controls:
    - Regime detection gate (only trade when regime matches)
    - Global trade rate limiter (max N trades/hour across ALL strategies)
    - Daily loss guard (hard stop at -X% daily)
    - Portfolio exposure cap (max total USD in open positions)
    """

    def __init__(
        self,
        client: HyperliquidClient,
        executor: HyperliquidVaultExecutor,
        regime_detector=None,
        liquidation_guard: Optional[LiquidationGuard] = None,
        hlp_gate: Optional[HLPSentimentGate] = None,
        risk_controller=None,
        funding_monitor=None,
        liquidation_tracker=None,
        # MoonDev profitability params
        max_global_trades_per_hour: int = 20,
        daily_loss_limit_pct: float = 2.0,
        max_portfolio_exposure_pct: float = 80.0,
    ):
        # T020: optional Hyperliquid native-WS candle feed (env-gated, default OFF).
        # Wraps the REST client so get_ohlcv is served from a live ws candle buffer with
        # REST candleSnapshot fallback (zero strategy/loop changes; a dead socket degrades
        # to today's REST behavior). Flip HL_WS_CANDLES=1 on Railway to activate + watch.
        if os.getenv("HL_WS_CANDLES", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                from src.lib.hyperliquid_ws_client import HyperliquidWsClient

                client = HyperliquidWsClient(client)
                logger.info("HL WS candle feed ENABLED (orchestrator client wrapped)")
            except Exception as e:  # noqa: BLE001 - never block startup on the ws upgrade
                logger.warning("HL WS candle feed disabled (wrap failed: %s); using REST client", e)
        self.client = client
        self.executor = executor
        self.regime_detector = regime_detector
        self.liquidation_guard = liquidation_guard
        self.hlp_gate = hlp_gate
        self._ob_gate = OrderBookImbalanceGate(client)
        self.risk_controller = risk_controller
        self.funding_monitor = funding_monitor
        self.liquidation_tracker = liquidation_tracker
        self._strategies: Dict[str, BaseStrategy] = {}
        self._strategy_types: Dict[str, str] = {}  # name -> strategy_type
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        # ── MoonDev Profitability Controls ──
        self.max_global_trades_per_hour = max_global_trades_per_hour
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct

        # Global trade tracking
        self._global_trades_this_hour: int = 0
        self._global_hour_start: datetime = datetime.now(timezone.utc)

        # Daily loss tracking
        self._daily_starting_balance: Optional[float] = None
        self._daily_loss_triggered: bool = False
        self._daily_loss_flattened: bool = False  # True once close_by_strategy called on halt
        self._day_start: Optional[datetime] = None

        # Regime gate stats
        self._regime_blocks: int = 0
        self._rate_limit_blocks: int = 0
        self._daily_loss_blocks: int = 0

        # Risk constants
        # max_portfolio_exposure_pct is already stored from __init__ arg (0-100 scale, e.g. 80.0)
        # BTC-delta cap: combined long notional must not exceed this multiple of equity
        self._btc_delta_cap_x: float = 1.5

        logger.info(
            "StrategyOrchestrator initialized | max_global_trades/hr=%d | daily_loss_limit=%.1f%% | max_exposure=%.0f%%",
            max_global_trades_per_hour, daily_loss_limit_pct, max_portfolio_exposure_pct,
        )

    def add_strategy(self, name: str, strategy_type: str, config: StrategyConfig) -> BaseStrategy:
        """Add and instantiate a strategy."""
        # Force config.name to match the orchestrator-level name so executor
        # position keys (f"{config.name}:{symbol}") match orchestrator lookups
        # (get_position(symbol, strategy_name=name)).
        config.name = name
        strategy = create_strategy(strategy_type, config)
        self._strategies[name] = strategy
        self._strategy_types[name] = strategy_type
        logger.info("Added strategy: %s (%s) on %s", name, strategy_type, config.symbol)
        return strategy

    def remove_strategy(self, name: str):
        """Remove a strategy (stops it first if running)."""
        if name in self._tasks:
            self._tasks[name].cancel()
            del self._tasks[name]
        if name in self._strategies:
            del self._strategies[name]
        self._strategy_types.pop(name, None)
        logger.info("Removed strategy: %s", name)

    def _compute_entry_adaptation(self, name: str, symbol: str, side: str) -> float:
        """Realtime adaptation multiplier for a new entry (ADR-0002). None-safe → 1.0."""
        if self.regime_detector is None:
            return 1.0
        regime_info = self.regime_detector.get_current_regime(symbol) or {}
        current_regime = regime_info.get("regime")
        favorable_types = set(regime_info.get("recommended_strategies") or [])
        atr_pct = None
        try:
            vol = self.regime_detector.get_volatility_status() or {}
            atr_pct = (vol.get(symbol) or {}).get("atr_pct")
        except Exception:
            atr_pct = None
        funding_bias = None
        if self.funding_monitor is not None:
            try:
                funding_bias = self.funding_monitor.get_funding_bias(symbol)
            except Exception:
                funding_bias = None
        strategy_type = self._strategy_types.get(name, "")
        return adaptation_multiplier(strategy_type, side, atr_pct, current_regime,
                                     favorable_types, funding_bias)

    def has_open_position(self, name: str) -> bool:
        """Best-effort sync check: does strategy `name` currently hold an open position?

        Reads the paper executor's in-memory `_positions` map. If the executor exposes
        no such map (e.g. a live executor whose positions live remotely), we cannot tell
        synchronously, so we answer conservatively (True) — callers use this only to
        decide whether to DEFER an exit-threshold change, and deferring is the safe side.
        """
        positions = getattr(self.executor, "_positions", None)
        if not isinstance(positions, dict):
            return True  # unknown structure → conservative: treat as open, defer tightening
        return any(getattr(p, "strategy_name", None) == name for p in positions.values())

    @staticmethod
    def _hard_stop_triggered(position, config) -> bool:
        """True iff the live position is at/beyond its configured ``max_loss_pct``.

        Execution-layer capital floor, enforced in the run loop independent of the
        strategy's own ``should_exit`` — backstops a strategy whose ``should_exit``
        raised (``run_iteration`` swallows the exception and emits no signal) or whose
        internal entry tracking desynced. Tolerant of a missing/None pnl or a non-dict
        position (answers False — never force-close on unknown PnL).
        """
        if not isinstance(position, dict):
            return False
        pnl_perc = position.get("pnl_perc")
        return pnl_perc is not None and pnl_perc <= config.max_loss_pct

    def update_live_params(self, name: str, fields: Dict, *, force: bool = False) -> Dict:
        """Apply config changes to a LIVE running strategy in-memory — no restart, no DB.

        This is the hot-apply primitive the compounder/boost paths already use inline;
        centralizing it lets the PATCH endpoint and the RBI promotion write-back actually
        reach the running process (closing the "DB-only write-back" gap).

        Field handling:
          - HOT_FORWARD_FIELDS (leverage/size_usd/lookback_days): applied now;
            forward-looking only, so safe even with an open position.
          - EXIT_THRESHOLD_FIELDS (target_pct/max_loss_pct): applied now only if the
            strategy is flat OR force=True; otherwise DEFERRED (would force-close an
            open position on the next bar via should_exit).
          - RESTART_REQUIRED_FIELDS (symbol/timeframe/interval_seconds/enabled): not
            honored by the running loop; reported (never claimed applied) so the caller
            can prompt a restart. enabled=false does NOT stop a running strategy —
            use POST /strategies/{name}/stop for that.
          - "params": shallow-merged into config.params (forward-looking for signals).

        Returns {"applied": [...], "deferred": [...], "restart_required": [...],
                 "running": bool}. A non-running strategy is a no-op with running=False.
        """
        strat = self.get_strategy(name)
        if strat is None:
            return {"applied": [], "deferred": [], "restart_required": [], "running": False}

        fields = fields or {}
        applied: List[str] = []
        deferred: List[str] = []
        restart_required: List[str] = []

        # Only pay for the open-position check when an exit-threshold change is requested.
        touches_exit = bool({k for k, v in fields.items() if v is not None} & EXIT_THRESHOLD_FIELDS)
        open_pos = self.has_open_position(name) if touches_exit else False

        for key, val in fields.items():
            if val is None:
                continue
            if key == "params" and isinstance(val, dict):
                strat.config.params.update(val)
                applied.append("params")
            elif key in RESTART_REQUIRED_FIELDS:
                restart_required.append(key)
            elif key in EXIT_THRESHOLD_FIELDS:
                if open_pos and not force:
                    deferred.append(key)
                else:
                    setattr(strat.config, key, val)
                    applied.append(key)
            elif key in HOT_FORWARD_FIELDS:
                setattr(strat.config, key, val)
                applied.append(key)
            # unknown keys are ignored (not live-applicable config attributes)

        if applied or deferred or restart_required:
            logger.info(
                "Live-tune %s | applied=%s deferred=%s restart_required=%s%s",
                name, applied, deferred, restart_required,
                " (forced)" if force else "",
            )
        return {"applied": applied, "deferred": deferred,
                "restart_required": restart_required, "running": True}

    async def start_strategy(self, name: str):
        """Start a single strategy's run loop."""
        if name not in self._strategies:
            raise ValueError(f"Strategy {name} not found")

        strategy = self._strategies[name]
        await strategy.on_start()

        # Capture starting balance on first strategy start
        if self._daily_starting_balance is None:
            self._daily_starting_balance = self._get_current_balance()
            self._day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            logger.info("Daily starting balance captured: $%.2f", self._daily_starting_balance)

        task = asyncio.create_task(self._run_strategy_loop(name, strategy))
        self._tasks[name] = task
        logger.info("Started strategy: %s", name)

    async def stop_strategy(self, name: str):
        """Stop a single strategy."""
        if name in self._tasks:
            self._tasks[name].cancel()
            try:
                await self._tasks[name]
            except asyncio.CancelledError:
                pass
            del self._tasks[name]

        if name in self._strategies:
            await self._strategies[name].on_stop()
            logger.info("Stopped strategy: %s", name)

    async def start_all(self):
        """Start all enabled strategies."""
        self._running = True
        for name, strategy in self._strategies.items():
            if strategy.config.enabled and name not in self._tasks:
                await self.start_strategy(name)
        logger.info("Started %d strategies", len(self._tasks))

    async def stop_all(self):
        """Stop all running strategies."""
        self._running = False
        names = list(self._tasks.keys())
        for name in names:
            await self.stop_strategy(name)
        logger.info("Stopped all strategies")

    def _get_current_balance(self) -> float:
        """Get current account balance from executor."""
        try:
            if hasattr(self.executor, 'balance'):
                return self.executor.balance
            if hasattr(self.executor, 'get_execution_stats'):
                stats = self.executor.get_execution_stats()
                return stats.get("current_balance", 10000.0)
        except Exception:
            pass
        return 10000.0

    def _get_total_open_notional(self) -> float:
        """Estimate total open notional (USD) across all active positions.

        For PaperExecutor: uses the `size_usd` field stored on each PaperPosition.
        For HyperliquidVaultExecutor: uses the per-strategy `config.size_usd` as a
        conservative proxy (actual fill may differ slightly, but this avoids an async
        account-state fetch inside the hot loop).
        Falls back to 0.0 on any error.
        """
        try:
            # PaperExecutor exposes _positions: Dict[str, PaperPosition]
            paper_positions = getattr(self.executor, '_positions', None)
            if isinstance(paper_positions, dict) and paper_positions:
                total = 0.0
                for pos in paper_positions.values():
                    # PaperPosition has size_usd; fall back to abs(size)*entry_price
                    sz_usd = getattr(pos, 'size_usd', None)
                    if sz_usd is not None and sz_usd > 0:
                        total += sz_usd
                    else:
                        ep = getattr(pos, 'entry_price', 0.0) or 0.0
                        sz = abs(getattr(pos, 'size', 0.0) or 0.0)
                        total += sz * ep
                return total

            # HyperliquidVaultExecutor exposes _active_positions: Dict[str, Dict]
            hl_positions = getattr(self.executor, '_active_positions', None)
            if isinstance(hl_positions, dict) and hl_positions:
                # Each entry doesn't store size_usd directly; use per-strategy
                # config.size_usd as a reasonable proxy.
                total = 0.0
                for pos_dict in hl_positions.values():
                    strat_name = pos_dict.get('strategy', '')
                    strat = self._strategies.get(strat_name)
                    if strat is not None:
                        total += float(strat.config.size_usd or 0.0)
                    else:
                        # Unknown strategy — can't estimate; treat as zero (fail-open)
                        pass
                return total
        except Exception as _exp_err:
            logger.debug("_get_total_open_notional error (defaulting 0): %s", _exp_err)
        return 0.0

    def _get_btc_long_notional(self) -> float:
        """Estimate total BTC long notional (USD) across all active positions.

        Same dual-executor strategy as _get_total_open_notional but filtered to
        BTC positions with side='long'.
        """
        try:
            paper_positions = getattr(self.executor, '_positions', None)
            if isinstance(paper_positions, dict) and paper_positions:
                total = 0.0
                for pos in paper_positions.values():
                    sym = getattr(pos, 'symbol', '') or ''
                    side = getattr(pos, 'side', '') or ''
                    if 'BTC' not in sym.upper() or side != 'long':
                        continue
                    sz_usd = getattr(pos, 'size_usd', None)
                    if sz_usd is not None and sz_usd > 0:
                        total += sz_usd
                    else:
                        ep = getattr(pos, 'entry_price', 0.0) or 0.0
                        sz = abs(getattr(pos, 'size', 0.0) or 0.0)
                        total += sz * ep
                return total

            hl_positions = getattr(self.executor, '_active_positions', None)
            if isinstance(hl_positions, dict) and hl_positions:
                total = 0.0
                for pos_dict in hl_positions.values():
                    sym = pos_dict.get('symbol', '') or ''
                    side = pos_dict.get('side', '') or ''
                    if 'BTC' not in sym.upper() or side != 'long':
                        continue
                    strat_name = pos_dict.get('strategy', '')
                    strat = self._strategies.get(strat_name)
                    if strat is not None:
                        total += float(strat.config.size_usd or 0.0)
                return total
        except Exception as _btc_err:
            logger.debug("_get_btc_long_notional error (defaulting 0): %s", _btc_err)
        return 0.0

    def _check_daily_loss(self) -> bool:
        """
        Check if daily loss limit has been breached.
        MoonDev: "Set a hard daily loss limit. When hit, close ALL positions
        and stop trading for 12 hours."
        Returns True if trading should be halted.
        """
        if self._daily_loss_triggered:
            return True

        if self._daily_starting_balance is None or self._daily_starting_balance <= 0:
            return False

        # Reset at midnight UTC
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self._day_start and today_start > self._day_start:
            self._day_start = today_start
            self._daily_starting_balance = self._get_current_balance()
            self._daily_loss_triggered = False
            self._daily_loss_flattened = False
            logger.info("Daily reset | new starting balance: $%.2f", self._daily_starting_balance)
            return False

        current_balance = self._get_current_balance()
        daily_pnl_pct = ((current_balance - self._daily_starting_balance) / self._daily_starting_balance) * 100

        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            self._daily_loss_triggered = True
            self._daily_loss_blocks += 1
            logger.critical(
                "DAILY LOSS LIMIT HIT | pnl=%.2f%% (limit=-%.1f%%) | balance=$%.2f → $%.2f | HALTING ALL STRATEGIES",
                daily_pnl_pct, self.daily_loss_limit_pct,
                self._daily_starting_balance, current_balance,
            )
            return True

        return False

    def _check_global_rate_limit(self) -> bool:
        """
        Check global trade rate limit across ALL strategies.
        MoonDev: "max 2-3 trades per hour TOTAL across all strategies"
        Returns True if rate limited (should NOT trade).
        """
        now = datetime.now(timezone.utc)

        # Reset hourly counter
        if (now - self._global_hour_start).total_seconds() >= 3600:
            self._global_trades_this_hour = 0
            self._global_hour_start = now

        if self._global_trades_this_hour >= self.max_global_trades_per_hour:
            self._rate_limit_blocks += 1
            return True

        return False

    def _record_global_trade(self):
        """Record a trade in the global counter."""
        self._global_trades_this_hour += 1

    def _check_regime_gate(self, name: str, symbol: str, data) -> bool:
        """
        Check if current regime allows this strategy to trade.
        MoonDev Video 71: "If regime = mismatch, do not trade."
        Returns True if regime ALLOWS trading, False if blocked.
        """
        if self.regime_detector is None:
            return True  # No detector = allow all

        strategy_type = self._strategy_types.get(name, "")

        # Update regime detection with latest data
        try:
            self.regime_detector.fit(symbol, data)
        except Exception as e:
            logger.debug("Regime fit failed for %s: %s", symbol, e)
            return True  # If regime detection fails, allow trading

        # Check if strategy should trade in current regime
        should_trade = self.regime_detector.should_trade(symbol, strategy_type)
        if not should_trade:
            self._regime_blocks += 1
            regime_info = self.regime_detector.get_current_regime(symbol)
            regime_name = regime_info.get("regime", "unknown") if regime_info else "unknown"
            logger.info(
                "REGIME GATE | %s (%s) blocked in %s regime | total blocks: %d",
                name, strategy_type, regime_name, self._regime_blocks,
            )

        return should_trade

    @staticmethod
    def _extract_atr(data, period: int = 14) -> Optional[float]:
        """
        Extract the ATR value from OHLCV data.

        If the dataframe already contains an 'atr' column (computed by a
        strategy), use it directly.  Otherwise compute a simple ATR on the
        fly from the high/low/close columns.
        """
        if data is None or data.empty:
            return None

        # Use pre-computed column if present
        if "atr" in data.columns:
            val = data["atr"].iloc[-1]
            if val and val > 0:
                return float(val)

        # Compute ATR from raw OHLCV
        try:
            high = data["high"]
            low = data["low"]
            close = data["close"]
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            true_range = tr1.combine(tr2, max).combine(tr3, max)
            atr = true_range.rolling(window=min(period, len(data))).mean().iloc[-1]
            return float(atr) if atr and atr > 0 else None
        except Exception:
            return None

    async def _run_strategy_loop(self, name: str, strategy: BaseStrategy):
        """
        Main loop for a single strategy.

        MoonDev profitability controls applied in order:
        1. Daily loss guard (hardest gate)
        2. Global trade rate limiter
        3. Regime detection gate
        4. Per-strategy anti-overtrading (in BaseStrategy.run_iteration)
        5. Signal strength filter (in BaseStrategy.run_iteration)
        """
        config = strategy.config
        symbol = config.symbol
        interval = config.timeframe
        sleep_seconds = config.interval_seconds

        logger.info(
            "Strategy loop started: %s | %s %s | interval=%ds",
            name, symbol, interval, sleep_seconds,
        )

        # Stagger startup to avoid simultaneous Hyperliquid candle fetches (429 burst)
        jitter = random.uniform(0.0, min(25.0, sleep_seconds * 0.4))
        if jitter > 0.5:
            logger.debug("Startup jitter for %s: %.1fs", name, jitter)
            await asyncio.sleep(jitter)

        while True:
            try:
                # ── Gate 1: Daily Loss Guard ──
                if self._check_daily_loss():
                    # On first trigger: flat ALL running strategies to honour the halt.
                    # Use a flag so we close exactly once per halt event (not once per
                    # strategy loop per tick, which would spam close_by_strategy).
                    if not self._daily_loss_flattened:
                        self._daily_loss_flattened = True
                        running_names = list(self._tasks.keys())
                        logger.critical(
                            "DAILY LOSS HALT: closing all positions for %d strategies: %s",
                            len(running_names), running_names,
                        )
                        for _strat_name in running_names:
                            try:
                                await self.executor.close_by_strategy(_strat_name)
                                logger.info(
                                    "DAILY LOSS HALT: closed positions for strategy %s",
                                    _strat_name,
                                )
                            except Exception as _halt_err:
                                logger.error(
                                    "DAILY LOSS HALT: close_by_strategy(%s) failed: %s",
                                    _strat_name, _halt_err,
                                )
                    logger.debug("Daily loss gate active | %s sleeping", name)
                    await asyncio.sleep(sleep_seconds * 4)  # Sleep longer when halted
                    continue

                # Fetch OHLCV data
                data = await asyncio.to_thread(
                    self.client.get_ohlcv,
                    symbol,
                    interval,
                    config.lookback_days,
                )

                if data.empty:
                    logger.warning("No data for %s %s, skipping", symbol, interval)
                    await asyncio.sleep(sleep_seconds)
                    continue

                # ── Gate 2: Regime Detection ──
                regime_allows = self._check_regime_gate(name, symbol, data)

                # Get current position (keyed per strategy for isolation)
                position = await self.executor.get_position(symbol, strategy_name=name)

                # ── Ruin Guard (reflex layer): force-close a position drifting into liquidation ──
                # hasattr guard is intentional: the live hl_executor doesn't implement the
                # per-position ruin guard yet (later phase), so this is paper-only for now.
                if position is not None and hasattr(self.executor, "check_position_ruin"):
                    # Buffer comes from RiskConfig when a risk controller is wired; otherwise
                    # check_position_ruin falls back to DEFAULT_RUIN_GUARD_BUFFER_PCT.
                    buffer_pct = (
                        self.risk_controller.config.ruin_guard_buffer_pct
                        if self.risk_controller is not None
                        and getattr(self.risk_controller, "config", None) is not None
                        else DEFAULT_RUIN_GUARD_BUFFER_PCT
                    )
                    # Reuse the mid get_position already fetched (avoids a 2nd price fetch).
                    known_mid = position.get("mark_price") if isinstance(position, dict) else None
                    should_close, dist_pct = await self.executor.check_position_ruin(
                        symbol, name, safety_buffer_pct=buffer_pct, mid=known_mid)
                    if should_close:
                        logger.critical(
                            "[RUIN-GUARD] %s %s within %.2f%% of liquidation — force-closing",
                            name, symbol, dist_pct)
                        await self.executor.close_by_strategy(name)
                        await asyncio.sleep(sleep_seconds)
                        continue

                # ── Hard stop (reflex layer): force-close at/beyond configured max_loss,
                # independent of strategy.should_exit. Backstops a should_exit that raised
                # (run_iteration swallows it) or a strategy whose entry tracking desynced
                # (e.g. after a restart), so an open position is always protected. ──
                if self._hard_stop_triggered(position, strategy.config):
                    logger.critical(
                        "[HARD-STOP] %s %s pnl=%.2f%% <= max_loss=%.2f%% — force-closing",
                        name, symbol, position.get("pnl_perc"), strategy.config.max_loss_pct)
                    await self.executor.close_by_strategy(name)
                    await asyncio.sleep(sleep_seconds)
                    continue

                # Run strategy iteration (includes per-strategy anti-overtrading)
                signal = await strategy.run_iteration(data, position)

                # Execute signal if present
                if signal and signal.signal_type.value != "none":

                    # For ENTRY signals, apply global gates
                    is_entry = signal.signal_type.value in ("long", "short")

                    if is_entry:
                        # ── Gate 2b: Regime must allow entries ──
                        if not regime_allows:
                            logger.debug(
                                "Entry blocked by regime | %s | %s",
                                name, signal.signal_type.value,
                            )
                            await asyncio.sleep(sleep_seconds)
                            continue

                        # ── Gate 3: Global Rate Limit ──
                        if self._check_global_rate_limit():
                            logger.debug(
                                "Entry blocked by global rate limit | %s | %d/%d trades this hour",
                                name, self._global_trades_this_hour,
                                self.max_global_trades_per_hour,
                            )
                            await asyncio.sleep(sleep_seconds)
                            continue

                        # ── Gate 3.5: RiskController ──
                        if self.risk_controller and hasattr(self.risk_controller, 'can_open_new_position'):
                            if not self.risk_controller.can_open_new_position():
                                logger.info("[Orchestrator] RiskController blocked entry for %s", signal.symbol)
                                await asyncio.sleep(sleep_seconds)
                                continue

                        # ── Gate 4: Liquidation Guard ──
                        if self.liquidation_guard:
                            is_long = signal.signal_type.value == "long"
                            entry_price = signal.price or (
                                float(data["close"].iloc[-1]) if not data.empty else 0.0
                            )
                            leverage = float(config.leverage)
                            atr_value = self._extract_atr(data)

                            safe, liq_price, dist, reason = self.liquidation_guard.is_entry_safe(
                                entry_price=entry_price,
                                leverage=leverage,
                                is_long=is_long,
                                symbol=symbol,
                                atr_value=atr_value,
                            )
                            if not safe:
                                logger.warning(
                                    "LIQUIDATION GUARD BLOCKED | %s | %s | liq_price=%.2f | dist=%.2f%% | %s",
                                    name, symbol, liq_price, dist, reason,
                                )
                                await asyncio.sleep(sleep_seconds)
                                continue

                        # ── Gate 4.3: Funding Guard (fresh ws activeAssetCtx) ──
                        # Net-tighten ONLY: hard-block entering INTO an extreme adverse-funding
                        # regime (crowded same-side — you'd pay funding AND join the crowd near a
                        # potential exhaustion). Reads fresh ws funding when the client exposes it
                        # (HL_WS_CANDLES on); fail-open and a no-op in REST mode or on missing data,
                        # so it can never block a normal qualified entry — only the extreme tail
                        # (> FUNDING_BLOCK_ANN annualized, default 100%) is removed.
                        _get_ctx = getattr(self.client, "get_asset_ctx", None)
                        if callable(_get_ctx):
                            try:
                                _ctx = _get_ctx(symbol)
                                _f8h = _ctx.get("funding") if _ctx else None
                                if _f8h is not None:
                                    _ann = float(_f8h) * 1095.0  # HL 8h funding -> annualized
                                    _block_ann = float(os.getenv("FUNDING_BLOCK_ANN", "1.0"))
                                    _is_long = signal.signal_type.value == "long"
                                    if (_is_long and _ann > _block_ann) or (not _is_long and _ann < -_block_ann):
                                        logger.info(
                                            "FUNDING GATE BLOCKED | %s | %s | funding=%+.0f%% ann (extreme %s-crowded) | OI=%s",
                                            name, signal.signal_type.value, _ann * 100.0,
                                            "long" if _ann > 0 else "short",
                                            _ctx.get("open_interest") if _ctx else None,
                                        )
                                        await asyncio.sleep(sleep_seconds)
                                        continue
                            except Exception as _fund_err:  # noqa: BLE001 - fail-open
                                logger.warning("Funding gate error (skipping): %s", _fund_err)

                        # ── Gate 4.4: Chaos Gate (volatility-spike proxy) ──
                        # MoonDev: skip entries during liquidation cascades. Proxy:
                        # current candle range > 3× 20-period avg = cascade signature.
                        # (No live liquidation event feed — OHLCV spike is always available.)
                        if len(data) >= 20:
                            try:
                                ranges = data["high"] - data["low"]
                                avg_range = float(ranges.iloc[:-1].tail(20).mean())
                                curr_range = float(ranges.iloc[-1])
                                if avg_range > 0 and curr_range > 3.0 * avg_range:
                                    logger.info(
                                        "CHAOS GATE BLOCKED | %s | %s | range=%.5f avg=%.5f (%.1fx spike)",
                                        name, signal.signal_type.value, curr_range, avg_range,
                                        curr_range / avg_range,
                                    )
                                    await asyncio.sleep(sleep_seconds)
                                    continue
                            except Exception as _chaos_err:
                                logger.warning("Chaos gate error (skipping): %s", _chaos_err)

                        # ── Gate 4.5: Funding-Cost Long Suppression ──
                        # Block new long entries when the 8h funding rate exceeds
                        # FUNDING_LONG_SUPPRESS_THRESHOLD (0.01%/8h). At that level
                        # longs bleed funding every 8 hours on top of taker fees.
                        # Shorts benefit from high positive funding — only longs suppressed.
                        # Fail-open: if the fetch fails, get_current_funding returns 0.0.
                        if signal.signal_type.value == "long" and self.funding_monitor is not None:
                            try:
                                _funding_rate = await self.funding_monitor.get_current_funding(symbol)
                                if _funding_rate > FUNDING_LONG_SUPPRESS_THRESHOLD:
                                    logger.warning(
                                        "[FUNDING-GATE] %s %s long suppressed: funding=%s/8h > threshold",
                                        name, symbol, f"{_funding_rate:.4%}",
                                    )
                                    await asyncio.sleep(sleep_seconds)
                                    continue
                            except Exception as _fg_err:  # noqa: BLE001 - fail-open
                                logger.warning("Funding-cost gate error (skipping): %s", _fg_err)

                        # ── Gate 4.6: LLM Advisory Gate ── (was 4.5 before funding gate)
                        try:
                            regime_info = (
                                self.regime_detector.get_current_regime(symbol)
                                if self.regime_detector else {}
                            )
                            regime_label = (regime_info or {}).get("regime", "unknown")
                            recent_pnl = [
                                round(t["pnl"], 2)
                                for t in self.executor.get_trade_history()
                                if t.get("action") == "exit"
                            ][-5:]
                            funding_rate = None
                            funding_bias = "neutral"
                            if self.funding_monitor:
                                funding_rate = self.funding_monitor.get_funding_rate(symbol)
                                funding_bias = self.funding_monitor.get_funding_bias(symbol)
                            gate_price = (
                                signal.price or
                                (float(data["close"].iloc[-1]) if not data.empty else 0.0)
                            )
                            gate_ctx = TradeContext(
                                strategy=name,
                                signal=signal.signal_type.value.upper(),
                                symbol=symbol,
                                price=gate_price,
                                regime=regime_label,
                                signal_strength=signal.strength,
                                recent_pnl=recent_pnl,
                                funding_rate=funding_rate,
                                funding_bias=funding_bias,
                            )
                            verdict = await llm_gate.evaluate(gate_ctx)
                            if not verdict.proceed:
                                logger.info(
                                    "LLM GATE BLOCKED | %s | %s | conf=%.2f | %s",
                                    name, signal.signal_type.value,
                                    verdict.confidence, verdict.reason,
                                )
                                await asyncio.sleep(sleep_seconds)
                                continue
                        except Exception as _llm_err:
                            logger.warning("LLM gate error (skipping gate): %s", _llm_err)

                        # ── Gate 4.7: HLP Sentiment Gate ──
                        # MoonDev Edge B: fade HLP when it has a concentrated losing position.
                        if is_entry and self.hlp_gate:
                            try:
                                hlp_blocked, hlp_reason = await self.hlp_gate.should_block(
                                    symbol, signal.signal_type.value == "long"
                                )
                                if hlp_blocked:
                                    logger.info(
                                        "HLP GATE BLOCKED | %s | %s | %s",
                                        name, signal.signal_type.value, hlp_reason,
                                    )
                                    await asyncio.sleep(sleep_seconds)
                                    continue
                            except Exception as _hlp_err:
                                logger.warning("HLP gate error (skipping): %s", _hlp_err)

                        # ── Gate 4.8: Order-Book Imbalance Gate ──
                        # Shadow mode (default): log what would be blocked, never skip entries.
                        # Flip OrderBookImbalanceGate.shadow_mode=False to activate blocking.
                        try:
                            _ob_side = signal.signal_type.value  # "long" or "short"
                            _ob_result = await self._ob_gate.check(symbol, _ob_side)
                            _would_block = "BLOCK" in _ob_result.reason
                            logger.info(
                                "[OB-GATE] %s %s %s imbalance=%.3f -> %s",
                                name, symbol, _ob_side,
                                _ob_result.imbalance_ratio,
                                "BLOCK" if _would_block else "ALLOW",
                            )
                            if not _ob_result.allowed:
                                # Shadow mode is off and the gate says block
                                await asyncio.sleep(sleep_seconds)
                                continue
                        except Exception as _ob_err:
                            logger.warning("OB gate error (skipping): %s", _ob_err)

                        # ── Gate 4.9: Portfolio Exposure Cap ──
                        # Skip entry if total open notional / equity already exceeds
                        # max_portfolio_exposure_pct (default 80.0 = 80%).
                        _equity = self._get_current_balance()
                        if _equity > 0:
                            _total_notional = self._get_total_open_notional()
                            _exposure_ratio = _total_notional / _equity
                            _exposure_limit = self.max_portfolio_exposure_pct / 100.0
                            if _exposure_ratio >= _exposure_limit:
                                logger.warning(
                                    "PORTFOLIO EXPOSURE CAP | %s | %s | notional=$%.0f equity=$%.0f"
                                    " ratio=%.1f%% >= limit=%.1f%% — entry skipped",
                                    name, signal.signal_type.value,
                                    _total_notional, _equity,
                                    _exposure_ratio * 100.0,
                                    self.max_portfolio_exposure_pct,
                                )
                                await asyncio.sleep(sleep_seconds)
                                continue

                        # ── Gate 4.10: Net BTC-Delta Cap ──
                        # For BTC long entries only: total BTC long notional must not exceed
                        # _btc_delta_cap_x × equity (default 1.5×).
                        _is_btc_long = (
                            signal.signal_type.value == "long"
                            and "BTC" in symbol.upper()
                        )
                        if _is_btc_long and _equity > 0:
                            _btc_long_notional = self._get_btc_long_notional()
                            _btc_cap = self._btc_delta_cap_x * _equity
                            if _btc_long_notional >= _btc_cap:
                                logger.warning(
                                    "BTC DELTA CAP | %s | BTC long notional=$%.0f >= cap=$%.0f"
                                    " (%.1fx equity) — entry skipped",
                                    name, _btc_long_notional, _btc_cap,
                                    self._btc_delta_cap_x,
                                )
                                await asyncio.sleep(sleep_seconds)
                                continue

                    # ── ADR-0002: attach realtime adaptation multiplier for entries ──
                    if is_entry:
                        signal.metadata["adaptation_multiplier"] = self._compute_entry_adaptation(
                            name, symbol, signal.signal_type.value)

                    # Execute the signal
                    result = await self.executor.execute_signal(signal, strategy)
                    if result.success:
                        if is_entry:
                            self._record_global_trade()
                        logger.info(
                            "Executed | %s | %s | %s | global_trades=%d/%d",
                            name, signal.signal_type.value, signal.reason,
                            self._global_trades_this_hour,
                            self.max_global_trades_per_hour,
                        )
                    else:
                        logger.warning(
                            "Execution failed | %s | %s | error=%s",
                            name, signal.signal_type.value, result.error,
                        )

                # ── Per-strategy circuit breaker check ──
                if strategy.state.circuit_breaker_triggered:
                    logger.critical(
                        "CIRCUIT BREAKER: %s auto-disabled | reason: %s",
                        name, strategy.state.circuit_breaker_reason,
                    )
                    break

                await asyncio.sleep(sleep_seconds)

            except asyncio.CancelledError:
                logger.info("Strategy loop cancelled: %s", name)
                break
            except Exception as e:
                logger.error("Strategy loop error | %s | %s", name, e)
                await strategy.on_error(e)

                if not strategy.is_healthy:
                    logger.critical(
                        "Strategy %s unhealthy (%d consecutive errors), stopping",
                        name, strategy.state.consecutive_errors,
                    )
                    break

                await asyncio.sleep(sleep_seconds)

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)

    def get_all_stats(self) -> List[Dict]:
        return [s.get_stats() for s in self._strategies.values()]

    def get_running_count(self) -> int:
        return len(self._tasks)

    def get_total_pnl(self) -> float:
        return sum(s.state.total_pnl for s in self._strategies.values())

    def get_total_trades(self) -> int:
        return sum(s.state.total_trades for s in self._strategies.values())

    def get_profitability_stats(self) -> Dict:
        """Get MoonDev profitability control stats."""
        return {
            "global_trades_this_hour": self._global_trades_this_hour,
            "max_global_trades_per_hour": self.max_global_trades_per_hour,
            "daily_starting_balance": self._daily_starting_balance,
            "current_balance": self._get_current_balance(),
            "daily_pnl_pct": (
                ((self._get_current_balance() - self._daily_starting_balance)
                 / self._daily_starting_balance * 100)
                if self._daily_starting_balance and self._daily_starting_balance > 0
                else 0.0
            ),
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "daily_loss_triggered": self._daily_loss_triggered,
            "regime_blocks": self._regime_blocks,
            "rate_limit_blocks": self._rate_limit_blocks,
            "daily_loss_blocks": self._daily_loss_blocks,
            "regime_detector_active": self.regime_detector is not None,
            "liquidation_guard_active": self.liquidation_guard is not None,
            "liquidation_guard_blocks": (
                self.liquidation_guard.total_blocks if self.liquidation_guard else 0
            ),
        }
