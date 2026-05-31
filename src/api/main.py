import asyncio
import json
import os
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
from src.engine.rbi_schedule import build_rbi_job_specs
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

logger = logging.getLogger(__name__)


def _auto_deploy_winners(db, orchestrator):
    """Deploy default winner strategies into an empty DB so they auto-start."""
    winners = [
        # ── Original 8 MoonDev winners ──
        {"name": "mm-eth", "strategy_type": "market_maker", "symbol": "ETH", "timeframe": "5m", "size_usd": 100, "leverage": 3, "params": {"num_bars": 180, "quartile": 0.33, "max_l2h": 0.05, "max_tr_pct": 0.02, "exit_pct": 0.010, "mm_stop_pct": 0.006, "time_limit_minutes": 480, "last_n_bars": 17, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        {"name": "mm-sol", "strategy_type": "market_maker", "symbol": "SOL", "timeframe": "5m", "size_usd": 100, "leverage": 3, "params": {"num_bars": 180, "quartile": 0.33, "max_l2h": 0.06, "max_tr_pct": 0.025, "exit_pct": 0.010, "mm_stop_pct": 0.006, "time_limit_minutes": 360, "last_n_bars": 17, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        {"name": "arb-eth", "strategy_type": "funding_arb", "symbol": "ETH", "timeframe": "1h", "size_usd": 150, "leverage": 3, "params": {"momentum_threshold": 0.015, "combined_target_pct": 0.8, "arb_max_loss_pct": -1.5, "min_hold_bars": 3, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.4}},
        {"name": "nw-eth", "strategy_type": "nadaraya_watson", "symbol": "ETH", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"kernel_bandwidth": 8.0, "kernel_lookback": 100, "overbought": 80, "oversold": 20, "adx_threshold": 35, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "nw-sol", "strategy_type": "nadaraya_watson", "symbol": "SOL", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"kernel_bandwidth": 8.0, "kernel_lookback": 100, "overbought": 80, "oversold": 20, "adx_threshold": 35, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "adx-eth", "strategy_type": "adx", "symbol": "ETH", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"adx_period": 14, "adx_threshold": 20, "exit_threshold": 15, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "vwap-btc", "strategy_type": "vwap_bot", "symbol": "BTC", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"vwap_bias_long": 0.7, "vwap_bias_short": 0.3, "min_vwap_distance": 0.0008, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        {"name": "mean-rev-eth", "strategy_type": "mean_reversion", "symbol": "ETH", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"zscore_entry": 1.5, "zscore_exit": 0.3, "reversion_target_pct": 0.008, "bb_std": 2.2, "dynamic_sizing": True, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        # ── Full fleet: breakout strategies ──
        {"name": "turtle-btc", "strategy_type": "turtle", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"lookback_period": 55, "atr_period": 20, "atr_multiplier": 3.0, "take_profit_pct": 0.075, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "bollinger-btc", "strategy_type": "bollinger", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"bb_period": 20, "bb_std": 2.0, "squeeze_threshold": 0.05, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "conspop-btc", "strategy_type": "consolidation_pop", "symbol": "BTC", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"atr_period": 14, "deviance_threshold": 0.45, "range_position_buy": 0.3, "range_position_sell": 0.7, "tp_pct": 0.02, "sl_pct": 0.015, "min_hold_bars": 3, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        # quarter-btc removed 2026-05-15: quarter_theory deliberately
        # disabled in src/strategies/registry.py (buggy TP/SL in crypto).
        # ── Full fleet: reversal strategies ──
        {"name": "sdz-btc", "strategy_type": "supply_demand_zone", "symbol": "BTC", "timeframe": "4h", "size_usd": 100, "leverage": 3, "interval_seconds": 60, "lookback_days": 30, "params": {"zone_lookback_days": 30, "zone_threshold": 0.015, "min_hold_bars": 2, "cooldown_seconds": 900, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "rsi-btc", "strategy_type": "rsi", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"rsi_period": 14, "oversold": 30, "overbought": 70, "trend_mode": False, "divergence_mode": False, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "pivot-btc", "strategy_type": "pivot_lines", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"pivot_lookback": 24, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "rsivwap-btc", "strategy_type": "rsi_vwap", "symbol": "BTC", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"rsi_period": 14, "oversold": 30, "overbought": 70, "min_hold_bars": 3, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        # ── Full fleet: trend strategies ──
        {"name": "sma-btc", "strategy_type": "sma_crossover", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"sma_period": 20, "support_lookback": 20, "adx_threshold": 15, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "vwma-btc", "strategy_type": "vwma", "symbol": "BTC", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"fast_period": 20, "mid_period": 41, "slow_period": 75, "min_hold_bars": 3, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        {"name": "macd-btc", "strategy_type": "macd", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9, "ma_filter_period": 50, "histogram_mode": True, "zero_cross_mode": True, "confirmation_bars": 1, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "ichimoku-btc", "strategy_type": "ichimoku", "symbol": "BTC", "timeframe": "4h", "size_usd": 100, "leverage": 3, "interval_seconds": 60, "params": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52, "min_hold_bars": 2, "cooldown_seconds": 900, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "emabb-btc", "strategy_type": "ema_bollinger", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"short_ema_period": 20, "long_ema_period": 50, "bb_period": 20, "bb_std": 2.0, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        {"name": "combo-btc", "strategy_type": "sma_adx_bb_vol", "symbol": "BTC", "timeframe": "1h", "size_usd": 100, "leverage": 3, "params": {"sma_period": 20, "adx_period": 14, "bb_period": 20, "bb_std": 2.0, "min_adx": 25, "volume_multiplier": 1.1, "min_hold_bars": 3, "cooldown_seconds": 600, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
        # ── Full fleet: statistical/pattern strategies ──
        {"name": "corr-sol", "strategy_type": "correlation", "symbol": "SOL", "timeframe": "15m", "size_usd": 100, "leverage": 3, "params": {"leader": "ETH", "correlation_window": 20, "lag_threshold": 0.003, "sl_pct": 0.005, "tp_pct": 0.015, "momentum_threshold": 0.015, "min_hold_bars": 3, "cooldown_seconds": 300, "max_trades_per_hour": 3, "min_signal_strength": 0.5}},
        # elliott-btc, ellpiv-btc removed 2026-05-15: elliott_wave and
        # elliott_pivot are deliberately disabled in
        # src/strategies/registry.py (unreliable in crypto). Re-add here
        # only when the registry re-enables them.
        {"name": "gridfib-btc", "strategy_type": "grid_fibonacci", "symbol": "BTC", "timeframe": "4h", "size_usd": 100, "leverage": 3, "interval_seconds": 60, "params": {"fib_lookback": 50, "proximity_pct": 0.8, "trend_period": 20, "take_profit_fib": 0.618, "stop_loss_fib": 1.0, "min_hold_bars": 2, "cooldown_seconds": 900, "max_trades_per_hour": 2, "min_signal_strength": 0.5}},
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
                size_usd=w.get("size_usd", 100), target_pct=9.0, max_loss_pct=-8.0,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
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
    app.state.solana_scanner = solana_scanner
    app.state.funding_monitor = funding_monitor

    # ── Restore paper trading state from disk ──
    if paper_mode and executor is not None:
        try:
            executor.load_state()
        except Exception as e:
            logger.warning("Paper state restore failed: %s", e)

    # ── Auto-start strategies that were running before shutdown ──
    if orchestrator is not None:
        try:
            from .database import SessionLocal
            from src.strategies.base_strategy import StrategyConfig, StrategyTier
            from src.strategies.registry import list_strategies

            db = SessionLocal()
            try:
                running_instances = db.query(models.StrategyInstance).filter(
                    models.StrategyInstance.status == "running"
                ).all()

                # If DB is empty (fresh deploy), auto-deploy winner strategies
                all_instances = db.query(models.StrategyInstance).count()
                if all_instances == 0:
                    logger.info("Auto-deploy: empty DB detected — deploying winner strategies")
                    _auto_deploy_winners(db, orchestrator)
                    running_instances = db.query(models.StrategyInstance).filter(
                        models.StrategyInstance.status == "running"
                    ).all()

                if running_instances:
                    logger.info("Auto-start: found %d strategies with status=running", len(running_instances))
                    tier_map = {"A": StrategyTier.A, "B": StrategyTier.B, "C": StrategyTier.C, "D": StrategyTier.D}
                    available = [s["strategy_type"] for s in list_strategies()]

                    for inst in running_instances:
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
                            logger.info("Auto-start: started %s (%s on %s)", inst.name, inst.strategy_type, inst.symbol)
                        except Exception as e:
                            logger.warning("Auto-start: failed to start %s — %s", inst.name, e)
                            inst.status = "error"
                            inst.error_message = f"Auto-start failed: {e}"
                            db.commit()
                else:
                    logger.info("Auto-start: no strategies with status=running")
            finally:
                db.close()
        except Exception as e:
            logger.warning("Auto-start: could not restore strategies — %s", e)

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

    # ── Winner High-Frequency + Leverage Boost (applied each startup) ──
    # Proven live: vwap +$3.12 | rsi +$3.09 | macd +$1.89 | pivot +$9.75
    # ETH/SOL live: adx-eth +$15.15/trade | macd-sol +$8.47/trade
    # MoonDev backtest: liqadx 494% Sharpe 2.81 | flip-flop 529% WR 81%
    _BTC_WINNERS = {
        "vwap-btc":     {"leverage": 3, "size_usd": 100, "cooldown_seconds": 120, "max_trades_per_hour": 6},
        "rsi-btc":      {"leverage": 3, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 4},
        "macd-btc":     {"leverage": 3, "size_usd": 100, "cooldown_seconds": 240, "max_trades_per_hour": 4},
        "pivot-btc":    {"leverage": 3, "size_usd": 100, "cooldown_seconds": 180, "max_trades_per_hour": 5},
        "adx-eth":      {"leverage": 3, "size_usd": 100, "cooldown_seconds": 180, "max_trades_per_hour": 5},
        "macd-sol":     {"leverage": 3, "size_usd": 100, "cooldown_seconds": 240, "max_trades_per_hour": 4},
        "turtle-btc":   {"leverage": 3, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 3},
        "liqadx-btc":   {"leverage": 4, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 3},
        "liqadx-eth":   {"leverage": 4, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 3},
        "flip-flop-btc":{"leverage": 4, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 3},
        "flip-flop-eth":{"leverage": 3, "size_usd": 100, "cooldown_seconds": 300, "max_trades_per_hour": 3},
    }
    try:
        from .database import SessionLocal as _BoostSL
        from sqlalchemy.orm.attributes import flag_modified as _flag_modified
        _boost_db = _BoostSL()
        try:
            for _wname, _wcfg in _BTC_WINNERS.items():
                _winst = _boost_db.query(models.StrategyInstance).filter(
                    models.StrategyInstance.name == _wname
                ).first()
                if _winst:
                    _winst.leverage = _wcfg["leverage"]
                    _winst.size_usd = _wcfg["size_usd"]
                    _winst.params = {
                        **(_winst.params or {}),
                        "cooldown_seconds": _wcfg["cooldown_seconds"],
                        "max_trades_per_hour": _wcfg["max_trades_per_hour"],
                    }
                    _flag_modified(_winst, "params")
                    if orchestrator:
                        _wstrat = orchestrator.get_strategy(_wname)
                        if _wstrat:
                            _wstrat.config.leverage = _wcfg["leverage"]
                            _wstrat.config.size_usd = _wcfg["size_usd"]
                            _wstrat.config.params["cooldown_seconds"] = _wcfg["cooldown_seconds"]
                            _wstrat.config.params["max_trades_per_hour"] = _wcfg["max_trades_per_hour"]
                    logger.info(
                        "Winner boost: %s | size=$%d | lev=%dx | cooldown=%ds | max_hr=%d",
                        _wname, _wcfg["size_usd"], _wcfg["leverage"],
                        _wcfg["cooldown_seconds"], _wcfg["max_trades_per_hour"],
                    )
            _boost_db.commit()
        finally:
            _boost_db.close()
    except Exception as _boost_err:
        logger.warning("Winner boost failed: %s", _boost_err)

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

    async def _rbi_job(strategy_type: str, strategy_id: int, symbol: str, timeframe: str):
        pipeline = _get_or_create_pipeline(strategy_type)
        try:
            event = await pipeline.run_cycle(
                strategy_type=strategy_type, strategy_id=strategy_id,
                symbol=symbol, timeframe=timeframe, lookback_days=90, n_trials=100,
            )
            if event.promoted:
                logger.info("Scheduler RBI promoted %s: %s", strategy_type, event.after_metrics)
        except Exception as e:
            logger.error("Scheduled RBI cycle failed for %s: %s", strategy_type, e)

    # Build DB-derived schedule, filtered to optimizer-supported strategy types.
    # Falls back to the hardcoded _RBI_SCHEDULE if no running instances exist.
    _supported_types: set[str] = set(PARAM_SPACES.keys())
    from .database import SessionLocal as _RBI_SL
    try:
        _rbi_db = _RBI_SL()
        try:
            _running_instances = _rbi_db.query(models.StrategyInstance).filter(
                models.StrategyInstance.status == "running"
            ).all()
        finally:
            _rbi_db.close()
    except Exception as _dbe:
        logger.warning("RBI scheduler: DB query failed (%s); using hardcoded schedule", _dbe)
        _running_instances = []

    if _running_instances:
        _job_specs = build_rbi_job_specs(_running_instances, _supported_types)
        _skipped = [i.strategy_type for i in _running_instances if i.strategy_type not in _supported_types]
        if _skipped:
            logger.warning("RBI scheduler: skipping unsupported strategy types (not in param_spaces): %s", sorted(set(_skipped)))
        for _spec in _job_specs:
            _scheduler.add_job(
                _rbi_job, "interval", hours=_spec["hours"],
                args=[_spec["strategy_type"], _spec["strategy_id"], _spec["symbol"], _spec["timeframe"]],
                id=f"rbi_{_spec['strategy_type']}_{_spec['strategy_id']}",
                replace_existing=True,
            )
        logger.info("RBI scheduler: %d DB-derived jobs scheduled (supported types: %s)", len(_job_specs), sorted(_supported_types))
    else:
        # Fallback: no running instances — use hardcoded schedule so behaviour
        # doesn't regress on an empty DB.
        _fallback_skipped = [stype for stype, *_ in _RBI_SCHEDULE if stype not in _supported_types]
        if _fallback_skipped:
            logger.warning("RBI scheduler (fallback): skipping unsupported types: %s", sorted(set(_fallback_skipped)))
        for stype, sid, sym, tf, hours in _RBI_SCHEDULE:
            if stype not in _supported_types:
                continue
            _scheduler.add_job(
                _rbi_job, "interval", hours=hours,
                args=[stype, sid, sym, tf],
                id=f"rbi_{stype}",
                replace_existing=True,
            )
        logger.info(
            "RBI scheduler: empty DB — fell back to %d hardcoded jobs",
            len([s for s, *_ in _RBI_SCHEDULE if s in _supported_types]),
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
                            _wr, _payoff, _n = executor._live_edge_stats(_cinst.name)
                            _cinst.edge_confidence_score = edge_confidence(_n, _wr, _payoff)

                # Dynamic winner detection from live stats
                _STATIC_WINNERS = {
                    'vwap-btc', 'rsi-btc', 'pivot-btc', 'turtle-btc',
                    'adx-eth', 'macd-btc', 'macd-sol',
                    'liqadx-btc', 'liqadx-eth',       # MoonDev: 494% return, Sharpe 2.81
                    'flip-flop-btc', 'flip-flop-eth',  # MoonDev: 529% return, WR 81%
                }
                _WINNER_MIN_TRADES = 1
                dynamic_winners = {
                    r.name for r in running
                    if _live_stats.get(r.name, {}).get("pnl", 0.0) > 0
                    and _live_stats.get(r.name, {}).get("trades", 0) >= _WINNER_MIN_TRADES
                }
                _WINNER_SET = dynamic_winners if dynamic_winners else _STATIC_WINNERS

                # Winners-first: concentrate investable on proven winners,
                # non-winners stay at $100 for signal discovery.
                _winners_running = [r for r in running if r.name in _WINNER_SET]
                _winner_unit = min(
                    round(investable / max(len(_winners_running), 1), 2), 3000.0
                ) if _winners_running else 100.0
                _OTHER_SIZE = 100.0

                for _cinst in running:
                    new_size = _winner_unit if _cinst.name in _WINNER_SET else _OTHER_SIZE
                    _cinst.size_usd = new_size
                    if orchestrator:
                        _cstrat = orchestrator.get_strategy(_cinst.name)
                        if _cstrat:
                            _cstrat.config.size_usd = new_size
                _cdb.commit()
                winner_size = _winner_unit
                other_size = _OTHER_SIZE
                logger.info(
                    "Compounder: balance=$%.2f | investable=$%.2f | winners=$%.2f | others=$%.2f | n_winners=%d/%d | winner_set=%s",
                    balance, investable, winner_size, other_size, len(_winners_running), n, sorted(_WINNER_SET),
                )
            finally:
                _cdb.close()
        except Exception as _ce:
            logger.error("Compound job error: %s", _ce)

    from datetime import datetime as _dt
    _scheduler.add_job(
        _compound_job, "interval", minutes=15,
        id="compounder",
        replace_existing=True,
        next_run_time=_dt.now(),
    )

    _scheduler.start()
    app.state.rbi_scheduler = _scheduler
    app.state.compound_initial_balance = _INITIAL_BALANCE
    app.state.compound_reserve_pct = _COMPOUND_RESERVE_PCT
    logger.info("RBI scheduler started with %d jobs + compounder", len(_scheduler.get_jobs()) - 1)

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
        agg_pnl = stats.get("total_realized_pnl", agg_pnl)
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


@app.get("/strategies/leaderboard")
async def strategy_leaderboard(request: Request):
    """Get strategies ranked by PnL with profitability metrics."""
    orchestrator = request.app.state.orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    db = next(get_db())
    try:
        strategies = db.query(models.StrategyInstance).all()

        leaderboard = []
        for s in strategies:
            win_rate = (s.winning_trades / s.total_trades * 100) if s.total_trades > 0 else 0
            leaderboard.append({
                "name": s.name,
                "strategy_type": s.strategy_type,
                "status": s.status,
                "total_pnl": round(s.total_pnl, 2),
                "total_trades": s.total_trades,
                "winning_trades": s.winning_trades,
                "losing_trades": s.losing_trades,
                "win_rate": round(win_rate, 1),
                "profitable": s.total_pnl > 0,
                "avg_pnl_per_trade": round(s.total_pnl / s.total_trades, 2) if s.total_trades > 0 else 0,
            })

        leaderboard.sort(key=lambda x: x["total_pnl"], reverse=True)

        profitable = [s for s in leaderboard if s["profitable"]]
        losing = [s for s in leaderboard if not s["profitable"] and s["total_trades"] > 0]
        inactive = [s for s in leaderboard if s["total_trades"] == 0]

        return {
            "leaderboard": leaderboard,
            "summary": {
                "total_strategies": len(leaderboard),
                "profitable_count": len(profitable),
                "losing_count": len(losing),
                "inactive_count": len(inactive),
                "total_pnl": round(sum(s["total_pnl"] for s in leaderboard), 2),
                "profitable_pnl": round(sum(s["total_pnl"] for s in profitable), 2),
                "losing_pnl": round(sum(s["total_pnl"] for s in losing), 2),
            },
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

    # Reset circuit breaker state
    if hasattr(strategy.state, 'circuit_breaker_triggered'):
        strategy.state.circuit_breaker_triggered = False
        strategy.state.circuit_breaker_reason = ""
        strategy.state.consecutive_losses = 0

    return {
        "name": name,
        "circuit_breaker_was_triggered": was_triggered,
        "circuit_breaker_reset": True,
        "message": f"Circuit breaker reset for {name}. Use /strategies/{name}/start to restart.",
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

    # Also reset strategy-level anti-overtrading state
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator and hasattr(orchestrator, "strategies"):
        for strategy in orchestrator.strategies.values():
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
            total_pnl = stats.get("total_realized_pnl", 0.0)
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
