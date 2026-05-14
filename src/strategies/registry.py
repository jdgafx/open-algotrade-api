"""
Strategy Registry — Maps strategy_type strings to BaseStrategy subclasses.

Every strategy that ships with Open Algotrade is registered here.
The orchestrator and API use this to discover, instantiate, and manage strategies.
"""

import logging
from typing import Dict, List, Type

from .base_strategy import BaseStrategy, StrategyConfig, StrategyTier

logger = logging.getLogger(__name__)

# Registry: strategy_type -> (class, tier, description, default_symbol, default_timeframe, default_params)
_REGISTRY: Dict[str, dict] = {}


def register_strategy(
    strategy_type: str,
    cls: Type[BaseStrategy],
    tier: StrategyTier,
    description: str,
    default_symbol: str = "BTC",
    default_timeframe: str = "1h",
    default_params: dict = None,
    category: str = "trend",
    risk_level: str = "medium",
):
    _REGISTRY[strategy_type] = {
        "cls": cls,
        "tier": tier,
        "description": description,
        "default_symbol": default_symbol,
        "default_timeframe": default_timeframe,
        "default_params": default_params or {},
        "category": category,
        "risk_level": risk_level,
    }
    logger.debug("Registered strategy: %s (%s)", strategy_type, tier.value)


def get_strategy_class(strategy_type: str) -> Type[BaseStrategy]:
    if strategy_type not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[strategy_type]["cls"]


def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
    cls = get_strategy_class(strategy_type)
    return cls(config)


def list_strategies() -> List[dict]:
    result = []
    for stype, info in _REGISTRY.items():
        result.append({
            "strategy_type": stype,
            "tier": info["tier"].value,
            "description": info["description"],
            "default_symbol": info["default_symbol"],
            "default_timeframe": info["default_timeframe"],
            "default_params": info["default_params"],
            "category": info.get("category", "trend"),
            "risk_level": info.get("risk_level", "medium"),
        })
    return result


def _register_all():
    """Import and register all built-in strategies."""
    from .turtle_hl_strategy import TurtleHLStrategy
    from .bollinger_strategy import BollingerStrategy
    from .sdz_strategy import SupplyDemandZoneStrategy
    from .vwap_bot_strategy import VWAPBotStrategy
    from .arb_strategy import FundingArbStrategy
    from .correlation_strategy import CorrelationStrategy
    from .consolidation_pop_strategy import ConsolidationPopStrategy
    from .nadaraya_watson_strategy import NadarayaWatsonStrategy
    from .market_maker_strategy import MarketMakerStrategy
    from .mean_reversion_strategy import MeanReversionStrategy
    from .sma_strategy import SMAStrategy
    from .rsi_strategy import RSIStrategy
    from .vwma_strategy import VWMAStrategy
    from .adx_strategy import ADXStrategy
    from .macd_strategy import MACDStrategy
    from .ichimoku_strategy import IchimokuStrategy
    # DISABLED: unreliable Elliott Wave pattern detection in crypto
    # from .elliott_wave_strategy import ElliottWaveStrategy
    from .pivot_lines_strategy import PivotLinesStrategy
    # DISABLED: buggy TP/SL behavior with quarter levels in crypto
    # from .quarter_theory_strategy import QuarterTheoryStrategy
    from .ema_bollinger_strategy import EMABollingerStrategy
    from .grid_fib_strategy import GridFibStrategy
    # DISABLED: unreliable Elliott Wave + pivot confluence in crypto
    # from .elliott_pivot_strategy import ElliottPivotStrategy
    from .sma_adx_bb_vol_strategy import SMAAdxBBVolStrategy
    from .rsi_vwap_strategy import RSIVWAPStrategy
    from .solana_sniper_strategy import SolanaSniperStrategy
    from .liquidation_dip_strategy import LiquidationDipStrategy
    from .closed_market_overnight_strategy import ClosedMarketOvernightStrategy

    # ═══════════════════════════════════════════════════════════════════
    # TIER A — Battle-tested HL native strategies
    # ═══════════════════════════════════════════════════════════════════

    # ── MoonDev profitability tuning applied ──
    # Key changes: longer cooldowns, fewer trades/hour, higher signal thresholds,
    # trailing stops, and regime-appropriate parameters.
    # Reference: MoonDev Videos 49, 53, 71, 74, 104, 115, 164, 191

    register_strategy(
        "turtle", TurtleHLStrategy, StrategyTier.A,
        "55-bar breakout with ATR trailing stops and take profit",
        "BTC", "1h",
        {
            "lookback_period": 40, "atr_period": 14, "atr_multiplier": 2.5,
            "take_profit_pct": 0.015,
            "min_hold_bars": 5, "cooldown_seconds": 600, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="breakout", risk_level="medium",
    )
    register_strategy(
        "bollinger", BollingerStrategy, StrategyTier.A,
        "MoonDev BB Squeeze + Keltner Channel + ADX>25 breakout (Harvard repo)",
        "BTC", "6h",
        {
            "bb_period": 20, "bb_std": 2.0,
            "keltner_period": 20, "keltner_atr_mult": 1.5,
            "adx_period": 14, "adx_threshold": 25,
            "min_hold_bars": 4, "cooldown_seconds": 180, "max_trades_per_hour": 3,
            "min_signal_strength": 0.65,
        },
        category="breakout", risk_level="medium",
    )
    register_strategy(
        "supply_demand_zone", SupplyDemandZoneStrategy, StrategyTier.A,
        "Supply/Demand zone detection with reversal entries",
        "BTC", "4h",
        {
            "zone_lookback_days": 21, "zone_threshold": 0.015,
            "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
            "min_signal_strength": 0.65,
        },
        category="reversal", risk_level="medium",
    )
    register_strategy(
        "vwap_bot", VWAPBotStrategy, StrategyTier.A,
        "VWAP-based probability bias trading (70/30 above/below)",
        "BTC", "15m",
        {
            "vwap_bias_long": 0.65, "vwap_bias_short": 0.35,
            "min_vwap_distance": 0.0008,
            "min_hold_bars": 4, "cooldown_seconds": 180, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="low",
    )

    register_strategy(
        "closed_market_overnight", ClosedMarketOvernightStrategy, StrategyTier.A,
        "MoonDev v.10 — trade only when NYSE is closed (overnight + weekends); breakout entry, force-close at market open",
        "BTC", "1h",
        {
            "momentum_lookback": 12, "breakout_pct": 0.002,
            "tp_pct": 0.010, "sl_pct": 0.008,
            "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
            "min_signal_strength": 0.65,
        },
        category="overnight", risk_level="low",
    )
    register_strategy(
        "funding_arb", FundingArbStrategy, StrategyTier.A,
        "Funding rate arbitrage between correlated assets (BTC/ETH)",
        "BTC", "1h",
        {
            "symbol_a": "BTC", "symbol_b": "ETH",
            "funding_threshold": 0.0008,
            "combined_target_pct": 2.0,
            "momentum_threshold": 0.015,
            "min_hold_bars": 8, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="arbitrage", risk_level="low",
    )

    # ═══════════════════════════════════════════════════════════════════
    # TIER B — Bonus algo strategies
    # ═══════════════════════════════════════════════════════════════════

    register_strategy(
        "correlation", CorrelationStrategy, StrategyTier.B,
        "Leader/follower correlation trading (ETH leads altcoins)",
        "SOL", "15m",
        {
            "leader": "ETH", "correlation_window": 30,
            "lag_threshold": 0.003,
            "sl_pct": 0.004, "tp_pct": 0.005,
            "momentum_threshold": 0.01,
            "min_hold_bars": 3, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="statistical", risk_level="medium",
    )
    register_strategy(
        "consolidation_pop", ConsolidationPopStrategy, StrategyTier.B,
        "Consolidation detection via ATR deviance, range breakout",
        "BTC", "15m",
        {
            "atr_period": 14, "deviance_threshold": 0.35,
            "range_position_buy": 0.25, "range_position_sell": 0.75,
            "tp_pct": 0.008, "sl_pct": 0.005,
            "min_hold_bars": 3, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="breakout", risk_level="medium",
    )
    register_strategy(
        "nadaraya_watson", NadarayaWatsonStrategy, StrategyTier.B,
        "Kernel regression envelope + Stochastic RSI signals",
        "BTC", "15m",
        {
            "kernel_bandwidth": 12.0, "kernel_lookback": 80,
            "stoch_period": 14, "stoch_k": 3, "stoch_d": 3,
            "overbought": 85, "oversold": 15,
            "min_hold_bars": 3, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="statistical", risk_level="medium",
    )
    register_strategy(
        "market_maker", MarketMakerStrategy, StrategyTier.B,
        "Spread-based market making with kill switch and ATR no-trade zones",
        "BTC", "1m",
        {
            "spread": 0.0015,
            "max_position_usd": 500.0, "kill_size_usd": 1000.0,
            "atr_period": 7, "refresh_seconds": 5,
            "exit_pct": 0.010, "mm_stop_pct": 0.006,
            "min_hold_bars": 2, "cooldown_seconds": 30, "max_trades_per_hour": 20,
            "min_signal_strength": 0.50,
        },
        category="market_making", risk_level="high",
    )
    register_strategy(
        "mean_reversion", MeanReversionStrategy, StrategyTier.B,
        "Multi-timeframe SMA mean reversion with trend filter",
        "ETH", "15m",
        {
            "sma_trend_period": 20, "sma_entry_period": 20,
            "trend_timeframe": "4h", "entry_timeframe": "15m",
            "reversion_target_pct": 0.006,
            "zscore_entry": 2.0, "zscore_exit": 0.3,
            "bb_period": 20, "bb_std": 2.2,
            "dynamic_sizing": True, "single_timeframe": True,
            "min_hold_bars": 3, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="mean_reversion", risk_level="medium",
    )

    # ═══════════════════════════════════════════════════════════════════
    # TIER C — Bootcamp bot strategies
    # ═══════════════════════════════════════════════════════════════════

    register_strategy(
        "sma_crossover", SMAStrategy, StrategyTier.C,
        "SMA crossover with support/resistance levels",
        "BTC", "1h",
        {
            "sma_period": 21, "support_lookback": 30,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="low",
    )
    register_strategy(
        "rsi", RSIStrategy, StrategyTier.C,
        "RSI overbought/oversold reversal strategy",
        "BTC", "1h",
        {
            "rsi_period": 14, "oversold": 25, "overbought": 75,
            "trend_mode": True, "divergence_mode": True,
            "trend_rsi_threshold_long": 58,
            "trend_rsi_threshold_short": 42,
            "divergence_lookback": 20,
            "rsi_momentum_period": 4,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="reversal", risk_level="low",
    )
    register_strategy(
        "vwma", VWMAStrategy, StrategyTier.C,
        "Volume-weighted moving average with multi-period alignment",
        "BTC", "15m",
        {
            "fast_period": 15, "mid_period": 34, "slow_period": 55,
            "min_hold_bars": 6, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="low",
    )

    # ═══════════════════════════════════════════════════════════════════
    # TIER D — Backtested strategies (ported from bt_code)
    # ═══════════════════════════════════════════════════════════════════

    register_strategy(
        "adx", ADXStrategy, StrategyTier.D,
        "ADX trend strength with +DI/-DI directional crossover",
        "BTC", "1h",
        {
            "adx_period": 14, "di_period": 14,
            "adx_threshold": 30, "exit_threshold": 22,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="medium",
    )
    register_strategy(
        "macd", MACDStrategy, StrategyTier.D,
        "MACD/Signal crossover with histogram confirmation and MA filter",
        "BTC", "1h",
        {
            "fast_period": 12, "slow_period": 26, "signal_period": 9,
            "ma_filter_period": 50,
            "confirmation_bars": 1,
            "use_ma_filter": True,
            "histogram_mode": True, "zero_cross_mode": True,
            "histogram_growth_bars": 2,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="medium",
    )
    register_strategy(
        "ichimoku", IchimokuStrategy, StrategyTier.D,
        "Full Ichimoku cloud system with TK cross and cloud positioning",
        "BTC", "4h",
        {
            "tenkan_period": 20, "kijun_period": 60, "senkou_b_period": 120,
            "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="medium",
    )
    # DISABLED: unreliable Elliott Wave pattern detection in crypto
    # register_strategy(
    #     "elliott_wave", ElliottWaveStrategy, StrategyTier.D,
    #     "Elliott Wave swing pattern detection with Fibonacci retracement validation",
    #     "BTC", "4h",
    #     {
    #         "swing_lookback": 5,
    #         "fib_retracement_min": 0.382, "fib_retracement_max": 0.618,
    #         "min_swing_pct": 0.5, "reversal_exit_pct": 3.0,
    #         "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
    #         "min_signal_strength": 0.65,
    #     },
    #     category="pattern", risk_level="high",
    # )
    register_strategy(
        "pivot_lines", PivotLinesStrategy, StrategyTier.D,
        "Classic pivot point support/resistance level trading",
        "BTC", "1h",
        {
            "pivot_lookback": 24,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="reversal", risk_level="low",
    )
    # DISABLED: buggy TP/SL behavior with quarter levels in crypto
    # register_strategy(
    #     "quarter_theory", QuarterTheoryStrategy, StrategyTier.D,
    #     "Quarter-point price level breakout strategy",
    #     "BTC", "1h",
    #     {
    #         "quarter_size": None, "breakout_pct": 0.1,
    #         "take_profit_quarters": 2, "stop_loss_quarters": 1,
    #         "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
    #         "min_signal_strength": 0.65,
    #     },
    #     category="breakout", risk_level="medium",
    # )
    register_strategy(
        "ema_bollinger", EMABollingerStrategy, StrategyTier.D,
        "Combined EMA crossover trend + Bollinger Band squeeze entries",
        "BTC", "1h",
        {
            "short_ema_period": 21, "long_ema_period": 55,
            "bb_period": 20, "bb_std": 2.0,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="medium",
    )
    register_strategy(
        "grid_fibonacci", GridFibStrategy, StrategyTier.D,
        "Grid trading at Fibonacci retracement levels with dynamic recalibration",
        "BTC", "4h",
        {
            "fib_lookback": 60,
            "proximity_pct": 0.5,
            "trend_period": 20,
            "take_profit_fib": 0.618, "stop_loss_fib": 0.786,
            "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
            "min_signal_strength": 0.65,
        },
        category="grid", risk_level="medium",
    )
    # DISABLED: unreliable Elliott Wave + pivot confluence in crypto
    # register_strategy(
    #     "elliott_pivot", ElliottPivotStrategy, StrategyTier.D,
    #     "Combined Elliott Wave patterns with pivot point levels for confluence",
    #     "BTC", "4h",
    #     {
    #         "swing_lookback": 5, "pivot_lookback": 24,
    #         "min_hold_bars": 2, "cooldown_seconds": 180, "max_trades_per_hour": 4,
    #         "min_signal_strength": 0.65,
    #     },
    #     category="pattern", risk_level="medium",
    # )
    register_strategy(
        "sma_adx_bb_vol", SMAAdxBBVolStrategy, StrategyTier.D,
        "Multi-indicator combo: SMA crossover + ADX strength + BB squeeze + volume confirmation",
        "BTC", "1h",
        {
            "sma_period": 21, "adx_period": 14,
            "bb_period": 20, "bb_std": 2.0,
            "min_adx": 25, "volume_multiplier": 1.3,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="trend", risk_level="medium",
    )
    register_strategy(
        "rsi_vwap", RSIVWAPStrategy, StrategyTier.D,
        "Combined RSI and VWAP confluence signals for trend entries",
        "BTC", "15m",
        {
            "rsi_period": 14,
            "oversold": 25, "overbought": 75,
            "min_hold_bars": 3, "cooldown_seconds": 90, "max_trades_per_hour": 6,
            "min_signal_strength": 0.65,
        },
        category="reversal", risk_level="low",
    )
    register_strategy(
        "solana_sniper", SolanaSniperStrategy, StrategyTier.A,
        "Solana token sniper - scans new launches, multi-filter quality entry, MA trend following",
        "SOL", "5m",
        {
            "sma_fast": 10, "sma_slow": 25,
            "sell_at_multiple": 5.0, "stop_loss_pct": -0.40,
            "max_top10_holder_pct": 0.50, "min_liquidity": 1000,
            "require_price_above_avg": True,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.65,
        },
        category="momentum", risk_level="high",
    )
    register_strategy(
        "liquidation_dip", LiquidationDipStrategy, StrategyTier.A,
        "MoonDev's Double Dip — buys the second touch of liquidation lows after bounce (681% backtest)",
        "BTC", "5m",
        {
            "liq_threshold_usd": 500_000,
            "bounce_pct": 0.005,
            "dip_tolerance_pct": 0.002,
            "max_liq_age_hours": 4,
            "mode": "double_dip",
            "take_profit_pct": 1.5,
            "stop_loss_pct": 10.0,
            "time_limit_hours": 24,
            "min_hold_bars": 3, "cooldown_seconds": 120, "max_trades_per_hour": 5,
            "min_signal_strength": 0.60,
        },
        category="liquidation", risk_level="high",
    )


# Auto-register on import
_register_all()
