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
):
    _REGISTRY[strategy_type] = {
        "cls": cls,
        "tier": tier,
        "description": description,
        "default_symbol": default_symbol,
        "default_timeframe": default_timeframe,
        "default_params": default_params or {},
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

    register_strategy(
        "turtle", TurtleHLStrategy, StrategyTier.A,
        "55-bar breakout with ATR trailing stops and take profit",
        "BTC", "1h",
        {"lookback_period": 55, "atr_period": 20, "atr_multiplier": 2.0, "take_profit_pct": 0.002},
    )
    register_strategy(
        "bollinger", BollingerStrategy, StrategyTier.A,
        "Bollinger Band squeeze breakout with band-width triggers",
        "BTC", "1h",
        {"bb_period": 20, "bb_std": 2.0, "squeeze_threshold": 0.03},
    )
    register_strategy(
        "supply_demand_zone", SupplyDemandZoneStrategy, StrategyTier.A,
        "Supply/Demand zone detection with reversal entries",
        "BTC", "4h",
        {"zone_lookback_days": 30, "zone_threshold": 0.02},
    )
    register_strategy(
        "vwap_bot", VWAPBotStrategy, StrategyTier.A,
        "VWAP-based probability bias trading (70/30 above/below)",
        "BTC", "15m",
        {"vwap_bias_long": 0.7, "vwap_bias_short": 0.3},
    )
    register_strategy(
        "funding_arb", FundingArbStrategy, StrategyTier.A,
        "Funding rate arbitrage between correlated assets (BTC/ETH)",
        "BTC", "1h",
        {"symbol_a": "BTC", "symbol_b": "ETH", "funding_threshold": 0.0005, "combined_target_pct": 3.0},
    )
    register_strategy(
        "correlation", CorrelationStrategy, StrategyTier.B,
        "Leader/follower correlation trading (ETH leads altcoins)",
        "SOL", "15m",
        {"leader": "ETH", "correlation_window": 20, "lag_threshold": 0.002, "sl_pct": 0.002, "tp_pct": 0.0025},
    )
    register_strategy(
        "consolidation_pop", ConsolidationPopStrategy, StrategyTier.B,
        "Consolidation detection via ATR deviance, range breakout",
        "BTC", "15m",
        {"atr_period": 14, "deviance_threshold": 0.4, "range_position_buy": 0.33, "range_position_sell": 0.67, "tp_pct": 0.003, "sl_pct": 0.0025},
    )
    register_strategy(
        "nadaraya_watson", NadarayaWatsonStrategy, StrategyTier.B,
        "Kernel regression envelope + Stochastic RSI signals",
        "BTC", "15m",
        {"kernel_bandwidth": 8.0, "kernel_lookback": 60, "stoch_period": 14, "stoch_k": 3, "stoch_d": 3, "overbought": 80, "oversold": 20},
    )
    register_strategy(
        "market_maker", MarketMakerStrategy, StrategyTier.B,
        "Spread-based market making with kill switch and ATR no-trade zones",
        "BTC", "1m",
        {"spread": 0.001, "max_position_usd": 1000.0, "kill_size_usd": 2000.0, "atr_period": 14, "refresh_seconds": 10},
    )
    register_strategy(
        "mean_reversion", MeanReversionStrategy, StrategyTier.B,
        "Multi-timeframe SMA mean reversion with trend filter",
        "ETH", "15m",
        {"sma_trend_period": 20, "sma_entry_period": 20, "trend_timeframe": "4h", "entry_timeframe": "15m", "reversion_target_pct": 0.003},
    )
    register_strategy(
        "sma_crossover", SMAStrategy, StrategyTier.C,
        "SMA crossover with support/resistance levels",
        "BTC", "1h",
        {"sma_period": 20, "support_lookback": 20},
    )
    register_strategy(
        "rsi", RSIStrategy, StrategyTier.C,
        "RSI overbought/oversold reversal strategy",
        "BTC", "1h",
        {"rsi_period": 14, "oversold": 30, "overbought": 70},
    )
    register_strategy(
        "vwma", VWMAStrategy, StrategyTier.C,
        "Volume-weighted moving average with multi-period alignment",
        "BTC", "15m",
        {"fast_period": 20, "mid_period": 41, "slow_period": 75},
    )


# Auto-register on import
_register_all()
