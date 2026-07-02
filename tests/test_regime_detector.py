"""Tests for RegimeDetector strategy map — regime/strategy compatibility."""
import pytest
from src.services.regime_detector import MarketRegime, REGIME_STRATEGY_MAP


class TestRegimeStrategyMap:
    """Verify that each strategy is allowed in the regimes where it has edge."""

    def test_consolidation_pop_allowed_in_mean_reverting(self):
        # ConsolidationPop thrives in tight ranges — the definition of mean-reverting.
        # This was the root cause of zero executions when BTC was range-bound.
        allowed = REGIME_STRATEGY_MAP[MarketRegime.MEAN_REVERTING]
        assert "consolidation_pop" in allowed, (
            "consolidation_pop must be allowed in MEAN_REVERTING: "
            "tight ranges are exactly when it has edge"
        )

    def test_consolidation_pop_allowed_in_low_volatility(self):
        # Low vol = compressed range = prime entry conditions for conspop.
        allowed = REGIME_STRATEGY_MAP[MarketRegime.LOW_VOLATILITY]
        assert "consolidation_pop" in allowed, (
            "consolidation_pop must be allowed in LOW_VOLATILITY: "
            "compressed ranges precede breakouts"
        )

    def test_consolidation_pop_allowed_in_trending_up(self):
        allowed = REGIME_STRATEGY_MAP[MarketRegime.TRENDING_UP]
        assert "consolidation_pop" in allowed

    def test_consolidation_pop_allowed_in_trending_down(self):
        allowed = REGIME_STRATEGY_MAP[MarketRegime.TRENDING_DOWN]
        assert "consolidation_pop" in allowed

    def test_consolidation_pop_allowed_in_high_volatility(self):
        allowed = REGIME_STRATEGY_MAP[MarketRegime.HIGH_VOLATILITY]
        assert "consolidation_pop" in allowed

    def test_rsi_allowed_in_mean_reverting(self):
        # Regression guard: RSI is the canonical mean-reversion tool.
        assert "rsi" in REGIME_STRATEGY_MAP[MarketRegime.MEAN_REVERTING]

    def test_liquidation_dip_allowed_in_trending_up(self):
        # Regression guard: liq dip needs momentum — trending regimes.
        assert "liquidation_dip" in REGIME_STRATEGY_MAP[MarketRegime.TRENDING_UP]

    def test_vwap_bot_allowed_everywhere(self):
        # vwap_bot is market-neutral — should always be allowed.
        for regime in MarketRegime:
            if regime == MarketRegime.UNKNOWN:
                continue
            allowed = REGIME_STRATEGY_MAP.get(regime, [])
            assert "vwap_bot" in allowed, f"vwap_bot must be allowed in {regime}"


class TestRegimeMapDiscriminates:
    """2026-07-02 sharpening: the map must BLOCK each family in the regime whose
    failure mode it is — the old map recommended ~39/40 types in trends."""

    def test_meanrev_blocked_in_trends(self):
        for regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            allowed = REGIME_STRATEGY_MAP[regime]
            for s in ("rsi", "bollinger", "mean_reversion", "grid_fibonacci"):
                assert s not in allowed, f"{s} must be blocked in {regime} (knife-catching)"

    def test_trend_blocked_in_ranges(self):
        for regime in (MarketRegime.MEAN_REVERTING, MarketRegime.LOW_VOLATILITY):
            allowed = REGIME_STRATEGY_MAP[regime]
            for s in ("trend_cross", "sma_crossover", "macd", "turtle"):
                assert s not in allowed, f"{s} must be blocked in {regime} (whipsaw)"

    def test_grid_blocked_in_high_vol(self):
        assert "grid_fibonacci" not in REGIME_STRATEGY_MAP[MarketRegime.HIGH_VOLATILITY]

    def test_neutrals_allowed_everywhere(self):
        for regime in MarketRegime:
            if regime == MarketRegime.UNKNOWN:
                continue
            allowed = REGIME_STRATEGY_MAP[regime]
            for s in ("xsec_driver", "xsec_carry", "market_maker", "funding_arb"):
                assert s in allowed, f"{s} (market-neutral) must be allowed in {regime}"

    def test_every_regime_blocks_something(self):
        # The map must discriminate: each known regime blocks >= 8 types.
        from src.services.regime_detector import _ALL_STRATEGIES
        for regime, allowed in REGIME_STRATEGY_MAP.items():
            if regime == MarketRegime.UNKNOWN:
                continue
            blocked = set(_ALL_STRATEGIES) - set(allowed)
            assert len(blocked) >= 8, f"{regime} blocks only {len(blocked)} types — rubber stamp"
