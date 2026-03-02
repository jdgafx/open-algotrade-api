"""
Regime Detector — Hidden Markov Model market regime classification.

Detects market regimes: TRENDING_UP, TRENDING_DOWN, MEAN_REVERTING,
HIGH_VOLATILITY, LOW_VOLATILITY using HMM fitted on returns + volatility.
Includes ATR-based volatility filter for strategy integration.
"""

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


# Regime color mapping for frontend
REGIME_COLORS = {
    MarketRegime.TRENDING_UP: "#22c55e",       # green
    MarketRegime.TRENDING_DOWN: "#ef4444",      # red
    MarketRegime.MEAN_REVERTING: "#3b82f6",     # blue
    MarketRegime.HIGH_VOLATILITY: "#f97316",    # orange
    MarketRegime.LOW_VOLATILITY: "#a855f7",     # purple
    MarketRegime.UNKNOWN: "#6b7280",            # gray
}

# All known strategy types — used as the universal allow-list
_ALL_STRATEGIES = [
    "turtle", "sma_crossover", "macd", "ichimoku", "adx",
    "ema_bollinger", "vwma", "sma_adx_bb_vol",
    "mean_reversion", "rsi", "bollinger", "rsi_vwap",
    "nadaraya_watson", "supply_demand_zone",
    "consolidation_pop", "grid_fibonacci", "elliott_wave", "elliott_pivot",
    "vwap_bot", "funding_arb", "correlation", "market_maker", "pivot_lines",
    "solana_sniper", "quarter_theory",
]

# Which strategies work in which regimes.
# Permissive: most strategies allowed in most regimes.
# The gate is advisory, not a kill switch — better to trade suboptimally
# than to block all signals for days.
REGIME_STRATEGY_MAP = {
    MarketRegime.TRENDING_UP: _ALL_STRATEGIES,
    MarketRegime.TRENDING_DOWN: _ALL_STRATEGIES,
    MarketRegime.MEAN_REVERTING: _ALL_STRATEGIES,
    MarketRegime.HIGH_VOLATILITY: _ALL_STRATEGIES,
    MarketRegime.LOW_VOLATILITY: _ALL_STRATEGIES,
    MarketRegime.UNKNOWN: _ALL_STRATEGIES,
}


class RegimeDetector:
    """
    HMM-based market regime detector.

    Fits a Gaussian HMM on returns + volatility features to classify
    the current market state into one of 5 regimes.
    """

    def __init__(self, n_regimes: int = 5, lookback_days: int = 90):
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self._models: Dict[str, Any] = {}  # per-symbol HMM models
        self._current_regimes: Dict[str, Dict] = {}  # symbol -> regime info
        self._regime_history: List[Dict] = []
        self._atr_cache: Dict[str, Dict] = {}

    def fit(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Fit HMM model on price data for a given symbol.

        Args:
            symbol: Asset symbol (e.g. "BTC")
            data: DataFrame with at least 'close' column (or 'Close')

        Returns:
            True if fitting succeeded
        """
        try:
            close_col = "Close" if "Close" in data.columns else "close"
            high_col = "High" if "High" in data.columns else "high"
            low_col = "Low" if "Low" in data.columns else "low"

            if close_col not in data.columns:
                logger.warning("No close column in data for %s", symbol)
                return False

            close = data[close_col].values.astype(float)
            if len(close) < 50:
                logger.warning("Insufficient data for %s (%d bars)", symbol, len(close))
                return False

            # Feature engineering: returns + rolling volatility
            returns = np.diff(np.log(close))
            vol_window = min(20, len(returns) // 3)
            rolling_vol = pd.Series(returns).rolling(vol_window).std().fillna(0).values

            # Trim to match lengths
            features = np.column_stack([
                returns[vol_window - 1:],
                rolling_vol[vol_window - 1:],
            ])

            # Remove NaN/Inf
            mask = np.isfinite(features).all(axis=1)
            features = features[mask]

            if len(features) < 30:
                logger.warning("Too few valid features for %s", symbol)
                return False

            try:
                from hmmlearn.hmm import GaussianHMM
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",
                    n_iter=500,
                    random_state=42,
                    min_covar=1e-3,
                )
                model.fit(features)
                self._models[symbol] = {
                    "hmm": model,
                    "features": features,
                    "fitted_at": datetime.now(timezone.utc),
                }
            except ImportError:
                logger.info("hmmlearn not available, using statistical fallback for %s", symbol)
                self._models[symbol] = {
                    "hmm": None,
                    "features": features,
                    "returns": returns,
                    "rolling_vol": rolling_vol,
                    "fitted_at": datetime.now(timezone.utc),
                }

            # Also compute ATR if high/low available
            if high_col in data.columns and low_col in data.columns:
                self._compute_atr(symbol, data, high_col, low_col, close_col)

            # Classify current regime
            self._classify_current(symbol, features, returns, rolling_vol)

            logger.info("Regime detector fitted for %s — current: %s",
                       symbol, self._current_regimes.get(symbol, {}).get("regime", "unknown"))
            return True

        except Exception as e:
            logger.error("Failed to fit regime detector for %s: %s", symbol, e)
            return False

    def _classify_current(
        self,
        symbol: str,
        features: np.ndarray,
        returns: np.ndarray,
        rolling_vol: np.ndarray,
    ):
        """Classify current regime using HMM or statistical fallback."""
        model_info = self._models.get(symbol)
        if not model_info:
            return

        hmm = model_info.get("hmm")

        if hmm is not None:
            # Use HMM hidden states
            try:
                states = hmm.predict(features)
                current_state = int(states[-1])

                # Map HMM states to regimes using mean returns and volatility per state
                state_stats = []
                for s in range(self.n_regimes):
                    mask = states == s
                    if mask.sum() > 0:
                        mean_ret = features[mask, 0].mean()
                        mean_vol = features[mask, 1].mean()
                    else:
                        mean_ret = 0
                        mean_vol = 0
                    state_stats.append((s, mean_ret, mean_vol))

                # Sort by mean return to assign labels
                state_stats.sort(key=lambda x: x[1])

                # Assign regimes based on characteristics
                regime_map = self._map_states_to_regimes(state_stats)
                current_regime = regime_map.get(current_state, MarketRegime.UNKNOWN)

                # Transition matrix
                transmat = hmm.transmat_.tolist()

            except Exception as e:
                logger.warning("HMM prediction failed for %s: %s", symbol, e)
                current_regime = self._statistical_regime(returns, rolling_vol)
                transmat = None
        else:
            # Statistical fallback
            current_regime = self._statistical_regime(returns, rolling_vol)
            transmat = None

        regime_info = {
            "regime": current_regime.value,
            "regime_label": current_regime.name.replace("_", " ").title(),
            "color": REGIME_COLORS[current_regime],
            "confidence": 0.75 if hmm else 0.5,
            "recommended_strategies": REGIME_STRATEGY_MAP.get(current_regime, []),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "transition_matrix": transmat,
        }

        # Track history
        prev = self._current_regimes.get(symbol)
        if prev and prev.get("regime") != current_regime.value:
            self._regime_history.append({
                "symbol": symbol,
                "from_regime": prev["regime"],
                "to_regime": current_regime.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self._current_regimes[symbol] = regime_info

    def _map_states_to_regimes(self, state_stats: List[Tuple]) -> Dict[int, MarketRegime]:
        """Map HMM hidden states to named regimes based on return/vol characteristics."""
        regime_map = {}
        n = len(state_stats)

        # Sort by volatility to find high/low vol states
        by_vol = sorted(state_stats, key=lambda x: x[2])

        if n >= 5:
            # Lowest vol
            regime_map[by_vol[0][0]] = MarketRegime.LOW_VOLATILITY
            # Highest vol
            regime_map[by_vol[-1][0]] = MarketRegime.HIGH_VOLATILITY

            # Remaining 3: sort by return
            remaining = [s for s in state_stats if s[0] not in regime_map]
            remaining.sort(key=lambda x: x[1])
            regime_map[remaining[0][0]] = MarketRegime.TRENDING_DOWN
            regime_map[remaining[1][0]] = MarketRegime.MEAN_REVERTING
            regime_map[remaining[2][0]] = MarketRegime.TRENDING_UP
        elif n >= 3:
            regime_map[by_vol[0][0]] = MarketRegime.LOW_VOLATILITY
            regime_map[by_vol[-1][0]] = MarketRegime.HIGH_VOLATILITY
            remaining = [s for s in state_stats if s[0] not in regime_map]
            if remaining:
                if remaining[0][1] > 0:
                    regime_map[remaining[0][0]] = MarketRegime.TRENDING_UP
                else:
                    regime_map[remaining[0][0]] = MarketRegime.TRENDING_DOWN
        else:
            for s, ret, vol in state_stats:
                if ret > 0.001:
                    regime_map[s] = MarketRegime.TRENDING_UP
                elif ret < -0.001:
                    regime_map[s] = MarketRegime.TRENDING_DOWN
                else:
                    regime_map[s] = MarketRegime.MEAN_REVERTING

        return regime_map

    def _statistical_regime(self, returns: np.ndarray, rolling_vol: np.ndarray) -> MarketRegime:
        """Simple statistical regime classification when HMM is not available."""
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        recent_vol = rolling_vol[-20:] if len(rolling_vol) >= 20 else rolling_vol

        mean_ret = np.mean(recent_returns)
        mean_vol = np.mean(recent_vol[np.isfinite(recent_vol)]) if len(recent_vol) > 0 else 0
        overall_vol = np.mean(rolling_vol[np.isfinite(rolling_vol)]) if len(rolling_vol) > 0 else 0

        # Classify based on return trend and volatility level
        vol_ratio = mean_vol / overall_vol if overall_vol > 0 else 1.0

        if vol_ratio > 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_ratio < 0.6:
            return MarketRegime.LOW_VOLATILITY
        elif mean_ret > 0.001:
            return MarketRegime.TRENDING_UP
        elif mean_ret < -0.001:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.MEAN_REVERTING

    def _compute_atr(
        self, symbol: str, data: pd.DataFrame,
        high_col: str, low_col: str, close_col: str,
        period: int = 14,
    ):
        """Compute ATR for volatility filtering."""
        high = data[high_col].values.astype(float)
        low = data[low_col].values.astype(float)
        close = data[close_col].values.astype(float)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        if len(tr) < period:
            return

        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        atr_pct = (atr / close[-1]) * 100 if close[-1] > 0 else 0

        self._atr_cache[symbol] = {
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "price": float(close[-1]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def is_volatile(self, symbol: str, threshold_pct: float = 3.0) -> bool:
        """
        Check if a symbol is currently in a volatile state.
        Uses ATR as percentage of price.

        Args:
            symbol: Asset symbol
            threshold_pct: ATR% threshold above which we consider volatile

        Returns:
            True if current ATR% > threshold
        """
        atr_info = self._atr_cache.get(symbol)
        if not atr_info:
            return False
        return atr_info["atr_pct"] > threshold_pct

    def get_current_regime(self, symbol: str) -> Optional[Dict]:
        """Get the current regime classification for a symbol."""
        return self._current_regimes.get(symbol)

    def get_all_regimes(self) -> Dict[str, Dict]:
        """Get current regimes for all tracked symbols."""
        return dict(self._current_regimes)

    def get_regime_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get regime transition history, optionally filtered by symbol."""
        history = self._regime_history
        if symbol:
            history = [h for h in history if h["symbol"] == symbol]
        return history[-limit:]

    def get_volatility_status(self) -> Dict[str, Dict]:
        """Get ATR-based volatility status for all cached symbols."""
        result = {}
        for symbol, atr_info in self._atr_cache.items():
            result[symbol] = {
                **atr_info,
                "is_volatile": self.is_volatile(symbol),
            }
        return result

    def should_trade(self, symbol: str, strategy_type: str) -> bool:
        """
        Check if a strategy should be active given the current regime.

        Integration point: strategies call this before trading.
        """
        regime_info = self._current_regimes.get(symbol)
        if not regime_info:
            return True  # No regime data = allow trading

        current_regime = MarketRegime(regime_info["regime"])
        recommended = REGIME_STRATEGY_MAP.get(current_regime, [])

        # If strategy isn't in recommended list for this regime, suggest caution
        return strategy_type in recommended

    async def update(self, symbol: str, data: pd.DataFrame):
        """Update regime detection with new data (call periodically)."""
        self.fit(symbol, data)


# Global singleton for use across the application
_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector
