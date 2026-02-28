"""
Nadaraya-Watson Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/4_nadarya_watson_algo/bot.py

- Gaussian kernel regression to smooth price (Nadaraya-Watson envelope)
- Stochastic RSI for momentum confirmation
- Long: price at lower envelope + StochRSI oversold
- Short: price at upper envelope + StochRSI overbought
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class NadarayaWatsonStrategy(BaseStrategy):

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params
        bandwidth = p.get("kernel_bandwidth", 8.0)
        lookback = p.get("kernel_lookback", 60)
        overbought = p.get("overbought", 80)
        oversold = p.get("oversold", 20)

        if len(data) < lookback + 5:
            return None

        data = self._add_indicators(data, bandwidth, lookback, p)
        current = data.iloc[-1]
        price = current["close"]

        nw_upper = current.get("nw_upper", None)
        nw_lower = current.get("nw_lower", None)
        stoch_k = current.get("stoch_k", 50)

        if nw_upper is None or pd.isna(nw_upper):
            return None

        # Long: price near lower envelope and StochRSI oversold
        if price <= nw_lower and stoch_k < oversold:
            # Strength: how deep below envelope + how oversold
            envelope_depth = (nw_lower - price) / max(nw_upper - nw_lower, 0.01) if nw_upper > nw_lower else 0.1
            stoch_depth = (oversold - stoch_k) / oversold
            strength = min(0.6 + envelope_depth * 0.2 + stoch_depth * 0.2, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"NW LONG: price {price:.2f} <= lower envelope {nw_lower:.2f}, StochRSI {stoch_k:.1f} < {oversold}",
                metadata={"nw_lower": nw_lower, "stoch_k": stoch_k},
            )

        # Short: price near upper envelope and StochRSI overbought
        if price >= nw_upper and stoch_k > overbought:
            envelope_depth = (price - nw_upper) / max(nw_upper - nw_lower, 0.01) if nw_upper > nw_lower else 0.1
            stoch_depth = (stoch_k - overbought) / (100 - overbought)
            strength = min(0.6 + envelope_depth * 0.2 + stoch_depth * 0.2, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"NW SHORT: price {price:.2f} >= upper envelope {nw_upper:.2f}, StochRSI {stoch_k:.1f} > {overbought}",
                metadata={"nw_upper": nw_upper, "stoch_k": stoch_k},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        bandwidth = p.get("kernel_bandwidth", 8.0)
        lookback = p.get("kernel_lookback", 60)

        data = self._add_indicators(data, bandwidth, lookback, p)
        current = data.iloc[-1]
        price = current["close"]

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        nw_mid = current.get("nw_mid", price)

        # Exit at mean reversion (NW midline)
        if is_long and price >= nw_mid:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"NW mean reversion exit: price {price:.2f} >= midline {nw_mid:.2f}",
            )
        if not is_long and price <= nw_mid:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason=f"NW mean reversion exit: price {price:.2f} <= midline {nw_mid:.2f}",
            )

        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _nadaraya_watson_kernel(prices: np.ndarray, bandwidth: float) -> np.ndarray:
        """Gaussian kernel regression (Nadaraya-Watson estimator)."""
        n = len(prices)
        smoothed = np.zeros(n)
        for i in range(n):
            weights = np.exp(-0.5 * ((np.arange(n) - i) / bandwidth) ** 2)
            smoothed[i] = np.sum(weights * prices) / np.sum(weights)
        return smoothed

    def _add_indicators(self, df: pd.DataFrame, bandwidth: float, lookback: int, params: dict) -> pd.DataFrame:
        if "nw_upper" in df.columns:
            return df
        df = df.copy()

        # Nadaraya-Watson kernel regression
        prices = df["close"].values[-lookback:]
        if len(prices) < lookback:
            df["nw_mid"] = df["close"]
            df["nw_upper"] = df["close"]
            df["nw_lower"] = df["close"]
            df["stoch_k"] = 50
            return df

        smoothed = self._nadaraya_watson_kernel(prices, bandwidth)

        # Calculate envelope from residuals
        residuals = prices - smoothed
        std = np.std(residuals) if len(residuals) > 1 else 0
        envelope_mult = 2.0

        nw_mid = pd.Series(index=df.index, dtype=float)
        nw_upper = pd.Series(index=df.index, dtype=float)
        nw_lower = pd.Series(index=df.index, dtype=float)

        nw_mid.iloc[-lookback:] = smoothed
        nw_upper.iloc[-lookback:] = smoothed + (std * envelope_mult)
        nw_lower.iloc[-lookback:] = smoothed - (std * envelope_mult)

        df["nw_mid"] = nw_mid
        df["nw_upper"] = nw_upper
        df["nw_lower"] = nw_lower

        # Stochastic RSI
        stoch_period = params.get("stoch_period", 14)
        stoch_k_smooth = params.get("stoch_k", 3)

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(stoch_period).mean()
        loss = (-delta.clip(upper=0)).rolling(stoch_period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling(stoch_period).min()
        rsi_max = rsi.rolling(stoch_period).max()
        stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, 1)) * 100
        df["stoch_k"] = stoch_rsi.rolling(stoch_k_smooth).mean()

        return df
