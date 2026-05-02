"""
Nadaraya-Watson Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/4_nadarya_watson_algo/bot.py

- Gaussian kernel regression to smooth price (Nadaraya-Watson envelope)
- Stochastic RSI for momentum confirmation
- Long: NW buy signal OR StochRSI oversold (MoonDev OR logic)
- Short: NW sell signal OR StochRSI overbought
- Exit: opposite signal with 2x confirmation (MoonDev original)
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
        lookback = p.get("kernel_lookback", 100)   # 100 bars: fast enough, still meaningful
        overbought = p.get("overbought", 80)       # Relaxed from 90 for more signals
        oversold = p.get("oversold", 20)            # Relaxed from 10 for more signals
        adx_period = p.get("adx_period", 14)
        adx_threshold = p.get("adx_threshold", 25)

        if len(data) < max(lookback + 5, adx_period * 2 + 2):
            return None

        data = self._add_indicators(data, bandwidth, lookback, p, adx_period)
        current = data.iloc[-1]
        price = current["close"]

        nw_upper = current.get("nw_upper", None)
        nw_lower = current.get("nw_lower", None)
        stoch_k = current.get("stoch_k", None)

        if nw_upper is None or pd.isna(nw_upper):
            return None

        # Mean reversion only works in ranging markets — skip when trending (ADX high)
        adx = current.get("adx", 0)
        if not pd.isna(adx) and adx >= adx_threshold:
            return None

        # NW envelope signals: direction change OR price below/above band
        nw_buy = bool(current.get("nw_buy", False))
        nw_sell = bool(current.get("nw_sell", False))

        # Also trigger when price touches the NW envelope bands (more frequent)
        nw_band_long = nw_lower is not None and not pd.isna(nw_lower) and price <= nw_lower * 1.002
        nw_band_short = nw_upper is not None and not pd.isna(nw_upper) and price >= nw_upper * 0.998

        stoch_oversold = stoch_k is not None and not pd.isna(stoch_k) and stoch_k < oversold
        stoch_overbought = stoch_k is not None and not pd.isna(stoch_k) and stoch_k > overbought

        # Long: NW buy signal OR price at lower band OR StochRSI oversold
        if nw_buy or nw_band_long or stoch_oversold:
            conditions_met = int(nw_buy) + int(stoch_oversold)
            strength = 0.65 if conditions_met == 1 else 0.9  # both = stronger signal
            parts = []
            if nw_buy:
                parts.append("NW buy signal")
            if stoch_oversold:
                parts.append(f"StochRSI {stoch_k:.1f} < {oversold}")
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"NW LONG: {' + '.join(parts)}",
                metadata={"nw_buy": nw_buy, "stoch_k": stoch_k, "nw_lower": nw_lower},
            )

        # Short: NW sell signal OR price at upper band OR StochRSI overbought
        if nw_sell or nw_band_short or stoch_overbought:
            conditions_met = int(nw_sell) + int(stoch_overbought)
            strength = 0.65 if conditions_met == 1 else 0.9
            parts = []
            if nw_sell:
                parts.append("NW sell signal")
            if stoch_overbought:
                parts.append(f"StochRSI {stoch_k:.1f} > {overbought}")
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=f"NW SHORT: {' + '.join(parts)}",
                metadata={"nw_sell": nw_sell, "stoch_k": stoch_k, "nw_upper": nw_upper},
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        bandwidth = p.get("kernel_bandwidth", 8.0)
        lookback = p.get("kernel_lookback", 200)
        overbought = p.get("overbought", 90)
        oversold = p.get("oversold", 10)

        adx_period = p.get("adx_period", 14)
        data = self._add_indicators(data, bandwidth, lookback, p, adx_period)
        current = data.iloc[-1]
        price = current["close"]

        is_long = position.get("is_long", position.get("size", 0) > 0)
        pnl_pct = position.get("pnl_perc", 0)

        # 2x confirmation exit (MoonDev original logic)
        stoch_exit_window = p.get("stoch_exit_window", 14)
        exit_times = p.get("exit_confirmation_times", 2)  # MoonDev requires 2x
        recent_stoch = data["stoch_k"].tail(stoch_exit_window)

        if is_long:
            # Exit long when StochRSI has been overbought 2+ times OR NW sell signal
            overbought_count = int((recent_stoch > overbought).sum()) if not recent_stoch.isna().all() else 0
            nw_sell_now = bool(current.get("nw_sell", False))
            if overbought_count >= exit_times or nw_sell_now:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"NW exit: overbought {overbought_count}x in {stoch_exit_window} bars" if overbought_count >= exit_times else "NW sell signal",
                )
        if not is_long:
            # Exit short when StochRSI has been oversold 2+ times OR NW buy signal
            oversold_count = int((recent_stoch < oversold).sum()) if not recent_stoch.isna().all() else 0
            nw_buy_now = bool(current.get("nw_buy", False))
            if oversold_count >= exit_times or nw_buy_now:
                return Signal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.config.symbol,
                    reason=f"NW exit: oversold {oversold_count}x in {stoch_exit_window} bars" if oversold_count >= exit_times else "NW buy signal",
                )

        # Fallback exits: target profit and max loss
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Target: {pnl_pct:.1f}%")
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(signal_type=close_type, symbol=self.config.symbol, reason=f"Max loss: {pnl_pct:.1f}%")

        return None

    @staticmethod
    def _nadaraya_watson_kernel(prices: np.ndarray, bandwidth: float) -> np.ndarray:
        """Gaussian kernel regression — vectorized O(n²) with numpy broadcasting."""
        n = len(prices)
        idx = np.arange(n, dtype=np.float64)
        # (n, n) weight matrix: weight[i,j] = exp(-0.5 * ((i-j)/bw)²)
        diff = (idx[:, None] - idx[None, :]) / bandwidth
        weights = np.exp(-0.5 * diff ** 2)
        smoothed = (weights * prices[None, :]).sum(axis=1) / weights.sum(axis=1)
        return smoothed

    def _add_indicators(self, df: pd.DataFrame, bandwidth: float, lookback: int, params: dict, adx_period: int = 14) -> pd.DataFrame:
        if "nw_upper" in df.columns and "adx" in df.columns:
            return df
        df = df.copy()

        # Nadaraya-Watson kernel regression
        prices = df["close"].values[-lookback:]
        if len(prices) < lookback:
            df["nw_mid"] = df["close"]
            df["nw_upper"] = df["close"]
            df["nw_lower"] = df["close"]
            df["nw_buy"] = False
            df["nw_sell"] = False
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

        # NW buy/sell signals: derivative direction change (MoonDev's original logic)
        nw_series = pd.Series(smoothed, index=df.index[-lookback:])
        nw_diff = nw_series.diff()
        df["nw_buy"] = False
        df["nw_sell"] = False
        # Buy signal: NW starts rising (diff goes from negative to positive)
        df.loc[df.index[-lookback:], "nw_buy"] = (nw_diff > 0) & (nw_diff.shift(1) < 0)
        # Sell signal: NW starts falling (diff goes from positive to negative)
        df.loc[df.index[-lookback:], "nw_sell"] = (nw_diff < 0) & (nw_diff.shift(1) > 0)

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

        # Wilder ADX — mean-reversion filter (enter only when ADX < threshold)
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        atr_safe = atr.replace(0, np.nan)
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr_safe
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr_safe
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()

        return df
