"""
Gap-Up Momentum Strategy.

MoonDev source: "ai coded 72 trade bot tests and found some gems (58,109% ROI)" (W0gooNe74qs).
Winner of 72-test sweep: Calmar 7.6, Sharpe 1.87, PF 2.43, expectancy 1.99 (optimized).

Logic:
  LONG when current bar opens ≥ gap_threshold above prev bar's close (gap up)
  AND Ultimate Oscillator < uo_oversold (momentum oversold → mean-reversion lift)
  AND ADX > adx_threshold (directional trend is strong)
  AND MFI < mfi_oversold (money flow oversold → potential reversal up)

Edge: Gap-ups in a directional environment with oversold oscillators signal an
exhaustion of selling pressure — the gap becomes a launching pad rather than
a re-fill target when all three confirmations align.

Optimized params: gap 0.5%, UO oversold 25, ADX 30, MFI oversold 35, TP 5%, SL 2%.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class GapUpMomentumStrategy(BaseStrategy):
    """
    Long-only gap-up momentum strategy with UO + ADX + MFI confirmation.

    State: IDLE → ENTERED (LONG on gap up + indicators) → IDLE (on exit).
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._entry_bar: Optional[int] = None
        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        n = len(df)
        if n < period + 2:
            return 0.0
        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)

        tr = np.maximum(highs[1:] - lows[1:],
               np.maximum(np.abs(highs[1:] - closes[:-1]),
                          np.abs(lows[1:] - closes[:-1])))
        dm_plus = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                           np.maximum(highs[1:] - highs[:-1], 0.0), 0.0)
        dm_minus = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                            np.maximum(lows[:-1] - lows[1:], 0.0), 0.0)

        def _smooth(arr: np.ndarray) -> np.ndarray:
            out = np.zeros(len(arr))
            out[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                out[i] = out[i - 1] - out[i - 1] / period + arr[i]
            return out

        atr = _smooth(tr)
        sdm_plus = _smooth(dm_plus)
        sdm_minus = _smooth(dm_minus)

        with np.errstate(invalid="ignore", divide="ignore"):
            di_plus = np.where(atr > 0, 100.0 * sdm_plus / atr, 0.0)
            di_minus = np.where(atr > 0, 100.0 * sdm_minus / atr, 0.0)
            dx = np.where((di_plus + di_minus) > 0,
                          100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0.0)

        # ADX = smoothed DX
        valid = dx[period - 1:]
        if len(valid) < period:
            return 0.0
        adx = valid[:period].mean()
        for i in range(period, len(valid)):
            adx = (adx * (period - 1) + valid[i]) / period
        return float(np.clip(adx, 0.0, 100.0))

    def _compute_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        n = len(df)
        if n < period + 1:
            return 50.0
        highs = df["high"].to_numpy(float)[-period - 1:]
        lows = df["low"].to_numpy(float)[-period - 1:]
        closes = df["close"].to_numpy(float)[-period - 1:]
        vols = df["volume"].to_numpy(float)[-period - 1:]

        tp = (highs + lows + closes) / 3.0
        mf = tp * vols

        pos_flow = np.where(tp[1:] > tp[:-1], mf[1:], 0.0)
        neg_flow = np.where(tp[1:] < tp[:-1], mf[1:], 0.0)

        sum_pos = pos_flow.sum()
        sum_neg = neg_flow.sum()
        if sum_neg == 0:
            return 100.0
        mfr = sum_pos / sum_neg
        return float(100.0 - 100.0 / (1.0 + mfr))

    def _compute_uo(self, df: pd.DataFrame,
                    short: int = 7, medium: int = 14, long_: int = 28) -> float:
        n = len(df)
        min_bars = long_ + 1
        if n < min_bars:
            return 50.0

        highs = df["high"].to_numpy(float)
        lows = df["low"].to_numpy(float)
        closes = df["close"].to_numpy(float)

        prev_closes = closes[:-1]
        bp = closes[1:] - np.minimum(lows[1:], prev_closes)   # buying pressure
        tr = np.maximum(highs[1:], prev_closes) - np.minimum(lows[1:], prev_closes)

        def _avg(period: int) -> float:
            bp_s = bp[-period:].sum()
            tr_s = tr[-period:].sum()
            return bp_s / tr_s if tr_s > 0 else 0.0

        avg_s = _avg(short)
        avg_m = _avg(medium)
        avg_l = _avg(long_)
        uo = 100.0 * (4.0 * avg_s + 2.0 * avg_m + avg_l) / 7.0
        return float(np.clip(uo, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Strategy logic
    # ------------------------------------------------------------------

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        self._bar_count += 1

        if self._entry_bar is not None:
            return None  # already in position

        p = self.config.params
        uo_long = int(p.get("uo_period_long", 28))
        adx_period = int(p.get("adx_period", 14))
        mfi_period = int(p.get("mfi_period", 14))
        min_bars = uo_long + adx_period + 5

        if len(data) < min_bars:
            return None

        gap_threshold = float(p.get("gap_threshold_pct", 0.005))
        uo_oversold = float(p.get("uo_oversold", 25.0))
        adx_threshold = float(p.get("adx_threshold", 30.0))
        mfi_oversold = float(p.get("mfi_oversold", 35.0))

        current_open = float(data["open"].iloc[-1])
        prev_close = float(data["close"].iloc[-2])

        if prev_close <= 0:
            return None

        gap_pct = (current_open - prev_close) / prev_close
        if gap_pct < gap_threshold:
            return None

        uo_short = int(p.get("uo_period_short", 7))
        uo_med = int(p.get("uo_period_med", 14))
        uo_val = self._compute_uo(data, short=uo_short, medium=uo_med, long_=uo_long)
        if uo_val >= uo_oversold:
            return None

        adx_val = self._compute_adx(data, period=adx_period)
        if adx_val < adx_threshold:
            return None

        mfi_val = self._compute_mfi(data, period=mfi_period)
        if mfi_val >= mfi_oversold:
            return None

        current_close = float(data["close"].iloc[-1])
        # Strength: weighted by how deep each oscillator is in oversold zone
        uo_depth = max(0.0, uo_oversold - uo_val) / uo_oversold
        mfi_depth = max(0.0, mfi_oversold - mfi_val) / mfi_oversold
        adx_excess = min((adx_val - adx_threshold) / adx_threshold, 1.0)
        gap_score = min(gap_pct / gap_threshold, 3.0) / 3.0
        strength = min(0.60 + uo_depth * 0.10 + mfi_depth * 0.10
                       + adx_excess * 0.10 + gap_score * 0.05, 0.92)

        self._entry_bar = self._bar_count - 1

        logger.info(
            "[GapUp] LONG | price=%.4f gap=%.3f%% UO=%.1f ADX=%.1f MFI=%.1f strength=%.2f",
            current_close, gap_pct * 100, uo_val, adx_val, mfi_val, strength,
        )
        return Signal(
            signal_type=SignalType.LONG,
            symbol=self.config.symbol,
            price=current_close,
            size_usd=self.config.size_usd,
            strength=strength,
            reason=(f"GapUp LONG: gap={gap_pct*100:.2f}% UO={uo_val:.1f} "
                    f"ADX={adx_val:.1f} MFI={mfi_val:.1f}"),
            metadata={"gap_pct": gap_pct, "uo": uo_val, "adx": adx_val,
                       "mfi": mfi_val, "strategy": "gap_up_momentum"},
        )

    async def should_exit(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[Signal]:
        if self._entry_bar is None:
            return None

        p = self.config.params
        hold_cap_bars = int(p.get("hold_cap_bars", 10))
        take_profit_pct = float(p.get("take_profit_pct", self.config.target_pct))
        stop_loss_pct = float(p.get("stop_loss_pct", abs(self.config.max_loss_pct)))

        pnl_pct = float(position.get("pnl_perc", 0))
        bars_held = self._bar_count - self._entry_bar

        if bars_held > hold_cap_bars:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"GapUp hold cap: {bars_held} bars",
            )
        if pnl_pct >= take_profit_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"GapUp TP: {pnl_pct:.1f}%",
            )
        if pnl_pct <= -stop_loss_pct:
            self._entry_bar = None
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"GapUp SL: {pnl_pct:.1f}%",
            )
        return None
