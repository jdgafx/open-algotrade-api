"""
Market Maker Strategy — BaseStrategy implementation.
Ported from ALGOS/6 Bonus Algos/5_market_maker/market-maker.py (MoonDev)

Full MoonDev logic:
- Range-based entry zones (buy near bottom, sell near top of lookback range)
- L2H max-range check (skip if market too volatile)
- Last-N-bars consolidation check (skip if trending — new highs/lows in recent bars)
- ATR / single-bar true-range filter
- Tight take-profit (0.4% default)
- Time-limit forced close
- Kill switch on excessive position size
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class MarketMakerStrategy(BaseStrategy):

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    # ------------------------------------------------------------------ #
    #  ENTRY — range-based zones with consolidation + volatility filters  #
    # ------------------------------------------------------------------ #

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        p = self.config.params

        # Configurable params with MoonDev defaults
        num_bars: int = p.get("num_bars", 180)           # lookback for range
        quartile: float = p.get("quartile", 0.33)        # entry zone %
        max_l2h: float = p.get("max_l2h", 0.05)          # max range as % of avg price (5%)
        max_tr_pct: float = p.get("max_tr_pct", 0.02)    # max single-bar true range %
        atr_period: int = p.get("atr_period", 14)
        last_n_bars: int = p.get("last_n_bars", 17)       # consolidation check window

        min_required = max(num_bars, atr_period + 2, last_n_bars + 2)
        if len(data) < min_required:
            return None

        # ── Lookback window for range calculation ──
        window = data.tail(num_bars)
        current = data.iloc[-1]
        price = float(current["close"])

        range_low = float(window["low"].min())
        range_high = float(window["high"].max())
        l2h = range_high - range_low
        avg_price = (range_high + range_low) / 2.0

        # ── 1. L2H check — range too wide ⇒ trending / volatile ──
        l2h_pct = l2h / avg_price if avg_price > 0 else 0.0
        if l2h_pct > max_l2h:
            logger.debug(
                "MM skip: l2h %.2f%% > max_l2h %.2f%% — too volatile",
                l2h_pct * 100,
                max_l2h * 100,
            )
            return None

        # ── 2. ATR / single-bar TR check ──
        data_with_atr = self._add_atr(data, atr_period)
        current_row = data_with_atr.iloc[-1]
        current_tr = float(current_row.get("tr", 0))
        if current_tr > 0 and price > 0:
            tr_pct = current_tr / price
            if tr_pct > max_tr_pct:
                logger.debug(
                    "MM skip: single-bar TR %.4f%% > max_tr_pct %.4f%%",
                    tr_pct * 100,
                    max_tr_pct * 100,
                )
                return None

        # ── 3. Last-N-bars consolidation check ──
        # If the overall range low or high was set within the last N bars the
        # market is still trending (making new extremes), not ranging.
        last_n = data.tail(last_n_bars)
        if float(last_n["low"].min()) <= range_low * 1.001:
            logger.debug("MM skip: new low in last %d bars — trending down", last_n_bars)
            return None
        if float(last_n["high"].max()) >= range_high * 0.999:
            logger.debug("MM skip: new high in last %d bars — trending up", last_n_bars)
            return None

        # ── 4. Calculate entry zones ──
        open_range = l2h * quartile
        buy_zone = range_low + open_range     # buy near bottom of range
        sell_zone = range_high - open_range    # sell near top of range

        # Signal strength: deeper into the zone ⇒ stronger signal
        if price < buy_zone:
            depth = (buy_zone - price) / open_range if open_range > 0 else 0.5
            strength = min(0.6 + depth * 0.3, 0.95)
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"MM range buy: price {price:.2f} < buy_zone {buy_zone:.2f} "
                    f"(range {range_low:.2f}-{range_high:.2f})"
                ),
                metadata={
                    "low": range_low,
                    "high": range_high,
                    "l2h_pct": round(l2h_pct, 5),
                    "zone": "buy",
                },
            )
        elif price > sell_zone:
            depth = (price - sell_zone) / open_range if open_range > 0 else 0.5
            strength = min(0.6 + depth * 0.3, 0.95)
            return Signal(
                signal_type=SignalType.SHORT,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"MM range sell: price {price:.2f} > sell_zone {sell_zone:.2f} "
                    f"(range {range_low:.2f}-{range_high:.2f})"
                ),
                metadata={
                    "low": range_low,
                    "high": range_high,
                    "l2h_pct": round(l2h_pct, 5),
                    "zone": "sell",
                },
            )

        # Price is in the middle of the range — no trade
        return None

    # ------------------------------------------------------------------ #
    #  EXIT — tight TP, stop loss, time limit, kill switch               #
    # ------------------------------------------------------------------ #

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params

        exit_pct: float = p.get("exit_pct", 0.004)            # 0.4% take profit (MoonDev default)
        stop_loss_pct: float = p.get("mm_stop_pct", 0.01)     # 1% stop loss
        time_limit: int = p.get("time_limit_minutes", 120)     # forced close after N minutes
        kill_size_usd: float = p.get("kill_size_usd", 2000.0)

        is_long = position.get("is_long", position.get("size", 0) > 0)
        size = abs(position.get("size", 0))
        entry_px = position.get("entry_px", position.get("entry_price", 0))
        pnl_pct = position.get("pnl_perc", 0)
        price = float(data.iloc[-1]["close"])

        position_usd = size * price

        # ── Kill switch: position too large ──
        if position_usd > kill_size_usd:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"MM kill switch: ${position_usd:.0f} > ${kill_size_usd:.0f}",
            )

        # ── Take profit / stop loss from entry price ──
        if entry_px > 0:
            if is_long:
                pnl_from_entry = (price - entry_px) / entry_px
            else:
                pnl_from_entry = (entry_px - price) / entry_px

            if pnl_from_entry >= exit_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=(
                        f"MM take profit: {pnl_from_entry * 100:.2f}% "
                        f">= {exit_pct * 100:.1f}%"
                    ),
                )

            if pnl_from_entry <= -stop_loss_pct:
                close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
                return Signal(
                    signal_type=close_type,
                    symbol=self.config.symbol,
                    reason=(
                        f"MM stop loss: {pnl_from_entry * 100:.2f}% "
                        f"<= -{stop_loss_pct * 100:.1f}%"
                    ),
                )

        # ── Time limit: forced close after N minutes ──
        # BaseStrategy increments self.state.entry_bar_count every iteration
        # while in a position, so bars * interval_seconds gives elapsed time.
        iterations_in_pos = self.state.entry_bar_count
        interval_s = max(self.config.interval_seconds, 1)
        time_limit_bars = int(time_limit * 60 / interval_s)
        if iterations_in_pos > time_limit_bars:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=(
                    f"MM time limit: {iterations_in_pos} bars > "
                    f"{time_limit_bars} ({time_limit}min)"
                ),
            )

        # ── Global target / max-loss from StrategyConfig ──
        if pnl_pct >= self.config.target_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Target: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            close_type = SignalType.CLOSE_LONG if is_long else SignalType.CLOSE_SHORT
            return Signal(
                signal_type=close_type,
                symbol=self.config.symbol,
                reason=f"Max loss: {pnl_pct:.1f}%",
            )

        return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Add ATR and true-range columns if not already present."""
        if "atr" in df.columns:
            return df
        df = df.copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = pd.concat(
            [
                (df["high"] - df["low"]).abs(),
                (df["high"] - df["prev_close"]).abs(),
                (df["low"] - df["prev_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = df["tr"].rolling(window=period).mean()
        return df
