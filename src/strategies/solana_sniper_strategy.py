"""
Solana Sniper Strategy -- Full Solana DEX integration.
Ported from ALGOS/Solana Sniper 2025/solana-sniper-2025-main/

This strategy has TWO operating modes:

1. ORCHESTRATOR MODE (symbol="SOL"):
   Runs as a standard BaseStrategy against SOL/USD OHLCV data on Hyperliquid.
   Uses MA crossover + momentum for entry/exit on the SOL perpetual.
   This is the fallback when the Solana scanner service is not available.

2. SCANNER MODE (when SolanaScanner is injected via params["scanner"]):
   Delegates token discovery to the SolanaScanner service, which runs the
   full MoonDev pipeline: Jupiter new tokens -> BirdEye security -> BirdEye
   metrics -> OHLCV technicals -> score -> paper buy.
   In this mode, should_enter() triggers a scan and returns a signal when
   high-scoring tokens are found.

Entry logic (orchestrator mode):
- Price above MA20, MA20 above MA40, price above avg close
- Higher highs + higher lows confirmation
- MA score >= 2 of 3 conditions + momentum

Entry logic (scanner mode):
- Token passes all 4 filter stages (security, metrics, OHLCV, score)
- Score >= min_score threshold

Exit logic (both modes):
- Take profit: price >= entry * sell_at_multiple
- Stop loss: PnL <= stop_loss_pct
- Trend break: price < slow MA (orchestrator mode)
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyConfig

logger = logging.getLogger(__name__)


class SolanaSniperStrategy(BaseStrategy):
    """Solana token sniper - dual mode: SOL trend following + Solana DEX token scanning."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Scanner is injected via params at runtime by the orchestrator or API
        self._scanner = config.params.get("scanner", None)
        self._last_scan_result = None

    @property
    def scanner_mode(self) -> bool:
        """True when the SolanaScanner service is available."""
        return self._scanner is not None

    async def should_enter(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Dual-mode entry logic:

        SCANNER MODE: Triggers a scan via SolanaScanner. If tokens pass all
        filters, returns a LONG signal with the best token info in metadata.

        ORCHESTRATOR MODE: Standard MA crossover on SOL OHLCV data.

        Params (configurable via strategy params):
        - sma_fast: Fast SMA period (default 20)
        - sma_slow: Slow SMA period (default 40)
        - require_price_above_avg: Require price > avg close (default True)
        - min_score: Min scanner score to trigger entry (default 60, scanner mode)
        """
        p = self.config.params

        # ---- Scanner Mode ----
        if self.scanner_mode:
            return await self._scanner_entry(p)

        # ---- Orchestrator Mode (SOL trend following) ----
        return await self._trend_entry(data, p)

    async def _scanner_entry(self, p: dict) -> Optional[Signal]:
        """Run a scan and return signal if any tokens pass."""
        min_score = p.get("min_score", 60)

        try:
            result = await self._scanner.run_scan()
            self._last_scan_result = result

            passed = result.get("passed_tokens", [])
            if not passed:
                return None

            # Find the best-scoring token
            best = max(passed, key=lambda t: t.get("score", 0))
            score = best.get("score", 0)

            if score < min_score:
                logger.debug(
                    "[SolSniper] Best token score %.1f < min_score %.1f",
                    score, min_score,
                )
                return None

            # Auto-buy the best token in paper mode
            buy_result = await self._scanner.paper_buy(best["address"])

            if buy_result.get("status") != "bought":
                logger.debug("[SolSniper] Paper buy failed: %s", buy_result.get("message"))
                return None

            strength = min(score / 100, 0.95)
            token_name = best.get("name", best["address"][:8])
            token_symbol = best.get("symbol", "???")

            logger.info(
                "[SolSniper] Scanner entry: %s (%s) score=%.1f | mc=$%s | liq=$%s",
                token_symbol, best["address"][:8], score,
                best.get("market_cap", "?"), best.get("liquidity", "?"),
            )

            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=best.get("price", 0),
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"Solana scanner LONG: {token_symbol} ({token_name}) "
                    f"score={score:.0f} mc=${best.get('market_cap', 0):,.0f} "
                    f"liq=${best.get('liquidity', 0):,.0f}"
                ),
                metadata={
                    "mode": "scanner",
                    "token_address": best["address"],
                    "token_name": token_name,
                    "token_symbol": token_symbol,
                    "score": score,
                    "market_cap": best.get("market_cap"),
                    "liquidity": best.get("liquidity"),
                    "trade_1h": best.get("trade_1h"),
                    "buy_pct": best.get("buy_pct"),
                    "tokens_scanned": result.get("tokens_fetched", 0),
                    "tokens_passed": result.get("tokens_passed", 0),
                },
            )

        except Exception as e:
            logger.error("[SolSniper] Scanner entry error: %s", e, exc_info=True)
            return None

    async def _trend_entry(self, data: pd.DataFrame, p: dict) -> Optional[Signal]:
        """Standard MA crossover entry on SOL OHLCV data."""
        sma_fast = p.get("sma_fast", 20)
        sma_slow = p.get("sma_slow", 40)
        require_above_avg = p.get("require_price_above_avg", True)

        min_len = sma_slow + 5
        if len(data) < min_len:
            return None

        data = self._add_sniper_indicators(data, sma_fast, sma_slow)
        current = data.iloc[-1]
        prev = data.iloc[-2]
        price = current["close"]

        ma_fast = current.get("sma_fast", None)
        ma_slow = current.get("sma_slow", None)

        if ma_fast is None or ma_slow is None or pd.isna(ma_fast) or pd.isna(ma_slow):
            return None

        # Technical conditions from the original sniper
        price_above_fast = price > ma_fast
        price_above_slow = price > ma_slow
        fast_above_slow = ma_fast > ma_slow

        # Price momentum: current close > previous close
        price_rising = price > prev["close"]

        # Price above average close (original sniper check)
        if require_above_avg:
            lookback = min(len(data), 120)
            avg_close = data["close"].iloc[-lookback:].mean()
            if price <= avg_close:
                return None

        # Higher highs / higher lows check (simplified)
        recent = data.iloc[-10:]
        highs_rising = recent["high"].iloc[-1] > recent["high"].iloc[0]
        lows_rising = recent["low"].iloc[-1] > recent["low"].iloc[0]

        # Combined signal: at least 2 of 3 MA conditions + momentum
        ma_score = sum([price_above_fast, price_above_slow, fast_above_slow])

        if ma_score >= 2 and price_rising and (highs_rising or lows_rising):
            strength = min(0.5 + ma_score * 0.15, 0.95)
            logger.info(
                "[SolSniper] Trend entry: price %.6f > MA%d=%.6f, MA%d=%.6f, "
                "momentum=%s, strength=%.2f",
                price, sma_fast, ma_fast, sma_slow, ma_slow, price_rising, strength,
            )
            return Signal(
                signal_type=SignalType.LONG,
                symbol=self.config.symbol,
                price=price,
                size_usd=self.config.size_usd,
                strength=strength,
                reason=(
                    f"Solana sniper LONG: price {price:.6f} above MA{sma_fast}={ma_fast:.6f}, "
                    f"MA{sma_slow}={ma_slow:.6f}, trend confirmed"
                ),
                metadata={
                    "mode": "trend",
                    "sma_fast": ma_fast,
                    "sma_slow": ma_slow,
                    "ma_score": ma_score,
                    "highs_rising": highs_rising,
                    "lows_rising": lows_rising,
                },
            )

        return None

    async def should_exit(
        self, data: pd.DataFrame, position: Dict[str, Any]
    ) -> Optional[Signal]:
        p = self.config.params
        sell_at_multiple = p.get("sell_at_multiple", 9.0)
        stop_loss_pct = p.get("stop_loss_pct", -0.6)
        sma_fast = p.get("sma_fast", 20)
        sma_slow = p.get("sma_slow", 40)

        # In scanner mode, also update scanner positions
        if self.scanner_mode:
            try:
                await self._scanner.update_positions()
            except Exception as e:
                logger.debug("[SolSniper] Scanner position update error: %s", e)

        data = self._add_sniper_indicators(data, sma_fast, sma_slow)
        current = data.iloc[-1]
        price = current["close"]

        is_long = position.get("is_long", True)
        entry_px = position.get("entry_px", position.get("entry_price", price))
        pnl_pct = position.get("pnl_perc", 0)

        if not is_long:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.config.symbol,
                reason="Sniper only goes long, closing short",
            )

        # Take profit: price >= entry * sell_at_multiple
        if entry_px > 0:
            current_multiple = price / entry_px
            if current_multiple >= sell_at_multiple:
                return Signal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.config.symbol,
                    reason=f"Sniper TP: {current_multiple:.1f}x >= {sell_at_multiple}x target",
                )

        # Stop loss
        if pnl_pct <= stop_loss_pct * 100:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Sniper SL: PnL {pnl_pct:.1f}% <= {stop_loss_pct*100:.0f}%",
            )

        # MA crossover exit: price falls below slow MA (trend broken)
        ma_slow_val = current.get("sma_slow", None)
        if ma_slow_val is not None and not pd.isna(ma_slow_val) and price < ma_slow_val:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Sniper trend exit: price {price:.6f} below MA{sma_slow}={ma_slow_val:.6f}",
            )

        # General PnL targets from config
        if pnl_pct >= self.config.target_pct:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Target: {pnl_pct:.1f}%",
            )
        if pnl_pct <= self.config.max_loss_pct:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.config.symbol,
                reason=f"Max loss: {pnl_pct:.1f}%",
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Extended stats including scanner info when in scanner mode."""
        stats = super().get_stats()

        stats["scanner_mode"] = self.scanner_mode
        if self.scanner_mode:
            try:
                scanner_stats = self._scanner.get_stats()
                stats["scanner"] = scanner_stats
            except Exception:
                stats["scanner"] = {"error": "could not fetch scanner stats"}

        if self._last_scan_result:
            stats["last_scan"] = {
                "tokens_fetched": self._last_scan_result.get("tokens_fetched", 0),
                "tokens_passed": self._last_scan_result.get("tokens_passed", 0),
                "scan_time_seconds": self._last_scan_result.get("scan_time_seconds", 0),
            }

        return stats

    @staticmethod
    def _add_sniper_indicators(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
        if "sma_fast" in df.columns and "sma_slow" in df.columns:
            return df
        df = df.copy()
        df["sma_fast"] = df["close"].rolling(window=fast).mean()
        df["sma_slow"] = df["close"].rolling(window=slow).mean()
        return df
