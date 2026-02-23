"""
Refactored Turtle Trading Strategy for Vaults
Based on the classic 55-bar breakout system with ATR-based stops
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TurtleConfig:
    """Configuration for turtle trading strategy"""

    symbol: str
    timeframe: str
    size: Decimal
    lookback_period: int = 55
    atr_period: int = 20
    atr_multiplier: Decimal = Decimal("2.0")
    take_profit_percentage: Decimal = Decimal("0.002")
    leverage: int = 5
    trading_hours_only: bool = True
    exit_friday: bool = True
    max_position_size: Decimal = Decimal("1000")


@dataclass
class Position:
    """Represents a trading position"""

    symbol: str
    size: Decimal
    side: str
    unrealized_pnl: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")


class VaultExecutor(ABC):
    """Abstract base class for vault execution"""

    @abstractmethod
    async def execute_order(
        self,
        side: str,
        size: Decimal,
        price: Optional[Decimal] = None,
        reduce_only: bool = False,
        order_type: str = "limit",
    ) -> bool:
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        pass

    @abstractmethod
    async def get_balance(self) -> Decimal:
        pass


class TurtleStrategy:
    """
    Refactored Turtle Trading System with VaultExecutor abstraction
    Features:
    - 55-bar breakout entry signals
    - 2x ATR stop losses
    - 0.2% take profit targets
    - Time-based trading (9:30 AM - 4:00 PM ET, Mon-Fri)
    - Friday position exit before close
    """

    def __init__(self, executor: VaultExecutor, config: TurtleConfig):
        self.executor = executor
        self.config = config

        self.current_position: Optional[Position] = None
        self.entry_price: Optional[Decimal] = None
        self.entry_time: Optional[float] = None
        self.stop_loss_price: Optional[Decimal] = None
        self.take_profit_price: Optional[Decimal] = None

        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal("0")

        logger.info(
            f"Initialized Turtle Strategy for {config.symbol} on {config.timeframe}"
        )

    async def on_candle(
        self, df: pd.DataFrame, current_price: Decimal, bid: Decimal, ask: Decimal
    ):
        """Process a new candle update"""
        try:
            if self.config.trading_hours_only and not self._is_trading_time():
                logger.debug("Outside trading hours, waiting...")
                return

            if self.config.exit_friday and self._is_friday_close():
                await self._close_friday_position()
                return

            self.current_position = await self.executor.get_position(self.config.symbol)

            df = self._calculate_atr(df)

            if len(df) < self.config.lookback_period + 1:
                logger.warning("Insufficient data for lookback")
                return

            last_candle = df.iloc[-2]

            if not self.current_position or abs(self.current_position.size) == 0:
                await self._check_entry_signals(
                    df, current_price, last_candle, bid, ask
                )
            else:
                await self._manage_position(df, current_price)

        except Exception as e:
            logger.error(f"Error in analysis: {e}")

    async def _check_entry_signals(
        self,
        df: pd.DataFrame,
        current_price: Decimal,
        last_candle: pd.Series,
        bid: Decimal,
        ask: Decimal,
    ):
        try:
            # Use previous completed candles for lookback to avoid repainting
            # and to correctly detect breakout of the PREVIOUS range
            lookback_data = df.iloc[-(self.config.lookback_period + 1) : -1]
            highest_high = lookback_data["high"].max()
            lowest_low = lookback_data["low"].min()

            current_open = df.iloc[-1]["open"]
            current_close = df.iloc[-1]["close"]
            current_atr = df.iloc[-1]["atr"]

            if (
                current_price >= highest_high
                and current_open < highest_high
                and current_close > last_candle["close"]
            ):
                logger.info(
                    f"🟢 LONG SIGNAL: Price {current_price} broke above 55-bar high {highest_high}"
                )
                await self._enter_long(bid, current_atr)

            elif (
                current_price <= lowest_low
                and current_open > lowest_low
                and current_close < last_candle["close"]
            ):
                logger.info(
                    f"🔴 SHORT SIGNAL: Price {current_price} broke below 55-bar low {lowest_low}"
                )
                await self._enter_short(ask, current_atr)

        except Exception as e:
            logger.error(f"Error checking entry signals: {e}")

    async def _enter_long(self, entry_price: Decimal, atr: Decimal):
        try:
            position_size = await self._calculate_position_size(entry_price, atr)

            if position_size == 0:
                logger.warning("Position size calculated as 0, skipping entry")
                return

            success = await self.executor.execute_order(
                side="buy", size=position_size, price=entry_price, order_type="limit"
            )

            if success:
                logger.info(
                    f"✅ Long entry order placed: {position_size} @ {entry_price}"
                )

                self.entry_price = entry_price
                self.entry_time = time.time()
                self.stop_loss_price = entry_price - (atr * self.config.atr_multiplier)
                self.take_profit_price = entry_price * (
                    Decimal("1") + self.config.take_profit_percentage
                )
                self.total_trades += 1

        except Exception as e:
            logger.error(f"Error entering long position: {e}")

    async def _enter_short(self, entry_price: Decimal, atr: Decimal):
        try:
            position_size = await self._calculate_position_size(entry_price, atr)

            if position_size == 0:
                logger.warning("Position size calculated as 0, skipping entry")
                return

            success = await self.executor.execute_order(
                side="sell", size=position_size, price=entry_price, order_type="limit"
            )

            if success:
                logger.info(
                    f"✅ Short entry order placed: {position_size} @ {entry_price}"
                )

                self.entry_price = entry_price
                self.entry_time = time.time()
                self.stop_loss_price = entry_price + (atr * self.config.atr_multiplier)
                self.take_profit_price = entry_price * (
                    Decimal("1") - self.config.take_profit_percentage
                )
                self.total_trades += 1

        except Exception as e:
            logger.error(f"Error entering short position: {e}")

    async def _manage_position(self, df: pd.DataFrame, current_price: Decimal):
        try:
            if not self.current_position:
                return

            current_atr = df.iloc[-1]["atr"]
            self._update_trailing_stop(current_price, current_atr)

            position_side = self.current_position.side

            if position_side == "long":
                if current_price <= self.stop_loss_price:
                    await self._exit_position("stop_loss")
                elif current_price >= self.take_profit_price:
                    await self._exit_position("take_profit")

            elif position_side == "short":
                if current_price >= self.stop_loss_price:
                    await self._exit_position("stop_loss")
                elif current_price <= self.take_profit_price:
                    await self._exit_position("take_profit")

        except Exception as e:
            logger.error(f"Error managing position: {e}")

    def _update_trailing_stop(self, current_price: Decimal, atr: Decimal):
        try:
            if not self.current_position:
                return

            position_side = self.current_position.side

            if position_side == "long":
                new_stop = current_price - (atr * self.config.atr_multiplier)
                if self.stop_loss_price is None or new_stop > self.stop_loss_price:
                    self.stop_loss_price = new_stop
                    logger.debug(f"Updated trailing stop to {self.stop_loss_price}")

            elif position_side == "short":
                new_stop = current_price + (atr * self.config.atr_multiplier)
                if self.stop_loss_price is None or new_stop < self.stop_loss_price:
                    self.stop_loss_price = new_stop
                    logger.debug(f"Updated trailing stop to {self.stop_loss_price}")

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

    async def _exit_position(self, exit_reason: str):
        try:
            if not self.current_position:
                return

            position_size = abs(self.current_position.size)
            position_side = self.current_position.side
            exit_side = "sell" if position_side == "long" else "buy"

            success = await self.executor.execute_order(
                side=exit_side,
                size=position_size,
                order_type="market",
                reduce_only=True,
            )

            if success:
                logger.info(
                    f"✅ Position exited via {exit_reason}: {position_size} {exit_side}"
                )

                if self.current_position.unrealized_pnl > 0:
                    self.winning_trades += 1
                self.total_pnl += self.current_position.unrealized_pnl

                self.current_position = None
                self.entry_price = None
                self.entry_time = None
                self.stop_loss_price = None
                self.take_profit_price = None

        except Exception as e:
            logger.error(f"Error exiting position: {e}")

    async def _calculate_position_size(
        self, entry_price: Decimal, atr: Decimal
    ) -> Decimal:
        try:
            available_balance = await self.executor.get_balance()

            # Ensure atr is Decimal
            if not isinstance(atr, Decimal):
                atr = Decimal(str(atr))

            max_risk_per_trade = min(
                available_balance * Decimal("0.01"),
                atr * self.config.atr_multiplier * self.config.size,
            )

            if atr > 0:
                position_size = max_risk_per_trade / (atr * self.config.atr_multiplier)
            else:
                position_size = self.config.size

            position_size = min(position_size, self.config.max_position_size)

            if position_size < self.config.size * Decimal("0.1"):
                position_size = Decimal("0")

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Decimal("0")

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["previous_close"] = df["close"].shift(1)
            df["high-low"] = abs(df["high"] - df["low"])
            df["high-pc"] = abs(df["high"] - df["previous_close"])
            df["low-pc"] = abs(df["low"] - df["previous_close"])
            df["tr"] = df[["high-low", "high-pc", "low-pc"]].max(axis=1)
            df["atr"] = df["tr"].rolling(window=self.config.atr_period).mean()
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            df["atr"] = 0
            return df

    def _is_trading_time(self) -> bool:
        try:
            current_time = datetime.now()
            et_time = current_time - timedelta(hours=5)

            if et_time.weekday() > 4:
                return False

            current_time_minutes = et_time.hour * 60 + et_time.minute
            start_time = 9 * 60 + 30
            end_time = 16 * 60

            return start_time <= current_time_minutes < end_time
        except Exception as e:
            logger.error(f"Error checking trading time: {e}")
            return True

    def _is_friday_close(self) -> bool:
        try:
            current_time = datetime.now()
            et_time = current_time - timedelta(hours=5)

            if et_time.weekday() == 4:
                current_time_minutes = et_time.hour * 60 + et_time.minute
                close_time = 15 * 60 + 45
                return current_time_minutes >= close_time
            return False
        except Exception as e:
            logger.error(f"Error checking Friday close: {e}")
            return False

    async def _close_friday_position(self):
        try:
            if self.current_position and abs(self.current_position.size) > 0:
                logger.info("📅 Friday close detected - exiting all positions")
                await self._exit_position("friday_close")
        except Exception as e:
            logger.error(f"Error closing Friday position: {e}")
