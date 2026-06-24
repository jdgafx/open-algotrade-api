"""F5 (R11/R8) — durable, uncapped per-trade ledger + redeploy-proof strategy state.

The pre-F5 executor persisted only a 500-capped JSON snapshot and held per-strategy
circuit-breaker state in memory, so:
  (a) the realized track record that feeds every live-edge / Wilson-CI /
      champion-challenger / de-freeze decision silently truncated past 500 trades
      across redeploys, and
  (b) a tripped circuit breaker (e.g. 5 consecutive losses) reset to zero on every
      redeploy, silently re-enabling a bleeding strategy.

F5 makes both durable on the persistent volume: an append-only JSONL ledger
(uncapped) plus replay-based StrategyState restoration on boot.
"""
import json

import pytest

from src.execution.paper_executor import PaperTradingExecutor, PaperTrade
from src.strategies.base_strategy import BaseStrategy, StrategyConfig


class _FakeStrategy(BaseStrategy):
    """Minimal concrete strategy — only needs real StrategyState + the inherited
    restore_from_pnls/_apply_trade_accounting under test."""

    async def should_enter(self, *args, **kwargs):
        return None

    async def should_exit(self, *args, **kwargs):
        return None


def _exit(i, strategy, pnl):
    return PaperTrade(
        id=i, symbol="BTC", side="long", action="exit", price=100.0,
        size=1.0, size_usd=50.0, pnl=pnl, pnl_pct=pnl / 50.0 * 100,
        reason="t", strategy_name=strategy,
    )


def _entry(i, strategy):
    # Entry rows carry pnl=0 and live in self._trades alongside exits.
    return PaperTrade(
        id=i, symbol="BTC", side="long", action="entry", price=100.0,
        size=1.0, size_usd=50.0, pnl=0.0, pnl_pct=0.0,
        reason="open", strategy_name=strategy,
    )


def _drive(ex, trades):
    """Mirror the two lines of _execute_exit: append to the hot cache AND the
    durable ledger (what production does on every realized exit)."""
    for t in trades:
        ex._trades.append(t)
        ex._append_ledger(t)


def test_ledger_recovers_all_trades_beyond_500_cap(tmp_path, monkeypatch):
    """600 exits → a simulated redeploy recovers all 600 from the ledger, even
    though the JSON snapshot caps at 500. (Covers R11.)"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(i, "bleeder-btc", 10.0 if i % 2 else -10.0) for i in range(1, 601)])
    ex.save_state()

    # The snapshot alone is lossy by design (hot cache).
    snap = json.loads((tmp_path / "paper_state.json").read_text())
    assert len(snap["trades"]) == 500

    # Fresh executor (simulated redeploy) recovers the FULL history from the ledger.
    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    assert ex2.load_state() is True
    assert len(ex2._trades) == 600
    assert ex2._trade_counter >= 600  # counter ahead of every rehydrated id


def test_redeploy_restores_tripped_circuit_breaker(tmp_path, monkeypatch):
    """5 consecutive losses trip the breaker; after a simulated redeploy the
    freshly-built strategy's state is rebuilt from the ledger and the breaker is
    still tripped — it no longer silently resets. (Covers R11/R8.)"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(i, "loser-btc", -10.0) for i in range(1, 6)])
    ex.save_state()

    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    ex2.load_state()
    strat = _FakeStrategy(StrategyConfig(name="loser-btc", symbol="BTC"))
    n = ex2.replay_strategy_state(strat)

    assert n == 5
    assert strat.state.consecutive_losses == 5
    assert strat.state.circuit_breaker_triggered is True  # default max_consecutive_losses=5


def test_replay_excludes_wall_clock_overtrading_state(tmp_path, monkeypatch):
    """Rehydration replays only the durable accounting half — it must NOT impose
    a spurious post-boot cooldown or inflate the hourly trade counter."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(i, "winner-btc", 10.0) for i in range(1, 11)])
    ex.save_state()

    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    ex2.load_state()
    strat = _FakeStrategy(StrategyConfig(name="winner-btc", symbol="BTC"))
    ex2.replay_strategy_state(strat)

    assert strat.state.total_trades == 10
    assert strat.state.consecutive_losses == 0
    assert strat.state.last_trade_close_time is None  # no spurious cooldown imposed
    assert strat.state.trades_this_hour == 0          # hourly counter not inflated


def test_replay_counts_only_realized_exits_not_entry_rows(tmp_path, monkeypatch):
    """Regression: self._trades interleaves entry rows (pnl=0) with exits. The
    durable ledger mirrors self._trades (entries preserved for the trades/stats
    endpoints), but replay_strategy_state must count ONLY exits — otherwise each
    entry row is miscounted as a loss, corrupting win-rate / drawdown / breaker on
    the first post-F5 boot (the seed-from-snapshot migration path)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    # 6 real wins, each preceded by its entry row → 12 rows, 6 of them exits.
    rows = []
    i = 0
    for _ in range(6):
        i += 1; rows.append(_entry(i, "winner-btc"))
        i += 1; rows.append(_exit(i, "winner-btc", 50.0))
    _drive(ex, rows)
    ex.save_state()

    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    ex2.load_state()
    # Faithful mirror: entries + exits all survive in self._trades (stats/endpoint).
    assert len(ex2._trades) == 12

    strat = _FakeStrategy(StrategyConfig(name="winner-btc", symbol="BTC"))
    n = ex2.replay_strategy_state(strat)
    assert n == 6                                       # exits only, not 12
    assert strat.state.total_trades == 6
    assert strat.state.winning_trades == 6
    assert strat.state.losing_trades == 0              # NO phantom losses from entries
    assert strat.state.consecutive_losses == 0
    assert strat.state.circuit_breaker_triggered is False  # a pure winner never trips


def test_replay_is_idempotent_guarded(tmp_path, monkeypatch):
    """A second replay onto an already-populated strategy is a no-op — restore is
    additive, so the guard prevents doubling its counters."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(i, "x-btc", -10.0) for i in range(1, 4)])
    ex.save_state()
    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    ex2.load_state()
    strat = _FakeStrategy(StrategyConfig(name="x-btc", symbol="BTC"))
    assert ex2.replay_strategy_state(strat) == 3
    assert strat.state.total_trades == 3
    assert ex2.replay_strategy_state(strat) == 0       # guarded: no second replay
    assert strat.state.total_trades == 3               # counters not doubled


def test_malformed_ledger_line_is_tolerated(tmp_path, monkeypatch):
    """A junk line and a truncated final line are skipped, not fatal. (Covers KTD-C.)"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(i, "x-btc", 10.0) for i in range(1, 4)])
    ex.save_state()
    with ex._ledger_path().open("a") as fh:
        fh.write("{not valid json\n")          # garbage line
        fh.write('{"id": 99, "symbol": "BTC"')  # truncated, no newline

    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    assert ex2.load_state() is True   # does not throw
    assert len(ex2._trades) == 3      # 3 good recovered, 2 bad skipped


def test_cold_first_boot_has_no_state(tmp_path, monkeypatch):
    """No snapshot and no ledger → clean empty boot, no error. (Covers KTD-C.)"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    assert ex.load_state() is False
    assert ex._trades == []
    strat = _FakeStrategy(StrategyConfig(name="fresh-btc", symbol="BTC"))
    assert ex.replay_strategy_state(strat) == 0  # nothing to replay


def test_ledger_seeded_from_snapshot_when_absent(tmp_path, monkeypatch):
    """First boot after F5 ships: a pre-F5 deploy has a snapshot but no ledger.
    load_state seeds the ledger from the snapshot so future exits append to a
    durable base (cannot recover already-truncated history; stops future loss)."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    pre_f5 = PaperTradingExecutor(initial_balance=1000.0)
    for i in range(1, 4):
        pre_f5._trades.append(_exit(i, "x-btc", 10.0))  # snapshot only, no ledger write
    pre_f5.save_state()
    assert not (tmp_path / "paper_trades.jsonl").exists()

    ex2 = PaperTradingExecutor(initial_balance=1000.0)
    ex2.load_state()
    assert (tmp_path / "paper_trades.jsonl").exists()  # seeded on first F5 boot
    assert len(ex2._load_ledger()) == 3


@pytest.mark.asyncio
async def test_forced_exit_path_appends_to_ledger(tmp_path, monkeypatch):
    """Regression: realized exits from the forced-close paths (risk_stop /
    strategy_disabled / emergency_close), not just _execute_exit, must land in the
    durable ledger — otherwise a risk-stopped strategy's losses vanish on redeploy
    and its tripped breaker silently re-enables. Drives close_by_strategy."""
    from datetime import datetime, timezone
    from src.execution.paper_executor import PaperPosition

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    ex._mid_prices = {"BTC": 90.0}            # exit below entry → realized loss
    ex._last_price_fetch = 9_999_999_999.0    # keep the cache fresh (no network)
    ex._positions["loser-btc:BTC"] = PaperPosition(
        symbol="BTC", side="long", size=1.0, entry_price=100.0,
        entry_time=datetime.now(timezone.utc), leverage=1, size_usd=100.0,
        strategy_name="loser-btc",
    )

    results = await ex.close_by_strategy("loser-btc")
    assert results and results[0].success

    ledger = ex._load_ledger()
    assert len(ledger) == 1
    assert ledger[0].action == "exit"
    assert ledger[0].strategy_name == "loser-btc"
    assert ledger[0].pnl < 0                  # realized loss is durably recorded


def test_durable_writes_target_persistent_volume_only(tmp_path, monkeypatch):
    """Every durable write goes to DATA_DIR (the persistent volume), never /app. (Covers R8.)"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    ex = PaperTradingExecutor(initial_balance=1000.0)
    _drive(ex, [_exit(1, "x-btc", 10.0)])
    assert ex._ledger_path().parent == tmp_path
    assert ex._state_path().parent == tmp_path
    assert (tmp_path / "paper_trades.jsonl").exists()
