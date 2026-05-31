"""Tests for StrategyOrchestrator.update_live_params / has_open_position.

These cover the "broken last inch" fix: a parameter change must reach the LIVE
in-memory strategy object (not just the DB row), with open-position safety so a
tightened exit threshold never force-closes a position that is currently open.
"""
from unittest.mock import MagicMock

import pytest

from src.engine.orchestrator import StrategyOrchestrator


def _make_orch(strategy_types=None):
    orch = StrategyOrchestrator(client=MagicMock(), executor=MagicMock())
    # real dict so has_open_position is deterministic (no open positions by default)
    orch.executor._positions = {}
    if strategy_types is not None:
        orch._strategy_types = strategy_types
    return orch


def _add_running(orch, name, cfg):
    strat = MagicMock()
    strat.config = cfg
    orch._strategies[name] = strat
    return strat


def test_update_live_params_applies_size_usd_to_running_strategy(make_strategy_config):
    """Tracer: a size_usd change mutates the live config object immediately."""
    orch = _make_orch()
    cfg = make_strategy_config(name="rsi-btc", size_usd=100.0)
    _add_running(orch, "rsi-btc", cfg)

    result = orch.update_live_params("rsi-btc", {"size_usd": 250.0})

    assert cfg.size_usd == 250.0
    assert "size_usd" in result["applied"]
    assert result["running"] is True


def _open_position(strategy_name, symbol="BTC"):
    pos = MagicMock()
    pos.strategy_name = strategy_name
    pos.symbol = symbol
    return pos


def test_exit_threshold_deferred_when_position_open(make_strategy_config):
    """Tightening max_loss_pct while a position is open must DEFER, not apply —
    otherwise should_exit() force-closes the open leg on the next bar."""
    orch = _make_orch()
    cfg = make_strategy_config(name="adx-eth", max_loss_pct=-8.0)
    _add_running(orch, "adx-eth", cfg)
    orch.executor._positions = {"adx-eth:ETH": _open_position("adx-eth", "ETH")}

    result = orch.update_live_params("adx-eth", {"max_loss_pct": -3.0})

    assert cfg.max_loss_pct == -8.0, "exit threshold must NOT change while position open"
    assert "max_loss_pct" in result["deferred"]
    assert "max_loss_pct" not in result["applied"]


def test_exit_threshold_applied_when_flat(make_strategy_config):
    """With no open position, an exit-threshold change applies immediately."""
    orch = _make_orch()
    cfg = make_strategy_config(name="adx-eth", max_loss_pct=-8.0)
    _add_running(orch, "adx-eth", cfg)
    orch.executor._positions = {}

    result = orch.update_live_params("adx-eth", {"max_loss_pct": -5.0})

    assert cfg.max_loss_pct == -5.0
    assert "max_loss_pct" in result["applied"]


def test_exit_threshold_applied_when_forced_despite_open_position(make_strategy_config):
    """force=True overrides the open-position deferral (explicit operator intent)."""
    orch = _make_orch()
    cfg = make_strategy_config(name="adx-eth", target_pct=9.0)
    _add_running(orch, "adx-eth", cfg)
    orch.executor._positions = {"adx-eth:ETH": _open_position("adx-eth", "ETH")}

    result = orch.update_live_params("adx-eth", {"target_pct": 4.0}, force=True)

    assert cfg.target_pct == 4.0
    assert "target_pct" in result["applied"]


def test_forward_fields_apply_even_with_open_position(make_strategy_config):
    """leverage/size_usd are forward-looking — safe to apply with a position open."""
    orch = _make_orch()
    cfg = make_strategy_config(name="adx-eth", leverage=3, size_usd=100.0)
    _add_running(orch, "adx-eth", cfg)
    orch.executor._positions = {"adx-eth:ETH": _open_position("adx-eth", "ETH")}

    result = orch.update_live_params("adx-eth", {"leverage": 5, "size_usd": 200.0})

    assert cfg.leverage == 5 and cfg.size_usd == 200.0
    assert set(result["applied"]) == {"leverage", "size_usd"}
    assert result["deferred"] == []


def test_restart_required_fields_reported_not_applied(make_strategy_config):
    """symbol/timeframe/interval_seconds are loop-local — never hot-applied."""
    orch = _make_orch()
    cfg = make_strategy_config(name="rsi-btc", symbol="BTC", interval_seconds=30)
    _add_running(orch, "rsi-btc", cfg)

    result = orch.update_live_params("rsi-btc", {"symbol": "ETH", "interval_seconds": 60})

    assert cfg.symbol == "BTC" and cfg.interval_seconds == 30
    assert set(result["restart_required"]) == {"symbol", "interval_seconds"}
    assert result["applied"] == []


def test_params_are_merged_not_replaced(make_strategy_config):
    """params is shallow-merged into the existing config.params (RBI write-back shape)."""
    orch = _make_orch()
    cfg = make_strategy_config(name="rsi-btc", params={"rsi_period": 14, "keep": 1})
    _add_running(orch, "rsi-btc", cfg)

    result = orch.update_live_params("rsi-btc", {"params": {"rsi_period": 21}})

    assert cfg.params == {"rsi_period": 21, "keep": 1}
    assert "params" in result["applied"]


def test_non_running_strategy_is_noop(make_strategy_config):
    """Updating a strategy the orchestrator isn't running returns running=False."""
    orch = _make_orch()
    result = orch.update_live_params("ghost", {"size_usd": 999.0})
    assert result == {"applied": [], "deferred": [], "restart_required": [], "running": False}


def test_has_open_position_detects_by_strategy_name(make_strategy_config):
    orch = _make_orch()
    orch.executor._positions = {
        "adx-eth:ETH": _open_position("adx-eth", "ETH"),
        "rsi-btc:BTC": _open_position("rsi-btc", "BTC"),
    }
    assert orch.has_open_position("adx-eth") is True
    assert orch.has_open_position("nope") is False
