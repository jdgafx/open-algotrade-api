"""U3 — the live optimizer ranks candidates by NET-of-cost fitness.

The optimizer's selection metric (composite_score) is now
backtest_fitness(net_trades, span)["score"] — Sortino − DD − turnover over
commission-net trades — replacing the gross 0.7*Sharpe + 0.3*PF composite.
These tests pin the behavior changes that matter live: a high-turnover
fee-bleeder no longer out-ranks a lean genuine edge, and a net-negative
strategy can never rank positive.
"""

from src.engine.rigor import backtest_fitness


def _net_trades(pnls):
    """The shape the optimizer feeds backtest_fitness: each backtester trade's
    commission-net USD pnl mapped onto the pnl_net contract key."""
    return [{"pnl_net": p} for p in pnls]


def _optimizer_transform(backtester_trades):
    """Exactly the mapping optimizer.optimize() applies to oos.trades — proves
    we feed backtest_fitness the commission-net 'pnl' under the 'pnl_net' key."""
    return [{"pnl_net": t["pnl"]} for t in backtester_trades]


def test_net_negative_strategy_scores_zero():
    """A gross edge smaller than round-trip cost (net-negative) can never rank
    positive — the fee-bleeder failure mode U3 targets. R4."""
    bleeder = _net_trades([0.5, -0.6, 0.4, -0.7, 0.3, -0.8] * 6)  # avg net < 0
    assert backtest_fitness(bleeder, span_days=30.0)["score"] == 0.0


def test_turnover_penalty_lowers_score_for_same_edge():
    """Identical net trades concentrated into a SHORTER span (higher
    trades_per_day) score lower — the turnover penalty is active. R4."""
    trades = _net_trades([3.0] * 24 + [-1.0] * 12)  # 36 net-positive trades
    slow = backtest_fitness(trades, span_days=90.0)["score"]
    fast = backtest_fitness(trades, span_days=15.0)["score"]
    assert slow > fast > 0


def test_lean_edge_outranks_high_turnover_bleeder():
    """The live failure U3 fixes: a lean net-positive strategy must rank above a
    churny net-negative one. Under the OLD gross composite the bleeder's gross
    Sharpe could win; under net fitness it scores 0. R4."""
    lean = _net_trades([4.0] * 20 + [-1.5] * 10)            # net-positive, 30 trades
    churny_bleeder = _net_trades([1.0, -1.2] * 40)          # 80 trades, net-negative
    lean_score = backtest_fitness(lean, span_days=30.0)["score"]
    bleeder_score = backtest_fitness(churny_bleeder, span_days=30.0)["score"]
    assert lean_score > 0
    assert bleeder_score == 0.0
    assert lean_score > bleeder_score


def test_optimizer_feeds_commission_net_pnl_under_contract_key():
    """The optimizer maps backtester trade dicts (key 'pnl', commission-net) onto
    the backtest_fitness 'pnl_net' contract. Scoring the transformed list must
    equal scoring the contract list — a wrong key would silently zero the edge. R4/R7."""
    backtester_trades = [
        {"pnl": 4.0, "pnl_pct": 2.0, "side": "long"},
        {"pnl": -1.5, "pnl_pct": -0.8, "side": "short"},
    ] * 15
    via_optimizer = backtest_fitness(_optimizer_transform(backtester_trades), span_days=30.0)
    via_contract = backtest_fitness(
        _net_trades([t["pnl"] for t in backtester_trades]), span_days=30.0
    )
    assert via_optimizer["score"] == via_contract["score"]
    assert via_optimizer["trade_count"] == 30
    assert via_optimizer["score"] > 0  # net-positive edge survives
