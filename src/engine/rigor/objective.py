"""Cost-aware, risk-adjusted fitness for a backtest's trade list (research 05 §6).

score = (sortino - lambda_dd*max_dd_frac - lambda_to*trades_per_day) * wilson
computed on NET-of-cost per-trade returns. Returns 0.0 below the min-trade gate
or when the average net is non-positive.
"""

# VENDORED from meta-repo src/autotuner/backtest/objective.py — do NOT hand-edit.
# Only this import line differs from the source. Parity enforced by
# backend/tests/test_rigor_parity.py. See plan U2 / KTD-2.
from ._constants import (
    MIN_TRADES_FOR_PROMOTION, FITNESS_LAMBDA_DD, FITNESS_LAMBDA_TO,
)

_EPS = 1e-9


def _max_drawdown_frac(nets) -> float:
    """Max drawdown of the cumulative-net equity curve, as a fraction of the
    running peak (peak measured from a 0 baseline; guarded for non-positive peak)."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in nets:
        cum += x
        if cum > peak:
            peak = cum
        drawdown = peak - cum
        denom = peak if peak > _EPS else 1.0
        max_dd = max(max_dd, drawdown / denom)
    return max_dd


def backtest_fitness(trades, span_days, min_trades=None) -> dict:
    """Net-of-cost, risk-adjusted fitness for a backtest's trade list.

    Contract (the live backend graft, plan U3, must honor this):
      - ``trades``: iterable of dicts, each carrying a ``'pnl_net'`` key whose
        value is that trade's PnL ALREADY net of fees + slippage + funding.
        No other key is read; gross-PnL records are a caller error.
      - ``span_days``: calendar span of the trade window, for the turnover
        penalty (``trades_per_day``).
      - ``min_trades``: promotion floor; defaults to ``MIN_TRADES_FOR_PROMOTION``.

    Returns a dict with ``score`` plus the components
    (``trade_count``, ``avg_net``, ``win_rate``, ``sortino``, ``max_dd_frac``,
    ``trades_per_day``). ``score`` is 0.0 below the min-trade gate or when the
    average net is non-positive — a net-negative edge can never score positive.
    """
    if min_trades is None:
        min_trades = MIN_TRADES_FOR_PROMOTION

    nets = [t['pnl_net'] for t in trades]
    trade_count = len(nets)
    if trade_count == 0:
        return {'trade_count': 0, 'avg_net': 0.0, 'win_rate': 0.0, 'sortino': 0.0,
                'max_dd_frac': 0.0, 'trades_per_day': 0.0, 'score': 0.0}

    wins = sum(1 for x in nets if x > 0)
    win_rate = wins / trade_count
    avg_net = sum(nets) / trade_count
    trades_per_day = trade_count / max(span_days, _EPS)
    max_dd_frac = _max_drawdown_frac(nets)

    base = {'trade_count': trade_count, 'avg_net': avg_net, 'win_rate': win_rate,
            'sortino': 0.0, 'max_dd_frac': max_dd_frac,
            'trades_per_day': trades_per_day, 'score': 0.0}

    if trade_count < min_trades or avg_net <= 0:
        return base

    downside_sq = sum(min(0.0, x) ** 2 for x in nets) / trade_count
    downside_dev = downside_sq ** 0.5
    sortino = avg_net / (downside_dev + _EPS)

    # Wilson lower bound at 95% (z=1.96) as a sample-size confidence factor
    z = 1.96
    z_sq = z * z
    p = wins / trade_count
    wilson = (
        p + z_sq / (2 * trade_count)
        - z * (p * (1 - p) / trade_count + z_sq / (4 * trade_count ** 2)) ** 0.5
    ) / (1 + z_sq / trade_count)

    score = (sortino - FITNESS_LAMBDA_DD * max_dd_frac
             - FITNESS_LAMBDA_TO * trades_per_day) * wilson

    base['sortino'] = sortino
    base['score'] = score
    return base
