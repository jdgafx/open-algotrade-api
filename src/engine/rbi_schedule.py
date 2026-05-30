"""RBI scheduler helper — pure, no DB/scheduler inside, unit-testable.

Callers (lifespan in main.py) are responsible for querying DB rows and
passing supported_types from PARAM_SPACES.
"""

from __future__ import annotations

from typing import Any


def build_rbi_job_specs(
    instances: list[Any],
    supported_types: set[str],
    default_hours_by_tf: dict[str, int] | None = None,
) -> list[dict]:
    """Build RBI scheduler job specs from live StrategyInstance rows.

    Returns a list of dicts {strategy_type, strategy_id, symbol, timeframe, hours},
    one per instance whose strategy_type is optimizer-supported.  Instances with an
    unsupported strategy_type are skipped (caller logs them).  Cadence by timeframe
    so intraday strategies re-optimize more often than daily ones.
    """
    hours_by_tf = default_hours_by_tf or {"5m": 4, "15m": 4, "1h": 8, "4h": 12, "1d": 24}
    specs: list[dict] = []
    for inst in instances:
        if inst.strategy_type not in supported_types:
            continue
        specs.append(
            {
                "strategy_type": inst.strategy_type,
                "strategy_id": inst.id,
                "symbol": inst.symbol,
                "timeframe": inst.timeframe,
                "hours": hours_by_tf.get(inst.timeframe, 8),
            }
        )
    return specs
