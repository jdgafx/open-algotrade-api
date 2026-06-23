"""Vendored overfitting-guard + objective rigor for the live promotion gate.

The canonical SOURCE + test home is the meta-repo `src/autotuner/backtest/`. This
is a VENDORED copy because the backend cannot import across the repo boundary on
Railway (the Dockerfile ships only `backend/src/`, and `src` is a shared top-level
package name). The two pure modules carry no SDK / no trading deps (the meta-repo
purity test guarantees it), so vendoring is safe; drift from the source is caught
by `backend/tests/test_rigor_parity.py`. See plan U2 / KTD-2.

Do NOT hand-edit `validation.py` / `objective.py` / `_constants.py` here — change
the meta-repo source, then re-vendor.
"""

from .validation import (
    deflated_sharpe_ratio,
    purged_combinatorial_kfold,
    cpcv_stability,
)
from .objective import backtest_fitness
from ._constants import DSR_MIN, MIN_TRADES_FOR_PROMOTION

__all__ = [
    "deflated_sharpe_ratio",
    "purged_combinatorial_kfold",
    "cpcv_stability",
    "backtest_fitness",
    "DSR_MIN",
    "MIN_TRADES_FOR_PROMOTION",
]
