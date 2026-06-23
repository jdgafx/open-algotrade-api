"""Hand-rolled overfitting guards — no mlfinlab (paywalled as of 2026).

deflated_sharpe_ratio: Bailey & Lopez de Prado (2014). purged_combinatorial_kfold:
combinatorial purged + embargoed CV for time-ordered records.
"""

from itertools import combinations

import numpy as np
from scipy.stats import norm, skew, kurtosis

# VENDORED from meta-repo src/autotuner/backtest/validation.py — do NOT hand-edit.
# Only this import line differs from the source (config constants are local here so
# the module is reachable inside the backend Railway container). Parity enforced by
# backend/tests/test_rigor_parity.py. See plan U2 / KTD-2.
from ._constants import (
    CPCV_N_SPLITS, CPCV_N_TEST_SPLITS, CPCV_EMBARGO_FRAC, CPCV_GAP_BARS,
)

_EULER_GAMMA = 0.5772156649015328606


def deflated_sharpe_ratio(sr_selected: float, sr_trials, returns) -> dict:
    """DSR = P(true SR > SR0), where SR0 is the expected max SR of N noise trials.
    All SRs and returns must be in the SAME (unannualized) per-period units."""
    sr_trials = np.asarray(sr_trials, dtype=float)
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    T = len(returns)
    N = len(sr_trials)
    if T < 2 or N < 1:
        return {'DSR': 0.0, 'SR0': 0.0, 'PSR_null': 0.0, 'T': T, 'N': N,
                'skew': 0.0, 'kurt': 3.0}

    gamma3 = float(skew(returns))
    gamma4 = float(kurtosis(returns, fisher=False))   # Pearson (normal = 3)

    var_sr = float(np.var(sr_trials, ddof=1)) if N > 1 else 0.0
    std_sr = var_sr ** 0.5

    z1 = norm.ppf(1.0 - 1.0 / N) if N > 1 else 0.0
    z2 = norm.ppf(1.0 - 1.0 / (N * np.e)) if N > 1 else 0.0
    SR0 = std_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    moment_denom = max(1.0 - gamma3 * SR0 + ((gamma4 - 1.0) / 4.0) * SR0 ** 2, 1e-12)
    DSR = float(norm.cdf((sr_selected - SR0) * np.sqrt(T - 1) / np.sqrt(moment_denom)))

    md0 = max(1.0 + ((gamma4 - 1.0) / 4.0) * 0.0, 1e-12)
    PSR_null = float(norm.cdf(sr_selected * np.sqrt(T - 1) / np.sqrt(md0)))

    return {'DSR': DSR, 'SR0': SR0, 'PSR_null': PSR_null, 'T': T, 'N': N,
            'skew': gamma3, 'kurt': gamma4}


def purged_combinatorial_kfold(n_samples, n_splits=None, n_test_splits=None,
                               gap=None, embargo_frac=None):
    """Yield (train_idx, test_idx) for each C(n_splits, n_test_splits) path, with
    train samples adjacent to a test block purged (+/- gap) and embargoed after."""
    n_splits = n_splits or CPCV_N_SPLITS
    n_test_splits = n_test_splits or CPCV_N_TEST_SPLITS
    gap = CPCV_GAP_BARS if gap is None else gap
    embargo_frac = CPCV_EMBARGO_FRAC if embargo_frac is None else embargo_frac

    embargo = int(np.floor(n_samples * embargo_frac))
    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_splits)

    for test_block_ids in combinations(range(n_splits), n_test_splits):
        test_block_ids = set(test_block_ids)
        test_idx = np.array([i for b in sorted(test_block_ids) for i in blocks[b]])

        train_idx = []
        for b in range(n_splits):
            if b in test_block_ids:
                continue
            block_inds = blocks[b].copy()
            for tb in test_block_ids:
                t_start, t_end = blocks[tb][0], blocks[tb][-1]
                block_inds = block_inds[
                    ~((block_inds >= t_start - gap) & (block_inds <= t_end + gap))
                ]
                embargo_range = np.arange(t_end + 1, t_end + 1 + embargo)
                block_inds = block_inds[~np.isin(block_inds, embargo_range)]
            train_idx.extend(block_inds.tolist())

        yield np.array(sorted(set(train_idx))), test_idx


def cpcv_stability(per_path_scores) -> dict:
    """Summarize CPCV-path scores: mean/std/min/frac_positive."""
    arr = np.asarray(list(per_path_scores), dtype=float)
    if arr.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'frac_positive': 0.0}
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0)),
            'min': float(arr.min()), 'frac_positive': float((arr > 0).mean())}
