"""Vendored constants for the rigor package — the subset of meta-repo
`src/autotuner/config.py` that `validation.py` + `objective.py` need.

Kept in sync with the meta-repo source by `backend/tests/test_rigor_parity.py`
(which asserts these values match `src/autotuner/config.py`). Do not drift these
without changing the canonical source first.
"""

# Promotion floor — minimum trades before a fitness score is non-zero.
MIN_TRADES_FOR_PROMOTION = 30

# Combinatorial purged CV (C(6,2) = 15 paths).
CPCV_N_SPLITS = 6
CPCV_N_TEST_SPLITS = 2
CPCV_EMBARGO_FRAC = 0.01          # embargo 1% of samples after each test block
CPCV_GAP_BARS = 5                 # purge gap ~ max holding period in bars

# Deflated Sharpe promotion threshold — block live promotion if DSR < this.
# Never lower: a relaxed gate is proven to ship money-losing overfit noise.
DSR_MIN = 0.95

# Net-of-cost fitness penalty weights.
FITNESS_LAMBDA_DD = 1.0           # weight on max-drawdown fraction
FITNESS_LAMBDA_TO = 0.05          # weight on trades-per-day (turnover)
