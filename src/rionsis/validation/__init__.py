"""Reusable self-consistency and regression-style checks.

These are promoted, reusable versions of sanity checks that used to live
inline in the analysis notebook (e.g. "does the truncated-geometric MLE
recover a known half-life from synthetic geometric samples?"). They are
exercised by ``tests/`` but are also useful standalone, e.g. when
adapting the pipeline to a new data set or detector.
"""

from rionsis.validation.self_tests import check_geometric_mle_recovery

__all__ = ["check_geometric_mle_recovery"]
