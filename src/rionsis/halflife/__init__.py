"""Half-life measurement from staircase-fit decay events.

Bridges the ``rionsis.levels`` staircase fit output to a decay-time
observable (``extract_terminal_events``), then fits and quantifies the
half-life with a doubly truncated discrete (geometric) maximum-likelihood
model that accounts for hard detection-window truncation.
"""

from rionsis.halflife.events import EVENT_DTYPE, extract_terminal_events
from rionsis.halflife.truncated_geometric import (
    BootstrapSummary,
    GeometricFitResult,
    HalflifeGrid,
    HalflifeScanConfig,
    bootstrap_halflife_grid,
    fit_truncated_geometric_halflife,
    scan_halflife_grid,
    summarize_bootstrap,
)

__all__ = [
    "EVENT_DTYPE",
    "extract_terminal_events",
    "BootstrapSummary",
    "GeometricFitResult",
    "HalflifeGrid",
    "HalflifeScanConfig",
    "bootstrap_halflife_grid",
    "fit_truncated_geometric_halflife",
    "scan_halflife_grid",
    "summarize_bootstrap",
]
