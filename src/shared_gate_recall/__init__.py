"""shared_gate_recall: canonical Youden gate + overlap recall."""

from .gate import (
    STRATEGIES,
    FrozenGate,
    bootstrap_tau_interval,
    candidate_taus,
    tpr_fpr_at,
    youden_calibrate,
)
from .recall import PatternMemory, normalize

__version__ = "0.1.0"

__all__ = [
    "youden_calibrate",
    "bootstrap_tau_interval",
    "candidate_taus",
    "tpr_fpr_at",
    "FrozenGate",
    "PatternMemory",
    "normalize",
    "STRATEGIES",
    "__version__",
]
