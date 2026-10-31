"""Canonical Youden-gate calibration + frozen gate.

This module is the single canonical extraction of the refusal-gate
logic that is duplicated across the public sparkainlp-x repos
(refusing-membrane, membrane-brain, membrane-heart, world-projection,
symmetry-quantum-flip-membrane; see the refusal-gate harness for the
vendored comparison). Other repos' READMEs may point here instead of
re-implementing the gate a sixth time.

GRID CONVENTION (exactly one, documented, never silently varied):
``grid`` is the NUMBER OF CANDIDATE TAUS. With ``grid=101`` there are
101 candidate taus, ``tau = k / (grid - 1)`` for ``k = 0..100``, i.e.
step 0.01 exactly, from 0.00 to 1.00 inclusive. This matches
refusing-membrane's convention. It DIFFERS from membrane-brain,
membrane-heart, and world-projection, which scan ``grid + 1`` points
at step ``1 / grid`` (102 points at step 1/101 when grid=101). The
canonical module does not offer both: if you need the 102-point
convention, say so and convert explicitly; do not silently swap it.

TIE-BREAK (also exactly one rule per call, chosen by the caller):
Youden's J = TPR - FPR, accept rule ``score >= tau``. When several
candidate taus share the maximal J (ties within 1e-12):
- ``strategy="first"``: the smallest maximal-J tau.
- ``strategy="midpoint"``: ``(first + last) / 2`` of the maximal-J
  taus, where first/last are the smallest/largest maximal-J
  candidates on the grid. This is NOT the longest-run midpoint used
  by membrane-heart's default; on a single contiguous interval the
  two agree, on split intervals they do not.

``strategy`` is keyword-only with NO default: omitting it is a
TypeError. A frozen tau that does not name its tie-break rule is
under-reported, and this module refuses to produce one.

All of this is classical, synthetic score arithmetic. No gate here
is a medical device, a quantum result, a brain model, or a safety
guarantee for any real system.
"""

import math
import random

__all__ = [
    "youden_calibrate",
    "bootstrap_tau_interval",
    "tpr_fpr_at",
    "FrozenGate",
    "STRATEGIES",
]

STRATEGIES = ("first", "midpoint")


def _validate_scores(in_scores, ood_scores):
    in_scores = list(in_scores)
    ood_scores = list(ood_scores)
    if not in_scores or not ood_scores:
        raise ValueError("calibration needs non-empty in and ood scores")
    if not all(math.isfinite(v) for v in in_scores + ood_scores):
        raise ValueError("calibration scores must be finite")
    return in_scores, ood_scores


def _validate_grid(grid):
    if not isinstance(grid, int) or isinstance(grid, bool) or grid < 2:
        raise ValueError("grid must be an integer >= 2 (number of candidate taus)")
    return grid


def candidate_taus(grid):
    """The grid=101 convention: grid points, step 1/(grid-1)."""
    grid = _validate_grid(grid)
    return [k / (grid - 1) for k in range(grid)]


def tpr_fpr_at(in_scores, ood_scores, tau):
    """TPR and FPR at a frozen tau under the accept rule score >= tau."""
    in_scores = list(in_scores)
    ood_scores = list(ood_scores)
    if not in_scores or not ood_scores:
        raise ValueError("tpr_fpr_at needs non-empty in and ood scores")
    tpr = sum(1 for v in in_scores if v >= tau) / len(in_scores)
    fpr = sum(1 for v in ood_scores if v >= tau) / len(ood_scores)
    return tpr, fpr


def youden_calibrate(in_scores, ood_scores, grid=101, *, strategy):
    """Calibrate a frozen tau by maximising Youden's J.

    Returns (tau, J). ``strategy`` is required and keyword-only:
    "first" or "midpoint" (see module docstring). Raises TypeError
    if omitted, ValueError for an unknown strategy, empty scores,
    non-finite scores, or an invalid grid.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    in_scores, ood_scores = _validate_scores(in_scores, ood_scores)
    taus = candidate_taus(grid)
    best_j = -2.0
    best_taus = []
    for tau in taus:
        tpr, fpr = tpr_fpr_at(in_scores, ood_scores, tau)
        j = tpr - fpr
        if j > best_j + 1e-12:
            best_j, best_taus = j, [tau]
        elif abs(j - best_j) <= 1e-12:
            best_taus.append(tau)
    if strategy == "first":
        return best_taus[0], best_j
    return (best_taus[0] + best_taus[-1]) / 2.0, best_j


def _quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("quantile requires at least one value")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def bootstrap_tau_interval(in_scores, ood_scores, grid=101, *, strategy,
                           repeats=500, seed=909):
    """Deterministic percentile bootstrap interval for the tau.

    Resamples in and ood scores with replacement (``random.Random``
    seeded with ``seed``), recalibrates with the same grid and
    strategy each replicate, and returns the 2.5th/97.5th percentiles
    of the replicate taus. Defaults (repeats=500, seed=909) match the
    world-projection bootstrap convention. ``strategy`` is required
    and keyword-only, as in :func:`youden_calibrate`. Returns a dict
    with lower, upper, half_width, repeats, seed, method, strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if repeats < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    in_scores, ood_scores = _validate_scores(in_scores, ood_scores)
    rng = random.Random(seed)
    taus = []
    for _ in range(repeats):
        in_sample = [in_scores[rng.randrange(len(in_scores))]
                     for _ in range(len(in_scores))]
        ood_sample = [ood_scores[rng.randrange(len(ood_scores))]
                      for _ in range(len(ood_scores))]
        tau, _ = youden_calibrate(in_sample, ood_sample, grid=grid,
                                  strategy=strategy)
        taus.append(tau)
    lower = _quantile(taus, 0.025)
    upper = _quantile(taus, 0.975)
    return {"lower": lower, "upper": upper,
            "half_width": (upper - lower) / 2.0,
            "repeats": repeats, "seed": seed, "strategy": strategy,
            "method": "percentile bootstrap"}


class FrozenGate:
    """A calibrated tau, frozen: decide() never re-tunes it.

    Construct from :func:`youden_calibrate` output. ``tau`` is
    read-only; there is no recalibration method on purpose. A probe
    is accepted iff ``score >= tau``.
    """

    def __init__(self, tau):
        if not math.isfinite(tau):
            raise ValueError("frozen tau must be finite")
        self._tau = float(tau)

    @property
    def tau(self):
        return self._tau

    def decide(self, score):
        """True (accept) iff score >= frozen tau, else False (refuse)."""
        return score >= self._tau
