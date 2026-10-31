"""Overlap recall over stored patterns — explicitly NOT W-matrix storage.

The membrane repos this module is extracted from sometimes describe
their storage as "outer-product holographic memory". The code in
those repos does not build an outer-product W matrix: it stores
pattern vectors in a list and recalls by inner-product overlap. This
module says what it does and does only that:

- store: patterns are kept as a list, each L2-normalised on store.
- recall: a probe is L2-normalised, overlaps are the inner products
  with each stored pattern, the winner is argmax |overlap|, and the
  reported probabilities are |m|^2 normalised to sum 1 (Born-like
  only in the sense of squared magnitudes; this is classical
  arithmetic, not quantum mechanics).

Edge case, documented not hidden: if every overlap is exactly 0, the
squared magnitudes sum to 0, so probs are all 0.0 and the winner is
the first pattern (index 0). A zero probe or a zero stored pattern
is rejected with ValueError at the point it is given, not silently
normalised.
"""

import math

__all__ = ["normalize", "PatternMemory"]


def normalize(vector):
    """L2-normalise a vector. Raises ValueError on empty/zero input."""
    vector = [float(v) for v in vector]
    if not vector:
        raise ValueError("cannot normalise an empty vector")
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalise a zero or non-finite vector")
    return [v / norm for v in vector]


def _dot(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    return sum(x * y for x, y in zip(a, b))


class PatternMemory:
    """Stored patterns + overlap recall + frozen-gate decision."""

    def __init__(self, patterns=()):
        self._patterns = []
        for pattern in patterns:
            self.store(pattern)

    def store(self, pattern):
        """Append a pattern (normalised on store). Returns its index."""
        self._patterns.append(normalize(pattern))
        return len(self._patterns) - 1

    def __len__(self):
        return len(self._patterns)

    def recall(self, probe):
        """Recall a probe.

        Returns a dict with overlaps (inner products with each
        stored pattern), probs (|m|^2 normalised, summing to 1
        unless every overlap is 0), and winner (argmax |overlap|).
        Raises ValueError if no pattern is stored.
        """
        if not self._patterns:
            raise ValueError("no patterns stored")
        probe = normalize(probe)
        overlaps = [_dot(probe, pattern) for pattern in self._patterns]
        squares = [m * m for m in overlaps]
        total = sum(squares)
        probs = [s / total for s in squares] if total > 0.0 else [0.0] * len(overlaps)
        winner = max(range(len(overlaps)), key=lambda i: abs(overlaps[i]))
        return {"overlaps": overlaps, "probs": probs, "winner": winner}

    def decide(self, probe, gate):
        """Recall + frozen-gate decision on the winner's |overlap|.

        Returns the recall dict plus "accepted" (winner |overlap|
        >= gate.tau) and "tau". The gate is never re-tuned here.
        """
        result = self.recall(probe)
        best = abs(result["overlaps"][result["winner"]])
        result["accepted"] = gate.decide(best)
        result["tau"] = gate.tau
        return result
