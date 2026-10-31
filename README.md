# shared-gate-recall v0.1.0

One canonical module for the Youden refusal gate and overlap recall
that is duplicated across the public sparkainlp-x repos
(refusing-membrane, membrane-brain, membrane-heart, world-projection,
symmetry-quantum-flip-membrane). This package is the extraction other
repos' READMEs can point at. It modifies no published repo.

**SYNTHETIC classical toy only.** No medical use, no quantum
hardware, no brain model, no consciousness claim, and no safety
guarantee for any real system. A gate that refuses is a property of
a toy, not a safety property of a deployment.

## What it is NOT: outer-product storage

Some of the source repos describe their storage as "outer-product
holographic memory". The code does not build an outer-product W
matrix: it stores pattern vectors in a list and recalls by
inner-product overlap. This module does exactly that and says so:

- store: patterns kept as a list, each L2-normalised on store.
- recall: probe L2-normalised, overlaps are inner products, winner
  is argmax |overlap|, probs are |m|^2 normalised to sum 1.
- If every overlap is exactly 0, probs are all 0.0 and the winner
  is the first pattern (documented edge case, not hidden).
- Zero probes and zero stored patterns are rejected with
  ValueError, not silently normalised.

## API

```python
from shared_gate_recall import (
    youden_calibrate, bootstrap_tau_interval, FrozenGate, PatternMemory,
)

tau, j = youden_calibrate(in_scores, ood_scores, strategy="midpoint")
gate = FrozenGate(tau)
mem = PatternMemory([pattern_a, pattern_b])
result = mem.decide(probe, gate)  # recall + accepted / tau, gate untouched
interval = bootstrap_tau_interval(in_scores, ood_scores, strategy="midpoint")
```

- `youden_calibrate(in_scores, ood_scores, grid=101, *, strategy)` —
  returns `(tau, J)` with J = TPR − FPR and accept rule
  `score >= tau`. `strategy` is keyword-only with **no default**:
  omitting it is a TypeError. `"first"` = smallest maximal-J tau;
  `"midpoint"` = `(first + last) / 2` of the maximal-J taus.
- `bootstrap_tau_interval(..., *, strategy, repeats=500, seed=909)` —
  deterministic percentile bootstrap (2.5th/97.5th percentiles of
  replicate taus), defaults matching the world-projection
  convention. Returns lower, upper, half_width, repeats, seed,
  strategy, method.
- `FrozenGate(tau)` — `decide(score)` accepts iff
  `score >= tau`. `tau` is read-only; there is no recalibration
  method, so held-out evaluation cannot silently re-tune the gate.

## The one grid convention

`grid` is the **number of candidate taus**. `grid=101` means 101
candidate taus, `tau = k / (grid - 1)` for `k = 0..100`, step 0.01
exactly, from 0.00 to 1.00 inclusive. This matches
refusing-membrane's convention. It differs from membrane-brain,
membrane-heart, and world-projection, which scan `grid + 1` points
at step `1 / grid` (102 points at step 1/101 when grid=101). This
module implements exactly one convention and states it; it does not
offer both and it does not silently swap between them. The harness
(`refusal-gate-harness`) documents and compares all of them.

## Measured numbers (from this package's own tests and demo)

Plateau probe set (in scores ≥ 0.6, ood scores ≤ 0.4, grid=101):

| rule | tau | J |
|---|---|---|
| first | 0.4100 | 1.000 |
| midpoint | 0.5050 | 1.000 |
| **delta (midpoint − first)** | **0.0950** | — |

The delta is reported, not hidden by picking one rule: on a wide
maximal-J interval the tie-break moves tau by almost a tenth of the
full range. On a unique maximal-J tau the delta is 0 (pinned by
test). Bootstrap on the same plateau set (500 repeats, seed 909):
interval [0.4550, 0.5550], half-width 0.0500 — the same order as the
tie-break delta, so both must be reported together.

## Tests

24 tests, stdlib-only (`python3 tests/tests.py` from the package
root). They pin: strategy omission = TypeError, unknown strategy =
ValueError, the plateau delta above, the grid=11 step check (first
0.5, midpoint 0.55, delta 0.05), grid shape (101 points, step 0.01),
bootstrap determinism (same seed = identical interval) and the
minimum-100-repeats rule, overlap/prob normalisation (probs sum to
1, scale-invariant overlaps), winner/prob behaviour on identical
and diagonal probes, zero-vector and empty-memory rejection, and
frozen-gate behaviour (tau unchanged after held-out decisions,
boundary accepts, tau read-only).

## Files

`src/shared_gate_recall/{__init__,gate,recall}.py`, `demo.py`,
`tests/tests.py`, `LICENSE` (AGPL-3.0-only), `CITATION.cff`,
`.zenodo.json` (Brisson only, licence `{"id":"AGPL-3.0-only"}`),
`CHANGELOG.md`.

## Provenance and credit

Extracted from the gate/recall logic of the public sparkainlp-x
repos listed above, as compared by the refusal-gate harness
(20/20 tests). Project author: Brisson, Jean-François (Spark AI
NLP). Mun Seok Lee is cited as inspiration only for the Digital to
Wave paradigm; no Lee code is included.
