# Changelog — shared-gate-recall

## 0.1.0 (2026-10-30)
- Initial release: canonical `youden_calibrate` (strategy keyword-only,
  no default; first | midpoint), `bootstrap_tau_interval` (500
  repeats, seed 909 defaults), `FrozenGate`, and `PatternMemory`
  overlap recall (normalize → overlap → winner/probs |m|^2).
- One documented grid convention: grid=101 means 101 candidate taus
  at step 0.01 (0.00–1.00), matching refusing-membrane and differing
  from the 102-point step-1/101 convention of membrane-brain,
  membrane-heart, and world-projection.
- Explicitly not outer-product W-matrix storage: patterns are stored
  as a list and recalled by inner-product overlap.
