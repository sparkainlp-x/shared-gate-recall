"""Demo: canonical gate + overlap recall, with the plateau delta printed."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from shared_gate_recall import (  # noqa: E402
    FrozenGate,
    PatternMemory,
    bootstrap_tau_interval,
    youden_calibrate,
)

IN = [0.6, 0.7, 0.8, 0.9, 0.65, 0.75]
OOD = [0.1, 0.2, 0.3, 0.4, 0.15, 0.35]


def main():
    tau_first, j = youden_calibrate(IN, OOD, strategy="first")
    tau_mid, _ = youden_calibrate(IN, OOD, strategy="midpoint")
    print(f"plateau: first={tau_first:.4f} midpoint={tau_mid:.4f} "
          f"delta={tau_mid - tau_first:.4f} J={j:.3f}")
    interval = bootstrap_tau_interval(IN, OOD, strategy="midpoint",
                                      repeats=500, seed=909)
    print(f"bootstrap (500, seed 909): [{interval['lower']:.4f}, "
          f"{interval['upper']:.4f}] half-width={interval['half_width']:.4f}")
    mem = PatternMemory([[1.0, 0.0], [0.0, 1.0]])
    gate = FrozenGate(tau_mid)
    for probe in ([1.0, 0.0], [1.0, 1.0], [0.0, 1.0]):
        result = mem.decide(probe, gate)
        print(f"probe={probe} winner={result['winner']} "
              f"probs={[round(p, 3) for p in result['probs']]} "
              f"accepted={result['accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
