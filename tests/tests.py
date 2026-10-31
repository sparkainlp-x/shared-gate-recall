"""Tests for shared_gate_recall. Run: python3 tests.py (from package root)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shared_gate_recall import (  # noqa: E402
    FrozenGate,
    PatternMemory,
    bootstrap_tau_interval,
    candidate_taus,
    normalize,
    tpr_fpr_at,
    youden_calibrate,
)

IN_PLATEAU = [0.6, 0.7, 0.8, 0.9, 0.65, 0.75]
OOD_PLATEAU = [0.1, 0.2, 0.3, 0.4, 0.15, 0.35]


class TestStrategyParameter(unittest.TestCase):
    def test_omitting_strategy_is_type_error(self):
        with self.assertRaises(TypeError):
            youden_calibrate(IN_PLATEAU, OOD_PLATEAU)

    def test_unknown_strategy_is_value_error(self):
        with self.assertRaises(ValueError):
            youden_calibrate(IN_PLATEAU, OOD_PLATEAU, strategy="median")

    def test_bootstrap_omitting_strategy_is_type_error(self):
        with self.assertRaises(TypeError):
            bootstrap_tau_interval(IN_PLATEAU, OOD_PLATEAU)

    def test_first_and_midpoint_are_both_accepted(self):
        tau_first, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, strategy="first")
        tau_mid, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, strategy="midpoint")
        self.assertIsInstance(tau_first, float)
        self.assertIsInstance(tau_mid, float)


class TestGridConvention(unittest.TestCase):
    def test_grid_101_is_101_points_step_001(self):
        taus = candidate_taus(101)
        self.assertEqual(len(taus), 101)
        self.assertEqual(taus[0], 0.0)
        self.assertEqual(taus[-1], 1.0)
        for a, b in zip(taus, taus[1:]):
            self.assertAlmostEqual(b - a, 0.01, places=9)

    def test_grid_11_plateau_pins_step(self):
        # 11 points, step 0.1: max-J taus are 0.5 and 0.6 only.
        tau_first, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, grid=11,
                                        strategy="first")
        tau_mid, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, grid=11,
                                      strategy="midpoint")
        self.assertAlmostEqual(tau_first, 0.5, places=9)
        self.assertAlmostEqual(tau_mid, 0.55, places=9)
        self.assertAlmostEqual(tau_mid - tau_first, 0.05, places=9)

    def test_invalid_grid_rejected(self):
        for bad in (1, 0, -5, 101.5, "101"):
            with self.assertRaises(ValueError):
                youden_calibrate(IN_PLATEAU, OOD_PLATEAU, grid=bad,
                                 strategy="first")


class TestPlateauDelta(unittest.TestCase):
    def test_plateau_first_vs_midpoint_delta(self):
        # Max-J taus on the 0.01 grid are 0.41..0.60: first 0.41,
        # midpoint (0.41 + 0.60) / 2 = 0.505, delta 0.095, J = 1.0.
        # The delta is REPORTED here, not hidden by picking one rule.
        tau_first, j_first = youden_calibrate(IN_PLATEAU, OOD_PLATEAU,
                                              strategy="first")
        tau_mid, j_mid = youden_calibrate(IN_PLATEAU, OOD_PLATEAU,
                                          strategy="midpoint")
        self.assertAlmostEqual(tau_first, 0.41, places=9)
        self.assertAlmostEqual(tau_mid, 0.505, places=9)
        self.assertAlmostEqual(tau_mid - tau_first, 0.095, places=9)
        self.assertAlmostEqual(j_first, 1.0, places=9)
        self.assertAlmostEqual(j_mid, 1.0, places=9)

    def test_unique_max_tau_has_zero_delta(self):
        # Single score pair 0.50 vs 0.49: only tau = 0.50 gives J = 1.
        tau_first, _ = youden_calibrate([0.50], [0.49], strategy="first")
        tau_mid, _ = youden_calibrate([0.50], [0.49], strategy="midpoint")
        self.assertAlmostEqual(tau_first, 0.50, places=9)
        self.assertAlmostEqual(tau_mid, 0.50, places=9)
        self.assertAlmostEqual(tau_mid - tau_first, 0.0, places=9)

    def test_identical_scores_give_j_zero(self):
        _, j = youden_calibrate([0.5], [0.5], strategy="first")
        self.assertAlmostEqual(j, 0.0, places=9)

    def test_tpr_fpr_at_known_values(self):
        tpr, fpr = tpr_fpr_at([0.9, 0.8, 0.4], [0.3, 0.2], 0.5)
        self.assertAlmostEqual(tpr, 2 / 3, places=9)
        self.assertAlmostEqual(fpr, 0.0, places=9)

    def test_empty_or_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            youden_calibrate([], OOD_PLATEAU, strategy="first")
        with self.assertRaises(ValueError):
            youden_calibrate(IN_PLATEAU, [], strategy="first")
        with self.assertRaises(ValueError):
            youden_calibrate([0.5, float("nan")], OOD_PLATEAU, strategy="first")
        with self.assertRaises(ValueError):
            youden_calibrate(IN_PLATEAU, [float("inf")], strategy="first")


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        in_scores = [0.7, 0.75, 0.8, 0.65, 0.9, 0.72]
        ood_scores = [0.2, 0.25, 0.3, 0.15, 0.35, 0.28]
        first = bootstrap_tau_interval(in_scores, ood_scores, strategy="midpoint",
                                       repeats=200, seed=909)
        second = bootstrap_tau_interval(in_scores, ood_scores, strategy="midpoint",
                                        repeats=200, seed=909)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower"], first["upper"])
        self.assertEqual(first["repeats"], 200)
        self.assertEqual(first["seed"], 909)
        self.assertEqual(first["strategy"], "midpoint")

    def test_bootstrap_rejects_few_repeats(self):
        with self.assertRaises(ValueError):
            bootstrap_tau_interval(IN_PLATEAU, OOD_PLATEAU, strategy="first",
                                   repeats=99, seed=909)


class TestOverlapRecall(unittest.TestCase):
    def test_identical_probe_wins_with_prob_one(self):
        mem = PatternMemory([[1.0, 0.0], [0.0, 1.0]])
        result = mem.recall([1.0, 0.0])
        self.assertEqual(result["winner"], 0)
        self.assertAlmostEqual(result["overlaps"][0], 1.0, places=9)
        self.assertAlmostEqual(result["probs"][0], 1.0, places=9)
        self.assertAlmostEqual(result["probs"][1], 0.0, places=9)

    def test_diagonal_probe_splits_probs_evenly(self):
        mem = PatternMemory([[1.0, 0.0], [0.0, 1.0]])
        result = mem.recall([1.0, 1.0])
        self.assertAlmostEqual(result["overlaps"][0], 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(result["overlaps"][1], 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(result["probs"][0], 0.5, places=9)
        self.assertAlmostEqual(result["probs"][1], 0.5, places=9)

    def test_probs_sum_to_one_and_scale_invariance(self):
        mem = PatternMemory([[1.0, 2.0, 2.0], [2.0, 0.0, 0.0]])
        small = mem.recall([1.0, 1.0, 0.0])
        big = mem.recall([10.0, 10.0, 0.0])
        self.assertAlmostEqual(sum(small["probs"]), 1.0, places=9)
        self.assertAlmostEqual(small["overlaps"][0], big["overlaps"][0], places=9)
        self.assertAlmostEqual(small["overlaps"][1], big["overlaps"][1], places=9)

    def test_zero_vectors_rejected(self):
        with self.assertRaises(ValueError):
            PatternMemory([[0.0, 0.0]])
        mem = PatternMemory([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            mem.recall([0.0, 0.0])

    def test_empty_memory_rejected(self):
        with self.assertRaises(ValueError):
            PatternMemory().recall([1.0, 0.0])

    def test_normalize_unit_length(self):
        vec = normalize([3.0, 4.0])
        self.assertAlmostEqual(vec[0], 0.6, places=9)
        self.assertAlmostEqual(vec[1], 0.8, places=9)


class TestFrozenGate(unittest.TestCase):
    def test_gate_never_retunes_on_held_out_data(self):
        tau, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, strategy="midpoint")
        gate = FrozenGate(tau)
        held_out = [0.95, 0.05, 0.55, 0.45, 0.99, 0.01] * 20
        for score in held_out:
            gate.decide(score)
        self.assertEqual(gate.tau, tau)

    def test_boundary_accepts_and_below_refuses(self):
        gate = FrozenGate(0.5)
        self.assertTrue(gate.decide(0.5))
        self.assertTrue(gate.decide(0.500001))
        self.assertFalse(gate.decide(0.499999))

    def test_tau_is_read_only(self):
        gate = FrozenGate(0.5)
        with self.assertRaises(AttributeError):
            gate.tau = 0.9

    def test_recall_plus_gate_decision(self):
        mem = PatternMemory([[1.0, 0.0], [0.0, 1.0]])
        tau, _ = youden_calibrate(IN_PLATEAU, OOD_PLATEAU, strategy="midpoint")
        gate = FrozenGate(tau)
        accepted = mem.decide([1.0, 0.0], gate)
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["tau"], tau)
        diagonal = mem.decide([1.0, 1.0], gate)
        # Winner |overlap| 0.707 > tau 0.505, so this diagonal probe
        # is accepted too; the gate checks the score, not the label.
        self.assertTrue(diagonal["accepted"])
        weak = mem.decide([1.0, 0.0], FrozenGate(1.01))
        self.assertFalse(weak["accepted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
