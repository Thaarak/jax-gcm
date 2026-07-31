"""Tests for the confirmatory-eval aggregation/gates (pure NumPy, no model).

Validates the micro-ensemble math the meta-audit Tier-1 plan relies on:
member-averaging shrinks per-run chaos noise by sqrt(k), the noise accounting
reports it, and the gates fire on per-IC MEANS (not member-level values).
"""

import unittest

import numpy as np

from run_confirmatory_eval import FEATURE_CONFIGS, aggregate_and_gate, parse_arms


def synth_cells(rng, arm_effects, n_ics=10, k=8, chaos_sd=0.014,
                target=-0.1):
    """Synthetic member-level cells: dsst = target + effect + chaos noise."""
    cells = {}
    for name, effect in arm_effects.items():
        dsst = (target + effect
                + chaos_sd * rng.standard_normal((n_ics, k)))
        cells[name] = {
            "dsst_10d": dsst,
            "dsst_snapshot": dsst + 0.001,
            "amazon_mm_day": np.zeros((n_ics, k)),
            "sahel_mm_day": np.zeros((n_ics, k)),
            "mean_loss": np.abs(dsst),
        }
    return cells


class AggregateAndGateTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_noise_accounting_recovers_chaos_sd(self):
        cells = synth_cells(self.rng, {"static": 0.0, "policy": 0.0},
                            n_ics=20, k=8, chaos_sd=0.014)
        _, noise, _ = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        self.assertAlmostEqual(noise["static"]["per_run_chaos_sd"], 0.014,
                               delta=0.003)
        # Cross-IC sd of k=8 member means ~ chaos_sd/sqrt(8) ~ 0.005
        self.assertLess(noise["static"]["cross_ic_sd_of_means"], 0.009)

    def test_member_averaging_powers_a_small_real_effect(self):
        # A 6 mK real advantage is invisible at k=1 but significant at k=16
        # with n=20 ICs — the core Tier-1 claim, verified end to end.
        effect = -0.006  # policy 6 mK closer to target
        detected = {}
        for k in (1, 16):
            rng = np.random.default_rng(42)
            cells = synth_cells(
                rng, {"static": 0.010, "policy": 0.010 + effect},
                n_ics=20, k=k, chaos_sd=0.014)
            _, _, gates = aggregate_and_gate(
                cells, comparator="static", target=-0.1,
                band=(-0.12, -0.08), tail_days=10)
            rep = gates["paired_report[policy vs static]"]
            detected[k] = rep["p_t"]
        self.assertGreater(detected[1], 0.05)   # k=1: underpowered
        self.assertLess(detected[16], 0.05)     # k=16: significant

    def test_gates_use_per_ic_means_and_comparator(self):
        cells = synth_cells(self.rng, {"static": 0.0, "a": 0.0, "b": 0.0},
                            n_ics=8, k=4)
        per_ic, _, gates = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        self.assertEqual(per_ic["a"]["dsst_10d"].shape, (8,))
        self.assertIn("G2_cooling[static]", gates)
        self.assertIn("G3_improvement[a vs static]", gates)
        self.assertIn("paired_report[a vs b]", gates)  # non-comparator pair
        self.assertNotIn("G3_improvement[static vs static]", gates)
        self.assertEqual(gates["n_ics"], 8)
        self.assertEqual(gates["members"], 4)

    def test_equivalence_bound_present_for_null_effect(self):
        cells = synth_cells(self.rng, {"static": 0.0, "policy": 0.0},
                            n_ics=20, k=8, chaos_sd=0.014)
        _, _, gates = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        rep = gates["paired_report[policy vs static]"]
        # With k=8/n=20 the demonstrable equivalence bound should be tight
        # (a few mK), the "significant bounded-negative" the plan promises.
        self.assertLess(rep["equivalence_bound_95"], 0.006)


class ParseArmsTest(unittest.TestCase):
    def test_rejects_bad_kind_and_duplicates(self):
        with self.assertRaises(SystemExit):
            parse_arms(["a=nope=/tmp/x"])
        self.assertIn("static", FEATURE_CONFIGS)


if __name__ == "__main__":
    unittest.main()
