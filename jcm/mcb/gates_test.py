"""Tests for the control-relative, significance-tested gates (gates.py).

These encode the PREREGISTRATION.md decision rules on synthetic per-IC data,
so the statistical logic is verified without running the model — the audit
found the project had no test of the very decisions that drove the roadmap.
"""

import unittest

import numpy as np

from jcm.mcb.gates import (
    cooling_gate,
    improvement_gate,
    learned,
    no_worse_gate,
    paired_stats,
)


class TestPairedStats(unittest.TestCase):
    def test_significant_difference(self):
        # policy consistently 0.05 above static, tiny scatter -> significant.
        static = np.zeros(10)
        policy = np.full(10, 0.05) + np.linspace(-1e-3, 1e-3, 10)
        st = paired_stats(policy, static)
        self.assertEqual(st["n"], 10)
        self.assertTrue(st["significant"])
        self.assertAlmostEqual(st["mean"], 0.05, places=3)

    def test_within_noise_is_not_significant(self):
        # Zero-centred difference with real scatter -> NOT significant (a coin
        # flip). This is the audit's Gate-3 situation.
        rng = np.random.default_rng(0)
        static = rng.normal(0, 0.03, 12)
        policy = static + rng.normal(0, 0.03, 12)  # no systematic offset
        st = paired_stats(policy, static)
        self.assertFalse(st["significant"])

    def test_n1_is_infinite_se(self):
        st = paired_stats([0.1], [0.0])
        self.assertEqual(st["se"], float("inf"))
        self.assertFalse(st["significant"])  # a single IC cannot support a verdict

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_stats([0.1, 0.2], [0.0])


class TestCoolingGate(unittest.TestCase):
    def test_in_band_and_powered_passes(self):
        # Tight cluster inside [-0.12, -0.08].
        g = cooling_gate([-0.101, -0.099, -0.100, -0.102, -0.098])
        self.assertEqual(g["verdict"], "PASS")

    def test_wide_scatter_is_underpowered(self):
        # Mean in band but per-IC scatter as wide as the band -> cannot resolve.
        g = cooling_gate([-0.02, -0.18, -0.05, -0.15, -0.10])
        self.assertEqual(g["verdict"], "underpowered")

    def test_out_of_band_but_powered_fails(self):
        g = cooling_gate([-0.161, -0.159, -0.160, -0.162, -0.158])
        self.assertEqual(g["verdict"], "FAIL")


class TestImprovementGate(unittest.TestCase):
    def test_clear_improvement_passes(self):
        # policy error much smaller than static, low scatter.
        static_err = np.full(10, 0.05) + np.linspace(-1e-3, 1e-3, 10)
        policy_err = np.full(10, 0.01) + np.linspace(-1e-3, 1e-3, 10)
        g = improvement_gate(policy_err, static_err)
        self.assertEqual(g["verdict"], "PASS")
        self.assertGreater(g["improvement"], 0)

    def test_marginal_improvement_is_underpowered(self):
        # 4% margin swamped by 0.03 scatter (the real Stage-3 situation).
        rng = np.random.default_rng(1)
        static_err = np.abs(rng.normal(0.05, 0.03, 6))
        policy_err = static_err - 0.002 + rng.normal(0, 0.03, 6)
        g = improvement_gate(policy_err, static_err)
        self.assertEqual(g["verdict"], "underpowered")


class TestNoWorseGate(unittest.TestCase):
    def test_significantly_better_passes(self):
        static_loss = np.full(10, 0.02) + np.linspace(-1e-4, 1e-4, 10)
        policy_loss = np.full(10, 0.015) + np.linspace(-1e-4, 1e-4, 10)
        g = no_worse_gate(policy_loss, static_loss)
        self.assertEqual(g["verdict"], "PASS")

    def test_significantly_worse_fails(self):
        static_loss = np.full(10, 0.015) + np.linspace(-1e-4, 1e-4, 10)
        policy_loss = np.full(10, 0.023) + np.linspace(-1e-4, 1e-4, 10)
        g = no_worse_gate(policy_loss, static_loss)
        self.assertEqual(g["verdict"], "FAIL")

    def test_within_noise_is_underpowered_not_pass(self):
        # PREREGISTRATION.md section 5: a margin within 2 s.e. must be
        # reported as underpowered, never converted into a PASS (the old
        # "PASS (not sig. worse)" label was an unregistered non-inferiority
        # framing — 2026-07-29 meta-audit).
        rng = np.random.default_rng(2)
        static_loss = np.abs(rng.normal(0.02, 0.005, 8))
        policy_loss = static_loss + rng.normal(0, 0.005, 8)
        g = no_worse_gate(policy_loss, static_loss)
        self.assertIn("underpowered", g["verdict"])
        self.assertNotIn("PASS", g["verdict"])

    def test_paired_stats_reports_formal_tests(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 1.0, 10)
        b = a + rng.normal(0.5, 0.3, 10)
        st = paired_stats(b, a)
        self.assertIn("p_t", st)
        self.assertIn("p_wilcoxon", st)
        self.assertIn("equivalence_bound_95", st)
        self.assertLess(st["p_t"], 0.05)  # 0.5 shift, sd 0.3 -> significant
        self.assertGreater(st["equivalence_bound_95"], abs(st["mean"]))


class TestLearned(unittest.TestCase):
    def test_improvement_below_noise_is_not_learned(self):
        # 5% loss drop with 11% per-epoch noise -> not learning (audit R2).
        g = learned(loss_improvement=0.0015, sigma_compile=0.0037)
        self.assertFalse(g["learned"])

    def test_improvement_above_noise_is_learned(self):
        g = learned(loss_improvement=0.02, sigma_compile=0.0037)
        self.assertTrue(g["learned"])


if __name__ == "__main__":
    unittest.main()
