"""Tests for the pre-committed Tier-2 primary analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_tier2 import analyze, group_err, holm


def synth_results(rng, arm_effects, n_ics=20, noise=0.005, target=-0.1,
                  etas=None):
    per_ic = {}
    if etas is None:
        etas = rng.uniform(0.6, 1.4, n_ics)
    for name, compensation in arm_effects.items():
        # dsst = target*eta compensated toward target by `compensation` in
        # [0, 1], plus noise: 1.0 = perfect feedback, 0.0 = static-like.
        dsst = target * (etas + (1 - etas) * compensation) \
            + noise * rng.standard_normal(n_ics)
        per_ic[name] = {"dsst_10d": dsst}
    return {"per_ic": per_ic, "efficacies": list(etas),
            "config": {"target_cooling": target, "band": (-0.12, -0.08)}}


class HolmTest(unittest.TestCase):
    def test_holm_adjustment(self):
        adj = holm({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adj["a"], 0.02)   # 2 * 0.01
        self.assertAlmostEqual(adj["b"], 0.04)   # max(0.02, 1 * 0.04)
        adj2 = holm({"a": 0.03, "b": 0.02})
        self.assertAlmostEqual(adj2["b"], 0.04)
        self.assertAlmostEqual(adj2["a"], 0.04)  # monotonicity enforced


class AnalyzeTest(unittest.TestCase):
    def test_detects_real_feedback_advantage(self):
        rng = np.random.default_rng(1)
        res = synth_results(rng, {
            "static": 0.0,
            "openloop_s42": 0.05, "openloop_s43": 0.0, "openloop_s44": 0.02,
            "feedback_s42": 0.8, "feedback_s43": 0.7, "feedback_s44": 0.75,
            "pi": 0.5,
        })
        out = analyze(res)
        self.assertTrue(out["primary"]["H2_nn_vs_static"]["significant_holm"])
        self.assertTrue(out["primary"]["H3_nn_vs_openloop"]["significant_holm"])
        self.assertLess(out["primary"]["H2_nn_vs_static"]["mean"], 0)
        self.assertIn("nn_vs_pi", out["secondary"])
        self.assertIn("stratified[eta<1]", out["secondary"])

    def test_null_is_not_significant_and_bounded(self):
        rng = np.random.default_rng(2)
        res = synth_results(rng, {
            "static": 0.0,
            "openloop_s42": 0.0, "openloop_s43": 0.0, "openloop_s44": 0.0,
            "feedback_s42": 0.0, "feedback_s43": 0.0, "feedback_s44": 0.0,
        }, noise=0.003)
        out = analyze(res)
        h2 = out["primary"]["H2_nn_vs_static"]
        self.assertFalse(h2["significant_holm"])
        self.assertLess(h2["equivalence_bound_95"], 0.01)

    def test_pooling_is_per_ic_mean_not_concatenation(self):
        rng = np.random.default_rng(3)
        res = synth_results(rng, {
            "static": 0.0,
            "feedback_s42": 0.5, "feedback_s43": 0.5, "feedback_s44": 0.5,
            "openloop_s42": 0.0, "openloop_s43": 0.0, "openloop_s44": 0.0,
        }, n_ics=12)
        err = group_err(res["per_ic"],
                        ["feedback_s42", "feedback_s43", "feedback_s44"],
                        -0.1)
        self.assertEqual(err.shape, (12,))  # n = ICs, never ICs x seeds
        out = analyze(res)
        self.assertEqual(out["n_ics"], 12)

    def test_significance_requires_correct_direction(self):
        # A feedback group significantly WORSE than static must not be
        # reported as a Holm-significant PASS.
        rng = np.random.default_rng(4)
        res = synth_results(rng, {
            "static": 0.8,
            "feedback_s42": 0.0, "feedback_s43": 0.0, "feedback_s44": 0.0,
            "openloop_s42": 0.8, "openloop_s43": 0.8, "openloop_s44": 0.8,
        })
        out = analyze(res)
        h2 = out["primary"]["H2_nn_vs_static"]
        self.assertLess(h2["p_holm"], 0.05)      # significant difference...
        self.assertFalse(h2["significant_holm"])  # ...but wrong direction


if __name__ == "__main__":
    unittest.main()
