"""Tests for the pre-committed ENSO primary analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_enso import analyze

ENSO_EFFECT_PER_K = 0.0525
TARGET = -0.1


def synth_results(rng, arm_compensation, n_ics=20, noise=0.004, amps=None):
    """Compensation in [0, 1]: fraction of the amp-driven miss removed.

    'ffmean' (if present) is handled specially: it compensates the MEAN
    amplitude exactly, so its signed miss is proportional to (A - mean(A)).
    """
    per_ic = {}
    if amps is None:
        half = rng.uniform(0.5, 2.0, n_ics // 2)
        amps = np.concatenate([half, 2.5 - half])
    effect = ENSO_EFFECT_PER_K * amps
    for name, comp in arm_compensation.items():
        if name == "ffmean":
            miss = ENSO_EFFECT_PER_K * (amps - amps.mean())
        else:
            miss = effect * (1.0 - comp)
        dsst = TARGET + miss + noise * rng.standard_normal(n_ics)
        per_ic[name] = {"dsst_10d": dsst}
    return {"per_ic": per_ic, "enso_amps": list(amps),
            "config": {"target_cooling": TARGET}}


class AnalyzeEnsoTest(unittest.TestCase):
    def test_detects_feedback_wins(self):
        rng = np.random.default_rng(1)
        res = synth_results(rng, {
            "static": 0.0, "ffmean": None, "pienso": 0.9,
            "imitation_s62": 0.85, "imitation_s63": 0.85,
            "imitation_s64": 0.85,
        })
        out = analyze(res)
        self.assertTrue(out["primary"]["H6_pienso_vs_static"]
                        ["significant_holm"])
        self.assertTrue(out["primary"]["H7_imitation_vs_static"]
                        ["significant_holm"])
        self.assertIn("pienso_vs_ffmean", out["secondary"])
        self.assertIn("stratified_H7[A>=median]", out["secondary"])
        # feedback must also beat the mean-feedforward schedule
        self.assertLess(out["secondary"]["pienso_vs_ffmean"]["mean"], 0)

    def test_null_when_imitation_matches_pi(self):
        rng = np.random.default_rng(2)
        res = synth_results(rng, {
            "static": 0.0, "pienso": 0.9,
            "imitation_s62": 0.9, "imitation_s63": 0.9,
            "imitation_s64": 0.9,
        }, noise=0.003)
        out = analyze(res)
        rep = out["secondary"]["imitation_vs_pienso"]
        self.assertFalse(rep["significant"])
        self.assertLess(rep["equivalence_bound_95"], 0.01)

    def test_direction_gate_blocks_wrong_side_wins(self):
        rng = np.random.default_rng(3)
        res = synth_results(rng, {
            "static": 0.9, "pienso": 0.0,        # pi WORSE than static
            "imitation_s62": 0.0, "imitation_s63": 0.0,
            "imitation_s64": 0.0,
        })
        out = analyze(res)
        self.assertFalse(out["primary"]["H6_pienso_vs_static"]
                         ["significant_holm"])
        self.assertFalse(out["primary"]["H7_imitation_vs_static"]
                         ["significant_holm"])

    def test_seed_pooling(self):
        rng = np.random.default_rng(4)
        res = synth_results(rng, {
            "static": 0.0, "pienso": 0.9,
            "imitation_s62": 0.7, "imitation_s63": 0.9,
            "imitation_s64": 0.8,
        }, noise=1e-6)
        out = analyze(res)
        self.assertEqual(out["imitation_arms"],
                         ["imitation_s62", "imitation_s63", "imitation_s64"])
        # pooled imitation error = mean of per-seed errors (comp 0.8 avg)
        im = out["arm_mean_err_mK"]
        pooled = np.mean([im["imitation_s62"], im["imitation_s63"],
                          im["imitation_s64"]])
        h7_err = out["primary"]["H7_imitation_vs_static"]
        st_err = im["static"]
        self.assertAlmostEqual(h7_err["mean"] * 1000, pooled - st_err,
                               places=1)


if __name__ == "__main__":
    unittest.main()
