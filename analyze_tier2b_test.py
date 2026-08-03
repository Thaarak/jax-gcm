"""Tests for the pre-committed Tier-2b primary analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_tier2b import analyze


def synth_results(rng, arm_compensation, n_ics=20, noise=0.004, target=-0.1,
                  etas=None):
    """Compensation in [0,1]: fraction of the eta-induced error removed."""
    per_ic = {}
    if etas is None:
        half = rng.uniform(0.6, 1.4, n_ics // 2)
        etas = np.concatenate([half, 2.0 - half])
    for name, comp in arm_compensation.items():
        dsst = target * (etas + (1 - etas) * comp) \
            + noise * rng.standard_normal(n_ics)
        per_ic[name] = {"dsst_10d": dsst}
    return {"per_ic": per_ic, "efficacies": list(etas),
            "config": {"target_cooling": target, "band": (-0.12, -0.08)}}


class AnalyzeTier2bTest(unittest.TestCase):
    def test_detects_ft_exceeding_pi(self):
        rng = np.random.default_rng(1)
        res = synth_results(rng, {
            "static": 0.0, "pi": 0.6, "imitation_s52": 0.55,
            "finetuned_s52": 0.9, "finetuned_s53": 0.85,
            "finetuned_s54": 0.88,
        })
        out = analyze(res)
        self.assertTrue(out["primary"]["H4_ft_vs_pi"]["significant_holm"])
        self.assertTrue(out["primary"]["H5_ft_vs_static"]["significant_holm"])
        self.assertIn("imitation_vs_pi", out["secondary"])
        self.assertIn("stratified_ft_vs_pi[eta<1]", out["secondary"])

    def test_ft_equals_pi_is_null_with_bound(self):
        rng = np.random.default_rng(2)
        res = synth_results(rng, {
            "static": 0.0, "pi": 0.6, "imitation_s52": 0.6,
            "finetuned_s52": 0.6, "finetuned_s53": 0.6, "finetuned_s54": 0.6,
        }, noise=0.003)
        out = analyze(res)
        h4 = out["primary"]["H4_ft_vs_pi"]
        self.assertFalse(h4["significant_holm"])
        self.assertLess(h4["equivalence_bound_95"], 0.008)
        # H5 must still fire: matching PI means crushing static.
        self.assertTrue(out["primary"]["H5_ft_vs_static"]["significant_holm"])

    def test_direction_gate(self):
        # Fine-tuning that DESTROYS the init (worse than PI) must not be
        # reported as a significant H4 win.
        rng = np.random.default_rng(3)
        res = synth_results(rng, {
            "static": 0.0, "pi": 0.8, "imitation_s52": 0.75,
            "finetuned_s52": 0.1, "finetuned_s53": 0.1, "finetuned_s54": 0.1,
        })
        out = analyze(res)
        h4 = out["primary"]["H4_ft_vs_pi"]
        self.assertLess(h4["p_holm"], 0.05)
        self.assertFalse(h4["significant_holm"])  # wrong direction


if __name__ == "__main__":
    unittest.main()
