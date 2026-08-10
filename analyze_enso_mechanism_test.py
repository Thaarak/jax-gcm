"""Tests for the post-hoc ENSO mechanism analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_enso_mechanism import analyze, linregress

TARGET = -0.1
SENS = 0.0525          # K of warming per K of Nino3.4 anomaly


def synth(rng, arm_spec, n_ics=20, k=4, chaos=0.016, bias=0.0):
    """Build an eval-shaped results dict.

    arm_spec maps arm -> (rejection, adapts): `rejection` in [0,1] is the
    fraction of the A-driven warming the arm removes; `adapts` controls
    whether its commanded forcing tracks A.
    """
    amps = np.concatenate([rng.uniform(0.5, 2.0, n_ics // 2)] * 2)[:n_ics]
    cells = {}
    for name, (rejection, adapts) in arm_spec.items():
        warming = SENS * amps * (1.0 - rejection)
        truth = TARGET + bias + warming
        cells[name] = {
            "dsst_10d": truth[:, None] + chaos * rng.standard_normal(
                (n_ics, k)),
            "mean_mcb_forcing": (
                (0.01 + 0.005 * amps)[:, None] * np.ones((1, k)) if adapts
                else np.full((n_ics, k), 0.01)),
        }
    return {"cells": cells, "enso_amps": list(amps)}


def synth_gcal(bias):
    return {"per_ic": {"static": {"dsst_10d": np.array(
        [TARGET + bias] * 6)}}}


class LinregressTest(unittest.TestCase):
    def test_recovers_known_slope(self):
        x = np.linspace(0, 2, 50)
        y = 3.0 * x - 1.0
        r = linregress(x, y)
        self.assertAlmostEqual(r["slope"], 3.0, places=6)
        self.assertAlmostEqual(r["intercept"], -1.0, places=6)
        self.assertLess(r["p"], 1e-6)

    def test_flat_relationship_is_not_significant(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.5, 2.0, 40)
        y = rng.standard_normal(40) * 0.01
        self.assertGreater(linregress(x, y)["p"], 0.05)


class MechanismAnalysisTest(unittest.TestCase):
    def test_detects_disturbance_rejection(self):
        rng = np.random.default_rng(1)
        res = synth(rng, {"static": (0.0, False), "pienso": (0.9, True)},
                    chaos=0.008)
        out = analyze(res, target=TARGET)
        st = out["arms"]["static"]
        pi = out["arms"]["pienso"]
        # static sensitivity ~ 52.5 mK/K; pienso ~ 5 mK/K
        self.assertAlmostEqual(st["sensitivity_mK_per_K"], 52.5, delta=8)
        self.assertLess(abs(pi["sensitivity_mK_per_K"]), 15)
        self.assertGreater(pi["sensitivity_reduction_pct"], 70)
        self.assertTrue(pi["adapts"])
        self.assertFalse(st["adapts"])
        self.assertGreater(pi["forcing_vs_A_r"], 0.9)

    def test_calibration_decomposition_and_debiasing(self):
        rng = np.random.default_rng(2)
        bias = 0.02
        res = synth(rng, {"static": (0.0, False), "ffmean": (0.4, False),
                          "pienso": (0.9, True)}, chaos=0.006, bias=bias)
        out = analyze(res, synth_gcal(bias), target=TARGET)
        self.assertAlmostEqual(out["calibration"]["deficit_mK"], bias * 1000,
                               places=6)
        st = out["arms"]["static"]
        # Debiasing must remove ~the full offset from a blind arm's error.
        self.assertAlmostEqual(st["abs_err_debiased_mK"],
                               st["abs_err_mK"] - bias * 1000, delta=3)
        # And the feedback conclusion must survive it.
        rep = out["debiased_comparisons"]["pienso vs static"]
        self.assertLess(rep["mean"], 0)
        self.assertTrue(rep["significant"])

    def test_residual_scatter_matches_chaos_when_fully_rejected(self):
        rng = np.random.default_rng(3)
        res = synth(rng, {"static": (0.0, False), "pienso": (1.0, True)},
                    n_ics=40, k=4, chaos=0.016)
        out = analyze(res, target=TARGET)
        pi = out["arms"]["pienso"]
        # A perfect rejector's leftover scatter IS the micro-ensemble se.
        self.assertAlmostEqual(pi["residual_sd_after_A_mK"],
                               pi["chaos_se_of_ic_mean_mK"], delta=3.0)
        self.assertGreater(pi["sensitivity_p"], 0.05)

    def test_seed_pooling_of_imitation_arms(self):
        rng = np.random.default_rng(4)
        res = synth(rng, {"static": (0.0, False), "pienso": (0.9, True),
                          "imitation_s62": (0.9, True),
                          "imitation_s63": (0.8, True)}, chaos=0.001)
        out = analyze(res, target=TARGET)
        self.assertIn("imitation(pooled)", out["arms"])
        pooled = out["arms"]["imitation(pooled)"]["sensitivity_mK_per_K"]
        a = out["arms"]["imitation_s62"]["sensitivity_mK_per_K"]
        b = out["arms"]["imitation_s63"]["sensitivity_mK_per_K"]
        self.assertAlmostEqual(pooled, (a + b) / 2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
