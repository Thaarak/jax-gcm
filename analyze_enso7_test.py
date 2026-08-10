"""Tests for the pre-committed Amendment-7 ENSO analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_enso7 import analyze, blind_gain_control, slope_report

TARGET = -0.1
SENS = 0.0716          # ocean-dSST K per K of Nino3.4 (metric-space)


def synth(rng, arm_rejection, n_ics=20, k=4, chaos=0.018, mu=-0.1):
    """Zero-mean disturbance; each arm rejects a fraction of the A effect."""
    half = rng.uniform(-1.4, 1.4, n_ics // 2)
    amps = np.concatenate([half, -half])
    cells = {}
    for name, rej in arm_rejection.items():
        truth = mu + SENS * amps * (1.0 - rej)
        cells[name] = {"dsst_10d": truth[:, None]
                       + chaos * rng.standard_normal((n_ics, k))}
    return {"cells": cells, "enso_amps": list(amps),
            "config": {"target_cooling": TARGET}}


class SlopeReportTest(unittest.TestCase):
    def test_recovers_slope_and_flags_flat(self):
        x = np.linspace(-1.4, 1.4, 40)
        r = slope_report(x, 0.05 * x + 0.01)
        self.assertAlmostEqual(r["slope"], 0.05, places=8)
        self.assertLess(r["p"], 1e-9)
        rng = np.random.default_rng(0)
        flat = slope_report(x, 0.001 * rng.standard_normal(40))
        self.assertGreater(flat["p"], 0.05)


class BlindControlTest(unittest.TestCase):
    def test_fixed_gain_is_exactly_slope_invariant(self):
        """The property the primary endpoint rests on: rescaling a blind
        controller by ANY constant leaves its disturbance slope untouched.
        """
        rng = np.random.default_rng(1)
        amps = rng.uniform(-1.4, 1.4, 20)
        static = -0.06 + SENS * amps + 0.01 * rng.standard_normal(20)
        s, mu = np.polyfit(amps, static, 1)
        chaos = static - (mu + s * amps)
        s_static = slope_report(amps, static - TARGET)["slope"]
        for g in (0.5, 1.0, 2.0, 3.7):
            rescaled = g * mu + s * amps + chaos
            self.assertAlmostEqual(
                slope_report(amps, rescaled - TARGET)["slope"], s_static,
                places=12)

    def test_loo_blind_slope_leakage_is_small(self):
        """The LOO-tuned gain varies per fold, so slope invariance is only
        approximate for the control arm. Bound that leakage: it must stay far
        below the rejection effects the endpoint is meant to detect (~60
        mK/K), or the control would be scoring adaptivity it does not have.
        """
        rng = np.random.default_rng(1)
        amps = rng.uniform(-1.4, 1.4, 20)
        static = -0.06 + SENS * amps + 0.01 * rng.standard_normal(20)
        blind = blind_gain_control(static, amps, TARGET)
        s_static = slope_report(amps, static - TARGET)["slope"]
        s_blind = slope_report(amps, blind - TARGET)["slope"]
        self.assertLess(abs(s_blind - s_static) * 1000, 10.0)

    def test_blind_gain_improves_mean_error_when_miscalibrated(self):
        rng = np.random.default_rng(2)
        amps = rng.uniform(-1.4, 1.4, 20)
        static = -0.05 + SENS * amps + 0.005 * rng.standard_normal(20)
        blind = blind_gain_control(static, amps, TARGET)
        self.assertLess(np.abs(blind - TARGET).mean(),
                        np.abs(static - TARGET).mean())


class AnalyzeEnso7Test(unittest.TestCase):
    def test_detects_genuine_disturbance_rejection(self):
        rng = np.random.default_rng(3)
        res = synth(rng, {"static": 0.0, "pienso": 0.85,
                          "imitation_s72": 0.8, "imitation_s73": 0.8,
                          "imitation_s74": 0.8})
        out = analyze(res)
        self.assertTrue(out["primary"]["H8_pienso_sensitivity_vs_static"]
                        ["significant_holm"])
        self.assertTrue(out["primary"]["H9_imitation_sensitivity_vs_static"]
                        ["significant_holm"])
        # And it must beat the blind control on the primary endpoint too.
        self.assertLess(out["secondary"]["slope[pienso vs blind_loo]"]["p"],
                        0.05)

    def test_non_adaptive_arm_cannot_pass(self):
        """A miscalibrated-but-blind arm must fail the primary endpoint even
        though it can look good on mean |miss|.
        """
        rng = np.random.default_rng(4)
        # 'pienso' here rejects nothing but sits at a better mean level.
        res = synth(rng, {"static": 0.0, "pienso": 0.0}, mu=-0.1)
        res["cells"]["pienso"]["dsst_10d"] += -0.02
        out = analyze(res)
        self.assertFalse(out["primary"]["H8_pienso_sensitivity_vs_static"]
                         ["significant_holm"])

    def test_direction_gate(self):
        rng = np.random.default_rng(5)
        res = synth(rng, {"static": 0.85, "pienso": 0.0})
        out = analyze(res)
        h8 = out["primary"]["H8_pienso_sensitivity_vs_static"]
        self.assertLess(h8["p"], 0.05)
        self.assertFalse(h8["significant_holm"])   # wrong direction

    def test_stratified_and_equivalence_present(self):
        rng = np.random.default_rng(6)
        res = synth(rng, {"static": 0.0, "pienso": 0.85,
                          "imitation_s72": 0.85})
        out = analyze(res)
        self.assertIn("stratified_mean_abs[pienso A>=0]", out["secondary"])
        self.assertIn("stratified_mean_abs[pienso A<0]", out["secondary"])
        eq = out["secondary"]["mean_abs[imitation vs pienso]"]
        self.assertIn("equivalence_bound_95", eq)

    def test_blind_control_is_reported_as_an_arm(self):
        rng = np.random.default_rng(7)
        res = synth(rng, {"static": 0.0, "pienso": 0.85})
        out = analyze(res)
        self.assertIn("blind_loo", out["sensitivity"])
        self.assertIn("blind_loo", out["mean_abs_err_mK"])


if __name__ == "__main__":
    unittest.main()
