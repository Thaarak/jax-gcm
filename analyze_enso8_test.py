"""Tests for the pre-committed Amendment 7 rev 2 ablation analysis."""

import unittest

import numpy as np

from analyze_enso8 import (
    analyze, hc3_se, hinge_slopes, paired_rms_boot, rms_a,
    slope_equivalence_bound, wild_bootstrap_slope)

TARGET = -0.1
SENS = 0.0547


def det_amps(n=24, a_max=1.8273):
    m = n // 2
    mags = a_max * np.sqrt((np.arange(1, m + 1) - 0.5) / m)
    return np.array([x for pair in zip(mags, -mags) for x in pair])


def synth(rng, spec, n=24, k=4, chaos=0.018):
    """spec: arm -> (rejection, warm_bias, cold_bias) in slope units."""
    amps = det_amps(n)
    cells = {}
    for name, (rej, wb, cb) in spec.items():
        base = SENS * amps * (1.0 - rej)
        skew = np.where(amps >= 0, wb, cb) * amps
        truth = TARGET + base + skew
        cells[name] = {"dsst_10d": truth[:, None]
                       + chaos * rng.standard_normal((n, k))}
    return {"cells": cells, "enso_amps": list(amps),
            "config": {"target_cooling": TARGET}}


class InferenceTest(unittest.TestCase):
    def test_hc3_exceeds_ols_under_heteroscedasticity(self):
        rng = np.random.default_rng(0)
        a = det_amps(40)
        y = 0.05 * a + np.abs(a) * 0.02 * rng.standard_normal(40)
        r = wild_bootstrap_slope(a, y, n_boot=2000)
        self.assertGreater(hc3_se(a, y), r["ols_se"])

    def test_wild_bootstrap_calibrated_under_null(self):
        """A true-null slope must not be called significant most of the time."""
        rej = 0
        for s in range(40):
            rng = np.random.default_rng(s)
            a = det_amps(24)
            y = 0.02 * rng.standard_normal(24)
            if wild_bootstrap_slope(a, y, n_boot=1500, seed=s)["p_boot"] < 0.05:
                rej += 1
        self.assertLessEqual(rej, 6)      # ~5% nominal, allow noise at n=40

    def test_detects_real_slope(self):
        rng = np.random.default_rng(1)
        a = det_amps(24)
        y = 0.05 * a + 0.01 * rng.standard_normal(24)
        r = wild_bootstrap_slope(a, y, n_boot=4000)
        self.assertLess(r["p_boot"], 0.01)
        self.assertAlmostEqual(r["slope"], 0.05, delta=0.01)

    def test_equivalence_bound_shrinks_with_precision(self):
        rng = np.random.default_rng(2)
        a = det_amps(24)
        noisy = slope_equivalence_bound(a, 0.05 * rng.standard_normal(24))
        clean = slope_equivalence_bound(a, 0.002 * rng.standard_normal(24))
        self.assertLess(clean, noisy)


class SignAgnosticTest(unittest.TestCase):
    def test_rms_a_is_not_fooled_by_cancellation(self):
        """The failure mode the audit found: a signed slope near zero built
        from warm under-correction and cold over-correction.
        """
        a = det_amps(24)
        cancelling = np.where(a >= 0, 0.012 * a, 0.012 * a)  # same sign slope
        opposing = np.where(a >= 0, 0.012 * a, -0.012 * a)   # cancels in OLS
        s_opp = wild_bootstrap_slope(a, opposing, n_boot=1000)["slope"]
        self.assertAlmostEqual(s_opp, 0.0, delta=2e-3)       # signed ~ 0
        self.assertGreater(rms_a(opposing), 0.5 * rms_a(cancelling))
        h = hinge_slopes(a, opposing)
        self.assertGreater(h["warm"]["slope"], 0)
        self.assertLess(h["cold"]["slope"], 0)

    def test_paired_rms_bootstrap_detects_improvement(self):
        rng = np.random.default_rng(3)
        a = det_amps(24)
        worse = 0.05 * a + 0.005 * rng.standard_normal(24)
        better = 0.005 * a + 0.005 * rng.standard_normal(24)
        r = paired_rms_boot(better, worse, n_boot=3000)
        self.assertLess(r["diff"], 0)
        self.assertLess(r["p_boot"], 0.05)


class AnalyzeTest(unittest.TestCase):
    def test_h10_detected_when_anticipation_helps(self):
        rng = np.random.default_rng(4)
        res = synth(rng, {"static": (0.0, 0, 0), "blindfb": (0.75, 0, 0),
                          "blindfb_t": (0.80, 0, 0), "pienso": (0.97, 0, 0)},
                    chaos=0.008)
        out = analyze(res, n_boot=3000)
        h10 = out["primary"]["pienso vs blindfb_t"]
        self.assertLess(h10["slope"], 0)
        self.assertTrue(h10["significant_holm"])

    def test_h10_null_gives_usable_bound_against_tuned_comparator(self):
        """The likely real outcome: a retuned reactive arm matches, and the
        null must come with an equivalence bound rather than a shrug.
        """
        rng = np.random.default_rng(5)
        res = synth(rng, {"static": (0.0, 0, 0), "blindfb": (0.75, 0, 0),
                          "blindfb_t": (0.96, 0, 0), "pienso": (0.97, 0, 0)},
                    chaos=0.012)
        out = analyze(res, n_boot=3000)
        h10 = out["primary"]["pienso vs blindfb_t"]
        self.assertFalse(h10["significant_holm"])
        self.assertLess(h10["equivalence_bound_95"] * 1000, 12.0)

    def test_untuned_comparator_would_have_manufactured_a_win(self):
        """Guards the blocking finding: pienso beats the UNTUNED blindfb even
        when it is statistically tied with the tuned one.
        """
        rng = np.random.default_rng(6)
        res = synth(rng, {"static": (0.0, 0, 0), "blindfb": (0.75, 0, 0),
                          "blindfb_t": (0.96, 0, 0), "pienso": (0.97, 0, 0)},
                    chaos=0.010)
        out = analyze(res, n_boot=3000)
        self.assertFalse(out["primary"]["pienso vs blindfb_t"]
                         ["significant_holm"])
        self.assertLess(out["secondary"]["pienso vs blindfb"]["p_boot"], 0.05)

    def test_hinge_reported_per_arm(self):
        rng = np.random.default_rng(7)
        res = synth(rng, {"static": (0.0, 0, 0), "blindfb_t": (0.9, 0, 0),
                          "pienso": (0.9, 0.01, -0.01)}, chaos=0.006)
        out = analyze(res, n_boot=1000)
        h = out["arms"]["pienso"]["hinge"]
        self.assertGreater(h["warm"]["slope"], h["cold"]["slope"])
        self.assertIn("rms_a_mK", out["arms"]["pienso"])


if __name__ == "__main__":
    unittest.main()
