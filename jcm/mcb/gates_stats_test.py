"""Tests for the formal paired tests (validated against SciPy where present)."""

import unittest

import numpy as np

from jcm.mcb.gates_stats import (
    equivalence_bound,
    paired_report,
    paired_t_test,
    t_ppf,
    t_sf,
    tost,
    wilcoxon_signed_rank,
)

try:
    import scipy.stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class TDistributionTest(unittest.TestCase):
    def test_known_critical_values(self):
        # Standard table values.
        self.assertAlmostEqual(t_ppf(0.975, 9), 2.262157, places=4)
        self.assertAlmostEqual(t_ppf(0.95, 9), 1.833113, places=4)
        self.assertAlmostEqual(t_ppf(0.975, 29), 2.045230, places=4)

    def test_sf_symmetry(self):
        self.assertAlmostEqual(t_sf(0.0, 7), 0.5, places=10)
        self.assertAlmostEqual(t_sf(2.0, 7) + t_sf(-2.0, 7), 1.0, places=10)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_sf_matches_scipy(self):
        for df in [3, 9, 19, 29]:
            for t in [-3.2, -1.0, 0.3, 2.262, 4.5]:
                self.assertAlmostEqual(
                    t_sf(t, df), float(sps.t.sf(t, df)), places=8,
                    msg=f"t={t}, df={df}")


class PairedTTest(unittest.TestCase):
    # The actual v3 G3 per-IC diffs (policy_err - static_err), from
    # eval_v3.pkl; scipy gives p=0.4597.
    G3_DIFFS = [0.0257568359375, 0.0020751953125, 0.0064697265625,
                -0.001361083984375011, -0.011883544921874989,
                0.00115966796875, -0.008190917968749989,
                -0.01743774414062499, 0.001391601562500011,
                0.050384521484375]

    def test_v3_g3_pvalue(self):
        res = paired_t_test(self.G3_DIFFS)
        self.assertAlmostEqual(res["p"], 0.4597, places=3)
        self.assertFalse(res["p"] < 0.05)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_matches_scipy(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            d = rng.normal(0.3, 1.0, size=12)
            ours = paired_t_test(d)
            t, p = sps.ttest_1samp(d, 0.0)
            self.assertAlmostEqual(ours["t"], float(t), places=8)
            self.assertAlmostEqual(ours["p"], float(p), places=8)


class WilcoxonTest(unittest.TestCase):
    def test_v3_g3_pvalue(self):
        res = wilcoxon_signed_rank(PairedTTest.G3_DIFFS)
        # scipy exact two-sided: 0.695
        self.assertAlmostEqual(res["p"], 0.695, places=2)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_matches_scipy_exact(self):
        rng = np.random.default_rng(3)
        for n in [8, 10, 15]:
            d = rng.normal(0.2, 1.0, size=n)
            ours = wilcoxon_signed_rank(d)
            w, p = sps.wilcoxon(d, mode="exact")
            self.assertAlmostEqual(ours["p"], float(p), places=6,
                                   msg=f"n={n}")


class EquivalenceTest(unittest.TestCase):
    def test_bound_formula(self):
        d = np.array(PairedTTest.G3_DIFFS)
        bound = equivalence_bound(d)
        mean, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        self.assertAlmostEqual(bound, abs(mean) + t_ppf(0.95, 9) * se,
                               places=8)
        # TOST must be consistent with the bound: equivalent just above it,
        # not equivalent just below it.
        self.assertTrue(tost(d, bound * 1.001)["equivalent"])
        self.assertFalse(tost(d, bound * 0.999)["equivalent"])

    def test_tight_data_gives_tight_bound(self):
        d = np.full(10, 0.001) + np.linspace(-1e-4, 1e-4, 10)
        self.assertLess(equivalence_bound(d), 0.0013)


class PairedReportTest(unittest.TestCase):
    def test_report_fields_and_convention(self):
        treatment = [1.0, 2.0, 3.0, 4.0]
        comparator = [1.5, 2.5, 3.5, 4.5]
        rep = paired_report(treatment, comparator)
        self.assertAlmostEqual(rep["mean"], -0.5, places=10)
        self.assertEqual(rep["n"], 4)
        self.assertIn("p_t", rep)
        self.assertIn("p_wilcoxon", rep)
        self.assertIn("equivalence_bound_95", rep)


if __name__ == "__main__":
    unittest.main()
