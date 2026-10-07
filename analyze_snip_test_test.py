"""Tests for the snip test's analysis (analyze_snip_test.py).

Uses Experiment 1's committed arrays as the truth. Stand-in "snip" Jacobians
then check that the merge is exact, and that Experiment 1's own results come
back unchanged.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_gradient_fidelity import analyze
from analyze_snip_test import (
    EXP1_PREFIX,
    choose_w_long,
    load,
    main,
    merge,
    summarize,
)

HAVE_EXP1 = Path(EXP1_PREFIX + ".npz").exists()


def _snip_like(exp1_arrays, exp1_meta, scales=(1.1, 1.2)):
    """Stand-in W = 21 and 30 Jacobians: W = 14's, scaled."""
    w14 = exp1_meta["windows"].index(14)
    jac = np.asarray(exp1_arrays["jacobians"], float)[:, :, w14]
    stacked = np.stack([s * jac for s in scales], axis=2)
    arrays = {"jacobians": stacked, "ic_index": exp1_arrays["ic_index"]}
    meta = {"windows": [21, 30], "horizons": exp1_meta["horizons"],
            "objective_names": exp1_meta["objective_names"],
            "a0": exp1_meta["config"]["a0"],
            "tail_days": exp1_meta["config"]["tail_days"]}
    return arrays, meta


@unittest.skipUnless(HAVE_EXP1, "Experiment 1's arrays are not here")
class MergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a1, cls.m1 = load(EXP1_PREFIX)
        cls.sa, cls.sm = _snip_like(cls.a1, cls.m1)
        cls.arrays, cls.meta = merge(cls.a1, cls.m1, cls.sa, cls.sm)
        cls.merged = analyze(cls.arrays, cls.meta, n_boot=400)
        cls.plain = analyze(cls.a1, cls.m1, n_boot=400)

    def test_window_order_and_values(self):
        self.assertEqual(self.meta["windows"], [1, 7, 14, 21, 30, 0])
        jac = self.arrays["jacobians"]
        old = np.asarray(self.a1["jacobians"])
        for w in (1, 7, 14, 0):
            np.testing.assert_array_equal(
                jac[:, :, self.meta["windows"].index(w)],
                old[:, :, self.m1["windows"].index(w)])
        np.testing.assert_allclose(jac[:, :, 3], 1.1 * jac[:, :, 2])

    def test_experiment1_results_come_back_unchanged(self):
        for key, cell in self.plain["cells"].items():
            got = self.merged["cells"][key]
            for name in ("angle_mean", "ratio", "median_single_angle",
                         "verdict", "angle_ci", "ratio_ci"):
                self.assertEqual(got[name], cell[name], f"{key} {name}")
        self.assertEqual(self.merged["primary"]["outcome"],
                         self.plain["primary"]["outcome"])

    def test_summary_on_stand_ins(self):
        s = summarize(self.merged)
        r14 = self.merged["cells"]["W14|T0@120"]["ratio"]
        self.assertAlmostEqual(s["ratios"]["21"], 1.1 * r14, places=9)
        self.assertTrue(s["P1_ratio_rises"])
        # Scaling leaves every angle unchanged, so P2 cannot hold.
        self.assertFalse(s["P2_single_angle_rises"])
        self.assertIn(s["outcome"], ("S-A", "S-B"))
        self.assertIn("W21|T0@120", s["cells"])

    def test_rejects_a_different_setup(self):
        bad = dict(self.sm, horizons=[15, 30])
        with self.assertRaises(ValueError):
            merge(self.a1, self.m1, self.sa, bad)
        bad = dict(self.sm, windows=[14, 30])
        with self.assertRaises(ValueError):
            merge(self.a1, self.m1, self.sa, bad)
        bad_arrays = dict(self.sa, ic_index=self.sa["ic_index"][::-1])
        with self.assertRaises(ValueError):
            merge(self.a1, self.m1, bad_arrays, self.sm)

    def test_main_writes_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "snip"
            np.savez(str(prefix) + ".npz", **self.sa)
            Path(str(prefix) + ".json").write_text(json.dumps(self.sm))
            out = Path(tmp) / "analysis.json"
            main(["--snip-prefix", str(prefix), "--output", str(out)])
            result = json.loads(out.read_text())
        self.assertEqual(result["windows"], [1, 7, 14, 21, 30, 0])
        self.assertIn(result["summary"]["outcome"], ("S-A", "S-B", "S-U"))


class RuleTest(unittest.TestCase):
    def test_w_long(self):
        def cell(r, v="useful"):
            return {"ratio": r, "verdict": v}
        self.assertEqual(choose_w_long({14: cell(0.69), 21: cell(0.80),
                                        30: cell(0.90)}), 30)
        # Within 0.05 of the best, the shorter window wins.
        self.assertEqual(choose_w_long({14: cell(0.69), 21: cell(0.88),
                                        30: cell(0.90)}), 21)
        self.assertEqual(choose_w_long({14: cell(0.69),
                                        21: cell(0.9, "failed"),
                                        30: cell(1.0, "inconclusive")}), 14)
        self.assertIsNone(choose_w_long({14: cell(0.7, "failed")}))


if __name__ == "__main__":
    unittest.main()
