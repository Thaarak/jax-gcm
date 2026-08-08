"""Tests for the retrospective precip side-effect analysis (pure NumPy)."""

import unittest

import numpy as np

from analyze_precip_sideeffects import (
    analyze_campaign, analyze_v3_breach, apply_holm_families, resolvability)

GROUPS = {"static": ["static"], "policy(2s)": ["policy_s1", "policy_s2"]}


def synth_cells(rng, arm_shift, n_ic=20, k=4, chaos_sd=1.0):
    """Cells with a per-arm mean precip shift buried in chaos noise."""
    cells = {}
    for arm, shift in arm_shift.items():
        cells[arm] = {
            f"{reg}_mm_day": shift + chaos_sd * rng.standard_normal((n_ic, k))
            for reg in ("amazon", "sahel")
        }
    return {"cells": cells}


class AnalyzePrecipTest(unittest.TestCase):
    def test_null_gives_withdrawn_gate_and_finite_bounds(self):
        rng = np.random.default_rng(1)
        res = synth_cells(rng, {"static": 0.0, "policy_s1": 0.0,
                                "policy_s2": 0.0})
        camp = analyze_campaign(res, "synth", GROUPS)
        apply_holm_families([camp])
        reg = camp["regions"]["amazon"]
        self.assertFalse(resolvability(reg))
        self.assertFalse(reg["vs_static"]["policy(2s)"]["significant_holm"])
        # k=4 members at sd 1.0 -> se ~ 0.11; bound must be tight-ish.
        self.assertLess(reg["arms"]["static"]["equivalence_bound_95"], 0.5)

    def test_large_drying_is_detected_after_holm(self):
        rng = np.random.default_rng(2)
        res = synth_cells(rng, {"static": -2.0, "policy_s1": -2.0,
                                "policy_s2": -2.0})
        camp = analyze_campaign(res, "synth", GROUPS)
        apply_holm_families([camp])
        reg = camp["regions"]["amazon"]
        self.assertTrue(resolvability(reg))
        self.assertLess(reg["arms"]["static"]["mean"], -1.0)

    def test_seed_pooling_averages_arms(self):
        rng = np.random.default_rng(3)
        res = synth_cells(rng, {"static": 0.0, "policy_s1": 1.0,
                                "policy_s2": 3.0}, chaos_sd=1e-6)
        camp = analyze_campaign(res, "synth", GROUPS)
        pooled = camp["regions"]["amazon"]["arms"]["policy(2s)"]["mean"]
        self.assertAlmostEqual(pooled, 2.0, places=3)

    def test_chaos_sd_estimate(self):
        rng = np.random.default_rng(4)
        res = synth_cells(rng, {"static": 0.0, "policy_s1": 0.0,
                                "policy_s2": 0.0}, n_ic=50, k=8, chaos_sd=2.5)
        camp = analyze_campaign(res, "synth", GROUPS)
        sd = camp["regions"]["amazon"]["arms"]["static"]["per_run_chaos_sd"]
        self.assertAlmostEqual(sd, 2.5, delta=0.3)

    def test_v3_breach_reports_stored_verdict_and_offset(self):
        rng = np.random.default_rng(5)
        offset = float(np.log(2.0) / 100.0)
        cells = {}
        for arm in ("stage1-static", "stage5-realistic"):
            for i in range(10):
                cells[(arm, i)] = {
                    "split": "heldout",
                    "precip_losses": {
                        "amazon": offset + 0.001 * rng.random(),
                        "sahel": offset,
                        "tropics": 0.0,
                    },
                }
                cells[(arm, 10 + i)] = {  # train rows must be excluded
                    "split": "train",
                    "precip_losses": {"amazon": 99.0, "sahel": 99.0,
                                      "tropics": 0.0},
                }
        res = {"gates": {"3_teleconnection_protection": False},
               "cells": cells}
        out = analyze_v3_breach(res)
        self.assertIs(out["stored_gate_verdict"], False)
        self.assertEqual(out["arms"]["stage1-static"]["amazon"]["n"], 10)
        # Offset-subtracted sahel mean must be ~0 (pure floor).
        self.assertAlmostEqual(
            out["arms"]["stage1-static"]["sahel"]["mean_offset_subtracted"],
            0.0, places=9)
        self.assertLess(
            out["arms"]["stage1-static"]["amazon"]["mean_raw"], 0.01)
        self.assertIn("policy_vs_static", out)


if __name__ == "__main__":
    unittest.main()
