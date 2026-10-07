"""Tests for run_experiment2.py and analyze_experiment2.py that need no model run.

* The design stage runs end to end on synthetic training files.
* Episode-shaped outputs go through the frozen analysis, in a world built so
  that the gradient arm is clearly better than the sunlight arm.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_experiment2 as ana
import run_experiment2 as rx

IX, IL = 6, 8
LATS = np.deg2rad(np.linspace(-70, 70, IL))


def _grid(tmp):
    ocean = np.ones((IX, IL), np.float32)
    ocean[0] = 0.0
    np.savez(tmp / "grid.npz", ocean_mask=ocean, land_mask=1 - ocean,
             latitudes_rad=LATS, longitudes_rad=np.linspace(0, 6, IX))
    return ocean


class ArgsTest(unittest.TestCase):
    def test_roles(self):
        rx.check_role("exp2_train", False)
        rx.check_role("exp2_eval", True)
        for role, allow, train_only in (("exp2_eval", False, False),
                                        ("exp2_eval", True, True),
                                        ("exp3_train", True, False)):
            with self.assertRaises(SystemExit):
                rx.check_role(role, allow, train_only)

    def test_validation_and_arms(self):
        rx.validate_args(rx.parse_args(["timing", "--ic-dir", "x",
                                        "--output", "t.json"]))
        with self.assertRaises(SystemExit):
            rx.validate_args(rx.parse_args(["timing", "--ic-dir", "x",
                                            "--days", "100", "--output",
                                            "t.json"]))
        self.assertEqual(set(rx.ARMS), set(ana.ARMS))
        for _, a, b, _ in ana.HYPOTHESES:
            self.assertIn(a, rx.ARMS)
            self.assertIn(b, rx.ARMS)

    def test_window_mean(self):
        daily = np.arange(182.0)[:, None, None] * np.ones((182, 2, 2))
        np.testing.assert_allclose(rx.window_mean(daily), np.mean(
            np.arange(98, 182)))


class DesignStageTest(unittest.TestCase):
    """The design stage on synthetic training outputs (linear world)."""

    def test_designs_for_every_arm(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            refs, resp, grad = tmp / "refs", tmp / "resp", tmp / "grad"
            for d in (refs, resp, grad):
                d.mkdir()
            ocean = _grid(refs)
            ics = []
            for s in range(3):
                idx = 100 * 2 * s + 1
                normal = 290.0 + 0.1 * rng.normal(size=(182, IX, IL))
                warmed_members = (290.2 + 0.05 * rng.normal(
                    size=(5, IX, IL))) * ocean + 290 * (1 - ocean)
                np.savez(refs / f"ic{idx:04d}_references.npz",
                         normal_sst=normal.astype(np.float32),
                         warmed_members_window_sst=warmed_members,
                         normal_cloud_cover=np.full((182, IX, IL), 0.4))
                ics.append({"entry": {"index": idx},
                            "start_sim_time_s": 30 * 86400.0})
                for layout, k in (("b5", 5), ("b13", 13)):
                    band = (warmed_members[None]
                            - 0.02 * rng.uniform(0.5, 1, (k, 1, IX, IL))
                            * ocean)
                    np.savez(resp / f"ic{idx:04d}_responses_{layout}.npz",
                             band_members_window_sst=band.astype(np.float32))
            (refs / "references_manifest.json").write_text(json.dumps(
                {"role": "exp2_train", "ics": ics}))
            (tmp / "timing.json").write_text(json.dumps(
                {"forward_s": 12.4, "jacobian_s": {"b5": 44.0, "b13": 93.0}}))
            for layout, mode in (("b5", "snipped"), ("b5", "bptt"),
                                 ("b13", "snipped")):
                k = 5 if layout == "b5" else 13
                (grad / f"gradient_{layout}_{mode}.json").write_text(
                    json.dumps({"design": [0.02] * k,
                                "final_training_objective": 0.003,
                                "design_seconds": 100.0}))
            out = tmp / "designs.json"
            rx.main(["design", "--references-dir", str(refs),
                     "--responses-dir", str(resp), "--timing",
                     str(tmp / "timing.json"), "--gradient-dir", str(grad),
                     "--output", str(out)])
            designs = json.loads(out.read_text())
        arms = designs["arms"]
        self.assertEqual(set(arms), set(rx.ARMS))
        for name, (layout, _) in rx.ARMS.items():
            k = 5 if layout == "b5" else 13
            self.assertEqual(len(arms[name]["amplitudes"]), k, name)
            self.assertTrue(all(0.0 <= x <= 0.15
                                for x in arms[name]["amplitudes"]), name)
        self.assertEqual(designs["costs"]["b5"]["equal_cost_members"], 2)
        self.assertEqual(designs["costs"]["b13"]["equal_cost_members"], 2)
        self.assertEqual(arms["brute5_eq"]["members"], 2)
        self.assertEqual(arms["brute13"]["members"], 5)
        self.assertGreater(sum(arms["brute5"]["amplitudes"]), 0.0)
        self.assertAlmostEqual(designs["start_day_of_year"], 30.0)


def _write_runs(tmp, n_macro=8):
    """Episode-shaped outputs: each arm's error is a fixed fraction of the warming."""
    rng = np.random.default_rng(1)
    refs = tmp / "refs"
    refs.mkdir()
    ocean = _grid(refs)
    lat = np.broadcast_to(LATS[None], (IX, IL))
    shape_pattern = np.sin(lat) * ocean            # a latitude profile
    leftover = {"uncontrolled": 1.0, "uniform": 0.45, "brute5": 0.35,
                "brute5_eq": 0.37, "grad5": 0.34, "bptt5": 0.9,
                "sunlight5": 0.6, "brute13": 0.3, "brute13_eq": 0.33,
                "grad13": 0.3, "sunlight13": 0.6}
    for macro in range(1, 2 * n_macro, 2):
        for branch in (0, 1):
            idx = 100 * macro + branch
            normal = 290.0 + np.zeros((182, IX, IL))
            np.savez(refs / f"ic{idx:04d}_references.npz",
                     normal_sst=normal, normal_land_temperature=normal,
                     normal_precipitation=np.ones((182, IX, IL)))
            for arm, frac in leftover.items():
                d = tmp / "runs" / arm
                d.mkdir(parents=True, exist_ok=True)
                noise = 0.01 * rng.normal(size=(IX, IL))
                sst = (normal + frac * (0.2 * shape_pattern + 0.1 * ocean)
                       + noise)
                np.savez(d / f"ic{idx:04d}.fields.npz", sst=sst,
                         land_temperature=normal,
                         precipitation=np.ones((182, IX, IL)))
                (d / f"ic{idx:04d}.json").write_text(json.dumps(
                    {"amplitudes": [0.02] * (13 if "13" in arm else 5),
                     "effort": 0.02, "members": 5, "seconds": 1.0}))
    return tmp / "runs", refs


class AnalysisTest(unittest.TestCase):
    def test_registered_tests_on_a_known_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _write_runs(Path(tmp))
            per_arm = ana.load_runs(runs, refs)
            result = ana.analyze(per_arm, n_boot=500)
        self.assertEqual(result["n_states"], 16)
        self.assertEqual(result["missing_arms"], [])
        h = result["hypotheses"]
        self.assertEqual(h["H2"]["verdict"], "better")      # grad13 vs sun13
        self.assertEqual(h["H3"]["verdict"], "better")      # grad5 vs bptt5
        self.assertIn(h["H1"]["verdict"], ("better", "negligible",
                                           "equivalent", "inconclusive"))
        self.assertLess(h["H2"]["rel"], -0.5)
        self.assertIn("grad13_vs_brute13", result["secondary"])
        self.assertIn("grad13", result["vs_uncontrolled"])

    def test_a_dropped_arm_is_not_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _write_runs(Path(tmp))
            per_arm = ana.load_runs(runs, refs)
        per_arm["bptt5"] = {}
        result = ana.analyze(per_arm, n_boot=200)
        self.assertEqual(result["hypotheses"]["H3"]["verdict"],
                         "not computed")
        self.assertIn("bptt5", result["missing_arms"])
        # Holm now runs over three hypotheses.
        h2 = result["hypotheses"]["H2"]
        self.assertLessEqual(h2["t_p_holm"], min(1.0, 3 * h2["t_p"]) + 1e-12)

    def test_smoke_runs_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _write_runs(Path(tmp), n_macro=1)
            js = runs / "grad13" / "ic0100.json"
            js.write_text(json.dumps({"amplitudes": [0.0] * 13, "effort": 0,
                                      "smoke": True}))
            with self.assertRaises(SystemExit):
                ana.load_runs(runs, refs)

    def test_costs(self):
        c = ana.costs({"costs": {"b5": {"gradient_cost_seconds": 100.0,
                                        "brute_cost_seconds_per_member": 50.0,
                                        "equal_cost_members": 2}}})
        self.assertEqual(c["b5"]["brute_5_members_s"], 250.0)
        self.assertEqual(c["b5"]["brute_5_over_gradient"], 2.5)


if __name__ == "__main__":
    unittest.main()
