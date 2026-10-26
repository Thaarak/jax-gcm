"""Tests for the registered Experiment 3f analysis (analyze_experiment3f.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import analyze_experiment3f as ex
from analyze_experiment3b_test import _world

MACROS = range(101, 132, 2)
SCALE_3D = {"uncontrolled": 1.0, "plan_learn": 0.3, "plan_naive": 0.5,
            "pi": 0.45, "adaptive": 0.5, "fixed": 0.8}
# The 16-week planner clearly better than two weeks; 8 weeks about equal.
SCALE_3F = {"learn14": 0.3, "learn112_mpc": 0.2, "learn56_mpc": 0.3,
            "learn112_hold": 0.4}
DESIGN = {"learner_noise_k": 0.00069104}
SEGMENTS, MEMBERS, K = 13, 3, 5


def _planner_runs(runs: Path, arms):
    """Give every run what a real one records."""
    for arm in arms:
        for js in sorted((runs / arm).glob("ic*.json")):
            s = json.loads(js.read_text())
            s["land_climatology"] = "monthly"
            s["seconds"] = 100.0
            if arm in ex.PLANNERS or arm == "plan_learn":
                look, window, schedule = ex.PLANNERS.get(
                    arm, (14, None, "none"))
                s["planner"] = {"lookahead_days": look,
                                "window_days": 99999 if window is None
                                else window,
                                "schedule": schedule, "sensing": "exact"}
                s["learner"] = {"config": {"noise_k":
                                           DESIGN["learner_noise_k"]}}
                if arm in ex.LONG:
                    s["target_extension"] = {"overlap_mean_abs_k": 0.004}
                truth = np.asarray(s["true_efficacy"])
                s["planner_logs"] = [[{"segment": j, "day": 14 * j,
                                       "seconds": 5.0 + look / 14.0,
                                       "efficacy_belief": (truth * (
                                           1.1 if j < 7 else 1.0)).tolist()}
                                      for j in range(SEGMENTS)]] * MEMBERS
            js.write_text(json.dumps(s))


def _worlds(tmp: Path):
    (tmp / "d").mkdir()
    (tmp / "f").mkdir()
    runs_3d, refs = _world(tmp / "d", SCALE_3D, macros=MACROS, seed=4,
                           arms=ex.ARMS_3D)
    runs_3f, _ = _world(tmp / "f", SCALE_3F, macros=MACROS, seed=4,
                        arms=ex.ARMS_3F)
    _planner_runs(runs_3d, ex.ARMS_3D)
    _planner_runs(runs_3f, ex.ARMS_3F)
    (tmp / "design.json").write_text(json.dumps(DESIGN))
    return runs_3f, runs_3d, refs


def _args(tmp, runs_3f, runs_3d, refs, *extra):
    return ["--runs-dir", str(runs_3f), "--runs-3d-dir", str(runs_3d),
            "--references-dirs", *map(str, refs), "--design",
            str(tmp / "design.json"), "--output", str(tmp / "a.json"), *extra]


class EndToEndTest(unittest.TestCase):
    def test_registered_analysis_on_a_synthetic_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3f, runs_3d, refs = _worlds(tmp)
            with redirect_stdout(io.StringIO()):
                ex.main(_args(tmp, runs_3f, runs_3d, refs))
            result = json.loads((tmp / "a.json").read_text())
        self.assertEqual((result["n_states"], result["n_macro"]), (48, 16))
        h = result["hypotheses"]
        self.assertEqual(sorted(h), ["H1", "H2", "H3"])
        self.assertEqual((h["H1"]["arm"], h["H1"]["comparator"]),
                         ("learn112_mpc", "learn14"))
        self.assertEqual(h["H1"]["verdict"], "better")
        self.assertEqual(h["H3"]["verdict"], "better")
        self.assertIn("better", result["h1_reading"])
        self.assertGreater(result["compute"]["learn112_mpc"]
                           ["replan_vs_learn14"], 1.5)
        self.assertAlmostEqual(
            result["checks"]["target_overlap_mean_abs_k_max"], 0.004)

    def test_dropped_hold_arm_skips_h3(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3f, runs_3d, refs = _worlds(tmp)
            with redirect_stdout(io.StringIO()):
                ex.main(_args(tmp, runs_3f, runs_3d, refs, "--arms",
                              "learn14", "learn112_mpc", "learn56_mpc"))
            result = json.loads((tmp / "a.json").read_text())
        self.assertEqual(sorted(result["hypotheses"]), ["H1", "H2"])

    def test_refuses_the_wrong_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3f, runs_3d, refs = _worlds(tmp)
            js = sorted((runs_3f / "learn112_mpc").glob("ic*.json"))[0]
            s = json.loads(js.read_text())
            s["planner"]["window_days"] = 14
            js.write_text(json.dumps(s))
            with self.assertRaises(SystemExit):
                ex.main(_args(tmp, runs_3f, runs_3d, refs))

    def test_refuses_a_target_that_is_another_climate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3f, runs_3d, refs = _worlds(tmp)
            js = sorted((runs_3f / "learn56_mpc").glob("ic*.json"))[0]
            s = json.loads(js.read_text())
            s["target_extension"]["overlap_mean_abs_k"] = 0.3
            js.write_text(json.dumps(s))
            with self.assertRaises(SystemExit):
                ex.main(_args(tmp, runs_3f, runs_3d, refs))

    def test_inconclusive_h1_is_read_as_a_bound(self):
        text = ex.h1_bound({"verdict": "inconclusive", "rel": -0.04,
                            "ci90_rel": [-0.15, 0.08]})
        self.assertIn("-15%", text)
        self.assertIn("+8%", text)


if __name__ == "__main__":
    unittest.main()
