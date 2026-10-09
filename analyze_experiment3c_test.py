"""Tests for the registered Experiment 3c analysis (analyze_experiment3c.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import analyze_experiment3c as cx
from analyze_experiment3b_test import _world

MACROS = range(17, 48, 2)
SCALE_3B = {"uncontrolled": 1.0, "plan_learn": 0.3, "plan_naive": 0.5,
            "plan_oracle": 0.28, "pi": 0.45, "adaptive": 0.5, "fixed": 0.8}
SCALE_3C = {"plan_learn": 0.35, "plan_naive": 0.5}


def _worlds(tmp: Path):
    """3b's runs and 3c's runs on the same states and references."""
    (tmp / "b").mkdir()
    (tmp / "c").mkdir()
    runs_3b, refs = _world(tmp / "b", SCALE_3B, macros=MACROS, seed=4)
    runs_3c, _ = _world(tmp / "c", SCALE_3C, macros=MACROS, seed=4,
                        arms=cx.ARMS_3C)
    return runs_3c, runs_3b, refs


def _set(summary_path: Path, **changes):
    s = json.loads(summary_path.read_text())
    s.update(changes)
    summary_path.write_text(json.dumps(s))


class EndToEndTest(unittest.TestCase):
    def test_registered_analysis_on_a_synthetic_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_3c, runs_3b, refs = _worlds(Path(tmp))
            out = Path(tmp) / "a.json"
            with redirect_stdout(io.StringIO()):
                cx.main(["--runs-dir", str(runs_3c), "--runs-3b-dir",
                         str(runs_3b), "--references-dirs", *map(str, refs),
                         "--output", str(out)])
            result = json.loads(out.read_text())
        self.assertEqual((result["n_states"], result["n_macro"]), (48, 16))
        h = result["hypotheses"]
        self.assertEqual((h["H1"]["arm"], h["H1"]["comparator"]),
                         ("plan_learn_rs", "pi"))
        self.assertAlmostEqual(h["H1"]["rel"], 0.35 ** 2 / 0.45 ** 2 - 1, 6)
        self.assertEqual(h["H1"]["verdict"], "better")
        self.assertEqual(h["H2"]["verdict"], "better")      # vs naive_rs
        self.assertEqual(h["H3"]["verdict"], "worse")       # vs 3b's learner
        self.assertNotIn("t_p_holm", h["H1"])
        self.assertIn("plan_naive_rs_vs_plan_naive", result["descriptive"])
        self.assertEqual(set(result["learning"]["by_arm"]),
                         {"plan_learn_rs", "plan_learn"})
        # 3c's arms get their gains from 3b's no-brightening runs.
        self.assertIn("G_sst@ocean", result["arm_means"]["plan_learn_rs"])

    def test_refuses_the_wrong_sensing_or_unequal_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_3c, runs_3b, _ = _worlds(Path(tmp))
            one = runs_3c / "plan_learn" / "ic1700.json"
            _set(one, planner={"sensing": "ocean"})
            cx.check_runs(runs_3c, runs_3b)
            _set(one, planner={"sensing": "exact"})
            with self.assertRaises(SystemExit):
                cx.check_runs(runs_3c, runs_3b)
            _set(one, planner={"sensing": "ocean"}, members=6)
            with self.assertRaises(SystemExit):
                cx.check_runs(runs_3c, runs_3b)
            _set(one, members=3)
            _set(runs_3b / "pi" / "ic1700.json", planner={"sensing": "ocean"})
            with self.assertRaises(SystemExit):
                cx.check_runs(runs_3c, runs_3b)


if __name__ == "__main__":
    unittest.main()
