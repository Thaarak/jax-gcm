"""Tests for the registered Experiment 3d analysis (analyze_experiment3d.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import analyze_experiment3d as dx
from analyze_experiment3b_test import _world

MACROS = range(101, 132, 2)
SCALE = {"uncontrolled": 1.0, "plan_learn": 0.3, "plan_naive": 0.5,
         "pi": 0.45, "adaptive": 0.5, "fixed": 0.8}


def _land(runs: Path, mode="monthly"):
    for js in runs.glob("*/ic*.json"):
        s = json.loads(js.read_text())
        s["land_climatology"] = mode
        js.write_text(json.dumps(s))


class EndToEndTest(unittest.TestCase):
    def test_registered_analysis_and_the_replication(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _world(Path(tmp), SCALE, macros=MACROS, arms=dx.ARMS)
            _land(runs)
            exp3b = Path(tmp) / "exp3b.json"
            exp3b.write_text(json.dumps({"hypotheses": {
                "H1": {"rel": -0.38, "verdict": "better"},
                "H2": {"rel": -0.19, "verdict": "better"},
                "H3": {"rel": +0.10, "verdict": "inconclusive"}}}))
            out = Path(tmp) / "a.json"
            with redirect_stdout(io.StringIO()):
                dx.main(["--runs-dir", str(runs), "--references-dirs",
                         *map(str, refs), "--exp3b-analysis", str(exp3b),
                         "--output", str(out)])
            result = json.loads(out.read_text())
        h = result["hypotheses"]
        self.assertEqual((result["n_states"], result["n_macro"]), (48, 16))
        self.assertAlmostEqual(h["H1"]["rel"], 0.3 ** 2 / 0.45 ** 2 - 1, 6)
        self.assertEqual([h[k]["verdict"] for k in ("H1", "H2", "H3")],
                         ["better"] * 3)
        rep = result["replication"]
        self.assertTrue(rep["H1"]["same_direction"])
        self.assertFalse(rep["H3"]["same_direction"])
        self.assertEqual(rep["H1"]["rel_3b"], -0.38)
        self.assertNotIn("plan_oracle", result["arm_means"])
        self.assertEqual(result["registered"]["land_climatology"], "monthly")

    def test_refuses_runs_from_the_old_land_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, _ = _world(Path(tmp), SCALE, macros=[101], arms=dx.ARMS)
            with self.assertRaises(SystemExit):     # no record: the old land
                dx.check_land(runs)
            _land(runs)
            dx.check_land(runs)
            _land(runs, "daily_index")
            with self.assertRaises(SystemExit):
                dx.check_land(runs)


if __name__ == "__main__":
    unittest.main()
