"""Tests for the registered Experiment 3b analysis (analyze_experiment3b.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import analyze_experiment3b as bx
from jcm.mcb.hidden_strength import hidden_strength

IX, IL, DAYS = 6, 4, 182
LATS = np.deg2rad(np.array([-45.0, -15.0, 15.0, 45.0]))


def _world(tmp: Path, scale: dict, macros=range(17, 48, 2), branches=(0, 1, 2),
           seed=0, arms=bx.ARMS):
    """Write references (two folders) and every arm's runs.

    Each arm's error is ``scale[arm] * base``, so J scales with its square.
    """
    rng = np.random.default_rng(seed)
    ocean = np.ones((IX, IL))
    ocean[0] = 0.0
    ref_dirs = [tmp / "refs_a", tmp / "refs_b"]
    for d in ref_dirs:
        d.mkdir()
        np.savez(d / "grid.npz", ocean_mask=ocean, land_mask=1.0 - ocean,
                 latitudes_rad=LATS, longitudes_rad=np.linspace(0, 6, IX))
    for macro in macros:
        for br in branches:
            idx = 100 * macro + br
            normal = 290.0 + rng.normal(0, 0.1, (DAYS, IX, IL))
            np.savez(ref_dirs[br % 2] / f"ic{idx:04d}_references.npz",
                     normal_sst=normal, normal_land_temperature=normal,
                     normal_precipitation=normal)
            truth = hidden_strength(macro, br)
            base = (0.1 + 0.05 * rng.normal()) * (1.0 + LATS[None, :] ** 2)
            base = base + 0.01 * rng.normal(size=(IX, IL))
            for arm in arms:
                d = tmp / "runs" / arm
                d.mkdir(parents=True, exist_ok=True)
                f = normal + scale.get(arm, 1.0) * base[None]
                np.savez(d / f"ic{idx:04d}.fields.npz", sst=f,
                         land_temperature=f, precipitation=f)
                summary = {"effort": 0.02, "members": 3, "seconds": 1.0,
                           "true_efficacy": truth.tolist()}
                if arm in ("uncontrolled", "fixed"):
                    summary["amplitudes"] = [0.03] * 5
                else:
                    summary["schedules"] = [[[0.03] * 5] * 13] * 3
                if arm == "plan_learn":
                    # Beliefs approach the truth over the 13 segments.
                    summary["learner"] = {"config": {}}
                    summary["planner_logs"] = [[
                        {"efficacy_belief": (np.ones(5) + (truth - 1.0)
                                             * min(1.0, s / 4)).tolist()}
                        for s in range(13)]] * 3
                (d / f"ic{idx:04d}.json").write_text(json.dumps(summary))
    return tmp / "runs", ref_dirs


class HelpersTest(unittest.TestCase):
    def test_overall_factor_from_the_strength(self):
        for macro, branch in ((17, 0), (17, 1), (17, 2), (46, 1)):
            self.assertEqual(bx.overall_factor(hidden_strength(macro, branch)),
                             bx.OVERALL_FACTORS[(macro + branch) % 3])

    def test_learning_curve_reaches_zero(self):
        truth = np.array([0.5, 2.0, 1.0, 1.5, 0.8])
        logs = [[{"efficacy_belief": [1.0] * 5},
                 {"efficacy_belief": truth.tolist()}]]
        curve = bx.learning_curve({"true_efficacy": truth.tolist(),
                                   "planner_logs": logs})
        self.assertGreater(curve[0], 0.2)
        self.assertAlmostEqual(curve[1], 0.0)


class EndToEndTest(unittest.TestCase):
    SCALE = {"uncontrolled": 1.0, "plan_learn": 0.3, "plan_naive": 0.5,
             "plan_oracle": 0.28, "pi": 0.45, "adaptive": 0.5, "fixed": 0.8}

    def test_registered_analysis_on_a_synthetic_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _world(Path(tmp), self.SCALE)
            per_arm = bx.load_runs(runs, refs)
            result = bx.analyze(per_arm, n_boot=300)
            out = Path(tmp) / "a.json"
            with redirect_stdout(io.StringIO()):
                bx.main(["--runs-dir", str(runs), "--references-dirs",
                         *map(str, refs), "--output", str(out)])
            saved = json.loads(out.read_text())
        self.assertEqual((result["n_states"], result["n_macro"]), (48, 16))
        h = result["hypotheses"]
        self.assertAlmostEqual(h["H1"]["rel"], 0.3 ** 2 / 0.45 ** 2 - 1, 6)
        self.assertEqual(h["H1"]["verdict"], "better")
        self.assertEqual(h["H2"]["verdict"], "better")
        self.assertEqual(h["H3"]["verdict"], "better")
        self.assertIn("t_p_holm", h["H2"])
        self.assertNotIn("t_p_holm", h["H1"])        # H1 stands alone
        for f in ("0.5", "1.0", "2.0"):
            self.assertEqual(result["by_overall_factor"][f]["n_states"], 16)
        curve = result["learning"]["mean_abs_log_error_by_segment"]
        self.assertEqual(len(curve), 13)
        self.assertGreater(curve[0], curve[-1])
        self.assertAlmostEqual(curve[-1], 0.0)
        m = result["arm_means"]["plan_learn"]
        self.assertAlmostEqual(m["J_zonal"], m["J_bias_term"]
                               + m["J_pattern_term"], 9)
        self.assertEqual(saved["registered"]["primary"][1], "plan_learn")

    def test_smoke_mode_and_missing_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _world(Path(tmp), self.SCALE, macros=[16],
                                branches=[0])
            buf = io.StringIO()
            with redirect_stdout(buf):
                bx.main(["--runs-dir", str(runs), "--references-dirs",
                         str(refs[0]), "--output", str(Path(tmp) / "s.json"),
                         "--smoke"])
            self.assertIn("every arm loaded", buf.getvalue())
            self.assertFalse((Path(tmp) / "s.json").exists())
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _world(Path(tmp), self.SCALE, macros=[16],
                                branches=[0], arms=bx.ARMS[:-1])
            with self.assertRaises(SystemExit):
                bx.main(["--runs-dir", str(runs), "--references-dirs",
                         str(refs[0]), "--output", str(Path(tmp) / "s.json"),
                         "--smoke"])


if __name__ == "__main__":
    unittest.main()
