"""Tests for the registered Experiment 3a analysis (analyze_experiment3a.py)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_experiment3a as ax

IX, IL, DAYS = 6, 4, 182
LATS = np.deg2rad(np.array([-45.0, -15.0, 15.0, 45.0]))


def _world(tmp: Path, scale: dict, n_macro=8, branches=3, seed=0):
    """Write references and every arm's runs; arm error = scale * base."""
    rng = np.random.default_rng(seed)
    ocean = np.ones((IX, IL))
    ocean[0] = 0.0
    refs = tmp / "refs"
    refs.mkdir()
    np.savez(refs / "grid.npz", ocean_mask=ocean, land_mask=1.0 - ocean,
             latitudes_rad=LATS, longitudes_rad=np.linspace(0, 6, IX))
    for macro in range(1, 2 * n_macro, 2):
        for br in range(2, 2 + branches):
            idx = 100 * macro + br
            normal = 290.0 + rng.normal(0, 0.1, (DAYS, IX, IL))
            np.savez(refs / f"ic{idx:04d}_references.npz",
                     normal_sst=normal, normal_land_temperature=normal,
                     normal_precipitation=normal)
            # A latitude-shaped warm error, different in every state.
            base = (0.1 + 0.05 * rng.normal()) * (1.0 + LATS[None, :] ** 2)
            base = base + 0.01 * rng.normal(size=(IX, IL))
            for arm in ax.ARMS:
                d = tmp / "runs" / arm
                d.mkdir(parents=True, exist_ok=True)
                f = normal + scale.get(arm, 1.0) * base[None]
                np.savez(d / f"ic{idx:04d}.fields.npz", sst=f,
                         land_temperature=f, precipitation=f)
                summary = {"schedules": [[[0.03] * 5] * 13], "effort": 0.02,
                           "members": 3, "seconds": 1.0}
                (d / f"ic{idx:04d}.json").write_text(json.dumps(summary))
    return tmp / "runs", refs


class StatsTest(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(ax.holm([0.01, 0.04, 0.03, 0.2]),
                                   [0.04, 0.09, 0.09, 0.2])

    def test_compare_and_verdicts(self):
        rng = np.random.default_rng(1)
        macros = np.repeat(np.arange(8), 3)
        b = rng.uniform(1.0, 2.0, 24)
        better = ax.compare(0.7 * b, b, macros, n_boot=500)
        self.assertAlmostEqual(better["rel"], -0.3)
        self.assertEqual(ax.verdict(better, better["t_p"]), "better")
        worse = ax.compare(1.3 * b, b, macros, n_boot=500)
        self.assertEqual(ax.verdict(worse, worse["t_p"]), "worse")
        same = ax.compare(b * (1 + 0.005 * rng.normal(size=24)), b, macros,
                          n_boot=500)
        self.assertEqual(ax.verdict(same, same["t_p"]), "equivalent")
        noisy = ax.compare(b * (1 + 0.5 * rng.normal(size=24)), b, macros,
                           n_boot=500)
        self.assertIn(ax.verdict(noisy, noisy["t_p"]),
                      ("inconclusive", "better", "worse"))

    def test_bootstrap_is_reproducible_and_hierarchical(self):
        macros = [1, 1, 3, 3]
        a, b = [1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]
        r1 = ax.hierarchical_rel(a, b, macros, n_boot=50)
        r2 = ax.hierarchical_rel(a, b, macros, n_boot=50)
        np.testing.assert_array_equal(r1, r2)
        # Only two macro states: every replicate is one of three values.
        self.assertTrue(set(np.round(r1, 6)) <= {-0.5, -0.25, 0.0})


class EndToEndTest(unittest.TestCase):
    def test_registered_analysis_on_a_synthetic_world(self):
        scale = {"uncontrolled": 1.0, "plan120": 0.3, "plan14": 0.6,
                 "plan60": 0.35, "linear_response": 0.5, "pi": 0.31,
                 "planner_average": 0.45, "uniform_cancel": 0.7,
                 "uniform_effort": 0.8, "adaptive": 0.6}
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _world(Path(tmp), scale)
            per_arm = ax.load_runs(runs, refs)
            result = ax.analyze(per_arm, n_boot=300)
            out = Path(tmp) / "a.json"
            ax.main(["--runs-dir", str(runs), "--references-dir", str(refs),
                     "--output", str(out)])
            saved = json.loads(out.read_text())
        self.assertEqual(result["n_states"], 24)
        h = result["hypotheses"]
        # J scales with the square of the error scale.
        self.assertAlmostEqual(h["H1"]["rel"], 0.3 ** 2 / 0.6 ** 2 - 1, 6)
        self.assertEqual(h["H1"]["verdict"], "better")
        self.assertEqual(h["H2"]["verdict"], "better")
        self.assertEqual(h["H4"]["verdict"], "better")
        self.assertIn(h["H3"]["verdict"], ("better", "negligible"))
        g = per_arm["plan120"][102]["G"]["sst@ocean"]
        self.assertAlmostEqual(g, 1 / 0.3 ** 2, 4)
        m = result["arm_means"]["plan120"]
        self.assertAlmostEqual(m["cap_share"], 0.2)
        self.assertGreater(m["J_map"], m["J_zonal"])
        self.assertEqual(saved["registered"]["primary"], "plan120")


if __name__ == "__main__":
    unittest.main()
