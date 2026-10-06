"""Tests for run_controllers.py that need no model run.

The ladder stage is run end to end on synthetic references and response
files. The model stages are smoke-tested on the coupled model by hand.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jcm.mcb.ladder import predicted_objective
from run_controllers import (
    experiment1_responses,
    main,
    parse_args,
    validate_args,
    window_inputs,
)

LATS = np.deg2rad(np.array([-45.0, -15.0, 15.0, 45.0]))
IX, IL, K, DAYS = 6, 4, 5, 28


def _write_world(tmp: Path, n_states=2, delta=0.1):
    """Write grid.npz, references and response files for a linear world."""
    rng = np.random.default_rng(0)
    ocean = np.ones((IX, IL), np.float32)
    ocean[0] = 0.0
    np.savez(tmp / "grid.npz", ocean_mask=ocean, land_mask=1.0 - ocean,
             latitudes_rad=LATS, longitudes_rad=np.linspace(0, 6, IX))
    days = np.arange(DAYS)[:, None, None]
    paths = []
    for s in range(n_states):
        idx = 2 + 200 * s
        normal = 290.0 + 0.1 * rng.normal(size=(DAYS, IX, IL))
        warmed = normal + 0.004 * days * ocean
        refs = {"normal_sst": normal, "warmed_sst": warmed,
                "normal_land_temperature": normal,
                "warmed_land_temperature": warmed}
        np.savez(tmp / f"ic{idx:04d}_references.npz",
                 **{k: v.astype(np.float32) for k, v in refs.items()})
        per_unit = -0.03 * rng.uniform(0.5, 1.5, (K, IX, IL)) * ocean
        runs = warmed[None] + delta * days[None] * per_unit[:, None]
        blocks = runs.reshape(K, DAYS // 7, 7, IX, IL).mean(axis=2)
        meta = {"delta": delta, "days": DAYS, "block_days": 7,
                "references_file": f"ic{idx:04d}_references.npz"}
        path = tmp / f"ic{idx:04d}_responses.npz"
        np.savez(path, sst_blocks=blocks.astype(np.float32),
                 indices=np.zeros((K, DAYS, 4), np.float32),
                 meta=json.dumps(meta))
        paths.append(path)
    return paths


class ArgsTest(unittest.TestCase):
    def test_validation(self):
        base = ["responses", "--ic-dir", "x", "--references", "r.npz",
                "--output-dir", "o"]
        validate_args(parse_args(base))
        for extra in (["--days", "10"], ["--delta", "0.5"]):
            with self.assertRaises(SystemExit):
                validate_args(parse_args(base + extra))
        fb = ["feedback", "--controller", "pi", "--ic-dir", "x",
              "--references", "r.npz", "--segments", "2", "--output", "o"]
        validate_args(parse_args(fb))
        with self.assertRaises(SystemExit):
            validate_args(parse_args(fb + ["--pattern", "1", "2"]))


class Experiment1Test(unittest.TestCase):
    def test_sensitivities_have_the_expected_signs(self):
        if not Path("mcb_experiments_gpu/exp1_gradient_fidelity.npz").exists():
            self.skipTest("Experiment 1's results are not on this machine")
        r = experiment1_responses()
        self.assertEqual(r.shape[1:], (3, 5))
        self.assertTrue(np.all(r[59, 0] < 0.0))           # T0 cools
        # T1 is north minus south: brightening 45N lowers it, 45S raises it.
        self.assertLess(r[59, 1, 0], 0.0)
        self.assertGreater(r[59, 1, 4], 0.0)


class LadderStageTest(unittest.TestCase):
    def test_designs_from_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = _write_world(tmp)
            plan = {"schedules": [[[0.02] * K, [0.04] * K]], "effort": 0.03,
                    "config": {"segment_days": 14}}
            with open(tmp / "plan.json", "w") as f:
                json.dump(plan, f)
            out = tmp / "ladder.json"
            main(["ladder", "--responses", *map(str, paths),
                  "--references-dir", str(tmp), "--window", "14", "28",
                  "--planner-summaries", str(tmp / "plan.json"),
                  "--output", str(out)])
            summary = json.loads(out.read_text())
            warm, resp = window_inputs(paths[0], tmp, (14, 28))
            with self.assertRaises(SystemExit):
                window_inputs(paths[0], tmp, (10, 28))
        pooled = summary["pooled"]
        self.assertEqual(set(pooled), {"uniform_cancel", "uniform_effort",
                                       "planner_average", "linear_response"})
        np.testing.assert_allclose(pooled["planner_average"]["amplitudes"],
                                   0.03)
        self.assertLessEqual(pooled["linear_response"]["predicted_objective"],
                             pooled["uniform_cancel"]["predicted_objective"])
        rungs = next(iter(summary["per_planner_run"].values()))
        np.testing.assert_allclose(rungs["planner_average"]["amplitudes"],
                                   0.03)
        self.assertEqual(warm.shape, (IX, IL))
        self.assertEqual(resp.shape, (K, IX, IL))
        self.assertGreater(predicted_objective(np.zeros(K), warm, resp,
                                               np.full((IX, IL), 1 / 24)),
                           0.0)

    def test_pooled_planner_runs_and_zonal_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = _write_world(tmp)
            plans = []
            for i, (level, eff) in enumerate(((0.02, 0.02), (0.06, 0.04))):
                plan = {"schedules": [[[level] * K] * 2], "effort": eff,
                        "config": {"segment_days": 14}}
                plans.append(tmp / f"plan{i}.json")
                plans[-1].write_text(json.dumps(plan))
            outs = {}
            for rep in ("map", "zonal"):
                out = tmp / f"ladder_{rep}.json"
                main(["ladder", "--responses", *map(str, paths),
                      "--references-dir", str(tmp), "--window", "14", "28",
                      "--planner-summaries", *map(str, plans),
                      "--representation", rep, "--output", str(out)])
                outs[rep] = json.loads(out.read_text())["pooled"]
        for pooled in outs.values():
            np.testing.assert_allclose(
                pooled["planner_average"]["amplitudes"], 0.04)
            self.assertLessEqual(
                pooled["linear_response"]["predicted_objective"],
                pooled["uniform_cancel"]["predicted_objective"] + 1e-12)
        # The zonal objective drops grid-point scatter, so it is smaller.
        self.assertLess(outs["zonal"]["uniform_cancel"]["predicted_objective"],
                        outs["map"]["uniform_cancel"]["predicted_objective"])
        u = [outs[r]["uniform_effort"]["uniform"] for r in outs]
        self.assertAlmostEqual(u[0], u[1])            # effort: not scored


if __name__ == "__main__":
    unittest.main()
