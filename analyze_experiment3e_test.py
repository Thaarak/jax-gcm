"""Tests for the registered Experiment 3e analysis (analyze_experiment3e.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import analyze_experiment3e as ex
from analyze_experiment3b_test import IL, _world

MACROS = range(101, 132, 2)
SCALE_3D = {"uncontrolled": 1.0, "plan_learn": 0.3, "plan_naive": 0.5,
            "pi": 0.45, "adaptive": 0.5, "fixed": 0.8}
SCALE_3E = {"plan_learn_a10": 0.32, "plan_naive_a10": 0.5,
            "plan_learn_a30": 0.36, "plan_learn_a100": 0.4}
# The share of the prior's log error each learner has left on day 98.
LEFT = {"plan_learn": 0.05, "plan_learn_a10": 0.3, "plan_learn_a30": 0.7,
        "plan_learn_a100": 0.95}
DESIGN = {"exact_learner_noise_k": 0.00069104,
          "levels": {"a10": {"analysis_error": 0.1, "learner_noise_k": 0.0057,
                             "detection_kappa": 0.0016},
                     "a30": {"analysis_error": 0.3, "learner_noise_k": 0.02,
                             "detection_kappa": 0.008},
                     "a100": {"analysis_error": 1.0, "learner_noise_k": 0.047,
                              "detection_kappa": 0.0145}}}
EXP3C = {"hypotheses": {"H1": {"rel": -0.15}},
         "learning": {"by_arm": {"plan_learn_rs": [0.59] * 7 + [0.57] * 6}}}
SEGMENTS, MEMBERS, K = 13, 3, 5


def _edit(path: Path, **changes):
    s = json.loads(path.read_text())
    s.update(changes)
    path.write_text(json.dumps(s))


def _planner_runs(runs: Path, arms, rng, segments=SEGMENTS):
    """Give every run what a real one records: land, sensing, learner, logs, misses."""
    for arm in arms:
        f = ex.ANALYSIS_ERROR.get(arm)
        for js in sorted((runs / arm).glob("ic*.json")):
            s = json.loads(js.read_text())
            s["land_climatology"] = "monthly"
            if f is None:                       # pi, adaptive, fixed, none
                js.write_text(json.dumps(s))
                continue
            sensing, _, noise = ex.expected(arm, DESIGN)
            s["planner"] = {"sensing": sensing, "analysis_error": f}
            truth = np.asarray(s["true_efficacy"])
            if noise is None:
                s["learner"], s["planner_efficacy"] = None, [1.0] * K
                beliefs = np.ones((segments, K))
            else:
                s["learner"] = {"config": {"noise_k": noise}}
                # The log error shrinks linearly to LEFT[arm] of the prior's
                # by day 98 (segment 7), and stays there.
                left = 1.0 - (1.0 - LEFT[arm]) * np.minimum(
                    np.arange(segments) / 7.0, 1.0)
                beliefs = np.exp(np.log(truth)[None] * (1.0 - left[:, None]))
            s["planner_logs"] = [[{"segment": j, "day": 14 * j,
                                   "efficacy_belief": beliefs[j].tolist()}
                                  for j in range(segments)]] * MEMBERS
            js.write_text(json.dumps(s))
            if arm not in ex.ARMS_3E:
                continue
            # One miss per re-plan after the first: brightening works (the
            # truth), plus a forecast error of the level's size.
            n = segments - 1
            sens = rng.uniform(0.0, 0.02, (MEMBERS, n, IL, K))
            belief = np.broadcast_to(beliefs[:n], (MEMBERS, n, K))
            predicted = rng.normal(size=(MEMBERS, n, IL))
            sd = DESIGN["levels"][ex.LEVEL[arm]]["learner_noise_k"]
            observed = (predicted + np.einsum("mnik,mnk->mni", sens,
                                              truth[None, None] - belief)
                        + sd * rng.normal(size=(MEMBERS, n, IL)))
            np.savez(js.with_suffix(".innovations.npz"), observed=observed,
                     predicted=predicted, sensitivity=sens, belief=belief,
                     true_efficacy=truth, profile_weights=np.full(IL, 1 / IL),
                     day=14 * np.arange(1, n + 1))


def _worlds(tmp: Path, macros=MACROS, segments=SEGMENTS):
    """3d's runs and 3e's runs on the same states and references."""
    rng = np.random.default_rng(7)
    (tmp / "d").mkdir()
    (tmp / "e").mkdir()
    runs_3d, refs = _world(tmp / "d", SCALE_3D, macros=macros, seed=4,
                           arms=ex.ARMS_3D)
    runs_3e, _ = _world(tmp / "e", SCALE_3E, macros=macros, seed=4,
                        arms=ex.ARMS_3E)
    _planner_runs(runs_3d, ex.ARMS_3D, rng, segments)
    _planner_runs(runs_3e, ex.ARMS_3E, rng, segments)
    (tmp / "design.json").write_text(json.dumps(DESIGN))
    (tmp / "exp3c.json").write_text(json.dumps(EXP3C))
    return runs_3e, runs_3d, refs


def _args(tmp: Path, runs_3e, runs_3d, refs, *extra):
    return ["--runs-dir", str(runs_3e), "--runs-3d-dir", str(runs_3d),
            "--references-dirs", *map(str, refs), "--design",
            str(tmp / "design.json"), "--exp3c-analysis",
            str(tmp / "exp3c.json"), "--output", str(tmp / "a.json"), *extra]


class EndToEndTest(unittest.TestCase):
    def test_registered_analysis_on_a_synthetic_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3e, runs_3d, refs = _worlds(tmp)
            with redirect_stdout(io.StringIO()):
                ex.main(_args(tmp, runs_3e, runs_3d, refs))
            result = json.loads((tmp / "a.json").read_text())
        self.assertEqual((result["n_states"], result["n_macro"]), (48, 16))
        h = result["hypotheses"]
        self.assertEqual(sorted(h), ["H1", "H2", "H3", "H4"])
        self.assertEqual((h["H1"]["arm"], h["H1"]["comparator"]),
                         ("plan_learn_a10", "pi"))
        self.assertAlmostEqual(h["H1"]["rel"], 0.32 ** 2 / 0.45 ** 2 - 1, 6)
        self.assertNotIn("t_p_holm", h["H1"])
        self.assertEqual(h["H1"]["verdict"], "better")
        # H2-H4 are one Holm family; H4 is on the strength error.
        for name in ("H2", "H3", "H4"):
            self.assertIn("t_p_holm", h[name])
            self.assertGreaterEqual(h[name]["t_p_holm"], h[name]["t_p"])
        self.assertEqual(h["H2"]["verdict"], "better")
        self.assertEqual(h["H3"]["verdict"], "worse")      # vs 3d's learner
        self.assertEqual(h["H4"]["endpoint"], "strength_error_day98")
        self.assertAlmostEqual(h["H4"]["rel"], 0.3 / 0.95 - 1, 6)
        self.assertEqual(h["H4"]["verdict"], "better")
        dose = result["dose_response"]
        self.assertEqual([d["analysis_error"] for d in dose],
                         [0.0, 0.1, 0.3, 1.0])
        self.assertAlmostEqual(dose[0]["rel_vs_pi"], 0.3 ** 2 / 0.45 ** 2 - 1,
                               6)
        errors = [d["strength_error_day98"] for d in dose]
        self.assertEqual(errors, sorted(errors))
        rep = result["replication_3c"]
        self.assertEqual((rep["rel_3c"], rep["strength_error_day98_3c"]),
                         (-0.15, 0.57))
        self.assertTrue(rep["same_direction"])
        self.assertEqual(set(result["learning"]["by_arm"]), set(ex.LEARNERS))
        self.assertIn("plan_naive_a10_vs_plan_naive", result["descriptive"])
        # 3e's arms get their gains from 3d's no-brightening runs.
        self.assertIn("G_sst@ocean", result["arm_means"]["plan_learn_a10"])
        det = result["detection"]
        self.assertEqual(set(det), set(ex.ARMS_3E))
        for arm, d in det.items():
            self.assertEqual(d["files"], 48)
            sd = DESIGN["levels"][ex.LEVEL[arm]]["learner_noise_k"]
            # The miss less the strength error's part is the forecast error.
            self.assertAlmostEqual(d["forecast_error_noise_k"] / sd, 1.0,
                                   delta=0.25)
            self.assertEqual(set(d["share_detected"]),
                             {str(x) for x in ex.DETECTION_DAYS})
        # Brightening works everywhere, so the statistic finds it.
        self.assertGreater(det["plan_learn_a10"]["share_detected"]["168"], 0.9)
        self.assertEqual(result["registered"]["learning_day"], 98)

    def test_smoke_mode_runs_on_two_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs_3e, runs_3d, refs = _worlds(tmp, macros=[101], segments=2)
            out = io.StringIO()
            with redirect_stdout(out):
                ex.main(_args(tmp, runs_3e, runs_3d, refs, "--smoke"))
            self.assertFalse((tmp / "a.json").exists())
        text = out.getvalue()
        self.assertIn("smoke analysis: every arm loaded", text)
        self.assertIn("plan_learn_a100", text)
        self.assertIn("miss files", text)


class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_dir.name)
        self.runs_3e, self.runs_3d, _ = _worlds(self.tmp, macros=[101, 103])
        self.design = ex.load_design(self.tmp / "design.json")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def check(self):
        ex.check_runs(self.runs_3e, self.runs_3d, self.design)

    def refuses(self, path, **changes):
        before = path.read_text()
        _edit(path, **changes)
        with self.assertRaises(SystemExit):
            self.check()
        path.write_text(before)
        self.check()

    def test_the_synthetic_world_passes(self):
        self.check()

    def test_refuses_runs_that_break_the_design(self):
        learn = self.runs_3e / "plan_learn_a10" / "ic10100.json"
        naive = self.runs_3e / "plan_naive_a10" / "ic10100.json"
        self.refuses(learn, land_climatology=None)
        self.refuses(learn, planner={"sensing": "ocean"})
        self.refuses(learn, planner={"sensing": "analysis",
                                     "analysis_error": 0.3})
        self.refuses(learn, learner={"config": {"noise_k": 0.02}})
        self.refuses(learn, members=6)
        self.refuses(naive, learner={"config": {"noise_k": 0.0057}})
        self.refuses(naive, planner_efficacy=[1.0, 1.0, 2.0, 1.0, 1.0])
        self.refuses(self.runs_3d / "plan_learn" / "ic10100.json",
                     planner={"sensing": "analysis", "analysis_error": 0.1})
        self.refuses(self.runs_3d / "pi" / "ic10100.json",
                     land_climatology="daily_index")

    def test_refuses_a_design_with_other_levels(self):
        bad = json.loads(json.dumps(DESIGN))
        bad["levels"]["a30"]["analysis_error"] = 0.2
        (self.tmp / "bad.json").write_text(json.dumps(bad))
        with self.assertRaises(SystemExit):
            ex.load_design(self.tmp / "bad.json")


class StrengthErrorTest(unittest.TestCase):
    def test_it_reads_the_belief_planned_with_on_the_day(self):
        truth = [2.0, 0.5, 1.0, 1.0, 1.0]
        logs = [[{"day": 0, "efficacy_belief": [1.0] * 5},
                 {"day": 98, "efficacy_belief": [2.0, 1.0, 1.0, 1.0, 1.0]},
                 {"day": 112, "efficacy_belief": truth}]] * 2
        s = {"true_efficacy": truth, "planner_logs": logs}
        self.assertAlmostEqual(ex.strength_error(s), np.log(2.0) / 5)
        self.assertAlmostEqual(ex.strength_error(s, day=None), 0.0)
        with self.assertRaises(SystemExit):
            ex.strength_error(s, day=42)


if __name__ == "__main__":
    unittest.main()
