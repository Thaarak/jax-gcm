"""Tests for the test-world driver's argument handling and guards."""

import unittest

import numpy as np

from jcm.mcb.test_world import BRIGHTENING_CAP
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from run_test_world import (
    check_role_allowed,
    check_warming_matches,
    episode_amplitudes,
    parse_args,
    planner_config,
    select_branches,
    strength_vector,
    validate_args,
)


class ReferencesArgsTest(unittest.TestCase):
    def _args(self, *extra):
        return parse_args(["references", "--ic-dir", "x", "--output-dir", "y",
                           *extra])

    def test_defaults_are_valid(self):
        args = self._args()
        validate_args(args)
        self.assertEqual((args.days, args.members, args.member_block_days),
                         (240, 5, 5))
        self.assertTrue(args.warmed)

    def test_rejects_bad_settings(self):
        for extra in (("--days", "0"), ("--members", "0"),
                      ("--days", "12", "--member-block-days", "5"),
                      ("--member-block-days", "-1")):
            with self.assertRaises(SystemExit, msg=str(extra)):
                validate_args(self._args(*extra))
        validate_args(self._args("--days", "12", "--member-block-days", "0"))

    def test_branch_selection(self):
        ics = [({"index": 1600 + b, "branch": b}, None) for b in (0, 1, 2)]
        self.assertEqual(select_branches(ics, None), ics)
        self.assertEqual([e["branch"] for e, _ in select_branches(ics, [0, 2])],
                         [0, 2])
        with self.assertRaises(SystemExit):
            select_branches(ics, [5])
        self.assertEqual(self._args("--branches", "1").branches, [1])
        self.assertIsNone(self._args().branches)

    def test_evaluation_roles_need_explicit_permission(self):
        with self.assertRaises(SystemExit):
            check_role_allowed("exp2_eval", allow_eval=False)
        check_role_allowed("exp2_eval", allow_eval=True)
        check_role_allowed("exp2_train", allow_eval=False)
        check_role_allowed("exp1", allow_eval=False)


class EpisodeArgsTest(unittest.TestCase):
    def _args(self, *extra):
        return parse_args(["episode", "--ic-dir", "x", "--references",
                           "r.npz", "--output", "o.json", "--segments", "3",
                           *extra])

    def test_amplitudes_uniform_or_explicit(self):
        np.testing.assert_allclose(
            episode_amplitudes(self._args("--uniform", "0.05")), [0.05] * 5)
        explicit = self._args("--amplitudes", "0", "0.01", "0.02", "0.03",
                              "0.04")
        validate_args(explicit)
        np.testing.assert_allclose(episode_amplitudes(explicit),
                                   [0, 0.01, 0.02, 0.03, 0.04])
        with self.assertRaises(SystemExit):
            self._args()          # one of the two is required

    def test_rejects_bad_settings(self):
        bad = [("--uniform", str(BRIGHTENING_CAP + 0.01)),
               ("--uniform", "-0.01"),
               ("--uniform", "0.05", "--score-start-day", "42"),
               ("--uniform", "0.05", "--score-end-day", "43"),
               ("--uniform", "0.05", "--segment-days", "0"),
               ("--uniform", "0.05", "--efficacy", "-1")]
        bad.append(("--uniform", "0.05", "--members", "0"))
        for extra in bad:
            with self.assertRaises(SystemExit, msg=str(extra)):
                validate_args(self._args(*extra))
        validate_args(self._args("--uniform", "0.05", "--score-start-day",
                                 "14", "--score-end-day", "42"))
        # By default the episode runs as many members as the references.
        self.assertIsNone(self._args("--uniform", "0.05").members)

    def test_warming_must_match_the_references(self):
        ref = {"warming_step_wm2": 4.0, "warming_ramp_wm2_per_day": 0.0}
        check_warming_matches(ref, self._args("--uniform", "0.05"))
        with self.assertRaises(SystemExit):
            check_warming_matches(ref, self._args(
                "--uniform", "0.05", "--warming-step-wm2", "2"))


class PlanArgsTest(unittest.TestCase):
    def _args(self, *extra):
        return parse_args(["plan", "--ic-dir", "x", "--references", "r.npz",
                           "--output", "o.json", "--segments", "3", *extra])

    def test_default_is_the_snipped_60_day_planner(self):
        args = self._args()
        validate_args(args)
        cfg = planner_config(args)
        self.assertEqual((cfg.lookahead_days, cfg.window_days, cfg.copies,
                          cfg.optimizer), (60, 14, 3, "adam"))

    def test_presets_and_overrides(self):
        cfg = planner_config(self._args("--preset", "short14"))
        self.assertEqual((cfg.lookahead_days, cfg.window_days),
                         (14, NO_TRUNCATION_DAYS))
        cfg = planner_config(self._args("--window-days", "0", "--optimizer",
                                        "gauss_newton", "--iterations", "3",
                                        "--mu", "0.5"))
        self.assertEqual((cfg.window_days, cfg.optimizer, cfg.iterations,
                          cfg.mu), (NO_TRUNCATION_DAYS, "gauss_newton", 3,
                                    0.5))

    def test_zonal_representation(self):
        cfg = planner_config(self._args("--optimizer", "gauss_newton",
                                        "--representation", "zonal"))
        self.assertEqual(cfg.representation, "zonal")
        self.assertEqual(planner_config(self._args()).representation, "map")

    def test_strength_per_band(self):
        np.testing.assert_allclose(strength_vector(1.0), np.ones(5))
        np.testing.assert_allclose(strength_vector([2.0]), np.full(5, 2.0))
        five = [0.5, 1.0, 1.5, 2.0, 0.8]
        np.testing.assert_allclose(strength_vector(five), five)
        with self.assertRaises(SystemExit):
            strength_vector([1.0, 2.0])
        args = self._args("--efficacy", *map(str, five),
                          "--planner-efficacy", "1")
        validate_args(args)
        with self.assertRaises(SystemExit):
            validate_args(self._args("--efficacy", "1", "-1", "1", "1", "1"))
        with self.assertRaises(SystemExit):
            validate_args(self._args("--planner-efficacy", "1", "0", "1",
                                     "1", "1"))

    def test_learning_planner_settings(self):
        good = ("--preset", "short14", "--optimizer", "gauss_newton",
                "--iterations", "1", "--copies", "1",
                "--representation", "zonal", "--learn-strength", "bands",
                "--learn-noise-k", "0.01")
        args = self._args(*good)
        validate_args(args)
        self.assertEqual(args.learn_strength, "bands")
        self.assertEqual(self._args().learn_strength, "off")
        base = ("--preset", "short14", "--iterations", "1", "--copies", "1",
                "--learn-strength", "bands")
        gn = ("--optimizer", "gauss_newton")
        noise = ("--learn-noise-k", "0.01")
        bad = {"no noise level": base + gn,
               "Adam": base + noise,
               "not starting at nominal": base + gn + noise
               + ("--planner-efficacy", "2"),
               "look-ahead longer than a segment": base + gn + noise
               + ("--segment-days", "7"),
               "zero noise": base + gn + ("--learn-noise-k", "0")}
        for why, extra in bad.items():
            with self.assertRaises(SystemExit, msg=why):
                validate_args(self._args(*extra))

    def test_rejects_bad_planner_settings(self):
        for extra in (("--copies", "0"), ("--lookahead-days", "0"),
                      ("--planner-efficacy", "0"), ("--learning-rate", "-1"),
                      ("--representation", "zonal")):        # Adam: no
            with self.assertRaises(SystemExit, msg=str(extra)):
                validate_args(self._args(*extra))


if __name__ == "__main__":
    unittest.main()
