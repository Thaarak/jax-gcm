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

    def test_rejects_bad_planner_settings(self):
        for extra in (("--copies", "0"), ("--lookahead-days", "0"),
                      ("--planner-efficacy", "0"), ("--learning-rate", "-1"),
                      ("--representation", "zonal")):        # Adam: no
            with self.assertRaises(SystemExit, msg=str(extra)):
                validate_args(self._args(*extra))


if __name__ == "__main__":
    unittest.main()
