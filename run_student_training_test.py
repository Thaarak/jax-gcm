"""Tests for run_student_training.py that need no model run."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_student_training import (
    budget_seconds,
    check_training_role,
    direct_config,
    make_specs,
    parse_args,
    teacher_config,
    validate_args,
)

COMMON = ["--ic-dir", "x", "--ic-positions", "0", "1", "--references-dir",
          "r", "--output-dir", "o"]


class SpecTest(unittest.TestCase):
    def test_episodes_are_fixed_and_inside_the_range(self):
        entries = [(0, {"index": 2}), (1, {"index": 202})]
        a = make_specs(entries, 2, 3, (0.5, 2.0), seed=4)
        b = make_specs(entries, 2, 3, (0.5, 2.0), seed=4)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2 * 2 * 3)
        self.assertTrue(all(0.5 <= s.efficacy <= 2.0 for s in a))
        self.assertEqual({(s.ic_index, s.member) for s in a},
                         {(2, 0), (2, 1), (202, 0), (202, 1)})
        self.assertNotEqual(a, make_specs(entries, 2, 3, (0.5, 2.0), seed=5))

    def test_only_training_roles(self):
        check_training_role("exp3_train")
        for role in ("exp3_eval", "exp1", "train_eval"):
            with self.assertRaises(SystemExit):
                check_training_role(role)


class ArgsTest(unittest.TestCase):
    def test_dagger_defaults_and_teacher(self):
        args = parse_args(["dagger", *COMMON, "--optimizer", "gauss_newton",
                           "--iterations", "3"])
        validate_args(args)
        cfg = teacher_config(args)
        self.assertEqual((cfg.optimizer, cfg.iterations, cfg.lookahead_days),
                         ("gauss_newton", 3, 60))
        with self.assertRaises(SystemExit):
            validate_args(parse_args(["dagger", *COMMON,
                                      "--efficacy-range", "2", "1"]))
        with self.assertRaises(SystemExit):
            validate_args(parse_args(["dagger", *COMMON, "--beta0", "3"]))

    def test_direct_methods_and_budgets(self):
        args = parse_args(["direct", *COMMON, "--method", "bptt",
                           "--budget-seconds", "100"])
        validate_args(args)
        self.assertEqual(direct_config(args).method, "bptt")
        self.assertEqual(budget_seconds(args), 100.0)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "dagger_log.json"
            log.write_text(json.dumps({"seconds_total": 1234.5}))
            args = parse_args(["direct", *COMMON, "--method", "eki",
                               "--budget-from", str(log)])
            self.assertEqual(budget_seconds(args), 1234.5)
        with self.assertRaises(SystemExit):
            parse_args(["direct", *COMMON, "--method", "eki"])
        with self.assertRaises(SystemExit):
            validate_args(parse_args(["direct", *COMMON, "--method", "eki",
                                      "--budget-seconds", "1",
                                      "--ensemble", "1"]))
        np.testing.assert_equal(direct_config(args).ensemble, 32)


if __name__ == "__main__":
    unittest.main()
