"""Tests for the planning-cost sweep's grid, guards and projections."""

import unittest

from run_planner_cost import (
    EPISODE_DAYS,
    FORWARD_S_PER_DAY,
    SEGMENT_DAYS,
    child_command,
    parse_args,
    projections,
    relative_difference,
    sweep_configs,
    validate_args,
)


class SweepTest(unittest.TestCase):
    def _args(self, *extra):
        return parse_args(["--ic-dir", "x", "--references", "r.npz",
                           "--output", "o.json", *extra])

    def test_default_grid(self):
        args = self._args("--full-bptt-check")
        validate_args(args)
        grid = sweep_configs(args)
        self.assertEqual(len(grid), 7)
        self.assertIn(("adam", 60, 14, False), grid)
        self.assertIn(("gauss_newton", 120, 14, False), grid)
        self.assertEqual(grid[-1], ("adam", 60, 0, True))

    def test_rejects_bad_settings(self):
        for extra in (("--lookaheads", "0"), ("--copies", "0"),
                      ("--repeats", "0"), ("--gn-iterations", "0")):
            with self.assertRaises(SystemExit, msg=str(extra)):
                validate_args(self._args(*extra))

    def test_child_command_round_trips(self):
        args = self._args("--copies", "2")
        cmd = child_command(args, "adam", 60, 0, True)
        child = parse_args(cmd[2:])
        self.assertEqual((child.single, child.copies, child.sequential_only),
                         ("adam:60:0", 2, True))


class ProjectionTest(unittest.TestCase):
    def test_projection_arithmetic(self):
        p = projections(10.0, 15)
        replans = EPISODE_DAYS // SEGMENT_DAYS
        self.assertEqual(replans, 13)
        self.assertEqual(p["replan_s"], 150.0)
        expected_h = (13 * 150.0 + EPISODE_DAYS * FORWARD_S_PER_DAY) / 3600
        self.assertAlmostEqual(p["episode_h"], expected_h)
        self.assertAlmostEqual(p["gpu_h_per_100_episodes"], 100 * expected_h)

    def test_relative_difference(self):
        self.assertEqual(relative_difference([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertAlmostEqual(relative_difference([1.0, 2.2], [1.0, 2.0]),
                               0.1)


if __name__ == "__main__":
    unittest.main()
