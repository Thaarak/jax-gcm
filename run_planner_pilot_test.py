"""Tests for the pure helpers of run_planner_pilot.py (Part 18 step 23)."""

import unittest

import numpy as np

from run_planner_pilot import (
    choose_warming,
    linear_solve,
    objective,
    paired,
    zonal_project,
    zonal_project_jacobian,
)


def _grid(ix=6, il=4, seed=0):
    rng = np.random.default_rng(seed)
    ocean = (rng.random((ix, il)) > 0.3).astype(np.float64)
    ocean[0] = 1.0                       # every latitude has ocean
    lats = np.linspace(-1.0, 1.0, il)
    w = np.cos(lats)[None, :] * ocean
    return ocean, w / w.sum(), rng


class ZonalTest(unittest.TestCase):

    def test_zonal_projection_is_constant_along_longitude_on_ocean(self):
        ocean, _, rng = _grid()
        z = zonal_project(rng.normal(size=ocean.shape), ocean)
        for j in range(ocean.shape[1]):
            vals = z[ocean[:, j] > 0, j]
            np.testing.assert_allclose(vals, vals[0])
        np.testing.assert_array_equal(z[ocean == 0], 0.0)

    def test_projection_keeps_the_weighted_mean_and_is_idempotent(self):
        ocean, w, rng = _grid()
        x = rng.normal(size=ocean.shape)
        z = zonal_project(x, ocean)
        self.assertAlmostEqual(np.sum(w * z), np.sum(w * x))
        np.testing.assert_allclose(zonal_project(z, ocean), z)

    def test_jacobian_projection_matches_per_band(self):
        ocean, _, rng = _grid()
        jac = rng.normal(size=ocean.shape + (3,))
        out = zonal_project_jacobian(jac, ocean)
        for b in range(3):
            np.testing.assert_allclose(out[..., b],
                                       zonal_project(jac[..., b], ocean))

    def test_objective_is_bias_plus_half_variance(self):
        ocean, w, rng = _grid()
        e = rng.normal(size=ocean.shape) * ocean
        mean = np.sum(w * e)
        var = np.sum(w * (e - mean) ** 2)
        self.assertAlmostEqual(objective(e, w), mean ** 2 + 0.5 * var)


class LinearSolveTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.big_r = rng.normal(size=(20, 3))
        self.a_star = np.array([0.02, 0.05, 0.08])
        self.point = np.full(3, 0.05)
        # r(point) such that the exact minimizer is a_star.
        self.r = self.big_r @ (self.point - self.a_star)

    def test_recovers_the_known_minimizer(self):
        a = linear_solve([(self.r, self.big_r)], self.point, cap=0.15,
                         damping=0.0)
        np.testing.assert_allclose(a, self.a_star, atol=1e-10)

    def test_mean_over_copies_equals_stacked_least_squares(self):
        a = linear_solve([(self.r, self.big_r)] * 3, self.point, cap=0.15,
                         damping=0.0)
        np.testing.assert_allclose(a, self.a_star, atol=1e-10)

    def test_penalty_shrinks_and_box_clips(self):
        free = linear_solve([(self.r, self.big_r)], self.point, 0.15,
                            damping=0.0)
        shrunk = linear_solve([(self.r, self.big_r)], self.point, 0.15,
                              mu=50.0, damping=0.0)
        self.assertLess(np.linalg.norm(shrunk), np.linalg.norm(free))
        clipped = linear_solve([(self.r, self.big_r)], self.point, 0.03,
                               damping=0.0)
        self.assertTrue(np.all(clipped <= 0.03) and np.all(clipped >= 0.0))

    def test_uniform_solution_is_the_best_single_value(self):
        a = linear_solve([(self.r, self.big_r)], self.point, 0.15,
                         uniform=True)
        self.assertTrue(np.allclose(a, a[0]))

        def cost(u):
            return np.sum((self.r + self.big_r @ (np.full(3, u)
                                                  - self.point)) ** 2)
        grid = np.linspace(0.0, 0.15, 3001)
        best = grid[np.argmin([cost(u) for u in grid])]
        self.assertAlmostEqual(a[0], best, places=4)


class RulesTest(unittest.TestCase):

    def test_paired_difference(self):
        p = paired([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(p["diff"], 0.0)
        self.assertAlmostEqual(p["se"], 1.0 / np.sqrt(3))

    def test_warming_rule_picks_smallest_ramp_meeting_three_to_one(self):
        def ratios(zonal):
            return {"sst@ocean:map": 0.1, "sst@ocean:zonal": zonal,
                    "sst@ocean:mean": 50.0,
                    "precipitation@land:zonal": 0.2}
        sn = {"ramps": {"4.0": {"ratios": ratios(1.5)},
                        "6.0": {"ratios": ratios(3.3)},
                        "8.0": {"ratios": ratios(5.9)}}}
        choice = choose_warming(sn)
        self.assertEqual(choice["ramp_end_wm2"], 6.0)
        self.assertEqual(choice["representation"], "zonal")
        self.assertFalse(choice["other_variables_scored_zonal"]
                         ["precipitation@land"])


if __name__ == "__main__":
    unittest.main()
