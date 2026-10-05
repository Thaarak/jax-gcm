"""Tests for the fixed-pattern ladder (jcm.mcb.ladder).

The synthetic tests check the algebra; the toy test checks the whole chain
(step runs, response maps, design, run) on the test world's linear toy, where
the linear model is exact up to float32 round-off.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.ladder import (
    ladder_designs,
    linear_response_design,
    linear_system,
    planner_average,
    predicted_objective,
    response_maps,
    uniform_at_effort,
    uniform_to_cancel,
)
from jcm.mcb.scores import area_weights, effort, segment_objective
from jcm.mcb.test_world import make_segment_fn, make_warming, time_means
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

RNG = np.random.default_rng(7)
IX, IL, K = 6, 4, 3


def _problem(n_states=1, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 1.5, (IX, IL))
    w[0] = 0.0                     # a "land" row with no weight
    w /= w.sum()
    resp = -rng.uniform(0.2, 1.0, (n_states, K, IX, IL))
    warm = rng.uniform(0.05, 0.15, (n_states, IX, IL))
    return w, warm, resp


class LinearSystemTest(unittest.TestCase):
    def test_sum_of_squares_equals_mean_objective(self):
        w, warm, resp = _problem(n_states=3)
        mu = 0.02
        r0, big_r = linear_system(warm, resp, w, 1.0, 0.5, mu)
        for _ in range(5):
            a = RNG.uniform(0.0, 0.2, K)
            r = r0 + big_r @ a
            direct = np.mean([float(segment_objective(
                jnp.asarray(e + np.tensordot(a, rr, axes=1)), jnp.asarray(w),
                jnp.asarray(a), jnp.asarray(a), 1.0, 0.5, mu, 0.0))
                for e, rr in zip(warm, resp)])
            self.assertAlmostEqual(float(r @ r), direct, delta=1e-6 * direct)

    def test_single_state_shapes_are_accepted(self):
        w, warm, resp = _problem()
        self.assertAlmostEqual(
            predicted_objective(np.zeros(K), warm[0], resp[0], w),
            predicted_objective(np.zeros(K), warm, resp, w))

    def test_rejects_mismatched_shapes(self):
        w, warm, resp = _problem(n_states=2)
        with self.assertRaises(ValueError):
            linear_system(warm, resp[:1], w)

    def test_response_maps(self):
        base = RNG.normal(size=(IX, IL))
        true = RNG.normal(size=(K, IX, IL))
        runs = base[None] + 0.05 * true
        np.testing.assert_allclose(response_maps(base, runs, 0.05), true,
                                   atol=1e-12)
        with self.assertRaises(ValueError):
            response_maps(base, runs, 0.0)


class DesignTest(unittest.TestCase):
    def test_interior_optimum_is_recovered_exactly(self):
        w, _, resp = _problem()
        a_true = np.array([0.03, 0.07, 0.05])
        warm = -np.tensordot(a_true, resp[0], axes=1)[None]
        a = linear_response_design(warm, resp, w, 1.0, 0.5, 0.0, cap=0.15)
        np.testing.assert_allclose(a, a_true, atol=1e-8)
        self.assertLess(predicted_objective(a, warm, resp, w), 1e-20)

    def test_bounds_hold_and_the_design_beats_every_feasible_setting(self):
        w, warm, resp = _problem(n_states=2, seed=3)
        warm = warm * 4.0          # too much warming for the cap
        cap = 0.1
        a = linear_response_design(warm, resp, w, 1.0, 0.5, 0.01, cap=cap)
        self.assertTrue(np.all(a >= 0.0) and np.all(a <= cap))
        best = predicted_objective(a, warm, resp, w, 1.0, 0.5, 0.01)
        for _ in range(200):
            trial = RNG.uniform(0.0, cap, K)
            self.assertLessEqual(best, predicted_objective(
                trial, warm, resp, w, 1.0, 0.5, 0.01) + 1e-12)

    def test_uniform_cancel_zeroes_the_mean_bias(self):
        w, warm, resp = _problem()
        a, info = uniform_to_cancel(warm, resp, w, cap=1.0)
        self.assertFalse(info["clipped"])
        np.testing.assert_allclose(a, a[0])
        d = warm[0] + np.tensordot(a, resp[0], axes=1)
        self.assertAlmostEqual(float(np.sum(w * d)), 0.0, places=12)
        a, info = uniform_to_cancel(warm, resp, w, cap=1e-3)
        self.assertTrue(info["clipped"])
        np.testing.assert_allclose(a, 1e-3)

    def test_uniform_at_effort_matches_the_effort(self):
        unit = np.abs(RNG.normal(size=(K, IX, IL)))
        sphere = np.full((IX, IL), 1.0 / (IX * IL))
        a, info = uniform_at_effort(0.02, unit, sphere, cap=1.0)
        self.assertAlmostEqual(effort(a[None], unit, sphere), 0.02, places=12)
        self.assertFalse(info["clipped"])
        a, info = uniform_at_effort(50.0, unit, sphere, cap=0.1)
        self.assertTrue(info["clipped"])
        np.testing.assert_allclose(a, 0.1)

    def test_planner_average_weights_days_and_members(self):
        s = np.array([[[0.0, 0.1], [0.2, 0.3]],
                      [[0.1, 0.1], [0.1, 0.1]]])        # (members, seg, K)
        np.testing.assert_allclose(planner_average(s, 2), [0.1, 0.15])
        np.testing.assert_allclose(planner_average(s[0], 2, start_day=2),
                                   [0.2, 0.3])
        with self.assertRaises(ValueError):
            planner_average(s, 2, start_day=3, end_day=3)

    def test_ladder_designs_returns_every_rung(self):
        w, warm, resp = _problem()
        unit = np.abs(RNG.normal(size=(K, IX, IL)))
        sphere = np.full((IX, IL), 1.0 / (IX * IL))
        out = ladder_designs(warm, resp, w, unit, sphere,
                             planner_schedules=np.full((3, K), 0.05),
                             planner_effort=0.01, segment_days=2, cap=1.0)
        self.assertEqual(set(out), {"uniform_cancel", "uniform_effort",
                                    "planner_average", "linear_response"})
        for rung in out.values():
            self.assertEqual(len(rung["amplitudes"]), K)
            self.assertGreaterEqual(rung["predicted_objective"], 0.0)
        # The best fixed design can only beat the other fixed designs.
        best = out["linear_response"]["predicted_objective"]
        for name in ("uniform_cancel", "uniform_effort", "planner_average"):
            self.assertLessEqual(best, out[name]["predicted_objective"]
                                 + 1e-12)
        self.assertEqual(set(ladder_designs(warm, resp, w, unit, sphere,
                                            cap=1.0)),
                         {"uniform_cancel", "linear_response"})


class ToyChainTest(unittest.TestCase):
    """Step runs -> response maps -> design -> run, on the linear toy."""

    def test_design_predicts_its_own_run(self):
        patterns = gaussian_band_patterns(LATS, OCEAN, centers_deg=(30.0,
                                                                   -30.0),
                                          width_deg=15.0)
        k = patterns.shape[0]
        carry = _carry()
        q_base = carry["ocn"]["forcing"].q_flux
        heat = jnp.asarray(OCEAN, jnp.float32)
        normal_w = make_warming(carry, heat)
        warm_w = make_warming(carry, heat, step_wm2=0.5)
        n, window, delta, cap = 12, (4, 12), 0.2, 5.0
        seg = make_segment_fn(_toy_step, patterns, n, fields_fn=_toy_fields)
        one = jnp.asarray(1.0)

        def mean_sst(amplitudes, warming):
            _, f = seg(carry, jnp.asarray(amplitudes, jnp.float32), one,
                       q_base, warming)
            return time_means({"sst": np.asarray(f["sst"])}, *window)["sst"]

        normal = mean_sst(np.zeros(k), normal_w)
        warm = mean_sst(np.zeros(k), warm_w)
        steps = [mean_sst(delta * np.eye(k)[j], warm_w) for j in range(k)]
        resp = response_maps(warm, steps, delta)
        w = np.asarray(area_weights(LATS, OCEAN))
        d_warm = warm - normal
        a = linear_response_design(d_warm, resp, w, 1.0, 0.5, 0.0, cap=cap)
        self.assertTrue(np.all(a > 0.0) and np.all(a < cap))
        actual = float(segment_objective(
            jnp.asarray(mean_sst(a, warm_w) - normal), jnp.asarray(w),
            jnp.asarray(a), jnp.asarray(a), 1.0, 0.5, 0.0, 0.0))
        predicted = predicted_objective(a, d_warm, resp, w, 1.0, 0.5)
        warm_j = predicted_objective(np.zeros(k), d_warm, resp, w, 1.0, 0.5)
        # float32 SSTs near 290 K carry about 3e-5 K of round-off.
        self.assertAlmostEqual(actual, predicted, delta=1e-6 + 1e-3 * warm_j)
        self.assertLess(actual, 0.2 * warm_j)
        # Run the uniform rung too: the best fixed design must beat it.
        u, _ = uniform_to_cancel(d_warm, resp, w, cap=cap)
        uniform_actual = float(segment_objective(
            jnp.asarray(mean_sst(u, warm_w) - normal), jnp.asarray(w),
            jnp.asarray(u), jnp.asarray(u), 1.0, 0.5, 0.0, 0.0))
        self.assertLess(actual, uniform_actual)


if __name__ == "__main__":
    unittest.main()
