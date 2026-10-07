"""Tests for the fixed-pattern designs of Experiment 2 (jcm.mcb.design).

The toy is the test world's small model, which is linear in the brightening,
so its window-mean map is exactly linear in the setting. That makes the
designs' optima known in closed form.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.design import (
    BAND_LAYOUTS,
    SCORE_WINDOW,
    accumulated_window_mean,
    band_patterns,
    brute_force_design,
    equal_cost_members,
    gauss_newton_design,
    make_window_jacobian_fn,
    make_window_sst_fn,
    slab_heat_capacity,
    sunlight_design,
    sunlight_guess_maps,
    sunlight_guess_warming,
    toa_insolation,
    unit_profiles,
    zonal_design,
    zonal_objective,
)
from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.scores import area_weights
from jcm.mcb.test_world import make_segment_fn, make_warming
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

DAYS, WINDOW = 8, (3, 8)
NONE = jnp.asarray(NO_TRUNCATION_DAYS)


class _Toy(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.weights = np.asarray(area_weights(LATS, OCEAN), np.float64)
        heat = jnp.asarray(OCEAN, jnp.float32)
        # Two "training states": two starting Q-fluxes.
        self.carries = [_carry(), _carry(q_offset=0.3)]
        self.warmings = [make_warming(c, heat, step_wm2=0.5)
                         for c in self.carries]
        self.normal = [make_warming(c, heat) for c in self.carries]
        self.f = jax.jit(make_window_sst_fn(_toy_step, self.patterns, DAYS,
                                            WINDOW))
        self.jac = make_window_jacobian_fn(_toy_step, self.patterns, DAYS,
                                           WINDOW)
        one = jnp.asarray(1.0)
        self.targets = np.stack([np.asarray(self.f(
            jnp.zeros(2), c, one, c["ocn"]["forcing"].q_flux, w, NONE))
            for c, w in zip(self.carries, self.normal)])

    def evaluate(self, scale=1.0):
        one = jnp.asarray(1.0)

        def evaluate(s, a):
            c = self.carries[s]
            jac, mean_map = self.jac(jnp.asarray(a, jnp.float32), c, one,
                                     c["ocn"]["forcing"].q_flux,
                                     self.warmings[s], NONE)
            return np.asarray(mean_map), scale * np.asarray(jac)

        return evaluate

    def exact_optimum(self, cap=5.0):
        warm, resp = [], []
        for s in range(2):
            m, j = self.evaluate()(s, np.zeros(2))
            warm.append(m - self.targets[s])
            resp.append(np.moveaxis(j, -1, 0))
        return zonal_design(np.stack(warm), np.stack(resp), OCEAN,
                            self.weights, cap=cap)[0]


class WindowFunctionTest(_Toy):
    def test_window_mean_matches_a_plain_run(self):
        a = jnp.asarray([0.3, 0.1])
        c, w = self.carries[0], self.warmings[0]
        seg = make_segment_fn(_toy_step, self.patterns, DAYS,
                              fields_fn=_toy_fields)
        _, fields = seg(c, a, jnp.asarray(1.0), c["ocn"]["forcing"].q_flux, w)
        expected = np.asarray(fields["sst"])[3:8].mean(0) - MAP_REFERENCE_K
        got = np.asarray(self.f(a, c, jnp.asarray(1.0),
                                c["ocn"]["forcing"].q_flux, w, NONE))
        np.testing.assert_allclose(got, expected, atol=3e-5)

    def test_jacobian_matches_finite_differences(self):
        a, h = np.array([0.3, 0.1]), 0.05
        _, jac = self.evaluate()(0, a)
        c, w = self.carries[0], self.warmings[0]
        for k in range(2):
            e = np.eye(2)[k] * h
            plus = np.asarray(self.f(jnp.asarray(a + e), c, jnp.asarray(1.0),
                                     c["ocn"]["forcing"].q_flux, w, NONE))
            minus = np.asarray(self.f(jnp.asarray(a - e), c,
                                      jnp.asarray(1.0),
                                      c["ocn"]["forcing"].q_flux, w, NONE))
            np.testing.assert_allclose(jac[..., k], (plus - minus) / (2 * h),
                                       atol=2e-3)

    def test_the_snip_changes_the_jacobian_not_the_map(self):
        c, w = self.carries[0], self.warmings[0]
        args = (jnp.asarray([0.2, 0.2]), c, jnp.asarray(1.0),
                c["ocn"]["forcing"].q_flux, w)
        j_full, m_full = self.jac(*args, NONE)
        j_snip, m_snip = self.jac(*args, jnp.asarray(1))
        np.testing.assert_array_equal(np.asarray(m_full), np.asarray(m_snip))
        self.assertGreater(float(jnp.max(jnp.abs(j_full - j_snip))), 0.0)

    def test_window_must_fit(self):
        with self.assertRaises(ValueError):
            make_window_sst_fn(_toy_step, self.patterns, 5, (3, 8))


class GaussNewtonTest(_Toy):
    def test_one_exact_step_lands_on_the_optimum(self):
        a_star = self.exact_optimum()
        self.assertTrue(np.all((a_star > 0.0) & (a_star < 5.0)), a_star)
        a, log = gauss_newton_design(self.evaluate(), self.targets, OCEAN,
                                     self.weights, 2, 1, cap=5.0)
        np.testing.assert_allclose(a, a_star, rtol=2e-3, atol=1e-4)
        self.assertEqual(log[0]["states_used"], [0, 1])
        self.assertGreater(log[0]["training_objective"], 0.0)

    def test_repeating_the_step_corrects_an_undercounted_jacobian(self):
        a_star = self.exact_optimum()
        a1, _ = gauss_newton_design(self.evaluate(0.65), self.targets, OCEAN,
                                    self.weights, 2, 1, cap=5.0)
        # One step with a Jacobian that is 65% of the truth overshoots by
        # about 1 / 0.65; four steps leave about |1 - 1/0.65|^4 = 9%.
        np.testing.assert_allclose(a1, a_star / 0.65, rtol=0.02)
        a4, log = gauss_newton_design(self.evaluate(0.65), self.targets,
                                      OCEAN, self.weights, 2, 4, cap=5.0)
        self.assertLess(np.linalg.norm(a4 - a_star),
                        0.12 * np.linalg.norm(a_star))
        objectives = [r["training_objective"] for r in log]
        self.assertLess(objectives[-1], objectives[1])

    def test_non_finite_states_sit_out(self):
        good = self.evaluate()

        def half_broken(s, a):
            m, j = good(s, a)
            return (m, j * np.nan) if s == 0 else (m, j)

        _, log = gauss_newton_design(half_broken, self.targets, OCEAN,
                                     self.weights, 2, 1, cap=5.0)
        self.assertEqual(log[0]["states_used"], [1])

        def broken(s, a):
            m, j = good(s, a)
            return m, j * np.nan

        a, log = gauss_newton_design(broken, self.targets, OCEAN,
                                     self.weights, 2, 2, cap=5.0,
                                     start=[0.1, 0.2])
        np.testing.assert_allclose(a, [0.1, 0.2])
        self.assertIsNone(log[0]["training_objective"])


class BruteForceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ix, self.il, self.k, self.s, self.m = 6, 4, 3, 2, 5
        self.mask = np.ones((self.ix, self.il))
        self.mask[0] = 0.0
        self.w = np.asarray(area_weights(np.deg2rad(np.linspace(-45, 45, 4)),
                                         self.mask), np.float64)
        self.true_resp = -rng.uniform(0.2, 1.0, (self.s, self.k, self.ix,
                                                 self.il)) * self.mask
        self.targets = rng.normal(size=(self.s, self.ix, self.il)) * 0.01
        self.warm = self.targets + 0.1 * self.mask
        self.rng = rng

    def members(self, noise):
        warmed = (self.warm[:, None] + noise * self.rng.normal(
            size=(self.s, self.m, self.ix, self.il)))
        band = (warmed[:, None] + 0.1 * self.true_resp[:, :, None]
                + noise * self.rng.normal(size=(self.s, self.k, self.m,
                                                self.ix, self.il)))
        return warmed, band

    def test_noise_free_runs_give_the_exact_design(self):
        warmed, band = self.members(0.0)
        a, info = brute_force_design(warmed, band, self.targets, 0.1,
                                     self.mask, self.w, cap=1.0)
        exact, _ = zonal_design(self.warm - self.targets, self.true_resp,
                                self.mask, self.w, cap=1.0)
        np.testing.assert_allclose(a, exact, atol=1e-9)
        self.assertEqual(info["members"], self.m)
        self.assertEqual(len(info["uniform_cancel"]), self.k)

    def test_member_subsets_use_the_first_samples(self):
        warmed, band = self.members(0.02)
        a1, _ = brute_force_design(warmed, band, self.targets, 0.1,
                                   self.mask, self.w, cap=1.0, members=1)
        b1, _ = brute_force_design(warmed[:, :1], band[:, :, :1],
                                   self.targets, 0.1, self.mask, self.w,
                                   cap=1.0)
        np.testing.assert_allclose(a1, b1)
        with self.assertRaises(ValueError):
            brute_force_design(warmed, band, self.targets, 0.1, self.mask,
                               self.w, members=6)

    def test_equal_cost_members(self):
        # 4 Jacobian runs of 44 s vs (5 + 1) forward runs of 12.4 s.
        self.assertEqual(equal_cost_members(4, 44.0, 5, 12.4), 2)
        self.assertEqual(equal_cost_members(4, 93.0, 13, 12.4), 2)
        self.assertEqual(equal_cost_members(1, 1.0, 13, 12.4), 1)
        self.assertEqual(equal_cost_members(40, 100.0, 5, 1.0), 5)
        with self.assertRaises(ValueError):
            equal_cost_members(0, 1.0, 5, 1.0)


class SunlightTest(unittest.TestCase):
    def test_insolation_values(self):
        equinox = 365.2425 / 4 - 10            # declination zero
        self.assertAlmostEqual(float(toa_insolation(0.0, equinox)),
                               1361.0 / np.pi, places=6)
        june = 365.2425 / 2 - 10                # declination +23.44 deg
        self.assertEqual(float(toa_insolation(np.radians(-80.0), june)), 0.0)
        self.assertGreater(float(toa_insolation(np.radians(80.0), june)),
                           float(toa_insolation(0.0, june)))
        annual = np.mean([toa_insolation(0.0, d) for d in range(365)])
        self.assertAlmostEqual(float(annual), 417.0, delta=3.0)

    def test_heat_capacity_and_accumulation(self):
        self.assertAlmostEqual(float(slab_heat_capacity(0.0)),
                               1025.0 * 3985.0 * 40.0)
        self.assertAlmostEqual(float(slab_heat_capacity(np.pi / 2)),
                               1025.0 * 3985.0 * 60.0, places=3)
        flux = np.full((10, 2, 2), 2.0)
        got = accumulated_window_mean(flux, (4, 10))
        np.testing.assert_allclose(got, 2.0 * 86400 * np.mean(np.arange(5,
                                                                       11)))
        with self.assertRaises(ValueError):
            accumulated_window_mean(flux, (4, 12))

    def test_guess_maps_and_design(self):
        lats = np.deg2rad(np.array([-30.0, 0.0, 30.0]))
        mask = np.ones((4, 3))
        mask[0] = 0.0
        pats = np.asarray(gaussian_band_patterns(lats, mask, (30.0, -30.0),
                                                 15.0))
        cloud = np.full((20, 4, 3), 0.5)
        resp = sunlight_guess_maps(lats, pats, cloud, 30.0, (10, 20))
        self.assertTrue(np.all(resp[:, mask == 1] < 0.0))
        np.testing.assert_array_equal(resp[:, mask == 0], 0.0)
        np.testing.assert_allclose(sunlight_guess_maps(lats, pats, 2 * cloud,
                                                       30.0, (10, 20)),
                                   2 * resp)
        warm = sunlight_guess_warming(lats, mask, 20, 0.0, 0.1, (10, 20))
        self.assertTrue(np.all(warm[mask == 1] > 0.0))
        w = np.asarray(area_weights(lats, mask), np.float64)
        a, info = sunlight_design(lats, mask, pats, cloud, 30.0, w, 20, 0.0,
                                  0.1, (10, 20), cap=10.0)
        self.assertEqual(a.shape, (2,))
        self.assertTrue(np.all(a > 0.0))
        self.assertGreater(info["guess_mean_warming_K"], 0.0)
        self.assertLess(info["predicted_objective"],
                        zonal_objective(warm, mask, w))


class LayoutTest(unittest.TestCase):
    def test_layouts(self):
        self.assertEqual(len(BAND_LAYOUTS["b5"][0]), 5)
        centers, width = BAND_LAYOUTS["b13"]
        self.assertEqual(len(centers), 13)
        np.testing.assert_allclose(np.diff(centers), -10.0)
        self.assertEqual(width, 5.0)
        lats = np.deg2rad(np.linspace(-87, 87, 48))
        mask = np.ones((96, 48))
        self.assertEqual(band_patterns("b13", lats, mask).shape, (13, 96, 48))
        self.assertEqual(unit_profiles("b5", lats, (96, 48)).shape,
                         (5, 96, 48))
        self.assertEqual(SCORE_WINDOW, (98, 182))


if __name__ == "__main__":
    unittest.main()
