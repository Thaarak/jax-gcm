"""Tests for the test world's objective and scores (jcm.mcb.scores)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.scores import (
    amplitude_penalty,
    area_weights,
    effort,
    gain,
    movement_penalty,
    objective_terms,
    pattern_objective,
    restoration_scores,
    segment_objective,
    weighted_mean,
    weighted_variance,
)

IX, IL = 8, 6
LATS = np.deg2rad(np.linspace(-75.0, 75.0, IL))


def _t30_latitudes():
    """Return the 48 Gaussian latitudes of the T30 grid (radians, S to N)."""
    nodes, _ = np.polynomial.legendre.leggauss(48)
    return np.arcsin(nodes)


class ObjectiveTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        mask = np.ones((IX, IL))
        mask[0] = 0.0
        self.w = area_weights(LATS, mask)
        self.d = jnp.asarray(rng.normal(0.3, 0.5, (IX, IL)))

    def test_weights_sum_to_one_and_respect_the_mask(self):
        self.assertAlmostEqual(float(jnp.sum(self.w)), 1.0, places=6)
        self.assertEqual(float(jnp.sum(self.w[0])), 0.0)

    def test_alpha_beta_one_is_the_weighted_mean_square_error(self):
        # Dubey et al.: for alpha = beta = 1 the two terms sum to <d^2>_w.
        j = pattern_objective(self.d, self.w, alpha=1.0, beta=1.0)
        self.assertAlmostEqual(float(j), float(jnp.sum(self.w * self.d ** 2)),
                               places=5)

    def test_beta_zero_keeps_only_the_bias(self):
        j = pattern_objective(self.d, self.w, alpha=1.0, beta=0.0)
        self.assertAlmostEqual(float(j), float(weighted_mean(self.d,
                                                             self.w)) ** 2,
                               places=6)
        flat = jnp.full((IX, IL), 0.7)
        self.assertAlmostEqual(float(weighted_variance(flat, self.w)), 0.0,
                               places=6)

    def test_penalties_and_terms_add_up(self):
        a, prev = jnp.array([0.1, 0.0, 0.2]), jnp.array([0.0, 0.1, 0.2])
        self.assertAlmostEqual(float(amplitude_penalty(a, 2.0)), 0.1,
                               places=6)
        self.assertAlmostEqual(float(movement_penalty(a, prev, 3.0)), 0.06,
                               places=6)
        total = segment_objective(self.d, self.w, a, prev, mu=2.0, lam=3.0)
        terms = objective_terms(self.d, self.w, a, prev, mu=2.0, lam=3.0)
        self.assertAlmostEqual(float(total), terms["total"], places=5)

    def test_gradients_are_correct(self):
        a, prev = jnp.array([0.1, 0.05, 0.2]), jnp.zeros(3)

        def f(err, amp):
            return segment_objective(err, self.w, amp, prev, mu=0.5, lam=0.2)

        check_grads(f, (self.d, a), order=1, modes=["fwd", "rev"],
                    atol=1e-2, rtol=1e-2)
        self.assertTrue(np.all(np.isfinite(jax.grad(f, argnums=1)(self.d, a))))


class GainEffortTest(unittest.TestCase):
    def test_gain_reference_values(self):
        w = np.full((IX, IL), 1.0 / (IX * IL))
        rng = np.random.default_rng(1)
        target = rng.normal(290.0, 2.0, (IX, IL))
        warmed = target + rng.normal(0.5, 0.3, (IX, IL))
        self.assertAlmostEqual(gain(warmed, warmed, target, w), 1.0)
        self.assertEqual(gain(target, warmed, target, w), float("inf"))
        halfway = target + 0.5 * (warmed - target)
        self.assertAlmostEqual(gain(halfway, warmed, target, w), 4.0)

    def test_effort_matches_dubey_unit_profile_area_mean(self):
        # Dubey et al.: the area mean of the summed unit profile of the five
        # default bands is 0.925, so a flat amplitude a has effort 0.925 a.
        lats = _t30_latitudes()
        ones = np.ones((96, 48))
        g = np.asarray(gaussian_band_patterns(lats, ones))
        w = np.asarray(area_weights(lats, ones))
        unit = effort(np.ones((1, 5)), g, w)
        self.assertAlmostEqual(unit, 0.925, delta=0.005)
        series = np.tile([0.02, 0.0, 0.04, 0.0, 0.02], (10, 1))
        self.assertAlmostEqual(effort(series, g, w),
                               effort(series[:1], g, w), places=12)
        self.assertAlmostEqual(effort(-series, g, w), effort(series, g, w),
                               places=12)

    def test_restoration_scores_cover_variables_and_domains(self):
        w = {d: np.full((IX, IL), 1.0 / (IX * IL))
             for d in ("ocean", "land", "global")}
        tgt = {"sst": np.zeros((IX, IL)), "precipitation": np.zeros((IX, IL))}
        warm = {k: v + 1.0 for k, v in tgt.items()}
        ctrl = {k: v + 0.5 for k, v in tgt.items()}
        out = restoration_scores(ctrl, warm, tgt, w)
        self.assertEqual(set(out), {"sst@ocean", "precipitation@land",
                                    "precipitation@global"})
        for g in out.values():
            self.assertAlmostEqual(g, 4.0)


if __name__ == "__main__":
    unittest.main()
