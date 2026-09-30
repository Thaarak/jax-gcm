"""Tests for the band controls and objectives (jcm.mcb.band_basis)."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import (
    DEFAULT_BAND_CENTERS_DEG,
    OBJECTIVE_NAMES,
    band_perturbation,
    gaussian_band_patterns,
    objective_values,
    objective_weights,
    stack_objective_weights,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


def _grid():
    lats = np.asarray(get_speedy_coords().horizontal.latitudes)
    ocean = np.ones((96, lats.size))
    ocean[10:20, :] = 0.0            # a land strip
    ocean[:, :3] = 0.0               # land near the south pole
    return lats, ocean, 1.0 - ocean


class GaussianBandPatternsTest(unittest.TestCase):
    def test_shape_peak_and_ocean_only(self):
        lats, ocean, land = _grid()
        pats = np.asarray(gaussian_band_patterns(lats, ocean))
        self.assertEqual(pats.shape, (len(DEFAULT_BAND_CENTERS_DEG), 96,
                                      lats.size))
        self.assertTrue(np.all(pats[:, land > 0] == 0.0))
        lat_deg = np.rad2deg(lats)
        for k, centre in enumerate(DEFAULT_BAND_CENTERS_DEG):
            il = int(np.argmin(np.abs(lat_deg - centre)))
            self.assertAlmostEqual(
                pats[k, 0, il],
                np.exp(-0.5 * ((lat_deg[il] - centre) / 10.0) ** 2),
                places=5)
            self.assertGreater(pats[k, 0, il], 0.9)
            self.assertAlmostEqual(pats[k].max(), pats[k, 0, il], places=6)

    def test_width_is_one_standard_deviation(self):
        lats = np.deg2rad(np.array([0.0, 10.0, 20.0]))
        pats = np.asarray(gaussian_band_patterns(
            lats, np.ones((2, 3)), centers_deg=(0.0,), width_deg=10.0))
        np.testing.assert_allclose(
            pats[0, 0], [1.0, np.exp(-0.5), np.exp(-2.0)], rtol=1e-6)

    def test_perturbation_is_linear_in_amplitudes(self):
        lats, ocean, _ = _grid()
        pats = gaussian_band_patterns(lats, ocean)
        a = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
        b = jnp.array([0.05, 0.0, -0.01, 0.02, 0.0])
        np.testing.assert_allclose(
            np.asarray(band_perturbation(a + b, pats)),
            np.asarray(band_perturbation(a, pats) + band_perturbation(b, pats)),
            atol=1e-7)


class ObjectiveWeightsTest(unittest.TestCase):
    def setUp(self):
        self.lats, self.ocean, self.land = _grid()
        self.w = objective_weights(self.lats, self.ocean, self.land)
        self.stack = stack_objective_weights(self.w)

    def test_normalisation(self):
        self.assertAlmostEqual(float(jnp.sum(self.w["T0"])), 1.0, places=5)
        self.assertAlmostEqual(float(jnp.sum(self.w["LAND"])), 1.0, places=5)
        # Zero-sum up to float32 round-off.
        self.assertAlmostEqual(float(jnp.sum(self.w["T1"])), 0.0, places=5)
        self.assertAlmostEqual(float(jnp.sum(self.w["T2"])), 0.0, places=5)
        self.assertTrue(np.all(np.asarray(self.w["T0"])[self.land > 0] == 0))
        self.assertTrue(np.all(np.asarray(self.w["LAND"])[self.ocean > 0]
                               == 0))

    def test_uniform_change_projects_on_t0_only(self):
        field = jnp.full(self.ocean.shape, -0.3)
        vals = np.asarray(objective_values(field, field, self.stack,
                                           reference_k=0.0))
        self.assertAlmostEqual(vals[0], -0.3, places=5)
        self.assertAlmostEqual(vals[1], 0.0, places=6)
        self.assertAlmostEqual(vals[2], 0.0, places=6)
        self.assertAlmostEqual(vals[3], -0.3, places=5)

    def test_reference_offset_keeps_differences_exact(self):
        # Two SST fields 2**-10 K apart near 290 K (both exactly
        # representable in float32): the difference of the objectives must
        # survive the weighted sums, which it would not without the offset.
        bump = 2.0 ** -10
        base = jnp.full(self.ocean.shape, 290.0, dtype=jnp.float32)
        bumped = jnp.full(self.ocean.shape, 290.0 + bump, dtype=jnp.float32)
        diff = np.asarray(objective_values(bumped, bumped, self.stack)
                          - objective_values(base, base, self.stack))
        self.assertAlmostEqual(diff[0], bump, delta=1e-5)
        self.assertAlmostEqual(diff[3], bump, delta=1e-5)

    def test_signs_of_the_contrasts(self):
        lat = np.broadcast_to(self.lats[None, :], self.ocean.shape)
        nh_warm = jnp.asarray(np.where(lat > 0, 1.0, 0.0))
        self.assertGreater(float(jnp.sum(self.w["T1"] * nh_warm)), 0.0)
        tropics_warm = jnp.asarray(np.where(np.abs(lat) < np.deg2rad(30),
                                            1.0, 0.0))
        self.assertLess(float(jnp.sum(self.w["T2"] * tropics_warm)), 0.0)

    def test_land_objective_reads_the_land_field(self):
        sst = jnp.zeros(self.ocean.shape)
        land_t = jnp.ones(self.ocean.shape)
        vals = np.asarray(objective_values(sst, land_t, self.stack,
                                           reference_k=0.0))
        np.testing.assert_allclose(vals, [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        self.assertEqual(len(vals), len(OBJECTIVE_NAMES))


if __name__ == "__main__":
    unittest.main()
