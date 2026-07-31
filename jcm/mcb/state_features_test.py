"""Tests for state_features masks and weights (regression for the radians bug)."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.mcb.state_features import (
    compute_area_weights,
    create_latitude_band_mask,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class LatitudeBandMaskTest(unittest.TestCase):
    """create_latitude_band_mask takes DEGREE bounds against radian latitudes.

    The 2026-07-29 meta-audit found the mask compared degree bounds directly
    to radian latitude values, so the tropical (-30, 30) band covered the
    whole globe and (30, 60) was empty. These tests pin the corrected
    behavior.
    """

    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()
        cls.lats_deg = np.rad2deg(np.asarray(cls.coords.horizontal.latitudes))

    def _band_fraction(self, lat_min, lat_max):
        mask = create_latitude_band_mask(self.coords, lat_min, lat_max)
        return float(jnp.mean(mask))

    def test_tropical_band_is_not_the_whole_globe(self):
        frac = self._band_fraction(-30.0, 30.0)
        self.assertGreater(frac, 0.2)
        self.assertLess(frac, 0.5)

    def test_midlatitude_band_is_not_empty(self):
        self.assertGreater(self._band_fraction(30.0, 60.0), 0.0)
        self.assertGreater(self._band_fraction(-60.0, -30.0), 0.0)

    def test_mask_matches_degree_reference(self):
        for lat_min, lat_max in [(-30.0, 30.0), (30.0, 60.0), (60.0, 90.0),
                                 (-90.0, -60.0), (0.0, 90.0)]:
            mask = np.asarray(
                create_latitude_band_mask(self.coords, lat_min, lat_max)
            )
            expected = (self.lats_deg >= lat_min) & (self.lats_deg <= lat_max)
            expected = np.broadcast_to(expected[None, :], mask.shape)
            np.testing.assert_array_equal(
                mask, expected,
                err_msg=f"band ({lat_min}, {lat_max}) mask mismatch",
            )

    def test_bands_partition_the_globe(self):
        tropical = create_latitude_band_mask(self.coords, -30.0, 30.0)
        nh_mid = create_latitude_band_mask(self.coords, 30.0, 60.0)
        sh_mid = create_latitude_band_mask(self.coords, -60.0, -30.0)
        nh_pol = create_latitude_band_mask(self.coords, 60.0, 90.0)
        sh_pol = create_latitude_band_mask(self.coords, -90.0, -60.0)
        # Bands overlap only at exact boundary latitudes (none on a Gaussian
        # grid), so cell counts must add up to the full grid.
        total = (jnp.sum(tropical) + jnp.sum(nh_mid) + jnp.sum(sh_mid)
                 + jnp.sum(nh_pol) + jnp.sum(sh_pol))
        self.assertEqual(int(total), int(np.prod(tropical.shape)))

    def test_hemispheres_are_disjoint_halves(self):
        nh = create_latitude_band_mask(self.coords, 0.0, 90.0)
        sh = create_latitude_band_mask(self.coords, -90.0, 0.0)
        self.assertAlmostEqual(float(jnp.mean(nh)), 0.5, places=6)
        self.assertAlmostEqual(float(jnp.mean(sh)), 0.5, places=6)
        self.assertEqual(float(jnp.max(nh + sh)), 1.0)


class AreaWeightsTest(unittest.TestCase):
    def test_pole_to_equator_ratio(self):
        coords = get_speedy_coords()
        w = np.asarray(compute_area_weights(coords))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
        ratio = float(w.max() / w.min())
        # cos-lat weighting on T30: ~20x equator/pole cell-weight ratio.
        self.assertGreater(ratio, 10.0)


if __name__ == "__main__":
    unittest.main()
