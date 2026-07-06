"""Tests for coupled feature extraction (Stage 4 absolute-SST features)."""

import unittest
from typing import NamedTuple

import jax.numpy as jnp

from jcm.mcb.coupled_features import (
    CoupledBaseline,
    CoupledFeatureConfig,
    extract_coupled_features,
    get_coupled_feature_dim,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class _SurfaceFlux(NamedTuple):
    tsfc: jnp.ndarray


class _Convection(NamedTuple):
    precnv: jnp.ndarray


class _Condensation(NamedTuple):
    precls: jnp.ndarray


class _Physics(NamedTuple):
    surface_flux: _SurfaceFlux
    convection: _Convection
    condensation: _Condensation


class _OceanState(NamedTuple):
    sea_surface_temperature: jnp.ndarray


def make_fake_carry(coords, sst_value=288.0, sst_field=None):
    """Build a minimal coupled carry compatible with feature extraction."""
    shape = coords.horizontal.nodal_shape
    sst = sst_field if sst_field is not None else jnp.full(shape, sst_value)
    return {
        "atm": {
            "derived": {
                "physics": _Physics(
                    surface_flux=_SurfaceFlux(tsfc=jnp.full(shape, 285.0)),
                    convection=_Convection(precnv=jnp.zeros(shape)),
                    condensation=_Condensation(precls=jnp.zeros(shape)),
                ),
                "total_heat_flux": jnp.zeros(shape),
                "mcb_perturbation": jnp.zeros(shape),
            }
        },
        "ocn": {"state": _OceanState(sea_surface_temperature=sst)},
    }


class TestCoupledFeatureDim(unittest.TestCase):
    """Tests for get_coupled_feature_dim with the absolute-SST flag."""

    def test_default_dim_is_11(self):
        """Default config keeps the 11-feature layout of old checkpoints."""
        self.assertEqual(get_coupled_feature_dim(CoupledFeatureConfig()), 11)

    def test_absolute_sst_adds_two(self):
        """include_absolute_sst adds exactly 2 features (11 -> 13)."""
        config = CoupledFeatureConfig(include_absolute_sst=True)
        self.assertEqual(get_coupled_feature_dim(config), 13)

    def test_default_flag_is_false(self):
        """The flag must default False to preserve existing checkpoints."""
        self.assertFalse(CoupledFeatureConfig().include_absolute_sst)


class TestAbsoluteSSTFeatures(unittest.TestCase):
    """Tests for the absolute-SST feature values and layout."""

    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()

    def _extract(self, carry, config, time_fraction=0.0):
        baseline = CoupledBaseline.from_coupled_carry(carry, self.coords)
        return extract_coupled_features(
            carry, baseline, self.coords, config, time_fraction=time_fraction
        )

    def test_feature_vector_lengths(self):
        """Feature vector length must match get_coupled_feature_dim."""
        carry = make_fake_carry(self.coords)
        for config in (CoupledFeatureConfig(),
                       CoupledFeatureConfig(include_absolute_sst=True)):
            features = self._extract(carry, config)
            self.assertEqual(features.shape, (get_coupled_feature_dim(config),))

    def test_new_features_appended_at_end(self):
        """First 11 features are unchanged by the flag (append-at-end)."""
        carry = make_fake_carry(self.coords, sst_value=290.0)
        f11 = self._extract(carry, CoupledFeatureConfig(), time_fraction=0.5)
        f13 = self._extract(
            carry, CoupledFeatureConfig(include_absolute_sst=True),
            time_fraction=0.5,
        )
        self.assertTrue(jnp.array_equal(f13[:11], f11))

    def test_global_mean_sst_offset(self):
        """Feature 11 is global-mean SST minus 288 K."""
        carry = make_fake_carry(self.coords, sst_value=290.0)
        f13 = self._extract(
            carry, CoupledFeatureConfig(include_absolute_sst=True)
        )
        # places=3: float32 precision near 288 K is ~2e-4
        self.assertAlmostEqual(float(f13[11]), 2.0, places=3)
        # Uniform SST -> zero hemispheric difference
        self.assertAlmostEqual(float(f13[12]), 0.0, places=3)

    def test_hemispheric_difference(self):
        """Feature 12 is NH-mean minus SH-mean SST."""
        lats = self.coords.horizontal.latitudes  # (il,)
        shape = self.coords.horizontal.nodal_shape
        asym = jnp.broadcast_to(jnp.where(lats > 0, 1.0, -1.0)[None, :], shape)
        carry = make_fake_carry(self.coords, sst_field=288.0 + asym)
        f13 = self._extract(
            carry, CoupledFeatureConfig(include_absolute_sst=True)
        )
        # Symmetric grid -> global mean offset ~0, NH-SH = 1 - (-1) = 2
        # (places=3: float32 precision near 288 K is ~2e-4)
        self.assertAlmostEqual(float(f13[11]), 0.0, places=3)
        self.assertAlmostEqual(float(f13[12]), 2.0, places=3)

    def test_distinguishes_ics_at_interval_zero(self):
        """Absolute-SST features differ across ICs even with zero anomalies."""
        config = CoupledFeatureConfig(include_absolute_sst=True)
        carry_a = make_fake_carry(self.coords, sst_value=288.0)
        carry_b = make_fake_carry(self.coords, sst_value=289.0)
        fa = self._extract(carry_a, config, time_fraction=0.0)
        fb = self._extract(carry_b, config, time_fraction=0.0)
        # Paired anomalies + time are identical (all zero)...
        self.assertTrue(jnp.allclose(fa[:11], fb[:11]))
        # ...but the absolute features are not
        self.assertFalse(jnp.allclose(fa[11:], fb[11:]))


class TestOceanMaskedFeatures(unittest.TestCase):
    """Stage 5 ocean-masking of area-weighted feature means."""

    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()
        cls.shape = cls.coords.horizontal.nodal_shape
        cls.config = CoupledFeatureConfig(include_absolute_sst=True)
        ix, il = cls.shape
        land = jnp.zeros(cls.shape).at[: ix // 2, :].set(1.0)
        cls.ocean_mask = 1.0 - land
        cls.land = land

    def _extract(self, carry, ocean_mask):
        baseline = CoupledBaseline.from_coupled_carry(carry, self.coords)
        return extract_coupled_features(
            carry, baseline, self.coords, self.config,
            time_fraction=0.3, ocean_mask=ocean_mask,
        )

    def test_invariant_to_land_sst_values(self):
        """Ocean-masked features are unchanged by arbitrary land SST values."""
        ocean_sst = jnp.full(self.shape, 289.0)
        sst_a = jnp.where(self.ocean_mask > 0, ocean_sst, 350.0)
        sst_b = jnp.where(self.ocean_mask > 0, ocean_sst, 150.0)

        fa = self._extract(
            make_fake_carry(self.coords, sst_field=sst_a), self.ocean_mask
        )
        fb = self._extract(
            make_fake_carry(self.coords, sst_field=sst_b), self.ocean_mask
        )
        self.assertTrue(jnp.allclose(fa, fb, atol=1e-5))

    def test_none_reproduces_unmasked(self):
        """ocean_mask=None matches an all-ones mask (aquaplanet behavior).

        The paths are mathematically identical but differ at float32
        precision: an all-ones mask renormalizes the area weights (which sum
        to 0.9999994, not exactly 1). atol=5e-4 covers the amplified
        renormalization noise on the ~288 K absolute-SST features (float32
        precision near 288 K is ~2e-4, consistent with the places=3 used in
        test_global_mean_sst_offset above).
        """
        carry = make_fake_carry(self.coords, sst_value=288.5)
        f_none = self._extract(carry, None)
        f_ones = self._extract(carry, jnp.ones(self.shape))
        self.assertTrue(jnp.allclose(f_none, f_ones, atol=5e-4))


class TestTerrainActivation(unittest.TestCase):
    """Smoke test that realistic terrain reaches the dynamics.

    Aquaplanet orography is identically zero; loading the T30 climatology
    terrain into the Model must produce a non-zero truncated_orography,
    proving orography reaches the spectral dynamical core.
    """

    def test_realistic_orography_nonzero(self):
        import jcm.model
        from jcm.terrain import TerrainData

        coords = get_speedy_coords()

        aqua = jcm.model.Model(coords=coords)
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(aqua.truncated_orography))), 0.0, places=6
        )

        terrain = TerrainData.from_file(
            "jcm/data/bc/t30/clim/terrain.nc", coords, lfluxland=True
        )
        realistic = jcm.model.Model(coords=coords, terrain=terrain)
        self.assertGreater(
            float(jnp.max(jnp.abs(realistic.truncated_orography))), 0.0
        )
        # And the land-sea mask marks some land.
        self.assertGreater(float(jnp.sum(terrain.fmask)), 0.0)


if __name__ == "__main__":
    unittest.main()
