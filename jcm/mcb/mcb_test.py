"""Tests for the MCB (Marine Cloud Brightening) forcing module.

Tests verify:
1. Data structure correctness (shapes, dtypes)
2. JAX compatibility (jit, grad, vmap)
3. Physical correctness (albedo bounds, ocean-only application)
4. Gradient validity (no NaNs, reasonable magnitudes)
"""

import unittest
import jax
import jax.numpy as jnp

from jcm.mcb.mcb_config import MCBConfig
from jcm.mcb.mcb_regions import (
    create_region_mask,
    create_ocean_mask,
    create_stratocumulus_mask,
    create_teleconnection_mask,
    STRATOCUMULUS_REGIONS,
    TELECONNECTION_REGIONS,
)
from jcm.mcb.mcb_forcing import (
    compute_mcb_sea_albedo,
    compute_mcb_radiative_forcing,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


class TestMCBConfigZeros(unittest.TestCase):
    """Tests for MCBConfig.zeros classmethod."""

    def test_zeros_shape(self):
        """MCBConfig.zeros should create correct shapes."""
        nodal_shape = (96, 48)
        config = MCBConfig.zeros(nodal_shape)

        self.assertEqual(config.albedo_perturbation.shape, nodal_shape)
        self.assertEqual(config.active_mask.shape, nodal_shape)
        self.assertEqual(config.temporal_weights.shape, (365,))

    def test_zeros_values(self):
        """MCBConfig.zeros should create zero perturbation and mask."""
        nodal_shape = (64, 32)
        config = MCBConfig.zeros(nodal_shape)

        self.assertTrue(jnp.allclose(config.albedo_perturbation, 0.0))
        self.assertTrue(jnp.allclose(config.active_mask, 0.0))
        self.assertTrue(jnp.allclose(config.temporal_weights, 1.0))

    def test_zeros_with_custom_temporal_weights(self):
        """MCBConfig.zeros should accept custom temporal weights."""
        nodal_shape = (64, 32)
        weights = jnp.linspace(0, 1, 365)
        config = MCBConfig.zeros(nodal_shape, temporal_weights=weights)

        self.assertTrue(jnp.allclose(config.temporal_weights, weights))


class TestMCBConfigUniform(unittest.TestCase):
    """Tests for MCBConfig.uniform classmethod."""

    def test_uniform_default_perturbation(self):
        """MCBConfig.uniform should default to 0.05 perturbation."""
        nodal_shape = (96, 48)
        config = MCBConfig.uniform(nodal_shape)

        self.assertTrue(jnp.allclose(config.albedo_perturbation, 0.05))
        self.assertTrue(jnp.allclose(config.active_mask, 1.0))

    def test_uniform_custom_perturbation(self):
        """MCBConfig.uniform should accept custom perturbation value."""
        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.1)

        self.assertTrue(jnp.allclose(config.albedo_perturbation, 0.1))

    def test_uniform_with_region_mask(self):
        """MCBConfig.uniform should apply provided region mask."""
        nodal_shape = (96, 48)
        mask = jnp.zeros(nodal_shape).at[:48, :24].set(1.0)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.1, region_mask=mask)

        self.assertTrue(jnp.allclose(config.active_mask, mask))
        self.assertTrue(jnp.allclose(config.albedo_perturbation, 0.1))


class TestMCBConfigCopy(unittest.TestCase):
    """Tests for MCBConfig.copy method."""

    def test_copy_no_changes(self):
        """MCBConfig.copy with no args should return identical config."""
        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)
        copied = config.copy()

        self.assertTrue(jnp.allclose(copied.albedo_perturbation, config.albedo_perturbation))
        self.assertTrue(jnp.allclose(copied.active_mask, config.active_mask))
        self.assertTrue(jnp.allclose(copied.temporal_weights, config.temporal_weights))

    def test_copy_with_new_perturbation(self):
        """MCBConfig.copy should allow updating perturbation field."""
        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)
        new_perturb = jnp.full(nodal_shape, 0.1)
        copied = config.copy(albedo_perturbation=new_perturb)

        self.assertTrue(jnp.allclose(copied.albedo_perturbation, 0.1))
        self.assertTrue(jnp.allclose(copied.active_mask, config.active_mask))


class TestMCBConfigJAXCompatibility(unittest.TestCase):
    """Tests for JAX compatibility of MCBConfig."""

    def test_jax_pytree_compatibility(self):
        """MCBConfig should work with JAX tree operations."""
        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)

        # Tree map should work
        doubled = jax.tree.map(lambda x: x * 2, config)
        self.assertTrue(jnp.allclose(doubled.albedo_perturbation, 0.1))

    def test_jax_jit_compatibility(self):
        """MCBConfig should work with JAX jit."""
        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)

        @jax.jit
        def get_mean_perturbation(cfg):
            return jnp.mean(cfg.albedo_perturbation)

        result = get_mean_perturbation(config)
        self.assertTrue(jnp.isclose(result, 0.05))


class TestMCBRegionMask(unittest.TestCase):
    """Tests for region mask creation utilities."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.grid = self.coords.horizontal

    def test_create_region_mask_shape(self):
        """Region mask should have correct shape."""
        mask = create_region_mask(
            self.grid,
            lat_bounds=(-30.0, -5.0),
            lon_bounds=(-120.0, -70.0),
        )

        self.assertEqual(mask.shape, self.grid.nodal_shape)

    def test_create_region_mask_binary(self):
        """Region mask should be binary (0 or 1)."""
        mask = create_region_mask(
            self.grid,
            lat_bounds=(-30.0, -5.0),
            lon_bounds=(-120.0, -70.0),
        )

        self.assertTrue(jnp.all((mask == 0.0) | (mask == 1.0)))

    def test_create_region_mask_nonzero(self):
        """Region mask should cover some area."""
        mask = create_region_mask(
            self.grid,
            lat_bounds=(-30.0, 30.0),
            lon_bounds=(-180.0, 180.0),
        )

        self.assertTrue(jnp.sum(mask) > 0)

    def test_create_ocean_mask(self):
        """Ocean mask should invert land-sea mask."""
        fmask = jnp.zeros(self.grid.nodal_shape)  # All ocean
        ocean_mask = create_ocean_mask(self.grid, fmask)

        self.assertTrue(jnp.allclose(ocean_mask, 1.0))

        fmask_land = jnp.ones(self.grid.nodal_shape)  # All land
        ocean_mask_land = create_ocean_mask(self.grid, fmask_land)

        self.assertTrue(jnp.allclose(ocean_mask_land, 0.0))

    def test_stratocumulus_mask_known_regions(self):
        """Stratocumulus mask should work with known region names."""
        fmask = jnp.zeros(self.grid.nodal_shape)  # All ocean

        for region_name in STRATOCUMULUS_REGIONS:
            mask = create_stratocumulus_mask(self.grid, fmask, [region_name])
            self.assertEqual(mask.shape, self.grid.nodal_shape)
            self.assertTrue(jnp.sum(mask) > 0, f"Region {region_name} should have nonzero mask")

    def test_stratocumulus_mask_invalid_region(self):
        """Stratocumulus mask should raise error for unknown region."""
        fmask = jnp.zeros(self.grid.nodal_shape)

        with self.assertRaises(ValueError):
            create_stratocumulus_mask(self.grid, fmask, ['invalid_region'])

    def test_teleconnection_mask_known_regions(self):
        """Teleconnection mask should work with known region names."""
        for region_name in TELECONNECTION_REGIONS:
            mask = create_teleconnection_mask(self.grid, region_name)
            self.assertEqual(mask.shape, self.grid.nodal_shape)
            self.assertTrue(jnp.sum(mask) > 0, f"Region {region_name} should have nonzero mask")


class TestMCBForcing(unittest.TestCase):
    """Tests for MCB forcing computation."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.nodal_shape = self.coords.horizontal.nodal_shape

    def test_compute_mcb_sea_albedo_no_perturbation(self):
        """With zero perturbation, albedo should equal base albedo."""
        config = MCBConfig.zeros(self.nodal_shape)
        base_albedo = 0.07

        result = compute_mcb_sea_albedo(config, base_albedo, day_of_year=0)

        self.assertTrue(jnp.allclose(result, base_albedo))

    def test_compute_mcb_sea_albedo_with_perturbation(self):
        """With perturbation, albedo should increase."""
        config = MCBConfig.uniform(self.nodal_shape, perturbation=0.1)
        base_albedo = 0.07

        result = compute_mcb_sea_albedo(config, base_albedo, day_of_year=0)

        self.assertTrue(jnp.all(result >= base_albedo))
        self.assertTrue(jnp.allclose(result, 0.17))

    def test_compute_mcb_sea_albedo_bounds(self):
        """Effective albedo should be bounded [base_albedo, 1.0]."""
        # Large perturbation that would exceed 1.0
        config = MCBConfig.uniform(self.nodal_shape, perturbation=2.0)
        base_albedo = 0.07

        result = compute_mcb_sea_albedo(config, base_albedo, day_of_year=0)

        self.assertTrue(jnp.all(result >= base_albedo))
        self.assertTrue(jnp.all(result <= 1.0))

    def test_compute_mcb_sea_albedo_temporal_modulation(self):
        """Temporal weights should modulate forcing strength."""
        # Create seasonal weights (summer=0, winter=1)
        temporal_weights = jnp.zeros(365)
        temporal_weights = temporal_weights.at[:182].set(1.0)  # First half = 1

        config = MCBConfig.uniform(
            self.nodal_shape,
            perturbation=0.1,
            temporal_weights=temporal_weights,
        )
        base_albedo = 0.07

        winter_result = compute_mcb_sea_albedo(config, base_albedo, day_of_year=0)
        summer_result = compute_mcb_sea_albedo(config, base_albedo, day_of_year=200)

        self.assertTrue(jnp.allclose(winter_result, 0.17))
        self.assertTrue(jnp.allclose(summer_result, base_albedo))

    def test_compute_mcb_radiative_forcing_sign(self):
        """Radiative forcing should be negative (cooling) for positive perturbation."""
        config = MCBConfig.uniform(self.nodal_shape, perturbation=0.1)
        base_albedo = 0.07
        incoming_sw = jnp.full(self.nodal_shape, 400.0)  # W/m^2

        forcing = compute_mcb_radiative_forcing(
            config, base_albedo, incoming_sw, day_of_year=0
        )

        self.assertTrue(jnp.all(forcing <= 0))  # Cooling is negative


class TestMCBGradients(unittest.TestCase):
    """Tests for gradient computation through MCB forcing."""

    def setUp(self):
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.nodal_shape = self.coords.horizontal.nodal_shape

    def test_gradient_no_nans_vjp(self):
        """VJP through MCB should not produce NaNs."""
        config = MCBConfig.uniform(self.nodal_shape, perturbation=0.05)

        def loss_fn(albedo_perturb):
            cfg = config.copy(albedo_perturbation=albedo_perturb)
            effective = compute_mcb_sea_albedo(cfg, 0.07, day_of_year=100)
            return jnp.mean(effective)

        primals, f_vjp = jax.vjp(loss_fn, config.albedo_perturbation)
        grad = f_vjp(jnp.ones_like(primals))[0]

        self.assertFalse(jnp.any(jnp.isnan(grad)))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))

    def test_gradient_no_nans_jvp(self):
        """JVP through MCB should not produce NaNs."""
        config = MCBConfig.uniform(self.nodal_shape, perturbation=0.05)

        def loss_fn(albedo_perturb):
            cfg = config.copy(albedo_perturbation=albedo_perturb)
            effective = compute_mcb_sea_albedo(cfg, 0.07, day_of_year=100)
            return jnp.mean(effective)

        tangent = jnp.ones(self.nodal_shape)
        primal, jvp_out = jax.jvp(loss_fn, (config.albedo_perturbation,), (tangent,))

        self.assertFalse(jnp.isnan(jvp_out))
        self.assertTrue(jnp.isfinite(jvp_out))

    def test_gradient_reasonable_magnitude(self):
        """Gradient should have reasonable magnitude."""
        config = MCBConfig.uniform(self.nodal_shape, perturbation=0.05)

        def loss_fn(albedo_perturb):
            cfg = config.copy(albedo_perturbation=albedo_perturb)
            effective = compute_mcb_sea_albedo(cfg, 0.07, day_of_year=100)
            return jnp.mean(effective)

        grad = jax.grad(loss_fn)(config.albedo_perturbation)

        # Gradient should be bounded and nonzero where mask is active
        self.assertTrue(jnp.all(jnp.abs(grad) < 10.0))


class TestMCBIntegration(unittest.TestCase):
    """Integration tests for MCB with SpeedyPhysics."""

    def test_speedy_physics_with_mcb_config(self):
        """SpeedyPhysics should accept mcb_config parameter."""
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics

        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)

        # Should not raise
        physics = SpeedyPhysics(mcb_config=config)
        self.assertIsNotNone(physics.mcb_config)

    def test_speedy_physics_without_mcb_config(self):
        """SpeedyPhysics should work without mcb_config (backward compatible)."""
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics

        # Should not raise
        physics = SpeedyPhysics()
        self.assertIsNone(physics.mcb_config)


if __name__ == '__main__':
    unittest.main()
