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

    def test_speedy_physics_with_mcb_config_requires_opt_in(self):
        """The legacy surface-albedo mcb_config path must be explicit opt-in.

        The 2026-07-12 audit (root cause R6) found this path perturbs the
        sea-surface albedo — anti-correlated with cloud cover, the opposite
        of the Twomey effect. It must raise unless explicitly allowed.
        """
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics

        nodal_shape = (64, 32)
        config = MCBConfig.uniform(nodal_shape, perturbation=0.05)

        with self.assertRaises(ValueError):
            SpeedyPhysics(mcb_config=config)

        physics = SpeedyPhysics(
            mcb_config=config, allow_legacy_surface_albedo_mcb=True
        )
        self.assertIsNotNone(physics.mcb_config)

    def test_speedy_physics_without_mcb_config(self):
        """SpeedyPhysics should work without mcb_config (backward compatible)."""
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics

        # Should not raise
        physics = SpeedyPhysics()
        self.assertIsNone(physics.mcb_config)


class TestMCBPolicyMLP(unittest.TestCase):
    """Tests for MCBPolicyMLP neural network."""

    def setUp(self):
        self.output_shape = (64, 32)
        self.input_dim = 12  # Default feature dimension

    def test_mlp_output_shape(self):
        """MLP should output correct spatial shape."""
        from jcm.mcb.policy import MCBPolicyMLP

        policy = MCBPolicyMLP(output_shape=self.output_shape)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_dim))
        output = policy.apply(params, jnp.zeros(self.input_dim))

        self.assertEqual(output.shape, self.output_shape)

    def test_mlp_output_bounds(self):
        """MLP output should be bounded [0, max_perturbation]."""
        from jcm.mcb.policy import MCBPolicyMLP

        max_perturb = 0.15
        policy = MCBPolicyMLP(output_shape=self.output_shape, max_perturbation=max_perturb)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_dim))

        # Test with random input
        rng = jax.random.PRNGKey(1)
        random_input = jax.random.normal(rng, (self.input_dim,))
        output = policy.apply(params, random_input)

        self.assertTrue(jnp.all(output >= 0.0))
        self.assertTrue(jnp.all(output <= max_perturb))

    def test_mlp_batched_input(self):
        """MLP should handle batched inputs."""
        from jcm.mcb.policy import MCBPolicyMLP

        batch_size = 4
        policy = MCBPolicyMLP(output_shape=self.output_shape)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_dim))

        batched_input = jnp.zeros((batch_size, self.input_dim))
        output = policy.apply(params, batched_input)

        self.assertEqual(output.shape, (batch_size,) + self.output_shape)

    def test_mlp_jit_compatible(self):
        """MLP should be JIT-compatible."""
        from jcm.mcb.policy import MCBPolicyMLP

        policy = MCBPolicyMLP(output_shape=self.output_shape)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_dim))

        @jax.jit
        def forward(p, x):
            return policy.apply(p, x)

        output = forward(params, jnp.zeros(self.input_dim))
        self.assertEqual(output.shape, self.output_shape)

    def test_mlp_gradient_flow(self):
        """MLP should allow gradient flow."""
        from jcm.mcb.policy import MCBPolicyMLP

        policy = MCBPolicyMLP(output_shape=self.output_shape)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_dim))

        def loss_fn(p):
            output = policy.apply(p, jnp.ones(self.input_dim))
            return jnp.mean(output)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Check no NaNs in gradients
        grad_leaves = jax.tree.leaves(grads)
        has_nans = any(jnp.any(jnp.isnan(g)) for g in grad_leaves)
        self.assertFalse(has_nans)

        # Check some non-zero gradients
        has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
        self.assertTrue(has_nonzero)


class TestMCBPolicyCNN(unittest.TestCase):
    """Tests for MCBPolicyCNN neural network."""

    def setUp(self):
        self.input_shape = (64, 32, 3)  # (ix, il, channels)

    def test_cnn_output_shape(self):
        """CNN should output same spatial shape as input."""
        from jcm.mcb.policy import MCBPolicyCNN

        policy = MCBPolicyCNN()
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_shape))
        output = policy.apply(params, jnp.zeros(self.input_shape))

        self.assertEqual(output.shape, self.input_shape[:2])

    def test_cnn_output_bounds(self):
        """CNN output should be bounded [0, max_perturbation]."""
        from jcm.mcb.policy import MCBPolicyCNN

        max_perturb = 0.15
        policy = MCBPolicyCNN(max_perturbation=max_perturb)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_shape))

        rng = jax.random.PRNGKey(1)
        random_input = jax.random.normal(rng, self.input_shape)
        output = policy.apply(params, random_input)

        self.assertTrue(jnp.all(output >= 0.0))
        self.assertTrue(jnp.all(output <= max_perturb))


class TestMCBPolicyResNet(unittest.TestCase):
    """Tests for MCBPolicyResNet neural network."""

    def setUp(self):
        self.input_shape = (64, 32, 3)

    def test_resnet_output_shape(self):
        """ResNet should output same spatial shape as input."""
        from jcm.mcb.policy import MCBPolicyResNet

        policy = MCBPolicyResNet(num_blocks=2)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_shape))
        output = policy.apply(params, jnp.zeros(self.input_shape))

        self.assertEqual(output.shape, self.input_shape[:2])

    def test_resnet_gradient_flow(self):
        """ResNet should have good gradient flow through residuals."""
        from jcm.mcb.policy import MCBPolicyResNet

        policy = MCBPolicyResNet(num_blocks=4)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(self.input_shape))

        def loss_fn(p):
            output = policy.apply(p, jnp.ones(self.input_shape))
            return jnp.mean(output)

        loss, grads = jax.value_and_grad(loss_fn)(params)

        grad_leaves = jax.tree.leaves(grads)
        has_nans = any(jnp.any(jnp.isnan(g)) for g in grad_leaves)
        self.assertFalse(has_nans)


class TestCreatePolicy(unittest.TestCase):
    """Tests for create_policy factory function."""

    def test_create_mlp(self):
        """create_policy should create MLP correctly."""
        from jcm.mcb.policy import create_policy, MCBPolicyMLP

        policy = create_policy('mlp', output_shape=(64, 32))
        self.assertIsInstance(policy, MCBPolicyMLP)

    def test_create_cnn(self):
        """create_policy should create CNN correctly."""
        from jcm.mcb.policy import create_policy, MCBPolicyCNN

        policy = create_policy('cnn', output_shape=(64, 32))
        self.assertIsInstance(policy, MCBPolicyCNN)

    def test_create_resnet(self):
        """create_policy should create ResNet correctly."""
        from jcm.mcb.policy import create_policy, MCBPolicyResNet

        policy = create_policy('resnet', output_shape=(64, 32))
        self.assertIsInstance(policy, MCBPolicyResNet)

    def test_create_invalid_policy(self):
        """create_policy should raise error for invalid type."""
        from jcm.mcb.policy import create_policy

        with self.assertRaises(ValueError):
            create_policy('invalid', output_shape=(64, 32))


class TestStateFeatures(unittest.TestCase):
    """Tests for state feature extraction."""

    def test_state_feature_config_defaults(self):
        """StateFeatureConfig should have sensible defaults."""
        from jcm.mcb.state_features import StateFeatureConfig

        config = StateFeatureConfig()

        self.assertTrue(config.include_temperature)
        self.assertTrue(config.include_precipitation)
        self.assertFalse(config.include_spatial)

    def test_get_feature_dim(self):
        """get_feature_dim should return expected dimension."""
        from jcm.mcb.state_features import get_feature_dim, StateFeatureConfig

        dim = get_feature_dim(StateFeatureConfig())
        self.assertEqual(dim, 12)

    def test_climate_baseline_global_mean(self):
        """ClimateBaseline.global_mean should create correct shapes."""
        from jcm.mcb.state_features import ClimateBaseline

        nodal_shape = (64, 32)
        baseline = ClimateBaseline.global_mean(nodal_shape, target_temp=288.0)

        self.assertEqual(baseline.surface_temperature.shape, nodal_shape)
        self.assertEqual(baseline.precipitation.shape, nodal_shape)
        self.assertEqual(baseline.temperature.shape, nodal_shape + (8,))


class TestLossWeights(unittest.TestCase):
    """Tests for LossWeights configuration."""

    def test_loss_weights_defaults(self):
        """LossWeights should have sensible defaults."""
        from jcm.mcb.loss import LossWeights

        weights = LossWeights()

        self.assertEqual(weights.temperature, 1.0)
        self.assertEqual(weights.amazon, 1.0)
        self.assertGreater(weights.regularization, 0.0)

    def test_loss_weights_custom(self):
        """LossWeights should accept custom values."""
        from jcm.mcb.loss import LossWeights

        weights = LossWeights(temperature=2.0, amazon=0.5)

        self.assertEqual(weights.temperature, 2.0)
        self.assertEqual(weights.amazon, 0.5)


class TestRegularizationLoss(unittest.TestCase):
    """Tests for MCB forcing regularization loss."""

    def test_regularization_loss_zero_forcing(self):
        """Zero forcing should give zero regularization loss."""
        from jcm.mcb.loss import regularization_loss

        forcing = jnp.zeros((64, 32))
        loss = regularization_loss(forcing)

        self.assertTrue(jnp.isclose(loss, 0.0))

    def test_regularization_loss_positive(self):
        """Non-zero forcing should give positive regularization loss."""
        from jcm.mcb.loss import regularization_loss

        forcing = jnp.ones((64, 32)) * 0.1
        loss = regularization_loss(forcing)

        self.assertTrue(loss > 0.0)

    def test_regularization_loss_gradient(self):
        """Regularization loss should be differentiable."""
        from jcm.mcb.loss import regularization_loss

        forcing = jnp.ones((64, 32)) * 0.1

        grad = jax.grad(regularization_loss)(forcing)

        self.assertFalse(jnp.any(jnp.isnan(grad)))
        self.assertTrue(jnp.any(grad != 0))


class TestSmoothnessLoss(unittest.TestCase):
    """Tests for MCB forcing smoothness loss."""

    def test_smoothness_loss_uniform(self):
        """Uniform forcing should give zero smoothness loss."""
        from jcm.mcb.loss import smoothness_loss

        forcing = jnp.ones((64, 32)) * 0.1
        loss = smoothness_loss(forcing)

        self.assertTrue(jnp.isclose(loss, 0.0))

    def test_smoothness_loss_nonuniform(self):
        """Non-uniform forcing should give positive smoothness loss."""
        from jcm.mcb.loss import smoothness_loss

        # Create checkerboard pattern
        x = jnp.arange(64)
        y = jnp.arange(32)
        xx, yy = jnp.meshgrid(x, y, indexing='ij')
        forcing = ((xx + yy) % 2).astype(jnp.float32) * 0.1

        loss = smoothness_loss(forcing)

        self.assertTrue(loss > 0.0)


class TestControllerConfig(unittest.TestCase):
    """Tests for ControllerConfig."""

    def test_controller_config_defaults(self):
        """ControllerConfig should have sensible defaults."""
        from jcm.mcb.controller import ControllerConfig

        config = ControllerConfig()

        self.assertEqual(config.control_interval_days, 30.0)
        self.assertEqual(config.total_days, 365.0)
        self.assertEqual(config.target_cooling, -0.5)
        self.assertTrue(config.use_checkpointing)

    def test_controller_config_custom(self):
        """ControllerConfig should accept custom values."""
        from jcm.mcb.controller import ControllerConfig

        config = ControllerConfig(
            control_interval_days=15.0,
            total_days=180.0,
            target_cooling=-1.0,
        )

        self.assertEqual(config.control_interval_days, 15.0)
        self.assertEqual(config.total_days, 180.0)
        self.assertEqual(config.target_cooling, -1.0)


class TestTrainingConfig(unittest.TestCase):
    """Tests for TrainingConfig."""

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        from jcm.mcb.train import TrainingConfig

        config = TrainingConfig()

        self.assertEqual(config.num_epochs, 100)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.optimizer, 'adam')

    def test_create_optimizer(self):
        """create_optimizer should create valid optimizer."""
        from jcm.mcb.train import TrainingConfig, create_optimizer

        config = TrainingConfig(learning_rate=1e-3, optimizer='adam')
        optimizer = create_optimizer(config)

        # Should be callable
        self.assertTrue(hasattr(optimizer, 'init'))
        self.assertTrue(hasattr(optimizer, 'update'))


if __name__ == '__main__':
    unittest.main()
