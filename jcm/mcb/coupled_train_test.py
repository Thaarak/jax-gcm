"""Tests for coupled ensemble training (Stage 4).

Uses a lightweight fake coupler whose step function cools SST in proportion
to the injected MCB perturbation, so gradients flow through the full
unroll -> features -> policy -> loss path without the real model.
"""

import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jcm.mcb.coupled_controller import (
    CoupledControllerConfig,
    create_coupled_step_fn,
    unroll_coupled_simple,
)
from jcm.mcb.coupled_features import (
    CoupledFeatureConfig,
    compute_baseline_trajectory,
)
from jcm.mcb.coupled_train import (
    create_coupled_eval_fn,
    create_coupled_grad_fn,
    ocean_fmask_from_coupler,
    ocean_mask_from_coupler,
    ocean_mask_from_fmask,
    train_coupled_policy_ensemble,
)
from jcm.mcb.policy import MCBPolicyMLP
from jcm.mcb.train import TrainingConfig
from jcm.physics.speedy.speedy_coords import get_speedy_coords

WORKFLOW = ["coupling", "atm", "ocn"]


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


def make_fake_carry(coords, sst_value=288.0):
    shape = coords.horizontal.nodal_shape
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
        "ocn": {"state": _OceanState(
            sea_surface_temperature=jnp.full(shape, sst_value)
        )},
    }


class FakeCoupler:
    """Minimal coupler: SST cools proportionally to the MCB perturbation."""

    def generate_step_function(self, workflow, jitted=True,
                               show_progress=False, verbose=False):
        def step_fn(carry, step_idx):
            derived = carry["atm"]["derived"]
            sst = carry["ocn"]["state"].sea_surface_temperature
            new_sst = sst - 0.05 * derived["mcb_perturbation"]
            new_carry = {
                "atm": {"derived": dict(derived)},
                "ocn": {"state": _OceanState(sea_surface_temperature=new_sst)},
            }
            return new_carry, None
        return step_fn


class TestOceanMaskFromFmask(unittest.TestCase):
    """The ocean mask must match the slab ocean model's land-sea convention.

    The slab ocean model (builtin_grid_generator.load_jcm_mask) treats a cell
    as land (pins SST to 288.15 K) only when fmask > 0.95, and evolves SST
    everywhere else. The MCB loss/features must weight exactly those
    evolving-SST cells, so the mask is binary (1 - bmask), NOT fractional.
    """

    def test_aquaplanet_all_ocean(self):
        """Zero land fraction -> all-ocean mask (aquaplanet unchanged)."""
        fmask = jnp.zeros((4, 3))
        self.assertTrue(jnp.array_equal(
            ocean_mask_from_fmask(fmask), jnp.ones((4, 3))
        ))

    def test_binary_threshold_at_0p95(self):
        """Cells are land iff fmask > 0.95; coastal cells stay ocean."""
        fmask = jnp.array([0.0, 0.5, 0.9, 0.95, 0.96, 1.0])
        # 0.95 is NOT > 0.95 -> ocean; only 0.96 and 1.0 are land.
        expected = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertTrue(jnp.array_equal(
            ocean_mask_from_fmask(fmask), expected
        ))

    def test_output_is_binary(self):
        """Fractional inputs never produce fractional mask values."""
        fmask = jnp.linspace(0.0, 1.0, 21)
        mask = ocean_mask_from_fmask(fmask)
        self.assertTrue(jnp.all((mask == 0.0) | (mask == 1.0)))

    def test_coastal_cells_treated_as_ocean(self):
        """A 90%-land coastal cell must be full-weight ocean, not 0.1.

        This is the correctness fix: the ocean model evolves SST there, so the
        loss must weight it fully rather than down-weighting to 1 - 0.9 = 0.1.
        """
        fmask = jnp.array([0.9])
        self.assertEqual(float(ocean_mask_from_fmask(fmask)[0]), 1.0)


class _FakeGrid:
    def __init__(self, bmask, fmask):
        self.bmask = bmask
        self.fmask = fmask


class _FakeOceanComponent:
    def __init__(self, bmask, fmask):
        self.raw_component = type(
            "_Raw", (), {"horizontal_grids": {"T": _FakeGrid(bmask, fmask)}}
        )()


class _FakeCouplerWithGrid:
    def __init__(self, bmask, fmask):
        self.components = {"ocn": _FakeOceanComponent(bmask, fmask)}


class TestOceanMaskFromCoupler(unittest.TestCase):
    """The authoritative mask must come from the ocean model's own grid.

    TerrainData.from_file's fmask is interpolated independently and disagrees
    with the ocean grid at coastlines; the loss/features must use the ocean
    grid's own bmask so masked cells match the SST-pinning cells exactly.
    """

    def test_uses_grid_bmask_not_terrain_fmask(self):
        """Ocean mask is 1 - bmask, independent of any terrain interpolation."""
        bmask = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        # fmask deliberately disagrees with bmask at the threshold to prove
        # bmask (not a fmask>0.95 recompute) is the source of truth.
        fmask = jnp.array([[0.96, 0.94], [0.5, 0.0]])
        coupler = _FakeCouplerWithGrid(bmask, fmask)
        mask = ocean_mask_from_coupler(coupler)
        self.assertTrue(jnp.array_equal(mask, 1.0 - bmask))

    def test_ocean_fmask_returns_grid_fmask(self):
        """ocean_fmask_from_coupler returns the ocean grid's fractional mask."""
        bmask = jnp.array([[0.0, 1.0]])
        fmask = jnp.array([[0.3, 0.97]])
        coupler = _FakeCouplerWithGrid(bmask, fmask)
        self.assertTrue(
            jnp.array_equal(ocean_fmask_from_coupler(coupler), fmask)
        )

    def test_grid_fmask_threshold_reproduces_bmask(self):
        """Passing the grid fmask through ocean_mask_from_fmask == 1 - bmask.

        This is the invariant the drivers rely on: feeding the trainer the
        ocean grid's fmask makes its internal binarization agree with the
        pinning bmask cell-for-cell.
        """
        bmask = jnp.array([[0.0, 1.0], [0.0, 1.0]])
        fmask = jnp.array([[0.5, 0.99], [0.95, 1.0]])  # >0.95 iff bmask==1
        coupler = _FakeCouplerWithGrid(bmask, fmask)
        grid_fmask = ocean_fmask_from_coupler(coupler)
        self.assertTrue(jnp.array_equal(
            ocean_mask_from_fmask(grid_fmask), 1.0 - bmask
        ))


class TestEnsembleGradients(unittest.TestCase):
    """Averaged per-IC gradients must equal the gradient of the mean loss."""

    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()
        cls.coupler = FakeCoupler()
        cls.config = CoupledControllerConfig(
            control_interval_steps=2,
            total_steps=4,
            target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
            use_checkpointing=True,
        )
        cls.ocean_mask = jnp.ones(cls.coords.horizontal.nodal_shape)
        cls.policy = MCBPolicyMLP(
            output_shape=cls.coords.horizontal.nodal_shape,
            hidden_dims=(8,),
            max_perturbation=0.15,
        )
        cls.params = cls.policy.init(jax.random.PRNGKey(0), jnp.zeros(13))

        step_fn = create_coupled_step_fn(cls.coupler, WORKFLOW)
        cls.carries = [
            make_fake_carry(cls.coords, 288.0),
            make_fake_carry(cls.coords, 289.5),
        ]
        cls.baselines = [
            compute_baseline_trajectory(c, step_fn, num_steps=4,
                                        coords=cls.coords)
            for c in cls.carries
        ]

    def test_averaged_grads_equal_grad_of_mean_loss(self):
        grad_fn = create_coupled_grad_fn(
            coupler=self.coupler,
            workflow=WORKFLOW,
            policy_fn=self.policy.apply,
            coords=self.coords,
            ocean_mask=self.ocean_mask,
            controller_config=self.config,
        )

        grads = []
        for carry, baseline in zip(self.carries, self.baselines):
            _, g = grad_fn(self.params, carry, baseline)
            grads.append(g)
        avg = jax.tree_util.tree_map(lambda a, b: (a + b) / 2.0, *grads)

        def mean_loss(p):
            losses = [
                unroll_coupled_simple(
                    coupler=self.coupler,
                    workflow=WORKFLOW,
                    policy_fn=self.policy.apply,
                    policy_params=p,
                    initial_carry=carry,
                    baseline_trajectory=baseline,
                    coords=self.coords,
                    ocean_mask=self.ocean_mask,
                    config=self.config,
                )
                for carry, baseline in zip(self.carries, self.baselines)
            ]
            return (losses[0] + losses[1]) / 2.0

        ref = jax.jit(jax.grad(mean_loss))(self.params)

        avg_leaves = jax.tree_util.tree_leaves(avg)
        ref_leaves = jax.tree_util.tree_leaves(ref)
        self.assertEqual(len(avg_leaves), len(ref_leaves))
        # Gradients must actually flow
        self.assertFalse(all(jnp.allclose(g, 0.0) for g in avg_leaves))
        for a, r in zip(avg_leaves, ref_leaves):
            self.assertTrue(
                jnp.allclose(a, r, rtol=1e-4, atol=1e-7),
                f"max abs diff {float(jnp.max(jnp.abs(a - r))):.3e}",
            )

    def test_eval_fn_matches_grad_fn_loss(self):
        grad_fn = create_coupled_grad_fn(
            coupler=self.coupler, workflow=WORKFLOW,
            policy_fn=self.policy.apply, coords=self.coords,
            ocean_mask=self.ocean_mask, controller_config=self.config,
        )
        eval_fn = create_coupled_eval_fn(
            coupler=self.coupler, workflow=WORKFLOW,
            policy_fn=self.policy.apply, coords=self.coords,
            ocean_mask=self.ocean_mask, controller_config=self.config,
        )
        loss_g, _ = grad_fn(self.params, self.carries[0], self.baselines[0])
        loss_e = eval_fn(self.params, self.carries[0], self.baselines[0])
        self.assertTrue(jnp.allclose(loss_g, loss_e, rtol=1e-6))


class TestEnsembleTraining(unittest.TestCase):
    """Smoke test for train_coupled_policy_ensemble on the fake coupler."""

    def test_two_epoch_run(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        config = CoupledControllerConfig(
            control_interval_steps=2,
            total_steps=4,
            target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
        )
        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape,
            hidden_dims=(8,),
            max_perturbation=0.15,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carries = [make_fake_carry(coords, v) for v in (288.0, 289.0, 290.0)]
        baselines = [
            compute_baseline_trajectory(c, step_fn, num_steps=4, coords=coords)
            for c in carries
        ]

        training_config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            log_interval=1,
            early_stopping_patience=None,
        )

        best_params, history = train_coupled_policy_ensemble(
            coupler=coupler,
            workflow=WORKFLOW,
            policy=policy,
            coords=coords,
            terrain_fmask=jnp.zeros(coords.horizontal.nodal_shape),
            train_carries=carries[:2],
            train_baselines=baselines[:2],
            heldout_carries=(carries[2],),
            heldout_baselines=(baselines[2],),
            training_config=training_config,
            controller_config=config,
            heldout_interval=1,
        )

        self.assertEqual(history['epochs_completed'], 2)
        self.assertEqual(len(history['loss_history']), 2)
        self.assertEqual(len(history['per_ic_loss_history']), 2)
        self.assertEqual(len(history['per_ic_loss_history'][0]), 2)
        self.assertEqual(len(history['heldout_loss_history']), 2)
        self.assertEqual(history['num_train_ics'], 2)
        self.assertEqual(history['num_heldout_ics'], 1)
        self.assertTrue(all(jnp.isfinite(v) for v in history['loss_history']))
        self.assertTrue(all(jnp.isfinite(v)
                            for v in history['grad_norm_history']))
        # Mean of per-IC losses equals the recorded mean loss
        for mean_v, per_ic in zip(history['loss_history'],
                                  history['per_ic_loss_history']):
            self.assertAlmostEqual(mean_v, sum(per_ic) / len(per_ic), places=6)
        # Returned params have the policy's structure
        self.assertEqual(
            jax.tree_util.tree_structure(best_params),
            jax.tree_util.tree_structure(
                policy.init(jax.random.PRNGKey(0), jnp.zeros(13))
            ),
        )


if __name__ == "__main__":
    unittest.main()
