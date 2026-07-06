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
