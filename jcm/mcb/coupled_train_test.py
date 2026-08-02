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
    evaluate_coupled_policy,
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
        # Tolerance is platform-sensitive: the two computations sum in
        # different orders under f32 (host-averaged per-IC grads vs
        # grad-of-mean), and ARM/x86 XLA fuse differently — measured up to
        # ~5e-5 abs diff on the GB10 (aarch64) for O(0.1-1)-norm grads. A
        # real bug (sign, factor, dropped IC) shows up at O(1e-2) or more.
        for a, r in zip(avg_leaves, ref_leaves):
            self.assertTrue(
                jnp.allclose(a, r, rtol=1e-3, atol=2e-4),
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

    def test_best_params_reproduces_best_loss(self):
        """Off-by-one regression: re-evaluating best_params must reproduce
        history['best_loss'].

        The trainer measures each epoch's loss at the PRE-update params, so
        best_params must be those pre-update params. The original code saved
        the POST-update params, whose loss was never measured — re-evaluating
        them would NOT match best_loss. On the deterministic fake coupler the
        match is exact to numerical precision.
        """
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
        carries = [make_fake_carry(coords, v) for v in (288.0, 289.5)]
        baselines = [
            compute_baseline_trajectory(c, step_fn, num_steps=4, coords=coords)
            for c in carries
        ]
        ocean_mask = jnp.ones(coords.horizontal.nodal_shape)
        # A learning rate large enough that the optimizer step visibly moves
        # the params, so a post-update checkpoint would differ from the
        # measured one — i.e. the off-by-one would be detectable.
        training_config = TrainingConfig(
            num_epochs=4, learning_rate=1e-2, log_interval=1,
            early_stopping_patience=None,
        )
        best_params, history = train_coupled_policy_ensemble(
            coupler=coupler, workflow=WORKFLOW, policy=policy, coords=coords,
            terrain_fmask=jnp.zeros(coords.horizontal.nodal_shape),
            train_carries=carries, train_baselines=baselines,
            heldout_carries=(), heldout_baselines=(),
            training_config=training_config, controller_config=config,
        )

        eval_fn = create_coupled_eval_fn(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy.apply,
            coords=coords, ocean_mask=ocean_mask, controller_config=config,
        )
        remeasured = sum(
            float(eval_fn(best_params, c, b))
            for c, b in zip(carries, baselines)
        ) / len(carries)
        self.assertAlmostEqual(remeasured, history['best_loss'], places=5)

    def test_select_on_heldout_gates_and_logs_coherence(self):
        """select_on_heldout must gate selection/early-stop on held-out loss,
        evaluate held-out every epoch, and record per-IC gradient coherence.
        """
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4, target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
        )
        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape, hidden_dims=(8,),
            max_perturbation=0.15,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carries = [make_fake_carry(coords, v) for v in (288.0, 289.0, 290.0)]
        baselines = [
            compute_baseline_trajectory(c, step_fn, num_steps=4, coords=coords)
            for c in carries
        ]
        training_config = TrainingConfig(
            num_epochs=3, learning_rate=1e-2, log_interval=1,
            early_stopping_patience=None,
        )
        best_params, history = train_coupled_policy_ensemble(
            coupler=coupler, workflow=WORKFLOW, policy=policy, coords=coords,
            terrain_fmask=jnp.zeros(coords.horizontal.nodal_shape),
            train_carries=carries[:2], train_baselines=baselines[:2],
            heldout_carries=(carries[2],), heldout_baselines=(baselines[2],),
            training_config=training_config, controller_config=config,
            heldout_interval=99,  # would suppress held-out logging if not gated
            select_on_heldout=True,
        )
        n = history['epochs_completed']
        self.assertEqual(history['selection_metric'], 'heldout')
        # Held-out evaluated EVERY epoch despite heldout_interval=99.
        self.assertEqual(len(history['heldout_loss_history']), n)
        self.assertEqual(len(history['coherence_history']), n)
        self.assertEqual(len(history['per_ic_grad_norm_history']), n)
        # Coherence is a ratio in (0, ~1]; per-IC norms present, one per IC.
        for c in history['coherence_history']:
            self.assertTrue(0.0 <= c <= 1.5)
        for norms in history['per_ic_grad_norm_history']:
            self.assertEqual(len(norms), 2)
        # best_loss == min over epochs of the mean held-out loss (selection
        # is on held-out, not train).
        heldout_means = [sum(hl) / len(hl)
                         for _, hl in history['heldout_loss_history']]
        self.assertAlmostEqual(history['best_loss'], min(heldout_means),
                               places=6)

    def test_select_on_heldout_requires_heldout_ics(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4, target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
        )
        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape, hidden_dims=(8,),
            max_perturbation=0.15,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carry = make_fake_carry(coords, 288.0)
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=4, coords=coords)
        with self.assertRaises(ValueError):
            train_coupled_policy_ensemble(
                coupler=coupler, workflow=WORKFLOW, policy=policy,
                coords=coords,
                terrain_fmask=jnp.zeros(coords.horizontal.nodal_shape),
                train_carries=[carry], train_baselines=[baseline],
                heldout_carries=(), heldout_baselines=(),
                training_config=TrainingConfig(num_epochs=1),
                controller_config=config, select_on_heldout=True,
            )


class TestTerminalDsstLossMode(unittest.TestCase):
    """loss_mode='terminal_dsst' must equal the gate metric (final dSST error).

    The historical 'summed' objective sums per-interval losses dominated by
    early-interval transients and the uniformity penalty (<0.2% gate-relevant
    and rewarding overcooling). 'terminal_dsst' trains on exactly what the
    pre-registered gate scores.
    """

    def _setup(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape, hidden_dims=(8,),
            max_perturbation=0.15,
        )
        params = policy.init(jax.random.PRNGKey(1), jnp.zeros(13))
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carry = make_fake_carry(coords, 289.0)
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=4, coords=coords)
        ocean_mask = jnp.ones(coords.horizontal.nodal_shape)
        return coords, coupler, policy, params, carry, baseline, ocean_mask

    def test_terminal_loss_equals_gate_dsst_error(self):
        coords, coupler, policy, params, carry, baseline, ocean_mask = \
            self._setup()
        cfg = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4, target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15, loss_mode="terminal_dsst",
            forcing_reg_weight=0.0,  # isolate the pure dSST term
        )
        loss = float(unroll_coupled_simple(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy.apply,
            policy_params=params, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords,
            ocean_mask=ocean_mask, config=cfg,
        ))
        metrics = evaluate_coupled_policy(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy.apply,
            policy_params=params, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords,
            ocean_mask=ocean_mask, config=cfg,
        )["metrics"]
        expected = (metrics["final_sst_change"] - cfg.target_cooling) ** 2
        # With forcing_reg=0 the training loss IS the squared gate error. The
        # loss subtracts-then-weights (more float32-stable) while the gate
        # metric weights-then-subtracts two ~289 K sums, so they agree only to
        # a float32 reduction-order epsilon (~1e-5 K in dSST, three orders
        # below the 0.013 K noise floor) — not bit-identical. places=4 still
        # separates this from the ~10x larger summed objective.
        self.assertAlmostEqual(loss, expected, places=4)

    def test_terminal_differs_from_summed(self):
        coords, coupler, policy, params, carry, baseline, ocean_mask = \
            self._setup()
        base = dict(
            control_interval_steps=2, total_steps=4, target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
        )
        summed = float(unroll_coupled_simple(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy.apply,
            policy_params=params, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords, ocean_mask=ocean_mask,
            config=CoupledControllerConfig(loss_mode="summed", **base),
        ))
        terminal = float(unroll_coupled_simple(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy.apply,
            policy_params=params, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords, ocean_mask=ocean_mask,
            config=CoupledControllerConfig(loss_mode="terminal_dsst", **base),
        ))
        self.assertNotAlmostEqual(summed, terminal, places=4)


class TestAreaWeights(unittest.TestCase):
    """R1 regression: compute_area_weights must be a genuine cos(lat) weighting.

    Latitudes are stored in RADIANS. The historical bug applied jnp.radians()
    to them a second time, collapsing the weights to near-uniform (pole/equator
    ratio ~1) and silently turning every 'area-weighted' global mean into a
    cell-count mean that over-weights the poles ~20x.
    """

    def test_pole_equator_ratio_is_cosine(self):
        from jcm.mcb.state_features import compute_area_weights
        coords = get_speedy_coords()
        w = compute_area_weights(coords)
        # Normalized to a probability distribution over the grid.
        self.assertAlmostEqual(float(jnp.sum(w)), 1.0, places=5)
        ratio = float(jnp.max(w) / jnp.min(w))
        # Correct cos(lat) weighting at T30 gives ~20; the double-radians bug
        # gives ~1. Guard hard against a regression to near-uniform.
        self.assertGreater(ratio, 10.0)

    def test_matches_cosine_of_latitude(self):
        import numpy as np
        from jcm.mcb.state_features import compute_area_weights
        coords = get_speedy_coords()
        lats = np.asarray(coords.horizontal.latitudes)
        w = np.asarray(compute_area_weights(coords))
        expected = np.cos(lats)
        expected = expected / expected.sum()  # per-latitude, before lon tiling
        # Every longitude row is identical; compare one column to cos(lat).
        got = w[0, :] / w[0, :].sum()
        self.assertTrue(np.allclose(got, expected, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()


class TestTailMeanMetric(unittest.TestCase):
    """The pre-registered final-10-day time-mean dSST (meta-audit Tier 1).

    With the FakeCoupler (dSST/step = -0.05 * perturbation) and a constant
    policy p, the dSST after t steps is -0.05*p*t exactly, so both the
    snapshot and the tail-mean have closed forms.
    """

    @classmethod
    def setUpClass(cls):
        from jcm.mcb.coupled_controller import evaluate_coupled_policy
        cls.evaluate_coupled_policy = staticmethod(evaluate_coupled_policy)
        cls.coords = get_speedy_coords()
        cls.coupler = FakeCoupler()
        cls.shape = cls.coords.horizontal.nodal_shape
        cls.ocean_mask = jnp.ones(cls.shape)
        cls.p = 0.02
        cls.total_steps = 8
        cls.config = CoupledControllerConfig(
            control_interval_steps=2,
            total_steps=cls.total_steps,
            target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
            use_checkpointing=True,
        )
        step_fn = create_coupled_step_fn(cls.coupler, WORKFLOW)
        cls.carry = make_fake_carry(cls.coords, 288.0)
        cls.baseline = compute_baseline_trajectory(
            cls.carry, step_fn, num_steps=cls.total_steps, coords=cls.coords)

    def _eval(self, tail_mean_days):
        policy_fn = lambda params, feats: jnp.full(self.shape, self.p)  # noqa: E731
        return self.evaluate_coupled_policy(
            coupler=self.coupler,
            workflow=WORKFLOW,
            policy_fn=policy_fn,
            policy_params={},
            initial_carry=self.carry,
            baseline_trajectory=self.baseline,
            coords=self.coords,
            ocean_mask=self.ocean_mask,
            config=self.config,
            tail_mean_days=tail_mean_days,
        )["metrics"]

    # Analytic tolerances are loose (2e-4) because 288 K SST quantizes at the
    # float32 ULP (~3e-5 K/step, accumulating); the sharp assertions are the
    # index-convention invariants, which must hold to float32 exactness.

    def test_snapshot_and_tail_mean(self):
        m = self._eval(tail_mean_days=4)
        rate = 0.05 * self.p
        # Snapshot at T=8 steps: -0.05*p*8
        self.assertAlmostEqual(m["final_sst_change"],
                               -rate * self.total_steps, delta=2e-4)
        # Tail mean over t = 5..8: -0.05*p*mean(5,6,7,8) = -0.05*p*6.5
        self.assertIn("final_sst_change_10d", m)
        self.assertEqual(m["tail_mean_days"], 4)
        self.assertAlmostEqual(m["final_sst_change_10d"],
                               -rate * 6.5, delta=2e-4)
        # INVARIANT: the last daily tail dSST is the same quantity as the
        # snapshot (same final state, same baseline index) — exact.
        self.assertAlmostEqual(float(m["dsst_daily_tail"][-1]),
                               m["final_sst_change"], places=6)
        # INVARIANT: the reported tail mean is the mean of the daily values.
        self.assertAlmostEqual(
            m["final_sst_change_10d"],
            float(jnp.mean(jnp.asarray(m["dsst_daily_tail"]))), places=6)
        # Daily tail dSSTs are strictly monotone (cooling accumulates), so an
        # off-by-one in the baseline index would break the analytic values.
        daily = [float(x) for x in m["dsst_daily_tail"]]
        for t, got in zip((5, 6, 7, 8), daily):
            self.assertAlmostEqual(got, -rate * t, delta=2e-4)

    def test_tail_disabled_reproduces_old_metrics(self):
        m = self._eval(tail_mean_days=0)
        self.assertNotIn("final_sst_change_10d", m)
        self.assertAlmostEqual(m["final_sst_change"],
                               -0.05 * self.p * self.total_steps, delta=2e-4)

    def test_tail_clipped_to_horizon(self):
        m = self._eval(tail_mean_days=99)
        self.assertEqual(m["tail_mean_days"], self.total_steps)
        # Mean over all t=1..8: -0.05*p*4.5
        self.assertAlmostEqual(m["final_sst_change_10d"],
                               -0.05 * self.p * 4.5, delta=2e-4)


class TestEfficacy(unittest.TestCase):
    """Per-episode MCB efficacy (Tier-2 meta-audit experiment).

    With the FakeCoupler (dSST/step = -0.05 * applied perturbation) and a
    constant commanded policy p, applied = eta * p exactly, so the terminal
    dSST scales linearly in eta.
    """

    @classmethod
    def setUpClass(cls):
        from jcm.mcb.coupled_controller import evaluate_coupled_policy
        cls.evaluate_coupled_policy = staticmethod(evaluate_coupled_policy)
        cls.coords = get_speedy_coords()
        cls.coupler = FakeCoupler()
        cls.shape = cls.coords.horizontal.nodal_shape
        cls.ocean_mask = jnp.ones(cls.shape)
        cls.p = 0.02
        cls.total_steps = 4
        cls.config = CoupledControllerConfig(
            control_interval_steps=2,
            total_steps=cls.total_steps,
            target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
            use_checkpointing=True,
        )
        step_fn = create_coupled_step_fn(cls.coupler, WORKFLOW)
        cls.carry = make_fake_carry(cls.coords, 288.0)
        cls.baseline = compute_baseline_trajectory(
            cls.carry, step_fn, num_steps=cls.total_steps, coords=cls.coords)

    def _dsst(self, efficacy):
        policy_fn = lambda params, feats: jnp.full(self.shape, self.p)  # noqa: E731
        out = self.evaluate_coupled_policy(
            coupler=self.coupler, workflow=WORKFLOW, policy_fn=policy_fn,
            policy_params={}, initial_carry=self.carry,
            baseline_trajectory=self.baseline, coords=self.coords,
            ocean_mask=self.ocean_mask, config=self.config,
            tail_mean_days=0, efficacy=efficacy,
        )
        m = dict(out["metrics"])
        # Per-cell field cooling: exact modulo per-step float32 rounding
        # (~1 ULP/step at 288 K), unlike the area-weighted mean whose f32
        # summation carries ~3e-4 absolute noise on this uniform field.
        final = out["final_carry"]["ocn"]["state"].sea_surface_temperature
        m["field_dsst"] = float(final[0, 0]) - 288.0
        return m

    def test_efficacy_scales_cooling_linearly(self):
        full = self._dsst(1.0)
        half = self._dsst(0.5)
        boosted = self._dsst(1.5)
        # Sharp assertion at the field level (exact to per-step rounding).
        ulp = 2.0 ** -15  # float32 ULP at 288 K
        self.assertAlmostEqual(half["field_dsst"],
                               0.5 * full["field_dsst"], delta=3 * ulp)
        self.assertAlmostEqual(boosted["field_dsst"],
                               1.5 * full["field_dsst"], delta=5 * ulp)
        # Metric level: tolerance covers the f32 weighted-sum noise (~3e-4),
        # which mostly cancels in the paired difference but not exactly.
        self.assertAlmostEqual(half["final_sst_change"],
                               0.5 * full["final_sst_change"], delta=5e-4)
        # Applied forcing scales too (mcb_forcing is the APPLIED field).
        self.assertAlmostEqual(half["mean_mcb_forcing"],
                               0.5 * full["mean_mcb_forcing"], places=6)
        self.assertEqual(half["efficacy"], 0.5)

    def test_default_efficacy_is_identity(self):
        explicit = self._dsst(1.0)["final_sst_change"]
        policy_fn = lambda params, feats: jnp.full(self.shape, self.p)  # noqa: E731
        default = self.evaluate_coupled_policy(
            coupler=self.coupler, workflow=WORKFLOW, policy_fn=policy_fn,
            policy_params={}, initial_carry=self.carry,
            baseline_trajectory=self.baseline, coords=self.coords,
            ocean_mask=self.ocean_mask, config=self.config,
            tail_mean_days=0,
        )["metrics"]["final_sst_change"]
        self.assertAlmostEqual(default, explicit, places=7)

    def test_grad_fn_accepts_traced_efficacy(self):
        from jcm.mcb.coupled_train import create_coupled_grad_fn
        policy_fn = lambda params, feats: jnp.full(self.shape, self.p) * (  # noqa: E731
            1.0 + 0.0 * params["scale"])
        grad_fn = create_coupled_grad_fn(
            coupler=self.coupler, workflow=WORKFLOW, policy_fn=policy_fn,
            coords=self.coords, ocean_mask=self.ocean_mask,
            controller_config=self.config,
        )
        params = {"scale": jnp.array(1.0)}
        l1, _ = grad_fn(params, self.carry, self.baseline, efficacy=1.0)
        l2, _ = grad_fn(params, self.carry, self.baseline, efficacy=0.5)
        self.assertNotAlmostEqual(float(l1), float(l2), places=6)


class TestEnsembleEfficacyRandomization(unittest.TestCase):
    """Tier-2 domain randomization plumbs per-episode efficacies end to end."""

    def test_efficacies_used_and_logged(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4, target_cooling=-0.1,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15,
        )
        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape, hidden_dims=(8,),
            max_perturbation=0.15,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carries = [make_fake_carry(coords, v) for v in (288.0, 288.0, 288.0)]
        baselines = [
            compute_baseline_trajectory(c, step_fn, num_steps=4, coords=coords)
            for c in carries
        ]
        training_config = TrainingConfig(
            num_epochs=2, learning_rate=1e-3, log_interval=1,
            early_stopping_patience=None,
        )

        def run(effs):
            _, h = train_coupled_policy_ensemble(
                coupler=coupler, workflow=WORKFLOW, policy=policy,
                coords=coords,
                terrain_fmask=jnp.zeros(coords.horizontal.nodal_shape),
                train_carries=carries[:2], train_baselines=baselines[:2],
                heldout_carries=(carries[2],),
                heldout_baselines=(baselines[2],),
                training_config=training_config, controller_config=config,
                select_on_heldout=True,
                train_efficacies=effs, heldout_efficacies=(effs[0],) if effs
                else (),
            )
            return h

        h_hi = run((1.5, 1.5))
        h_lo = run((0.5, 0.5))
        self.assertEqual(h_hi['train_efficacies'], [1.5, 1.5])
        self.assertEqual(h_lo['heldout_efficacies'], [0.5])
        # Identical ICs and identical policy init: only efficacy differs, so
        # epoch-0 train losses must differ between the two runs (the applied
        # forcing is 3x larger in the high-efficacy run).
        self.assertNotAlmostEqual(h_hi['loss_history'][0],
                                  h_lo['loss_history'][0], places=6)
        # Default (no efficacies) must log 1.0s.
        h_def = run(())
        self.assertEqual(h_def['train_efficacies'], [1.0, 1.0])


class TestTailDsstLoss(unittest.TestCase):
    """tail_dsst loss (Amendment 3) matches the registered tail-mean metric.

    FakeCoupler analytic: constant policy p, eta=1, T=8 steps, tail=4:
    dsst_tail = -0.05*p*mean(5,6,7,8) = -0.05*p*6.5; setting target to the
    analytic value makes the terminal loss ~ 0 + forcing_reg * p^2.
    """

    def test_tail_loss_analytic(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        shape = coords.horizontal.nodal_shape
        p = 0.02
        analytic_tail = -0.05 * p * 6.5
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=8,
            target_cooling=analytic_tail,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15, use_checkpointing=True,
            loss_mode="tail_dsst", tail_days=4, forcing_reg_weight=0.001,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carry = make_fake_carry(coords, 288.0)
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=8, coords=coords)
        policy_fn = lambda params, feats: jnp.full(shape, p)  # noqa: E731
        loss = float(unroll_coupled_simple(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy_fn,
            policy_params={}, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords,
            ocean_mask=jnp.ones(shape), config=config,
        ))
        reg = 0.001 * p ** 2  # applied forcing is p everywhere at eta=1
        # Terminal term ~ (f32 mean-noise ~3e-4)^2 ~ 1e-7; reg = 4e-10.
        self.assertLess(abs(loss - reg), 5e-7)
        # A mis-set target must show up quadratically.
        config_off = config.copy(target_cooling=analytic_tail - 0.01) if \
            hasattr(config, "copy") else None
        if config_off is None:
            import dataclasses
            try:
                config_off = dataclasses.replace(
                    config, target_cooling=analytic_tail - 0.01)
            except TypeError:
                config_off = CoupledControllerConfig(
                    control_interval_steps=2, total_steps=8,
                    target_cooling=analytic_tail - 0.01,
                    feature_config=CoupledFeatureConfig(
                        include_absolute_sst=True),
                    max_perturbation=0.15, use_checkpointing=True,
                    loss_mode="tail_dsst", tail_days=4,
                    forcing_reg_weight=0.001,
                )
        loss_off = float(unroll_coupled_simple(
            coupler=coupler, workflow=WORKFLOW, policy_fn=policy_fn,
            policy_params={}, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords,
            ocean_mask=jnp.ones(shape), config=config_off,
        ))
        self.assertAlmostEqual(loss_off, 0.01 ** 2 + reg, delta=1e-5)

    def test_tail_loss_gradient_flows(self):
        coords = get_speedy_coords()
        coupler = FakeCoupler()
        shape = coords.horizontal.nodal_shape
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4, target_cooling=-0.01,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=0.15, use_checkpointing=True,
            loss_mode="tail_dsst", tail_days=2,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carry = make_fake_carry(coords, 288.0)
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=4, coords=coords)
        grad_fn = create_coupled_grad_fn(
            coupler=coupler, workflow=WORKFLOW,
            policy_fn=lambda prm, f: jnp.full(shape, 0.01) * prm["s"],
            coords=coords, ocean_mask=jnp.ones(shape),
            controller_config=config,
        )
        loss, grads = grad_fn({"s": jnp.array(1.0)}, carry, baseline)
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertNotEqual(float(grads["s"]), 0.0)
