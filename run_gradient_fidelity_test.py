"""Tests for the Experiment 1 driver and the macro-IC plan.

Fast tests cover the pure helpers. The two slow tests run the REAL coupled
model on CPU for a few days (the first call compiles the step, ~4 min):
the truncation wrapper must leave forward values bit-identical, a window
longer than the rollout must reproduce full BPTT exactly, and the
same-day-only gradient (W = 1) must keep the direct actuator path alive.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from run_generate_macro_ics import (
    DEFAULT_PLAN,
    branch_seed,
    ic_index,
    macro_diagnostics,
    validate_plan,
)
from run_gradient_fidelity import (
    amplitude_for,
    parse_args,
    run_labels,
    validate_args,
)


class RunLabelsTest(unittest.TestCase):
    def test_order_and_amplitudes(self):
        labels = run_labels(2, zero_run=True)
        self.assertEqual(labels, ["zero", "center", "plus_0", "minus_0",
                                  "plus_1", "minus_1"])
        a0 = jnp.array([0.03, 0.03])
        np.testing.assert_allclose(amplitude_for("zero", a0, 0.02), [0, 0])
        np.testing.assert_allclose(amplitude_for("center", a0, 0.02), a0)
        np.testing.assert_allclose(amplitude_for("plus_1", a0, 0.02),
                                   [0.03, 0.05])
        np.testing.assert_allclose(amplitude_for("minus_0", a0, 0.02),
                                   [0.01, 0.03])

    def test_labels_without_zero_run(self):
        self.assertEqual(run_labels(1, zero_run=False),
                         ["center", "plus_0", "minus_0"])


class ValidateArgsTest(unittest.TestCase):
    def _args(self, *extra):
        return parse_args(["--ic-dir", "x", "--output", "y", *extra])

    def test_defaults_are_valid(self):
        validate_args(self._args())

    def test_rejects_bad_settings(self):
        bad = [("--grad-members", "9", "--fd-members", "2"),
               ("--horizons", "5", "--tail-days", "10"),
               ("--a0", "0.01", "--delta", "0.02"),
               ("--windows", "-1")]
        for extra in bad:
            with self.assertRaises(SystemExit):
                validate_args(self._args(*extra))


class MacroPlanTest(unittest.TestCase):
    def test_registered_plan_is_valid_and_disjoint(self):
        flat = validate_plan(DEFAULT_PLAN, 16)
        pairs = [(m, b) for _, m, b, _, _ in flat]
        self.assertEqual(len(pairs), len(set(pairs)))
        eval_macros = {m for role, m, _, _, _ in flat if role.endswith("eval")}
        other_macros = {m for role, m, _, _, _ in flat
                        if not role.endswith("eval")}
        self.assertFalse(eval_macros & other_macros)
        counts = {}
        for role, *_ in flat:
            counts[role] = counts.get(role, 0) + 1
        self.assertEqual(counts, {"exp1": 8, "exp2_train": 8, "exp2_eval": 16,
                                  "exp3_train": 24, "exp3_eval": 24})

    def test_rejects_reuse_and_out_of_range(self):
        reuse = {"a": [{"macro": [0], "branches": [0], "split": "train",
                        "horizon": 5}],
                 "b": [{"macro": [0], "branches": [0], "split": "heldout",
                        "horizon": 5}]}
        with self.assertRaises(SystemExit):
            validate_plan(reuse, 2)
        with self.assertRaises(SystemExit):
            validate_plan(DEFAULT_PLAN, 8)

    def test_seeds_and_indices_are_unique(self):
        flat = validate_plan(DEFAULT_PLAN, 16)
        seeds = {branch_seed(12000, m, b) for _, m, b, _, _ in flat}
        idx = {ic_index(m, b) for _, m, b, _, _ in flat}
        self.assertEqual(len(seeds), len(flat))
        self.assertEqual(len(idx), len(flat))

    def test_macro_diagnostics(self):
        w = np.full((4, 3), 1.0 / 12)
        a = np.zeros((4, 3))
        b = np.ones((4, 3)) * 0.5
        diag = macro_diagnostics([a, b, a], w)
        np.testing.assert_allclose(diag["ocean_mean_sst_K"], [0.0, 0.5, 0.0])
        self.assertEqual(len(diag["neighbour_similarity"]), 2)
        self.assertAlmostEqual(diag["neighbour_similarity"][0]["rms_diff_K"],
                               0.5)


def _real_model_setup(days):
    import jax_datetime as jdt

    from jcm.mcb.band_basis import (
        gaussian_band_patterns,
        objective_weights,
        stack_objective_weights,
    )
    from jcm.mcb.coupled_controller import create_coupled_step_fn
    from jcm.mcb.coupled_train import ocean_mask_from_coupler
    from run_coupled_training import coupler_workflow, setup_coupled_model
    from run_gradient_fidelity import land_mask_from_coupler

    coupler, coords, _, _ = setup_coupled_model(
        jdt.to_datetime("2000-01-01"), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    # initialize() first: the slab land model loads stl_clim there.
    carry = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                     jitted=True)
    lats = coords.horizontal.latitudes
    ocean = ocean_mask_from_coupler(coupler)
    land = land_mask_from_coupler(coupler, coords.horizontal.nodal_shape)
    patterns = gaussian_band_patterns(lats, ocean, centers_deg=(20.0, -20.0))
    weights = stack_objective_weights(objective_weights(lats, ocean, land))
    return step_fn, patterns, weights, carry


@pytest.mark.slow
class RealModelTruncationTest(unittest.TestCase):
    """Real coupled model on CPU (compiles the coupled step: minutes)."""

    @classmethod
    def setUpClass(cls):
        from jcm.mcb.gradient_fidelity import (
            make_jacobian_fn,
            make_objective_fn,
            make_series_fn,
        )
        from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
        cls.days = 3
        cls.no_trunc = NO_TRUNCATION_DAYS
        step_fn, patterns, weights, carry = _real_model_setup(cls.days)
        cls.carry = carry
        cls.a0 = jnp.array([0.03, 0.03])
        cls.t0 = carry["ocn"]["state"].sim_time
        # Kept in a dict: jitted functions stored as class attributes would
        # bind like methods and receive the test instance as an argument.
        cls.fns = {
            "series": jax.jit(make_series_fn(step_fn, patterns, weights,
                                             cls.days)),
            "objective": jax.jit(make_objective_fn(step_fn, patterns,
                                                   weights, cls.days, 1)),
            "jac": make_jacobian_fn(step_fn, patterns, weights, cls.days, 1),
        }
        cls.full = np.asarray(cls.fns["jac"](
            cls.a0, carry, jnp.asarray(NO_TRUNCATION_DAYS), cls.t0))

    def test_long_window_reproduces_full_bptt_exactly(self):
        long = np.asarray(self.fns["jac"](self.a0, self.carry,
                                          jnp.asarray(self.days + 1),
                                          self.t0))
        np.testing.assert_array_equal(long, self.full)

    def test_same_day_window_keeps_the_direct_actuator_path(self):
        w1 = np.asarray(self.fns["jac"](self.a0, self.carry, jnp.asarray(1),
                                        self.t0))
        t0_full, t0_w1 = self.full[0], w1[0]     # T0 row: ocean-mean SST
        self.assertTrue(np.all(t0_w1 < 0.0), msg=f"W=1 T0 row {t0_w1}")
        self.assertTrue(np.all(t0_full < 0.0), msg=f"full T0 row {t0_full}")
        # Over 3 days chaos has barely grown: most of the response is the
        # direct cloud-albedo -> surface-sunlight -> ocean path.
        ratio = t0_w1 / t0_full
        self.assertTrue(np.all((ratio > 0.3) & (ratio < 1.5)),
                        msg=f"W=1 / full ratio {ratio}")

    def test_forward_values_do_not_depend_on_truncation(self):
        series = np.asarray(self.fns["series"](self.a0, self.carry))
        self.assertEqual(series.shape, (self.days, 4))
        self.assertTrue(np.all(np.isfinite(series)))
        cut = np.asarray(self.fns["objective"](self.a0, self.carry,
                                               jnp.asarray(1), self.t0))
        uncut = np.asarray(self.fns["objective"](self.a0, self.carry,
                                                 jnp.asarray(self.no_trunc),
                                                 self.t0))
        # Same compiled program, different traced window: bit-identical.
        np.testing.assert_array_equal(cut, uncut)
        # Different compiled programs (series vs objective): round-off only.
        np.testing.assert_allclose(uncut, series[-1], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
