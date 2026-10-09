"""Tests for the Experiment 1 driver and the macro-IC plan.

Fast tests cover the pure helpers. The two slow tests run the REAL coupled
model on CPU for a few days (the first call compiles the step, ~4 min):
the truncation wrapper must leave forward values bit-identical, a window
longer than the rollout must reproduce full BPTT exactly, and the
same-day-only gradient (W = 1) must keep the direct actuator path alive.
"""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from run_generate_macro_ics import (
    DEFAULT_PLAN,
    EVAL_MACROS,
    EXP3B_PLAN,
    EXP3D_PLAN,
    TRAIN_MACROS,
    branch_seed,
    check_exp3b_settings,
    check_registered_settings,
    ic_index,
    macro_diagnostics,
    split_macros,
    validate_plan,
)
from run_generate_macro_ics import parse_args as macro_parse_args
from run_gradient_fidelity import (
    amplitude_for,
    forward_reverse_check,
    parse_args,
    run_labels,
    truth_alignment_error,
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
               ("--windows", "-1"),
               ("--damped-efold-days", "0"),
               ("--damped-efold-days", "3", "-1")]
        for extra in bad:
            with self.assertRaises(SystemExit):
                validate_args(self._args(*extra))

    def test_damped_estimators_default_and_can_be_skipped(self):
        # Revision 0.1 registers tau = 3 and 7 days; an empty list skips them.
        self.assertEqual(self._args().damped_efold_days, [3.0, 7.0])
        skipped = self._args("--damped-efold-days")
        self.assertEqual(skipped.damped_efold_days, [])
        validate_args(skipped)

    def test_maps_are_on_by_default_in_registered_blocks(self):
        # Revision 0.4: 5-day blocks tile every registered tail window.
        args = self._args()
        self.assertTrue(args.save_maps and args.map_jacobians)
        self.assertEqual(args.map_block_days, 5)
        self.assertFalse(self._args("--no-map-jacobians").map_jacobians)

    def test_map_blocks_must_tile_every_tail_window(self):
        for block in ("0", "3", "4"):
            with self.assertRaises(SystemExit):
                validate_args(self._args("--map-block-days", block))
        validate_args(self._args("--map-block-days", "3", "--no-maps"))
        validate_args(self._args("--horizons", "2", "3", "--tail-days", "1",
                                 "--map-block-days", "1"))


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

    def test_training_and_evaluation_are_interleaved(self):
        # Revision 0.4: even macro states train, odd ones evaluate.
        self.assertEqual(TRAIN_MACROS, [0, 2, 4, 6, 8, 10, 12, 14])
        self.assertEqual(EVAL_MACROS, [1, 3, 5, 7, 9, 11, 13, 15])
        train, evaluation = split_macros(validate_plan(DEFAULT_PLAN, 16))
        self.assertEqual((train, evaluation), (TRAIN_MACROS, EVAL_MACROS))
        # Both sides span the whole control run; states on one side sit two
        # spacings apart.
        self.assertEqual(np.diff(train).tolist(), [2] * 7)
        self.assertEqual(np.diff(evaluation).tolist(), [2] * 7)

    def test_rejects_reuse_and_out_of_range(self):
        reuse = {"a": [{"macro": [0], "branches": [0], "split": "train",
                        "horizon": 5}],
                 "b": [{"macro": [0], "branches": [0], "split": "heldout",
                        "horizon": 5}]}
        with self.assertRaises(SystemExit):
            validate_plan(reuse, 2)
        with self.assertRaises(SystemExit):
            validate_plan(DEFAULT_PLAN, 8)

    def test_rejects_an_evaluation_state_used_by_another_role(self):
        shared = {"exp9_train": [{"macro": [0, 1], "branches": [0],
                                  "split": "train", "horizon": 5}],
                  "exp9_eval": [{"macro": [1], "branches": [1],
                                 "split": "heldout", "horizon": 5}]}
        with self.assertRaises(SystemExit):
            validate_plan(shared, 2)
        smoke = {"exp1": [{"macro": [0, 1], "branches": [0],
                           "split": "heldout", "horizon": 3}]}
        self.assertEqual(len(validate_plan(smoke, 2)), 2)

    def test_seeds_and_indices_are_unique(self):
        flat = validate_plan(DEFAULT_PLAN, 16)
        seeds = {branch_seed(12000, m, b) for _, m, b, _, _ in flat}
        idx = {ic_index(m, b) for _, m, b, _, _ in flat}
        self.assertEqual(len(seeds), len(flat))
        self.assertEqual(len(idx), len(flat))

    def test_exp3b_plan_is_fresh_interleaved_and_disjoint(self):
        flat = validate_plan(EXP3B_PLAN, 33, offset=15)
        counts = {}
        for role, *_ in flat:
            counts[role] = counts.get(role, 0) + 1
        self.assertEqual(counts, {"exp3b_train": 32, "exp3b_eval": 48})
        train, evaluation = split_macros(flat)
        self.assertEqual(train, list(range(16, 48, 2)))
        self.assertEqual(evaluation, list(range(17, 48, 2)))
        # Nothing is shared with the states of revisions 0.4 and 1.
        old = validate_plan(DEFAULT_PLAN, 16)
        self.assertFalse({m for _, m, *_ in flat} & {m for _, m, *_ in old})
        both = flat + old
        self.assertEqual(len({ic_index(m, b) for _, m, b, *_ in both}),
                         len(both))
        self.assertEqual(len({branch_seed(12000, m, b)
                              for _, m, b, *_ in both}), len(both))

    def test_offset_bounds_the_global_macro_indices(self):
        with self.assertRaises(SystemExit):
            validate_plan(EXP3B_PLAN, 33)            # offset 0: 16-47 too big
        with self.assertRaises(SystemExit):
            validate_plan(EXP3B_PLAN, 32, offset=15)  # 47 just outside
        smoke = {"exp9_eval": [{"macro": [5], "branches": [0],
                                "split": "heldout", "horizon": 3}]}
        self.assertEqual(len(validate_plan(smoke, 2, offset=4)), 1)

    def test_exp3b_registered_settings_are_enforced(self):
        base = ["--base-carry", "x", "--output-root", "y", "--plan", "exp3b"]
        good = ["--macro-offset", "15", "--num-macro", "33",
                "--spacing-days", "365"]
        check_exp3b_settings(macro_parse_args(base + good))
        for i in (1, 3, 5):
            bad = list(good)
            bad[i] = "7"
            with self.assertRaises(SystemExit):
                check_exp3b_settings(macro_parse_args(base + bad))
        # The original plan is unaffected by the check.
        check_exp3b_settings(macro_parse_args(["--base-carry", "x",
                                               "--output-root", "y"]))

    def test_exp3d_plan_is_fresh_interleaved_and_disjoint(self):
        flat = validate_plan(EXP3D_PLAN, 33, offset=99)
        counts = {}
        for role, *_ in flat:
            counts[role] = counts.get(role, 0) + 1
        self.assertEqual(counts, {"exp3d_train": 32, "exp3d_eval": 48})
        train, evaluation = split_macros(flat)
        self.assertEqual(train, list(range(100, 132, 2)))
        self.assertEqual(evaluation, list(range(101, 132, 2)))
        # No index or seed is shared with any earlier state.
        old = (validate_plan(DEFAULT_PLAN, 16)
               + validate_plan(EXP3B_PLAN, 33, offset=15))
        both = flat + old
        self.assertEqual(len({ic_index(m, b) for _, m, b, *_ in both}),
                         len(both))
        self.assertEqual(len({branch_seed(12000, m, b)
                              for _, m, b, *_ in both}), len(both))

    def test_exp3d_needs_its_settings_and_the_fixed_land(self):
        base = ["--base-carry", "x", "--output-root", "y", "--plan", "exp3d"]
        good = ["--macro-offset", "99", "--num-macro", "33",
                "--spacing-days", "365"]
        with mock.patch.dict(os.environ, {"JCM_LAND_CLIMATOLOGY": "monthly"}):
            check_registered_settings(macro_parse_args(base + good))
            bad = list(good)
            bad[1] = "15"
            with self.assertRaises(SystemExit):
                check_registered_settings(macro_parse_args(base + bad))
        with mock.patch.dict(os.environ,
                             {"JCM_LAND_CLIMATOLOGY": "daily_index"}):
            with self.assertRaises(SystemExit):
                check_registered_settings(macro_parse_args(base + good))

    def test_macro_diagnostics(self):
        w = np.full((4, 3), 1.0 / 12)
        a = np.zeros((4, 3))
        b = np.ones((4, 3)) * 0.5
        diag = macro_diagnostics([a, b, a], w)
        np.testing.assert_allclose(diag["ocean_mean_sst_K"], [0.0, 0.5, 0.0])
        self.assertEqual(len(diag["neighbour_similarity"]), 2)
        self.assertAlmostEqual(diag["neighbour_similarity"][0]["rms_diff_K"],
                               0.5)
        self.assertEqual(len(diag["two_apart_similarity"]), 1)
        self.assertAlmostEqual(diag["two_apart_similarity"][0]["rms_diff_K"],
                               0.0)
        self.assertNotIn("split_balance", diag)
        self.assertNotIn("trend_K_per_decade", diag)

    def test_macro_diagnostics_trend_and_split_balance(self):
        rng = np.random.default_rng(0)
        w = np.full((4, 3), 1.0 / 12)
        # Ocean means rise by 0.05 K per macro state (730 d apart) plus a
        # spatial pattern that averages to zero.
        pattern = rng.normal(0.0, 0.2, (4, 3))
        pattern -= pattern.mean()
        ssts = [290.0 + 0.05 * m + pattern for m in range(6)]
        diag = macro_diagnostics(ssts, w, [0, 2, 4], [1, 3, 5],
                                 spacing_days=730)
        per_decade = 0.05 * 3652.425 / 730
        self.assertAlmostEqual(diag["trend_K_per_decade"]["slope"],
                               per_decade, places=6)
        self.assertAlmostEqual(diag["trend_K_per_decade"]["se"], 0.0,
                               places=6)
        bal = diag["split_balance"]
        self.assertAlmostEqual(bal["eval_minus_train_K"], 0.05, places=9)
        self.assertEqual((bal["train_macros"], bal["eval_macros"]),
                         ([0, 2, 4], [1, 3, 5]))


class MapChecksTest(unittest.TestCase):
    """The two logged checks of revision 0.4, on synthetic arrays."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.weights = rng.normal(0.0, 1.0, (4, 3, 2))
        self.sst = rng.normal(2.0, 0.5, (6, 3, 2))       # 6 one-day blocks
        self.land = rng.normal(-5.0, 0.5, (6, 3, 2))
        w = self.weights
        self.series = np.stack([
            np.concatenate([np.einsum("oxy,xy->o", w[:3], s),
                            [np.sum(w[3] * land)]])
            for s, land in zip(self.sst, self.land)])

    def test_aligned_maps_pass_and_misaligned_maps_do_not(self):
        err = truth_alignment_error(self.series, self.sst, self.land,
                                    self.weights, [3, 6], 2, 1)
        self.assertLess(err, 1e-12)
        shifted = np.roll(self.sst, 1, axis=0)
        self.assertGreater(truth_alignment_error(
            self.series, shifted, self.land, self.weights, [3, 6], 2, 1),
            1e-3)

    def test_forward_reverse_check(self):
        sst_j = self.sst[None].repeat(2, axis=0) * np.array([1.0, -2.0])[
            :, None, None, None]
        land_j = self.land[None].repeat(2, axis=0)
        reverse = np.stack([np.stack([self.series[h - 2:h].mean(axis=0),
                                      np.concatenate([
                                          -2.0 * self.series[h - 2:h, :3]
                                          .mean(axis=0),
                                          self.series[h - 2:h, 3:]
                                          .mean(axis=0)])], axis=1)
                            for h in (3, 6)])
        out = forward_reverse_check(sst_j, land_j, reverse, self.weights,
                                    [3, 6], 2, 1)
        self.assertEqual(set(out), {"3", "6"})
        for cell in out.values():
            self.assertLess(cell["rel_diff"], 1e-12)
        out = forward_reverse_check(sst_j, land_j, reverse * 1.01,
                                    self.weights, [3, 6], 2, 1)
        self.assertGreater(min(c["rel_diff"] for c in out.values()), 5e-3)


def _drop_modules_that_hold_jcm():
    """Forget every cached module that refers to jcm classes.

    The root conftest deletes the jcm modules after every test, but jem and
    the run_* scripts keep references to the classes they imported first. A
    model built from a mix of old and new classes fails with "arguments have
    different tree structures", so a real-model setup re-imports all of them.
    """
    import sys
    for key in list(sys.modules):
        root = key.split(".")[0]
        if root in ("jcm", "jem") or (root.startswith("run_")
                                       and not root.endswith("_test")):
            del sys.modules[key]


def _real_model_setup(days):
    _drop_modules_that_hold_jcm()
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
        cls.days = 3
        step_fn, patterns, weights, carry = _real_model_setup(cls.days)
        # Imported after the setup re-imported jcm (see
        # _drop_modules_that_hold_jcm).
        from jcm.mcb.gradient_fidelity import (
            make_jacobian_fn,
            make_objective_fn,
            make_series_fn,
        )
        from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
        cls.no_trunc = NO_TRUNCATION_DAYS
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


@pytest.mark.slow
class RealModelMapsTest(unittest.TestCase):
    """Revision 0.4 maps on the real coupled model (CPU; compiles: minutes)."""

    @classmethod
    def setUpClass(cls):
        cls.days = 3
        step_fn, patterns, weights, carry = _real_model_setup(cls.days)
        # Imported after the setup re-imported jcm (see
        # _drop_modules_that_hold_jcm).
        from jcm.mcb.gradient_fidelity import (
            make_jacobian_fn,
            make_map_jacobian_fn,
            make_series_and_maps_fn,
            make_series_fn,
        )
        from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
        cls.weights = weights
        a0 = jnp.array([0.03, 0.03])
        t0 = carry["ocn"]["state"].sim_time
        one = jnp.asarray(1.0, dtype=jnp.float32)
        cls.series = np.asarray(jax.jit(make_series_fn(
            step_fn, patterns, weights, cls.days))(a0, carry))
        cls.truth = [np.asarray(x) for x in jax.jit(make_series_and_maps_fn(
            step_fn, patterns, weights, cls.days, 1))(a0, carry)]
        reverse = make_jacobian_fn(step_fn, patterns, weights, cls.days, 1)
        forward = make_map_jacobian_fn(step_fn, patterns, weights, cls.days,
                                       1)
        cls.pairs = {}
        for name, window in (("W1", 1), ("full", NO_TRUNCATION_DAYS)):
            args = (a0, carry, jnp.asarray(window), t0, one)
            cls.pairs[name] = (np.asarray(reverse(*args)),
                               [np.asarray(x) for x in forward(*args)])

    def test_truth_runs_keep_the_registered_series(self):
        # Same rollout, different compiled program: round-off only.
        np.testing.assert_allclose(self.truth[0], self.series, rtol=0,
                                   atol=1e-5)
        self.assertEqual(self.truth[1].shape, (self.days, 96, 48))
        self.assertTrue(np.all(np.isfinite(self.truth[1]))
                        and np.all(np.isfinite(self.truth[2])))

    def test_truth_maps_reproduce_their_own_series(self):
        from run_gradient_fidelity import truth_alignment_error
        err = truth_alignment_error(self.truth[0], self.truth[1],
                                    self.truth[2], self.weights,
                                    [1, 2, 3], 1, 1)
        # float32 sums over the grid give a few 1e-6 K; a one-day
        # misalignment would give at least about 1e-2 K.
        self.assertLess(err, 5e-5)

    def test_forward_mode_maps_agree_with_reverse_mode(self):
        # Part 18 step 16 on the real model: forward- and backward-mode
        # derivatives of the same truncated rollout must agree. Measured on
        # CPU (2026-10-01): about 1e-3 per objective, worst 3.3e-3 (LAND at
        # 3 days), the same for W = 1 and full BPTT: float32 round-off over
        # the model's sub-steps. Faults such as a misaligned day or a lost
        # truncation move these Jacobians by tens of percent (W = 1 and
        # full BPTT differ by up to 70% over 3 days, see
        # RealModelTruncationTest).
        from run_gradient_fidelity import forward_reverse_check
        for name, (reverse, (sst_j, land_j)) in self.pairs.items():
            self.assertEqual(sst_j.shape, (2, self.days, 96, 48))
            check = forward_reverse_check(sst_j, land_j, reverse[None],
                                          self.weights, [self.days], 1,
                                          1)[str(self.days)]
            worst = max(check["rel_diff_by_objective"].values())
            self.assertLess(worst, 1e-2, msg=f"{name}: {check}")


if __name__ == "__main__":
    unittest.main()
