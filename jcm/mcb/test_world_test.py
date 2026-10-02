"""Tests for the Experiment 2-3 test world (jcm.mcb.test_world).

A small gridded toy with the real carry layout:
- the ocean state (sim_time, SST) and an ocean forcing that holds a monthly
  Q-flux;
- the land state;
- the atmosphere state with derived fluxes and the actuator slot.

The toy ocean reads its heat flux from the Q-flux, the same way the real
slab does, so warming through that slot shows up in the SST. The toy is
LINEAR in the brightening, so central differences of the look-ahead objective
(quadratic in the amplitudes) are exact.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.mcb.band_basis import (
    gaussian_band_patterns,
    objective_values,
    objective_weights,
    stack_objective_weights,
)
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.scores import area_weights
from jcm.mcb.test_world import (
    constant_policy,
    domain_weights,
    index_anomalies,
    make_lookahead_objective,
    make_segment_fn,
    make_warming,
    member_seed,
    perturb_member,
    reference_ensemble,
    run_episode,
    schedule_policy,
    time_means,
    warming_flux,
    wrap_step_fn_with_warming,
)

DT = 86400.0
IX, IL = 4, 3
LATS = np.deg2rad(np.array([-30.0, 0.0, 30.0]))
OCEAN = np.ones((IX, IL))
OCEAN[0, :] = 0.0
LAND = 1.0 - OCEAN
R_T, C_T, D_T, K_T = 0.6, 0.2, 0.8, 0.05


@jax.tree_util.register_pytree_node_class
class _OceanState:
    def __init__(self, sim_time, sst):
        self.sim_time = sim_time
        self.sea_surface_temperature = sst

    def copy(self, updates):
        return _OceanState(updates.get("sim_time", self.sim_time),
                           updates.get("sea_surface_temperature",
                                       self.sea_surface_temperature))

    def tree_flatten(self):
        return (self.sim_time, self.sea_surface_temperature), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class _OceanForcing:
    def __init__(self, q_flux):
        self.q_flux = q_flux

    def copy(self, updates):
        return _OceanForcing(updates.get("q_flux", self.q_flux))

    def tree_flatten(self):
        return (self.q_flux,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class _Land:
    def __init__(self, temperature):
        self.land_surface_temperature = temperature

    def tree_flatten(self):
        return (self.land_surface_temperature,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _toy_step(carry, step_idx):
    atm, ocn, lnd = carry["atm"], carry["ocn"], carry["lnd"]
    sst = ocn["state"].sea_surface_temperature
    m = atm["derived"]["mcb_perturbation"]
    x_new = R_T * atm["state"]["x"] + m + C_T * (sst - 290.0)
    q_now = jnp.mean(ocn["forcing"].q_flux, axis=-1)     # upward-positive
    new_sst = sst - K_T * (atm["derived"]["total_heat_flux"] + q_now) * OCEAN
    derived = {"total_heat_flux": x_new + D_T * m,
               "land_heat_flux": jnp.mean(x_new) * jnp.ones_like(x_new),
               "mcb_perturbation": m}
    new_lnd = _Land(lnd["state"].land_surface_temperature
                    - K_T * atm["derived"]["land_heat_flux"])
    return {"atm": {"state": {"x": x_new}, "derived": derived},
            "ocn": {"state": _OceanState(ocn["state"].sim_time + DT, new_sst),
                    "forcing": ocn["forcing"]},
            "lnd": {"state": new_lnd}}, None


def _toy_fields(carry):
    return {"sst": carry["ocn"]["state"].sea_surface_temperature,
            "land_temperature": carry["lnd"]["state"].land_surface_temperature,
            "precipitation": carry["atm"]["state"]["x"],
            "evaporation": carry["atm"]["derived"]["total_heat_flux"]}


def _carry(start_day=3, q_offset=0.0):
    z = jnp.zeros((IX, IL))
    q = jnp.asarray(np.linspace(-2.0, 2.0, 12)[None, None, :] * OCEAN[..., None]
                    + q_offset, jnp.float32)
    return {"atm": {"state": {"x": z + 0.1},
                    "derived": {"total_heat_flux": z, "land_heat_flux": z,
                                "mcb_perturbation": z}},
            "ocn": {"state": _OceanState(jnp.asarray(start_day * DT),
                                         z + 290.0),
                    "forcing": _OceanForcing(q)},
            "lnd": {"state": _Land(z + 280.0)}}


def _run(step, carry, n):
    for s in range(n):
        carry, _ = step(carry, s)
    return carry


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.carry = _carry()
        self.q_base = self.carry["ocn"]["forcing"].q_flux
        self.pattern = jnp.asarray(OCEAN, jnp.float32)


class WarmingTest(_Fixture):
    def test_zero_warming_changes_nothing_bit_for_bit(self):
        w = make_warming(self.carry, self.pattern)
        warmed = _run(wrap_step_fn_with_warming(_toy_step, self.q_base, w),
                      self.carry, 5)
        plain = _run(_toy_step, self.carry, 5)
        for a, b in zip(jax.tree_util.tree_leaves(warmed),
                        jax.tree_util.tree_leaves(plain)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_step_warming_heats_only_the_pattern(self):
        flux, n = 4.0, 6
        w = make_warming(self.carry, self.pattern, step_wm2=flux)
        warmed = _run(wrap_step_fn_with_warming(_toy_step, self.q_base, w),
                      self.carry, n)
        plain = _run(_toy_step, self.carry, n)
        diff = np.asarray(warmed["ocn"]["state"].sea_surface_temperature
                          - plain["ocn"]["state"].sea_surface_temperature)
        # In the toy the extra heat feeds back through the atmosphere, but
        # its first effect is K_T * F per day on the pattern cells only.
        self.assertTrue(np.all(diff[OCEAN == 1] > 0.0))
        np.testing.assert_array_equal(diff[OCEAN == 0], 0.0)
        one = _run(wrap_step_fn_with_warming(_toy_step, self.q_base, w),
                   self.carry, 1)
        d1 = np.asarray(one["ocn"]["state"].sea_surface_temperature
                        - _run(_toy_step, self.carry, 1)["ocn"]["state"]
                        .sea_surface_temperature)
        # Differences of float32 SSTs near 290 K come in steps of 3.05e-5 K.
        np.testing.assert_allclose(d1[OCEAN == 1], K_T * flux, rtol=0,
                                   atol=3.1e-5)

    def test_ramp_is_evaluated_at_mid_step_and_never_compounds(self):
        w = make_warming(self.carry, self.pattern, step_wm2=1.0,
                         ramp_wm2_per_day=0.5)
        step = wrap_step_fn_with_warming(_toy_step, self.q_base, w)
        carry = self.carry
        for day in range(4):
            carry, _ = step(carry, day)
            expected_q = (np.asarray(self.q_base)
                          - float(warming_flux(w, day + 0.5))
                          * OCEAN[..., None])
            np.testing.assert_allclose(
                np.asarray(carry["ocn"]["forcing"].q_flux), expected_q,
                rtol=1e-6, atol=1e-6)


class SegmentEpisodeTest(_Fixture):
    def _seg(self, days):
        return make_segment_fn(_toy_step, self.patterns, days,
                               fields_fn=_toy_fields)

    def test_segment_records_daily_fields(self):
        w = make_warming(self.carry, self.pattern, step_wm2=2.0)
        carry, fields = self._seg(5)(self.carry, jnp.array([0.03, 0.01]),
                                     jnp.asarray(1.0), self.q_base, w)
        self.assertEqual(set(fields), {"sst", "land_temperature",
                                       "precipitation", "evaporation"})
        self.assertEqual(fields["sst"].shape, (5, IX, IL))
        np.testing.assert_array_equal(np.asarray(fields["sst"][-1]),
                                      np.asarray(carry["ocn"]["state"]
                                                 .sea_surface_temperature))

    def test_efficacy_scales_the_brightening(self):
        seg, w = self._seg(4), make_warming(self.carry, self.pattern)
        a = jnp.array([0.04, 0.02])
        _, f_half = seg(self.carry, a, jnp.asarray(0.5), self.q_base, w)
        _, f_scaled = seg(self.carry, 0.5 * a, jnp.asarray(1.0), self.q_base,
                          w)
        np.testing.assert_allclose(np.asarray(f_half["sst"]),
                                   np.asarray(f_scaled["sst"]), rtol=0,
                                   atol=1e-5)

    def test_constant_policy_equals_one_long_segment(self):
        w = make_warming(self.carry, self.pattern, step_wm2=1.0,
                         ramp_wm2_per_day=0.2)
        a = np.array([0.03, 0.05], np.float32)
        ep = run_episode(self.carry, self._seg(3), constant_policy(a),
                         n_segments=3, segment_days=3, k_bands=2, warming=w)
        _, long = self._seg(9)(self.carry, jnp.asarray(a), jnp.asarray(1.0),
                               self.q_base, w)
        np.testing.assert_allclose(ep.fields["sst"], np.asarray(long["sst"]),
                                   rtol=0, atol=1e-5)
        np.testing.assert_array_equal(ep.amplitudes, np.tile(a, (3, 1)))
        np.testing.assert_array_equal(
            np.asarray(ep.final_carry["ocn"]["forcing"].q_flux),
            np.asarray(self.q_base))

    def test_policy_sees_segment_day_and_previous_settings(self):
        seen = []
        schedule = np.array([[0.01, 0.0], [0.02, 0.01], [0.0, 0.03]],
                            np.float32)
        inner = schedule_policy(schedule)

        def policy(state):
            seen.append((state.segment, state.day, state.previous.copy(),
                         state.last_fields is None))
            return inner(state)

        run_episode(self.carry, self._seg(2), policy, n_segments=3,
                    segment_days=2, k_bands=2,
                    warming=make_warming(self.carry, self.pattern))
        self.assertEqual([s[:2] for s in seen], [(0, 0), (1, 2), (2, 4)])
        np.testing.assert_array_equal(seen[0][2], [0.0, 0.0])
        np.testing.assert_array_equal(seen[2][2], schedule[1])
        self.assertEqual([s[3] for s in seen], [True, False, False])

    def test_bad_policy_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            run_episode(self.carry, self._seg(2), constant_policy([0.1]),
                        n_segments=1, segment_days=2, k_bands=2,
                        warming=make_warming(self.carry, self.pattern))


class ReferenceTest(_Fixture):
    def test_member_zero_is_the_ic_and_seeds_match_the_project(self):
        from run_generate_ics_independent import perturb_sst
        self.assertIsNone(member_seed(93000, 200, 0))
        self.assertEqual(member_seed(93000, 200, 2), 93000 + 97 * 200 + 2)
        a = perturb_member(self.carry, 7, 0.001)
        b = perturb_sst(self.carry, 7, 0.001)
        np.testing.assert_array_equal(
            np.asarray(a["ocn"]["state"].sea_surface_temperature),
            np.asarray(b["ocn"]["state"].sea_surface_temperature))

    def test_ensemble_mean_and_blocks(self):
        days = 6
        seg = make_segment_fn(_toy_step, self.patterns, days,
                              fields_fn=_toy_fields)
        w = make_warming(self.carry, self.pattern, step_wm2=1.0)
        ref = reference_ensemble(self.carry, seg, days, 2, w, members=3,
                                 seed0=93000, ic_index=0, member_amp=0.01,
                                 block_days=3)
        runs = []
        for m in range(3):
            s = member_seed(93000, 0, m)
            c = self.carry if s is None else perturb_member(self.carry, s,
                                                            0.01)
            runs.append(np.asarray(seg(c, jnp.zeros(2), jnp.asarray(1.0),
                                       self.q_base, w)[1]["sst"]))
        np.testing.assert_allclose(ref["mean"]["sst"], np.mean(runs, axis=0),
                                   rtol=0, atol=1e-4)
        self.assertEqual(ref["member_blocks"]["sst"].shape, (3, 2, IX, IL))
        np.testing.assert_allclose(ref["member_blocks"]["sst"][1, 0],
                                   runs[1][:3].mean(axis=0), atol=1e-4)
        with self.assertRaises(ValueError):
            reference_ensemble(self.carry, seg, days, 2, w, 1, 93000, 0,
                               block_days=4)


class LookaheadObjectiveTest(_Fixture):
    DAYS = 8

    def setUp(self):
        super().setUp()
        self.w_ocean = area_weights(LATS, OCEAN)
        self.warm = make_warming(self.carry, self.pattern, step_wm2=3.0)
        seg = make_segment_fn(_toy_step, self.patterns, self.DAYS,
                              fields_fn=_toy_fields)
        _, normal = seg(self.carry, jnp.zeros(2), jnp.asarray(1.0),
                        self.q_base, make_warming(self.carry, self.pattern))
        self.target = jnp.mean(normal["sst"], axis=0)     # normal climate
        self.prev = jnp.zeros(2)

    def _J(self, mu=0.0, lam=0.0):
        return make_lookahead_objective(_toy_step, self.patterns,
                                        self.w_ocean, self.DAYS, mu=mu,
                                        lam=lam)

    def _args(self, a, window):
        return (a, self.carry, self.target, self.prev, jnp.asarray(1.0),
                self.q_base, self.warm, jnp.asarray(window))

    def test_unwarmed_uncontrolled_run_scores_zero(self):
        cold = make_warming(self.carry, self.pattern)
        j = self._J()(jnp.zeros(2), self.carry, self.target, self.prev,
                      jnp.asarray(1.0), self.q_base, cold,
                      jnp.asarray(NO_TRUNCATION_DAYS))
        self.assertAlmostEqual(float(j), 0.0, places=8)

    def test_forward_value_does_not_depend_on_the_window(self):
        a = jnp.array([0.02, 0.01])
        vals = [float(jax.jit(self._J())(*self._args(a, w)))
                for w in (1, 3, NO_TRUNCATION_DAYS)]
        self.assertEqual(vals[0], vals[1])
        self.assertEqual(vals[0], vals[2])

    def test_full_gradient_matches_finite_differences_on_linear_toy(self):
        a, h = jnp.array([0.02, 0.01]), 1e-2
        J = jax.jit(self._J(mu=0.5, lam=0.2))
        grad = np.asarray(jax.grad(lambda x: J(*self._args(
            x, NO_TRUNCATION_DAYS)))(a))
        fd = []
        for k in range(2):
            e = jnp.zeros(2).at[k].set(h)
            fd.append((float(J(*self._args(a + e, NO_TRUNCATION_DAYS)))
                       - float(J(*self._args(a - e, NO_TRUNCATION_DAYS))))
                      / (2 * h))
        np.testing.assert_allclose(grad, fd, rtol=2e-2, atol=1e-6)

    def test_truncation_semantics(self):
        a = jnp.array([0.02, 0.01])
        J = self._J()

        def g(window):
            return np.asarray(jax.grad(lambda x: J(*self._args(x, window)))(a))

        np.testing.assert_allclose(g(self.DAYS + 1), g(NO_TRUNCATION_DAYS),
                                   rtol=1e-6)
        self.assertFalse(np.allclose(g(1), g(NO_TRUNCATION_DAYS)))


class HelpersTest(unittest.TestCase):
    def test_index_anomalies_and_time_means(self):
        w = stack_objective_weights(objective_weights(LATS, OCEAN, LAND))
        sst = jnp.full((IX, IL), 291.0)
        land = jnp.full((IX, IL), 281.0)
        self.assertTrue(np.allclose(index_anomalies(sst, land, sst, land, w),
                                    0.0))
        an = np.asarray(index_anomalies(sst + 0.5, land, sst, land, w))
        np.testing.assert_allclose(an, np.asarray(
            objective_values(sst + 0.5, land, w) - objective_values(sst, land,
                                                                   w)))
        self.assertAlmostEqual(float(an[0]), 0.5, places=5)
        fields = {"sst": np.arange(4.0)[:, None, None] * np.ones((4, IX, IL))}
        np.testing.assert_allclose(time_means(fields, 1, 3)["sst"], 1.5)
        with self.assertRaises(ValueError):
            time_means(fields, 2, 5)

    def test_domain_weights(self):
        d = domain_weights(LATS, OCEAN, LAND)
        for name in ("ocean", "land", "global"):
            self.assertAlmostEqual(float(jnp.sum(d[name])), 1.0, places=6)
        self.assertEqual(float(jnp.sum(d["ocean"][0])), 0.0)


def _drop_modules_that_hold_jcm():
    """Forget cached modules that refer to jcm classes (see conftest).

    The root conftest deletes the jcm modules after every test, but jem and
    the run_* scripts keep the classes they imported first, so a real-model
    setup re-imports all of them.
    """
    import sys
    for key in list(sys.modules):
        root = key.split(".")[0]
        if root in ("jcm", "jem") or (root.startswith("run_")
                                       and not root.endswith("_test")):
            del sys.modules[key]


@pytest.mark.slow
class RealModelTestWorldTest(unittest.TestCase):
    """The real coupled model with the corrected Q-flux (CPU; minutes)."""

    DAYS = 3

    @classmethod
    def setUpClass(cls):
        _drop_modules_that_hold_jcm()
        import jax_datetime as jdt

        from jcm.mcb.band_basis import gaussian_band_patterns as bands
        from jcm.mcb.coupled_controller import create_coupled_step_fn
        from jcm.mcb.coupled_train import ocean_mask_from_coupler
        from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS as FULL
        from jcm.mcb.qflux import load_qflux, set_qflux
        from jcm.mcb.scores import area_weights as weights
        from jcm.mcb import test_world as tw
        from run_coupled_training import coupler_workflow, setup_coupled_model

        coupler, coords, _, _ = setup_coupled_model(
            jdt.to_datetime("2000-01-01"), jdt.to_timedelta(1, "day"),
            realistic_terrain=True)
        carry = coupler.initialize()
        carry = set_qflux(carry, load_qflux(
            "mcb_experiments/qflux/qflux_monthly_t30_v2.nc"))
        step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                         jitted=True)
        # A cold start has a dry, resting atmosphere: rain spins up over about
        # two weeks (0 mm/day on day 1, about 2.4 by day 15).
        for s in range(15):
            carry, _ = step_fn(carry, s)
        lats = coords.horizontal.latitudes
        ocean = ocean_mask_from_coupler(coupler)
        patterns = bands(lats, ocean)
        cls.ocean = np.asarray(ocean)
        q_base = carry["ocn"]["forcing"].q_flux
        seg = tw.make_segment_fn(step_fn, patterns, cls.DAYS)
        zero, one = jnp.zeros(5), jnp.asarray(1.0)
        cold = tw.make_warming(carry, ocean)
        hot = tw.make_warming(carry, ocean, step_wm2=20.0)
        _, cls.normal = seg(carry, zero, one, q_base, cold)
        _, cls.warmed = seg(carry, zero, one, q_base, hot)
        # The same days without the warming wrapper at all.
        controlled = tw.apply_band_control(carry, zero, patterns)
        plain = controlled
        for s in range(cls.DAYS):
            plain, _ = step_fn(plain, s)
        cls.plain_sst = np.asarray(plain["ocn"]["state"]
                                   .sea_surface_temperature)
        w_ocean = weights(lats, ocean)
        cls.w_ocean = np.asarray(w_ocean)
        cls.w_global = np.asarray(weights(lats, jnp.ones_like(ocean)))
        target = jnp.mean(cls.normal["sst"], axis=0)
        objective = tw.make_lookahead_objective(step_fn, patterns, w_ocean,
                                                cls.DAYS)

        def j(a, window):
            return objective(a, carry, target, zero, one, q_base, hot,
                             jnp.asarray(window))

        a = jnp.full((5,), 0.03)
        cls.values = [float(j(a, w)) for w in (1, FULL)]
        cls.grads = {w: np.asarray(jax.grad(j)(a, w)) for w in (1, FULL)}

    def test_zero_warming_matches_the_unwrapped_model(self):
        np.testing.assert_array_equal(np.asarray(self.normal["sst"][-1]),
                                      self.plain_sst)

    def test_warming_heats_the_ocean_by_about_f_t_over_c(self):
        dsst = np.asarray(self.warmed["sst"][-1] - self.normal["sst"][-1])
        rise = float(np.sum(self.w_ocean * dsst))
        # A 50 m slab: C = 1025 * 3985 * 50 J m-2 K-1; F t / C for 3 days.
        expected = 20.0 * self.DAYS * 86400.0 / (1025.0 * 3985.0 * 50.0)
        self.assertGreater(rise, 0.5 * expected)
        self.assertLess(rise, 1.5 * expected)
        np.testing.assert_array_equal(dsst[self.ocean == 0], 0.0)

    def test_rainfall_and_evaporation_are_plausible(self):
        # Units check: the settled Step-0 climate rains 2.9-3.5 mm/day; this
        # spun-up cold start about 2.4.
        means = {}
        for name in ("precipitation", "evaporation"):
            field = np.asarray(self.normal[name])
            self.assertTrue(np.all(np.isfinite(field)), msg=name)
            means[name] = float(np.sum(self.w_global * field.mean(axis=0)))
            self.assertTrue(1.5 < means[name] < 5.0,
                            msg=f"{name} {means[name]} mm/day")
        # Water balance: global rainfall about equals evaporation.
        self.assertLess(abs(means["precipitation"] - means["evaporation"])
                        / means["evaporation"], 0.3, msg=str(means))

    def test_lookahead_objective_and_its_gradients(self):
        self.assertEqual(self.values[0], self.values[1])
        self.assertGreater(self.values[0], 0.0)
        for w, g in self.grads.items():
            self.assertTrue(np.all(np.isfinite(g)), msg=f"window {w}")
        # Brightening cools a warmed ocean, so it lowers the objective.
        self.assertTrue(np.all(self.grads[1] < 0.0), msg=str(self.grads[1]))


if __name__ == "__main__":
    unittest.main()
