"""Tests for the receding-horizon planner (jcm.mcb.planner).

On the linear toy of ``jcm.mcb.test_world_test`` the look-ahead SST map is
exactly linear in the band settings, so the objective is exactly quadratic
and its minimizer is known in closed form (Part 18 step 16: "check that it
finds the known best answer on a toy model").
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.planner import (
    PRESETS,
    W_STAR_DAYS,
    LearningPlanner,
    Planner,
    PlannerConfig,
    gauss_newton_residuals,
    noise_to_signal,
    to_amplitudes,
    to_logits,
)
from jcm.mcb.scores import (
    area_weights,
    pattern_objective,
    segment_objective,
    zonal_projection,
)
from jcm.mcb.strength_estimator import EstimatorConfig
from jcm.mcb.test_world import (
    EpisodeState,
    make_lookahead_sst_fn,
    make_segment_fn,
    make_warming,
    run_episode,
)
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

L = 6


class _Toy(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.w = area_weights(LATS, OCEAN)
        self.carry = _carry()
        self.q_base = self.carry["ocn"]["forcing"].q_flux
        self.warm = make_warming(self.carry, OCEAN, step_wm2=3.0)
        seg = make_segment_fn(_toy_step, self.patterns, 3 * L,
                              fields_fn=_toy_fields)
        _, normal = seg(self.carry, jnp.zeros(2), jnp.asarray(1.0),
                        self.q_base, make_warming(self.carry, OCEAN))
        self.target = np.asarray(normal["sst"])          # normal climate

    def planner(self, **overrides):
        # The toy's optimum is about 2.1 per band, so its cap is 5.
        cfg = dict(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                   copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0)
        cfg.update(overrides)
        return Planner(_toy_step, self.patterns, self.w, self.target,
                       self.q_base, self.warm, 1.0, PlannerConfig(**cfg))

    def state(self, segment=0, day=0, previous=(0.0, 0.0)):
        return EpisodeState(segment=segment, day=day, carry=self.carry,
                            previous=np.asarray(previous, np.float32),
                            last_fields=None)

    def known_optimum(self, mu=0.0, lam=0.0, previous=(0.0, 0.0),
                      zonal=False):
        """Exact minimizer: the toy's look-ahead map is linear in a."""
        import jax
        f = make_lookahead_sst_fn(_toy_step, self.patterns, L)

        def mean_map(a):
            return f(a, self.carry, jnp.asarray(1.0), self.q_base, self.warm,
                     jnp.asarray(NO_TRUNCATION_DAYS))

        a0 = jnp.zeros(2)
        jac = np.asarray(jax.jacfwd(mean_map)(a0))
        target = self.target[:L].mean(axis=0) - MAP_REFERENCE_K
        error = np.asarray(mean_map(a0), np.float64) - target
        if zonal:
            error = zonal_projection(error, OCEAN)
            jac = np.stack([zonal_projection(np.asarray(jac[..., k],
                                                        np.float64), OCEAN)
                            for k in range(2)], axis=-1)
        r, big_r = gauss_newton_residuals(error, jac, self.w, np.zeros(2),
                                          previous, 1.0, 0.5, mu, lam)
        return np.linalg.lstsq(big_r, -r, rcond=None)[0]


class KnownOptimumTest(_Toy):
    def test_the_toy_optimum_is_interior(self):
        a_star = self.known_optimum()
        self.assertTrue(np.all((a_star > 0.0) & (a_star < 5.0)), a_star)

    def test_one_gauss_newton_step_lands_on_it(self):
        planner = self.planner(optimizer="gauss_newton", iterations=1)
        a = planner(self.state())
        np.testing.assert_allclose(a, self.known_optimum(), rtol=1e-3,
                                   atol=1e-5)

    def test_gauss_newton_with_penalties_matches_the_penalized_optimum(self):
        prev = (0.2, 0.1)
        planner = self.planner(optimizer="gauss_newton", iterations=2,
                               mu=1e-4, lam=1e-4)
        a = planner(self.state(segment=1, previous=prev))
        np.testing.assert_allclose(a, self.known_optimum(1e-4, 1e-4, prev),
                                   rtol=1e-3, atol=1e-5)

    def test_zonal_gauss_newton_lands_on_the_zonal_optimum(self):
        # The toy is the same at every longitude, so here the zonal optimum
        # equals the map optimum; HelpersTest checks the projection itself.
        a_zonal = self.known_optimum(zonal=True)
        planner = self.planner(optimizer="gauss_newton", iterations=1,
                               representation="zonal")
        np.testing.assert_allclose(planner(self.state()), a_zonal,
                                   rtol=1e-3, atol=1e-5)

    def test_adam_converges_to_it(self):
        planner = self.planner(optimizer="adam", iterations=300,
                               learning_rate=0.05)
        a = planner(self.state())
        np.testing.assert_allclose(a, self.known_optimum(), rtol=0.02,
                                   atol=1e-3)
        values = planner.log[0]["objective"]
        self.assertLess(values[-1], values[0])


class SafeguardTest(_Toy):
    def test_bands_stay_inside_the_cap(self):
        for opt in ("adam", "gauss_newton"):
            planner = self.planner(optimizer=opt, iterations=5, cap=0.01)
            a = planner(self.state())
            self.assertTrue(np.all((a >= 0.0) & (a <= 0.01)), msg=f"{opt} {a}")

    def test_noise_to_signal_is_logged(self):
        same = self.planner(copies=3, copy_amp=0.0, iterations=1)
        same(self.state())
        self.assertEqual(same.log[0]["noise_to_signal"], 0.0)
        alone = self.planner(copies=1, iterations=1)
        alone(self.state())
        self.assertIsNone(alone.log[0]["noise_to_signal"])
        nudged = self.planner(copies=3, copy_amp=0.5, iterations=1,
                              optimizer="gauss_newton")
        nudged(self.state())
        self.assertGreater(nudged.log[0]["noise_to_signal"], 0.0)

    def test_last_iterate_is_applied_and_logged(self):
        planner = self.planner(iterations=4)
        a = planner(self.state())
        record = planner.log[0]
        np.testing.assert_allclose(record["final"], a)
        self.assertEqual(len(record["objective"]), 4)
        self.assertEqual((record["segment"], record["day"]), (0, 0))

    def test_window_reaches_the_gradient(self):
        # Gauss-Newton steps use the response's size, not just its sign
        # (Adam's first steps do), so the window must change the answer.
        a_full = self.planner(optimizer="gauss_newton")(self.state())
        a_cut = self.planner(optimizer="gauss_newton",
                             window_days=1)(self.state())
        self.assertFalse(np.allclose(a_full, a_cut))

    def test_lookahead_must_fit_the_target(self):
        planner = self.planner()
        with self.assertRaises(ValueError):
            planner.target_mean(3 * L - 2)


class SideBySideTest(_Toy):
    """Copies run side by side (one vmapped call) give the same plan."""

    def test_batched_copies_match_sequential(self):
        for opt, iters in (("adam", 4), ("gauss_newton", 2)):
            kw = dict(optimizer=opt, iterations=iters, copies=3,
                      copy_amp=0.05)
            seq, bat = self.planner(**kw), self.planner(batch_copies=True,
                                                        **kw)
            np.testing.assert_allclose(bat(self.state()), seq(self.state()),
                                       rtol=1e-5, atol=1e-7, err_msg=opt)
            np.testing.assert_allclose(bat.log[0]["objective"],
                                       seq.log[0]["objective"], rtol=1e-5,
                                       err_msg=opt)
            self.assertAlmostEqual(bat.log[0]["noise_to_signal"],
                                   seq.log[0]["noise_to_signal"], places=5,
                                   msg=opt)

    def test_evaluate_copies_shapes(self):
        gn = self.planner(optimizer="gauss_newton", copies=3,
                          batch_copies=True)
        jacs, maps = gn.evaluate_copies(np.full(2, 0.5),
                                        gn.prepare_copies(self.carry, 0))
        self.assertEqual(jacs.shape, (3,) + OCEAN.shape + (2,))
        self.assertEqual(maps.shape, (3,) + OCEAN.shape)
        adam = self.planner(optimizer="adam", copies=3)
        vals, grads = adam.evaluate_copies(np.zeros(2),
                                           adam.prepare_copies(self.carry, 0),
                                           adam.target_mean(0), np.zeros(2))
        self.assertEqual((vals.shape, grads.shape), ((3,), (3, 2)))


class PolicyTest(_Toy):
    def test_planner_drives_an_episode(self):
        planner = self.planner(iterations=2)
        seg = make_segment_fn(_toy_step, self.patterns, 2,
                              fields_fn=_toy_fields)
        ep = run_episode(self.carry, seg, planner, n_segments=3,
                         segment_days=2, k_bands=2, warming=self.warm)
        self.assertEqual(ep.amplitudes.shape, (3, 2))
        self.assertEqual([r["day"] for r in planner.log], [0, 2, 4])
        np.testing.assert_allclose(planner.log[1]["start"],
                                   ep.amplitudes[0])


class StrengthLearningTest(_Toy):
    """Experiment 3b: forecasts are checked, and the strength is learned."""

    TRUTH = np.array([0.6, 1.8], np.float32)

    def make(self, efficacy=1.0, **overrides):
        cfg = dict(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                   copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0,
                   optimizer="gauss_newton", iterations=1)
        cfg.update(overrides)
        return Planner(_toy_step, self.patterns, self.w, self.target,
                       self.q_base, self.warm, efficacy, PlannerConfig(**cfg))

    def learner(self, **overrides):
        cfg = dict(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                   copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0,
                   optimizer="gauss_newton", iterations=1)
        cfg.update(overrides)
        return LearningPlanner(_toy_step, self.patterns, self.w, self.target,
                               self.q_base, self.warm,
                               EstimatorConfig(noise_k=1e-4, prior_sd=10.0),
                               PlannerConfig(**cfg))

    def episode(self, policy, truth, segment_days=L, n_segments=3):
        seg = make_segment_fn(_toy_step, self.patterns, segment_days,
                              fields_fn=_toy_fields)
        return run_episode(self.carry, seg, policy, n_segments=n_segments,
                           segment_days=segment_days, k_bands=2,
                           warming=self.warm, efficacy=truth)

    def test_belief_scales_the_plan(self):
        # The toy responds to strength x setting, so believing a band is
        # twice as strong halves its setting at the (interior) optimum.
        base = self.make()(self.state())
        belief = np.array([2.0, 0.5], np.float32)
        np.testing.assert_allclose(self.make(belief)(self.state()),
                                   base / belief, rtol=1e-3)
        changed = self.make()
        changed.efficacy = belief                 # no rebuild needed
        np.testing.assert_allclose(changed(self.state()), base / belief,
                                   rtol=1e-3)

    def test_forecasts_are_checked_against_what_happened(self):
        oracle = self.make(self.TRUTH)
        self.episode(oracle, self.TRUTH)
        misses = [r.get("innovation_ms") for r in oracle.log]
        self.assertIsNone(misses[0])              # nothing to check yet
        self.assertLess(max(misses[1:]), 1e-8)    # exact on the linear toy
        naive = self.make(1.0)
        self.episode(naive, self.TRUTH)
        self.assertGreater(naive.log[1]["innovation_ms"], 1e-6)

    def test_learning_planner_recovers_the_hidden_strength(self):
        learner = self.learner()
        self.episode(learner, self.TRUTH)
        np.testing.assert_allclose(learner.log[0]["efficacy_belief"], [1, 1])
        np.testing.assert_allclose(learner.log[1]["efficacy_belief"],
                                   self.TRUTH, rtol=1e-2)
        # Once learned, it plans as the oracle does from the same state.
        oracle = self.make(self.TRUTH)
        np.testing.assert_allclose(learner.plan(self.state()),
                                   oracle(self.state()), rtol=2e-2)

    def test_first_move_does_not_depend_on_the_hidden_strength(self):
        a = self.learner()
        b = self.learner()
        self.episode(a, self.TRUTH)
        self.episode(b, np.array([1.5, 0.4], np.float32))
        np.testing.assert_allclose(a.log[0]["final"], b.log[0]["final"])
        self.assertFalse(np.allclose(a.log[1]["efficacy_belief"],
                                     b.log[1]["efficacy_belief"]))

    def test_learning_needs_gauss_newton_and_matching_segments(self):
        with self.assertRaises(ValueError):
            self.learner(optimizer="adam")
        with self.assertRaises(ValueError):
            self.episode(self.learner(), self.TRUTH, segment_days=2,
                         n_segments=3)


class OceanSensingTest(_Toy):
    """Experiment 3c: the planner sees the ocean but not the weather."""

    def make(self, sensing="ocean", background=None, **overrides):
        cfg = dict(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                   copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0,
                   optimizer="gauss_newton", iterations=1, sensing=sensing)
        cfg.update(overrides)
        return Planner(_toy_step, self.patterns, self.w, self.target,
                       self.q_base, self.warm, 1.0, PlannerConfig(**cfg),
                       background=background)

    def weather(self, x):
        """The toy state with another atmosphere (its weather)."""
        carry = dict(self.carry)
        carry["atm"] = {"state": {"x": jnp.full_like(
            self.carry["atm"]["state"]["x"], x)},
            "derived": self.carry["atm"]["derived"]}
        return carry

    def at(self, carry):
        return EpisodeState(segment=0, day=0, carry=carry,
                            previous=np.zeros(2, np.float32),
                            last_fields=None)

    def test_with_the_true_weather_it_plans_as_the_exact_planner(self):
        np.testing.assert_allclose(
            self.make(background=self.carry)(self.state()),
            self.make(sensing="exact")(self.state()), rtol=1e-6)

    def test_it_cannot_see_the_true_weather(self):
        calm, stormy = self.weather(0.1), self.weather(3.0)
        exact = self.make(sensing="exact")
        self.assertFalse(np.allclose(exact(self.at(calm)),
                                     exact(self.at(stormy)), rtol=1e-3))
        guess = self.weather(-1.0)
        np.testing.assert_allclose(
            self.make(background=guess)(self.at(calm)),
            self.make(background=guess)(self.at(stormy)), rtol=1e-6)

    def test_it_sees_the_true_ocean(self):
        warm = dict(self.carry)
        warm["ocn"] = dict(self.carry["ocn"])
        warm["ocn"]["state"] = self.carry["ocn"]["state"].copy(
            {"sea_surface_temperature":
             self.carry["ocn"]["state"].sea_surface_temperature + 0.5})
        planner = self.make(background=self.carry)
        self.assertFalse(np.allclose(planner(self.state()),
                                     self.make(background=self.carry)(
                                         self.at(warm)), rtol=1e-3))

    def test_the_next_forecast_starts_where_this_one_ended(self):
        guess = self.weather(-1.0)
        planner = self.make(background=guess)
        planner(self.state())
        f = make_lookahead_sst_fn(_toy_step, self.patterns, L,
                                  return_final=True)
        start = jnp.full(2, 0.5 * 5.0)          # the first guess
        estimated = dict(guess)
        estimated["ocn"] = self.carry["ocn"]
        _, final = f(start, estimated, jnp.ones(2), self.q_base, self.warm,
                     jnp.asarray(NO_TRUNCATION_DAYS))
        np.testing.assert_allclose(
            np.asarray(planner.background["atm"]["state"]["x"]),
            np.asarray(final["atm"]["state"]["x"]), rtol=1e-6)

    def test_episode_keeps_the_background_in_step(self):
        seg = make_segment_fn(_toy_step, self.patterns, L,
                              fields_fn=_toy_fields)
        planner = self.make(background=self.weather(-1.0))
        run_episode(self.carry, seg, planner, n_segments=3, segment_days=L,
                    k_bands=2, warming=self.warm)
        self.assertEqual(len(planner.log), 3)
        # A segment shorter than the look-ahead leaves the background behind.
        seg2 = make_segment_fn(_toy_step, self.patterns, 2,
                               fields_fn=_toy_fields)
        with self.assertRaises(ValueError):
            run_episode(self.carry, seg2, self.make(background=self.carry),
                        n_segments=2, segment_days=2, k_bands=2,
                        warming=self.warm)

    def test_learner_with_ocean_sensing_still_learns(self):
        truth = np.array([0.6, 1.8], np.float32)
        cfg = PlannerConfig(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                            copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0,
                            optimizer="gauss_newton", iterations=1,
                            sensing="ocean")
        learner = LearningPlanner(_toy_step, self.patterns, self.w,
                                  self.target, self.q_base, self.warm,
                                  EstimatorConfig(noise_k=1e-4, prior_sd=10.0),
                                  cfg, background=self.weather(-1.0))
        seg = make_segment_fn(_toy_step, self.patterns, L,
                              fields_fn=_toy_fields)
        run_episode(self.carry, seg, learner, n_segments=3, segment_days=L,
                    k_bands=2, warming=self.warm, efficacy=truth)
        # The wrong weather costs accuracy, but the belief still moves from
        # nominal toward the truth in both bands.
        belief = np.asarray(learner.log[-1]["efficacy_belief"])
        self.assertTrue(np.all(np.abs(np.log(belief / truth))
                               < np.abs(np.log(1.0 / truth))))

    def test_needs_gauss_newton_and_a_background(self):
        with self.assertRaises(ValueError):
            PlannerConfig(sensing="ocean", optimizer="adam").validate()
        with self.assertRaises(ValueError):
            PlannerConfig(sensing="weather").validate()
        with self.assertRaises(ValueError):
            self.make(background=None)


class HelpersTest(unittest.TestCase):
    def test_transform_round_trip_and_bounds(self):
        a = np.array([0.01, 0.075, 0.14])
        z = to_logits(a, 0.15, 0.02)
        np.testing.assert_allclose(np.asarray(to_amplitudes(jnp.asarray(z),
                                                            0.15)), a,
                                   rtol=1e-5)
        edge = np.asarray(to_amplitudes(jnp.asarray(to_logits(
            [0.0, 0.15], 0.15, 0.02)), 0.15))
        np.testing.assert_allclose(edge, [0.003, 0.147], rtol=1e-4)

    def test_residuals_reproduce_the_objective(self):
        rng = np.random.default_rng(0)
        w = np.asarray(area_weights(LATS, OCEAN))
        e = rng.normal(0.2, 0.3, OCEAN.shape)
        a, prev = np.array([0.03, 0.05]), np.array([0.01, 0.07])
        r, big_r = gauss_newton_residuals(e, np.zeros(OCEAN.shape + (2,)), w,
                                          a, prev, 1.0, 0.5, 0.3, 0.2)
        j = segment_objective(jnp.asarray(e), jnp.asarray(w), a, prev, 1.0,
                              0.5, 0.3, 0.2)
        self.assertAlmostEqual(float(r @ r), float(j), places=6)
        self.assertEqual(big_r.shape, (r.size, 2))

    def test_zonal_projection(self):
        rng = np.random.default_rng(1)
        w = np.asarray(area_weights(LATS, OCEAN))
        x = rng.normal(size=OCEAN.shape)
        z = zonal_projection(x, OCEAN)
        self.assertIsInstance(z, np.ndarray)
        for j in range(OCEAN.shape[1]):
            vals = z[OCEAN[:, j] > 0, j]
            if vals.size:
                np.testing.assert_allclose(vals, vals[0])
        np.testing.assert_array_equal(z[OCEAN == 0], 0.0)
        self.assertAlmostEqual(float(np.sum(w * z)), float(np.sum(w * x)))
        np.testing.assert_allclose(zonal_projection(z, OCEAN), z)
        np.testing.assert_allclose(
            np.asarray(zonal_projection(jnp.asarray(x, jnp.float32),
                                        OCEAN)), z, rtol=1e-5, atol=1e-6)
        # Zonal residuals reproduce the objective of the projected error.
        r, _ = gauss_newton_residuals(z, np.zeros(OCEAN.shape + (2,)), w,
                                      np.zeros(2), np.zeros(2), 1.0, 0.5,
                                      0.0, 0.0)
        self.assertAlmostEqual(float(r @ r), float(pattern_objective(z, w)))
        # The zonal objective never exceeds the map objective.
        self.assertLessEqual(float(pattern_objective(z, w)),
                             float(pattern_objective(x, w)) + 1e-12)

    def test_noise_to_signal(self):
        self.assertIsNone(noise_to_signal([np.ones(3)]))
        self.assertEqual(noise_to_signal([np.ones(3), np.ones(3)]), 0.0)
        g = [np.array([1.0, 0.0]), np.array([3.0, 0.0])]
        self.assertAlmostEqual(noise_to_signal(g), np.sqrt(2.0) / 2.0)

    def test_presets_and_validation(self):
        self.assertEqual(W_STAR_DAYS, 14)
        snip, short, bptt = (PRESETS[k] for k in ("snipped60", "short14",
                                                  "bptt60"))
        self.assertEqual((snip.lookahead_days, snip.window_days), (60, 14))
        self.assertEqual((short.lookahead_days, short.window_days),
                         (14, NO_TRUNCATION_DAYS))
        self.assertEqual((bptt.lookahead_days, bptt.window_days),
                         (60, NO_TRUNCATION_DAYS))
        for bad in ({"copies": 0}, {"optimizer": "sgd"}, {"cap": 0.0},
                    {"first_guess": 1.0}, {"edge_margin": 0.6},
                    {"mu": -1.0}, {"lookahead_days": 0},
                    {"representation": "bands"},
                    {"representation": "zonal", "optimizer": "adam"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                PlannerConfig(**bad).validate()


@pytest.mark.slow
class RealModelPlannerTest(unittest.TestCase):
    """One re-plan on the real coupled model (CPU; compiles: minutes)."""

    @classmethod
    def setUpClass(cls):
        from jcm.mcb.test_world_test import _drop_modules_that_hold_jcm
        _drop_modules_that_hold_jcm()
        import jax_datetime as jdt

        from jcm.mcb import planner as pl
        from jcm.mcb import test_world as tw
        from jcm.mcb.band_basis import gaussian_band_patterns as bands
        from jcm.mcb.coupled_controller import create_coupled_step_fn
        from jcm.mcb.coupled_train import ocean_mask_from_coupler
        from jcm.mcb.qflux import load_qflux, set_qflux
        from jcm.mcb.scores import area_weights as weights
        from run_coupled_training import coupler_workflow, setup_coupled_model

        coupler, coords, _, _ = setup_coupled_model(
            jdt.to_datetime("2000-01-01"), jdt.to_timedelta(1, "day"),
            realistic_terrain=True)
        carry = set_qflux(coupler.initialize(), load_qflux(
            "mcb_experiments/qflux/qflux_monthly_t30_v2.nc"))
        step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                         jitted=True)
        lats = coords.horizontal.latitudes
        ocean = ocean_mask_from_coupler(coupler)
        patterns = bands(lats, ocean)
        q_base = carry["ocn"]["forcing"].q_flux
        seg = tw.make_segment_fn(step_fn, patterns, 3)
        _, normal = seg(carry, jnp.zeros(5), jnp.asarray(1.0), q_base,
                        tw.make_warming(carry, ocean))
        warm = tw.make_warming(carry, ocean, step_wm2=20.0)
        state = tw.EpisodeState(segment=0, day=0, carry=carry,
                                previous=np.zeros(5, np.float32),
                                last_fields=None)
        cls.results = {}
        for opt, iters in (("adam", 2), ("gauss_newton", 1)):
            cfg = pl.PlannerConfig(lookahead_days=3, window_days=1,
                                   copies=2, optimizer=opt, iterations=iters)
            planner = pl.Planner(step_fn, patterns, weights(lats, ocean),
                                 np.asarray(normal["sst"]), q_base, warm,
                                 1.0, cfg)
            cls.results[opt] = (planner(state), planner.log[0])

    def test_one_replan_per_optimizer(self):
        for opt, (a, record) in self.results.items():
            self.assertEqual(a.shape, (5,), msg=opt)
            self.assertTrue(np.all((a >= 0.0) & (a <= 0.15)), msg=f"{opt} {a}")
            self.assertTrue(np.all(np.isfinite(record["objective"])), msg=opt)
            self.assertIsNotNone(record["noise_to_signal"], msg=opt)


if __name__ == "__main__":
    unittest.main()
