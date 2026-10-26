"""Tests for the block-schedule planner of Experiment 3f (jcm.mcb.planner).

On the linear toy of ``jcm.mcb.test_world_test`` every block's SST map is
exactly linear in the settings, so one Gauss-Newton step reaches the
schedule objective's minimum, and a planner that is told nothing about the
strength learns it from its first two-day forecast.
"""

import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.planner import LearningPlanner, Planner, PlannerConfig
from jcm.mcb.scores import area_weights
from jcm.mcb.strength_estimator import EstimatorConfig
from jcm.mcb.test_world import (
    EpisodeState,
    make_lookahead_sst_fn,
    make_schedule_sst_fn,
    make_segment_fn,
    make_warming,
    run_episode,
)
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

L, B = 6, 2          # look-ahead and block (= segment) length, in toy days


class _Toy(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.w = area_weights(LATS, OCEAN)
        self.carry = _carry()
        self.q_base = self.carry["ocn"]["forcing"].q_flux
        # A growing warming, as in the test world, so the best schedule has
        # a trend.
        self.warm = make_warming(self.carry, OCEAN, step_wm2=1.0,
                                 ramp_wm2_per_day=0.5)
        seg = make_segment_fn(_toy_step, self.patterns, 3 * L,
                              fields_fn=_toy_fields)
        _, normal = seg(self.carry, jnp.zeros(2), jnp.asarray(1.0),
                        self.q_base, make_warming(self.carry, OCEAN))
        self.target = np.asarray(normal["sst"])

    def cfg(self, **overrides):
        cfg = dict(lookahead_days=L, window_days=NO_TRUNCATION_DAYS,
                   copies=1, copy_amp=0.0, cap=5.0, gn_damping=0.0,
                   optimizer="gauss_newton", iterations=1, block_days=B)
        cfg.update(overrides)
        return PlannerConfig(**cfg)

    def planner(self, efficacy=1.0, **overrides):
        return Planner(_toy_step, self.patterns, self.w, self.target,
                       self.q_base, self.warm, efficacy, self.cfg(**overrides))

    def state(self, segment=0, day=0, previous=(0.0, 0.0)):
        return EpisodeState(segment=segment, day=day, carry=self.carry,
                            previous=np.asarray(previous, np.float32),
                            last_fields=None)

    def blocks(self, level, trend, n=L // B):
        f = make_schedule_sst_fn(_toy_step, self.patterns, n, B, cap=5.0)
        return np.asarray(f(jnp.asarray(level, jnp.float32),
                            jnp.asarray(trend, jnp.float32), self.carry,
                            jnp.asarray(1.0), self.q_base, self.warm,
                            jnp.asarray(NO_TRUNCATION_DAYS)))


class ScheduleForecastTest(_Toy):
    def test_held_blocks_average_to_the_lookahead_mean(self):
        a = [0.7, 1.3]
        held = self.blocks(a, [0.0, 0.0]).mean(axis=0)
        f = make_lookahead_sst_fn(_toy_step, self.patterns, L)
        mean = np.asarray(f(jnp.asarray(a, jnp.float32), self.carry,
                            jnp.asarray(1.0), self.q_base, self.warm,
                            jnp.asarray(NO_TRUNCATION_DAYS)))
        np.testing.assert_allclose(held, mean, rtol=1e-5, atol=1e-6)

    def test_block_zero_is_the_next_segment_and_ignores_the_trend(self):
        a = [0.7, 1.3]
        flat, sloped = self.blocks(a, [0, 0]), self.blocks(a, [0.4, -0.2])
        np.testing.assert_allclose(flat[0], sloped[0], rtol=1e-6)
        self.assertGreater(np.abs(flat[-1] - sloped[-1]).max(), 1e-4)
        seg = make_segment_fn(_toy_step, self.patterns, B,
                              fields_fn=_toy_fields)
        _, fields = seg(self.carry, jnp.asarray(a, jnp.float32),
                        jnp.asarray(1.0), self.q_base, self.warm)
        from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
        np.testing.assert_allclose(
            flat[0], np.asarray(fields["sst"]).mean(axis=0) - MAP_REFERENCE_K,
            rtol=1e-5, atol=1e-6)


class SchedulePlannerTest(_Toy):
    def test_hold_plans_exactly_as_the_original_planner(self):
        for zonal in (False, True):
            rep = "zonal" if zonal else "map"
            kw = dict(mu=1e-3, lam=1e-3, representation=rep)
            prev = (0.3, 0.1)
            old = self.planner(**kw)(self.state(1, 0, prev))
            new = self.planner(schedule="hold", **kw)(self.state(1, 0, prev))
            np.testing.assert_allclose(new, old, rtol=1e-4, atol=1e-6)

    def test_one_linear_step_reaches_the_schedule_optimum(self):
        p = self.planner(schedule="linear", mu=1e-3)
        first = p.plan(self.state())
        trend = p.trend.copy()
        again = p.plan(self.state(1, 0, first))  # warm start at the optimum
        np.testing.assert_allclose(again, first, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(p.trend, trend, rtol=1e-3, atol=1e-5)
        self.assertGreater(np.abs(trend).max(), 1e-3)   # it uses the trend
        self.assertLess(p.log[1]["objective"][0], p.log[0]["objective"][0])
        self.assertEqual(p.log[0]["schedule"], "linear")

    def test_planning_the_trend_beats_holding_one_setting(self):
        # Scored block by block, a held setting is the trend-0 special case
        # of the linear plan, so it cannot fit the blocks better.
        a_hold = self.planner(schedule="hold").plan(self.state())
        lin = self.planner(schedule="linear")
        a_lin = lin.plan(self.state())
        trend_lin = lin.trend.copy()
        lin.trend = np.zeros(2)
        lin.plan(self.state(1, 0, a_hold))        # objective at the held plan
        at_hold = lin.log[-1]["objective"][0]
        lin.trend = trend_lin
        lin.plan(self.state(1, 0, a_lin))         # objective at its own plan
        self.assertLess(lin.log[-1]["objective"][0], at_hold)

    def test_rejects_settings_it_cannot_use(self):
        for bad in (dict(schedule="linear", optimizer="adam"),
                    dict(schedule="linear", sensing="ocean"),
                    dict(schedule="linear", lookahead_days=5),
                    dict(schedule="sideways")):
            with self.assertRaises(ValueError):
                self.cfg(**bad).validate()


class ScheduleLearningTest(_Toy):
    TRUTH = np.array([0.6, 1.8], np.float32)

    def learner(self, schedule):
        return LearningPlanner(_toy_step, self.patterns, self.w, self.target,
                               self.q_base, self.warm,
                               EstimatorConfig(noise_k=1e-4, prior_sd=10.0),
                               self.cfg(schedule=schedule))

    def episode(self, policy):
        seg = make_segment_fn(_toy_step, self.patterns, B,
                              fields_fn=_toy_fields)
        return run_episode(self.carry, seg, policy, n_segments=3,
                           segment_days=B, k_bands=2, warming=self.warm,
                           efficacy=self.TRUTH)

    def test_long_lookahead_planners_learn_from_each_segment(self):
        for schedule in ("hold", "linear"):
            learner = self.learner(schedule)
            ep = self.episode(learner)
            self.assertEqual(ep.amplitudes.shape, (3, 2))
            np.testing.assert_allclose(learner.log[0]["efficacy_belief"],
                                       [1, 1])
            np.testing.assert_allclose(learner.log[1]["efficacy_belief"],
                                       self.TRUTH, rtol=1e-2)
            self.assertEqual(len(learner.innovation_log), 2)
            self.assertLess(learner.log[2]["innovation_ms"], 1e-8)


class ExtendedTargetTest(unittest.TestCase):
    def test_own_days_then_the_longer_files_days(self):
        from run_test_world import extended_target
        own = np.random.default_rng(0).normal(size=(20, 3, 2))
        longer = own.copy()
        longer = np.concatenate([longer + 0.01,
                                 np.full((10, 3, 2), 7.0)], axis=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ext.npz"
            np.savez(path, normal_sst=longer)
            target, rec = extended_target({"normal_sst": own}, str(path), 26)
        self.assertEqual(target.shape, (26, 3, 2))
        np.testing.assert_array_equal(target[:20], own)
        np.testing.assert_array_equal(target[20:], 7.0)
        self.assertEqual(rec["own_days"], 20)
        self.assertEqual(rec["extension_days"], 6)
        self.assertAlmostEqual(rec["overlap_mean_abs_k"], 0.01, places=6)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ext.npz"
            np.savez(path, normal_sst=longer)
            with self.assertRaises(SystemExit):
                extended_target({"normal_sst": own}, str(path), 40)


if __name__ == "__main__":
    unittest.main()
