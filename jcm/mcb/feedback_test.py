"""Tests for the classical feedback controllers (jcm.mcb.feedback).

Most tests close the loop around an "integrator world": daily index anomalies
grow at ``eta * S a + disturbance`` (the plant model of the module), and the
controller sees two-week means. That checks the control laws exactly. One test
runs the GLENS-style controller as a policy on the test world's toy, which
checks the sensing and the episode plumbing.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import (
    gaussian_band_patterns,
    objective_weights,
    stack_objective_weights,
)
from jcm.mcb.feedback import (
    AdaptiveConfig,
    AdaptiveController,
    IndexPIController,
    PIConfig,
    bands_for_rates,
    last_segment_means,
    ocean_indices,
    pi_gains,
    segment_index_means,
    sensitivity_rates,
)
from jcm.mcb.test_world import (
    EpisodeState,
    make_segment_fn,
    make_warming,
    run_episode,
)
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

DAYS = 14
# Index rates per unit band (K/day), shaped like Experiment 1's measurements
# (T0 cools everywhere; T1 and T2 change sign across the bands).
S = np.array([[-0.0033, -0.0099, -0.0147, -0.0109, -0.0096],
              [-0.0030, -0.0040, -0.0010, +0.0040, +0.0080],
              [+0.0000, +0.0025, +0.0060, +0.0030, -0.0025]])


def _pi(**kw):
    return IndexPIController(S, np.zeros((4, 2, 2)), np.zeros((400, 2, 2)),
                             PIConfig(**kw))


def _run_pi(ctrl, eta, disturbance, n_seg, feedforward=False, noise=0.0,
            seed=0):
    """Close the loop in the integrator world; return daily errors and settings."""
    rng = np.random.default_rng(seed)
    e = np.zeros(len(ctrl.rows))
    s_rows = S[ctrl.rows]
    errors, settings, last = [], [], None
    for n in range(n_seg):
        measured = np.zeros_like(e) if last is None else last + \
            noise * rng.normal(size=e.shape)
        ff = None
        if feedforward:
            ff = np.mean([disturbance(n * DAYS + d)[ctrl.rows]
                          for d in range(DAYS)], axis=0)
        a = ctrl.step(measured, ff)
        settings.append(a)
        seg = []
        for d in range(DAYS):
            e = e + eta * s_rows @ a + disturbance(n * DAYS + d)[ctrl.rows]
            seg.append(e.copy())
        errors += seg
        last = np.mean(seg, axis=0)
    return np.array(errors), np.array(settings)


class PlantModelTest(unittest.TestCase):
    def test_gains(self):
        kp, ki = pi_gains(42.0, 1.0)
        self.assertAlmostEqual(kp, 2.0 / 42.0)
        self.assertAlmostEqual(ki, 1.0 / 42.0 ** 2)
        with self.assertRaises(ValueError):
            pi_gains(0.0)

    def test_sensitivity_rates_recover_a_line(self):
        days = np.arange(60) + 1.0
        series = days[:, None, None] * S[None]
        np.testing.assert_allclose(sensitivity_rates(series), S, atol=1e-15)
        noisy = series + 1e-3 * np.random.default_rng(1).normal(
            size=series.shape)
        np.testing.assert_allclose(sensitivity_rates(noisy, range(10, 60)),
                                   S, atol=2e-5)

    def test_bands_meet_a_feasible_demand_exactly(self):
        a_true = np.array([0.01, 0.03, 0.02, 0.04, 0.05])
        demand = -S @ a_true
        a = bands_for_rates(demand, S, cap=0.15)
        np.testing.assert_allclose(S @ a, -demand, atol=1e-7)
        self.assertLessEqual(np.linalg.norm(a), np.linalg.norm(a_true) + 1e-9)

    def test_bands_stay_in_bounds(self):
        for demand in ([1.0, 0.0, 0.0], [-1.0, 0.5, -0.5]):
            a = bands_for_rates(demand, S, cap=0.15)
            self.assertTrue(np.all(a >= 0.0) and np.all(a <= 0.15))


class IndexPITest(unittest.TestCase):
    def test_integral_action_removes_a_steady_miss(self):
        d0 = np.array([0.002, 0.0005, -0.0003])      # constant warming rates
        ctrl = _pi()
        errors, settings = _run_pi(ctrl, 1.0, lambda t: d0, 16)
        tail = np.abs(errors[-2 * DAYS:]).max(axis=0)
        # Proportional control alone would settle at d0 / kp.
        self.assertTrue(np.all(tail < 0.1 * np.abs(d0) / ctrl.kp))
        self.assertTrue(np.all(settings >= 0.0) and np.all(settings <= 0.15))

    def test_holds_only_the_chosen_indices(self):
        ctrl = _pi(indices=("T0",))
        errors, _ = _run_pi(ctrl, 1.0, lambda t: np.array([0.002, 0, 0]), 16)
        self.assertEqual(errors.shape[1], 1)
        self.assertLess(abs(errors[-DAYS:, 0]).max(), 0.01)

    def test_feedforward_beats_feedback_alone_on_a_growing_warming(self):
        def ramp(t):
            return np.array([2e-5 * t, 0.0, 0.0])
        plain, _ = _run_pi(_pi(), 1.0, ramp, 13)
        ff, _ = _run_pi(_pi(), 1.0, ramp, 13, feedforward=True)
        self.assertLess(np.abs(ff[-4 * DAYS:, 0]).mean(),
                        0.5 * np.abs(plain[-4 * DAYS:, 0]).mean())

    def test_anti_windup_limits_the_overshoot(self):
        def burst(t):
            return np.array([0.02 if t < 84 else 0.0, 0.0, 0.0])
        overshoot = {}
        for flag in (True, False):
            errors, _ = _run_pi(_pi(anti_windup=flag), 1.0, burst, 26)
            overshoot[flag] = -errors[84:, 0].min()
        self.assertLess(overshoot[True], 0.5 * overshoot[False])

    def test_noise_keeps_the_settings_bounded(self):
        _, settings = _run_pi(_pi(), 1.0, lambda t: np.array([0.002, 0, 0]),
                              13, noise=0.03, seed=4)
        self.assertTrue(np.all(settings >= 0.0) and np.all(settings <= 0.15))

    def test_feedforward_needs_a_forecast(self):
        with self.assertRaises(ValueError):
            IndexPIController(S, np.zeros((4, 2, 2)), np.zeros((40, 2, 2)),
                              PIConfig(feedforward=True))
        with self.assertRaises(ValueError):
            PIConfig(indices=("T3",)).validate()


class SensingTest(unittest.TestCase):
    def setUp(self):
        land = 1.0 - OCEAN
        self.ws = np.asarray(stack_objective_weights(
            objective_weights(LATS, OCEAN, land)))
        self.target = 290.0 + np.zeros((3 * DAYS, 4, 3))

    def test_measures_the_last_segment_against_the_same_days(self):
        anomaly = np.random.default_rng(2).normal(size=(4, 3)) * 0.1
        target = self.target + np.arange(3 * DAYS)[:, None, None] * 0.01
        obs = target[DAYS:2 * DAYS] + anomaly
        state = EpisodeState(segment=2, day=2 * DAYS, carry=None,
                             previous=np.zeros(5),
                             last_fields={"sst": obs})
        o, r = last_segment_means(state, target, DAYS)
        np.testing.assert_allclose(o - r, anomaly, atol=1e-9)
        ctrl = IndexPIController(S, self.ws, target, PIConfig())
        np.testing.assert_allclose(ctrl.measure(state),
                                   ocean_indices(anomaly, self.ws), atol=1e-9)
        first = state._replace(segment=0, day=0, last_fields=None)
        self.assertIsNone(last_segment_means(first, target, DAYS))
        np.testing.assert_allclose(ctrl.measure(first), 0.0)

    def test_segment_index_means_of_a_uniform_anomaly(self):
        daily = self.target + 0.25
        means = segment_index_means(daily, self.ws, DAYS, self.target)
        self.assertEqual(means.shape, (3, 3))
        np.testing.assert_allclose(means[:, 0], 0.25, atol=1e-9)
        np.testing.assert_allclose(means[:, 1:], 0.0, atol=1e-6)


class AdaptiveTest(unittest.TestCase):
    """The ported Tier 2 law, with a hidden strength, in the integrator world."""

    def _run(self, eta, n_seg=13, relaxation=0.5):
        land = 1.0 - OCEAN
        ws = np.asarray(stack_objective_weights(objective_weights(LATS, OCEAN,
                                                                  land)))
        days = np.arange(n_seg * DAYS + DAYS)
        warming = 4e-6 * (days + 1.0) ** 2              # a ramp, integrated
        target = 290.0 + np.zeros((days.size, 4, 3))
        forecast = target + warming[:, None, None]
        pattern = np.array([0.2, 0.6, 1.0, 0.8, 0.7])
        ctrl = AdaptiveController(pattern, S[0], ws, target, forecast,
                                  AdaptiveConfig(relaxation=relaxation))
        cooled, anomalies, last = 0.0, [], None
        for n in range(n_seg):
            gain = ctrl.step(n, last)
            seg = []
            for d in range(DAYS):
                cooled += eta * ctrl.rate * gain
                seg.append(warming[n * DAYS + d] - cooled)
            anomalies += seg
            last = float(np.mean(seg))
        return ctrl, np.array(anomalies), warming[:len(anomalies)]

    def test_recovers_the_hidden_strength_and_cancels_the_warming(self):
        for eta in (0.5, 1.0, 2.0):
            ctrl, anomalies, warming = self._run(eta)
            self.assertAlmostEqual(ctrl.eta_hat, eta, delta=0.2 * eta)
            tail = slice(-4 * DAYS, None)
            self.assertLess(np.abs(anomalies[tail]).mean(),
                            0.1 * warming[tail].mean(), msg=f"eta {eta}")

    def test_gains_stay_in_bounds(self):
        ctrl, _, _ = self._run(0.3)
        self.assertTrue(all(0.0 <= g <= 0.15 for g in ctrl.gains))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            AdaptiveConfig(relaxation=0.0).validate()
        with self.assertRaises(ValueError):
            AdaptiveController(-np.ones(5), S[0], np.zeros((4, 2, 2)),
                               np.zeros((28, 2, 2)), np.zeros((28, 2, 2)))


class ToyEpisodeTest(unittest.TestCase):
    """The GLENS-style controller drives the test world's toy as a policy."""

    def test_feedback_reduces_the_ocean_mean_anomaly(self):
        patterns = gaussian_band_patterns(LATS, OCEAN, centers_deg=(30.0,
                                                                   -30.0),
                                          width_deg=15.0)
        k = patterns.shape[0]
        land = 1.0 - OCEAN
        ws = np.asarray(stack_objective_weights(objective_weights(LATS, OCEAN,
                                                                  land)))
        carry = _carry()
        q_base = carry["ocn"]["forcing"].q_flux
        heat = jnp.asarray(OCEAN, jnp.float32)
        n_seg, seg_days, delta = 6, 4, 0.2
        n_days = n_seg * seg_days
        one = jnp.asarray(1.0)
        long_seg = make_segment_fn(_toy_step, patterns, n_days,
                                   fields_fn=_toy_fields)

        def run(amplitudes, warming):
            _, f = long_seg(carry, jnp.asarray(amplitudes, jnp.float32), one,
                            q_base, warming)
            return np.asarray(f["sst"], np.float64)

        normal = run(np.zeros(k), make_warming(carry, heat))
        warm_w = make_warming(carry, heat, step_wm2=0.3)
        warmed = run(np.zeros(k), warm_w)
        series = np.stack([np.stack([ocean_indices(m, ws) for m in
                                     (run(delta * np.eye(k)[j], warm_w)
                                      - warmed)], axis=0)
                           for j in range(k)], axis=-1) / delta
        rates = sensitivity_rates(series)                  # (3, K)
        ctrl = IndexPIController(rates, ws, normal, PIConfig(
            closed_loop_days=8.0, segment_days=seg_days, cap=5.0,
            indices=("T0",)))
        seg = make_segment_fn(_toy_step, patterns, seg_days,
                              fields_fn=_toy_fields)
        ep = run_episode(carry, seg, ctrl, n_seg, seg_days, k, warm_w)
        controlled = segment_index_means(ep.fields["sst"], ws, seg_days,
                                         normal)[:, 0]
        uncontrolled = segment_index_means(warmed, ws, seg_days, normal)[:, 0]
        self.assertEqual(len(ctrl.log), n_seg)
        self.assertGreater(ep.amplitudes[1:].sum(), 0.0)
        self.assertLess(abs(controlled[-1]), 0.5 * abs(uncontrolled[-1]))


if __name__ == "__main__":
    unittest.main()
