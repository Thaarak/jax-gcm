"""Tests for the per-band strength estimator of Experiment 3b."""

import unittest

import numpy as np

from jcm.mcb.strength_estimator import EstimatorConfig, StrengthEstimator

N, K = 24, 5


def _sensitivity(rng):
    """Smooth, overlapping band responses on a latitude profile."""
    lat = np.linspace(-1.0, 1.0, N)
    centres = np.linspace(-0.8, 0.8, K)
    g = -0.02 * np.exp(-((lat[:, None] - centres[None, :]) / 0.35) ** 2)
    return g * rng.uniform(0.8, 1.2, (1, K))


class StrengthEstimatorTest(unittest.TestCase):
    def test_starts_at_the_prior(self):
        est = StrengthEstimator(K, EstimatorConfig(noise_k=0.01))
        np.testing.assert_allclose(est.strength(), np.ones(K))

    def test_recovers_the_true_strength_without_noise(self):
        rng = np.random.default_rng(0)
        truth = np.array([0.5, 1.5, 1.0, 2.0, 0.8])
        w = np.full(N, 1.0 / N)
        est = StrengthEstimator(K, EstimatorConfig(noise_k=1e-4,
                                                   prior_sd=10.0))
        belief = np.ones(K)
        for _ in range(4):
            g = _sensitivity(rng)
            miss = g @ (truth - belief)          # the linear model, exactly
            belief = est.update(miss, g, belief, w)
        np.testing.assert_allclose(belief, truth, rtol=1e-3)

    def test_global_mode_learns_one_shared_factor(self):
        rng = np.random.default_rng(1)
        truth = np.full(K, 1.7)
        w = np.full(N, 1.0 / N)
        est = StrengthEstimator(K, EstimatorConfig(noise_k=1e-4,
                                                   mode="global"))
        belief = np.ones(K)
        for _ in range(3):
            g = _sensitivity(rng)
            belief = est.update(g @ (truth - belief), g, belief, w)
        np.testing.assert_allclose(belief, truth, rtol=1e-3)
        self.assertEqual(est.theta.shape, (1,))

    def test_noise_and_prior_set_how_far_one_segment_moves_it(self):
        rng = np.random.default_rng(2)
        g = _sensitivity(rng)
        w = np.full(N, 1.0 / N)
        miss = g @ np.full(K, 1.0)                # truth 2, belief 1
        moved = []
        for noise in (1e-4, 1e-2, 1.0):
            est = StrengthEstimator(K, EstimatorConfig(noise_k=noise))
            moved.append(np.mean(est.update(miss, g, np.ones(K), w)) - 1.0)
        self.assertGreater(moved[0], moved[1])
        self.assertGreater(moved[1], moved[2])
        self.assertLess(abs(moved[2]), 1e-3)      # noise >> signal: stays put

    def test_noisy_measurements_converge_with_more_segments(self):
        rng = np.random.default_rng(3)
        truth = np.array([0.6, 1.4, 1.0, 2.0, 0.5])
        w = np.full(N, 1.0 / N)
        noise = 0.002
        errs = []
        for segments in (2, 40):
            est = StrengthEstimator(K, EstimatorConfig(noise_k=noise))
            belief = np.ones(K)
            for _ in range(segments):
                g = _sensitivity(rng)
                miss = g @ (truth - belief) + rng.normal(0, noise, N)
                belief = est.update(miss, g, belief, w)
            errs.append(np.abs(np.log(belief / truth)).mean())
        self.assertLess(errs[1], errs[0])

    def test_estimates_are_clipped(self):
        w = np.full(N, 1.0 / N)
        g = _sensitivity(np.random.default_rng(4))
        est = StrengthEstimator(K, EstimatorConfig(noise_k=1e-5,
                                                   prior_sd=100.0))
        out = est.update(g @ np.full(K, 50.0), g, np.ones(K), w)
        self.assertTrue(np.all(out <= 5.0))

    def test_rejects_bad_settings(self):
        for kwargs in ({"noise_k": 0.0}, {"noise_k": 1.0, "mode": "x"},
                       {"noise_k": 1.0, "prior_sd": 0.0},
                       {"noise_k": 1.0, "bounds": (1.0, 2.0)}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                EstimatorConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
