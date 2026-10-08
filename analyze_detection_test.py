"""Tests for the one-Earth detection analysis (analyze_detection.py)."""

import unittest

import numpy as np

from analyze_detection import (
    detection_probability,
    fingerprint_snr,
    first_day,
    index_snr,
    noise_covariances,
    window_means,
    zonal_profiles,
)


class StatisticsTest(unittest.TestCase):
    def test_zonal_profiles_average_ocean_cells(self):
        ocean = np.array([[1.0, 0.0], [1.0, 1.0]])
        field = np.array([[2.0, 9.0], [4.0, 6.0]])
        np.testing.assert_allclose(zonal_profiles(field, ocean, np.array(
            [True, True])), [3.0, 6.0])

    def test_window_means(self):
        x = np.arange(8.0)[:, None] * np.ones((8, 2))
        w = window_means(x, 3)
        self.assertTrue(np.all(np.isnan(w[:2])))
        np.testing.assert_allclose(w[2:, 0], [1, 2, 3, 4, 5, 6])

    def test_snrs_and_probability(self):
        area = np.array([0.5, 0.5])
        cov = np.repeat(np.diag([0.01, 0.01])[None], 3, axis=0)
        signal = np.repeat(np.array([[0.1, 0.1]]), 3, axis=0)
        # Index: |0.1| / sqrt(0.25*0.01*2) = 0.1/0.0707 = 1.414.
        np.testing.assert_allclose(index_snr(signal, cov, area), 1.4142,
                                   atol=1e-3)
        fp = fingerprint_snr(signal, cov)
        self.assertTrue(np.all(fp >= index_snr(signal, cov, area) - 1e-9))
        # A pattern the index cannot see (opposite signs) the fingerprint can.
        dipole = np.repeat(np.array([[0.1, -0.1]]), 3, axis=0)
        self.assertLess(float(index_snr(dipole, cov, area)[0]), 1e-12)
        self.assertGreater(float(fingerprint_snr(dipole, cov)[0]), 1.0)
        p = detection_probability(np.array([1.6448536269514722, 5.0]))
        self.assertAlmostEqual(p[0], 0.5)
        self.assertGreater(p[1], 0.99)
        self.assertEqual(first_day(np.array([0.1, 0.6, 0.97]), 0.95), 15)
        self.assertIsNone(first_day(np.array([0.1, 0.2]), 0.5))

    def test_forecast_noise_from_members(self):
        rng = np.random.default_rng(0)
        members = rng.normal(0.0, 0.2, (6, 5, 12, 3))
        cov = noise_covariances(members, np.full(3, 1 / 3))
        self.assertEqual(cov.shape, (12, 3, 3))
        # A 30-day (6-block) mean of independent noise: variance 0.04/6,
        # times (1 + 1/5) for the forecast's own error.
        self.assertAlmostEqual(float(np.mean(np.diagonal(cov[-1]))),
                               0.04 / 6 * 1.2, delta=0.003)

    def test_ledoit_wolf_is_well_conditioned(self):
        from analyze_detection import ledoit_wolf
        rng = np.random.default_rng(1)
        x = rng.normal(size=(20, 30))            # fewer samples than dims
        c = ledoit_wolf(x - x.mean(0))
        self.assertGreater(np.linalg.eigvalsh(c).min(), 0.0)


if __name__ == "__main__":
    unittest.main()
