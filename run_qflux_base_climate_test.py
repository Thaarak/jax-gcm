"""Tests for the Q-flux base-climate driver's pure helpers (revision 0.2)."""

import unittest

import numpy as np

from jcm.mcb.qflux import DAYS_PER_YEAR, SECONDS_PER_DAY
from run_qflux_base_climate import (
    BIAS_MAX_K,
    DRIFT_MAX_K_PER_60D,
    block_means,
    deseasonalize,
    drift_per_60d,
    fit_exponential_approach,
    gate,
    newton_correction,
    parse_args,
    validate_args,
)


class DriftTest(unittest.TestCase):
    def test_recovers_a_trend_under_a_seasonal_cycle(self):
        t = np.arange(1460.0)
        season = (2.0 * np.cos(2 * np.pi * t / DAYS_PER_YEAR)
                  + 0.5 * np.sin(4 * np.pi * t / DAYS_PER_YEAR))
        for per_60d in (0.0, 0.01, -0.05):
            series = 290.0 + season + per_60d / 60.0 * t
            self.assertAlmostEqual(drift_per_60d(series), per_60d, places=6)

    def test_weather_noise_alone_gives_a_small_drift(self):
        rng = np.random.default_rng(0)
        t = np.arange(1460.0)
        series = (290.0 + np.cos(2 * np.pi * t / DAYS_PER_YEAR)
                  + rng.normal(0.0, 0.05, t.size))
        self.assertLess(abs(drift_per_60d(series)), 0.002)


class NewtonCorrectionTest(unittest.TestCase):
    """Revision 0.3: the correction recovers a known slab response."""

    def _series(self, tau=1200.0, t_inf=288.6, amp=1.6, noise=0.0, n=3650):
        t = np.arange(n, dtype=float)
        rng = np.random.default_rng(1)
        return (t_inf + amp * np.exp(-t / tau)
                + 2.0 * np.cos(2 * np.pi * t / DAYS_PER_YEAR)
                + 0.4 * np.sin(4 * np.pi * t / DAYS_PER_YEAR)
                + rng.normal(0.0, noise, n))

    def test_recovers_tau_and_equilibrium_under_a_seasonal_cycle(self):
        tm, ym = block_means(deseasonalize(self._series(), 1460), 60)
        fit = fit_exponential_approach(tm[tm >= 180], ym[tm >= 180])
        self.assertAlmostEqual(fit["tau_days"], 1200.0,
                               delta=0.02 * 1200.0)
        self.assertAlmostEqual(fit["t_inf"], 288.6, delta=0.01)
        self.assertFalse(fit["tau_at_grid_edge"])

    def test_heat_correction_is_lambda_times_the_equilibrium_bias(self):
        cap = 2.0e8
        res = newton_correction(self._series(noise=0.02), 1460, 290.4, cap)
        lam = cap / (1200.0 * SECONDS_PER_DAY)
        self.assertAlmostEqual(res["lambda_wm2_per_k"], lam,
                               delta=0.05 * lam)
        self.assertAlmostEqual(res["equilibrium_bias_k"], -1.8, delta=0.03)
        self.assertAlmostEqual(res["heat_into_ocean_wm2"], lam * 1.8,
                               delta=0.07 * lam * 1.8)

    def test_flat_series_hits_the_grid_edge(self):
        t = np.arange(40.0)
        fit = fit_exponential_approach(t, np.full(40, 290.0)
                                       + np.random.default_rng(2).normal(
                                           0.0, 1e-3, 40))
        self.assertIsInstance(fit["tau_at_grid_edge"], bool)


class GateTest(unittest.TestCase):
    def test_registered_thresholds(self):
        self.assertEqual((DRIFT_MAX_K_PER_60D, BIAS_MAX_K), (0.02, 0.5))
        self.assertTrue(gate(0.01, -0.3)["pass"])
        self.assertFalse(gate(0.03, 0.0)["pass"])
        self.assertFalse(gate(0.0, 0.6)["pass"])
        v = gate(-0.021, -0.51)
        self.assertFalse(v["G1_drift_pass"] or v["G2_bias_pass"])


class ArgsTest(unittest.TestCase):
    def test_registered_defaults(self):
        d = parse_args(["diagnose", "--output-dir", "x"])
        self.assertEqual((d.tau_days, d.spinup_days, d.record_days),
                         (5.0, 365, 1460))
        validate_args(d)
        s = parse_args(["settle", "--qflux", "q.nc", "--output-dir", "x",
                        "--base-carry-out", "b.pkl"])
        self.assertEqual((s.days, s.eval_days), (3650, 1460))
        validate_args(s)

    def test_rejects_lengths_that_do_not_fit_the_chunks(self):
        bad = [["diagnose", "--output-dir", "x", "--record-days", "100"],
               ["diagnose", "--output-dir", "x", "--tau-days", "0"],
               ["settle", "--qflux", "q", "--output-dir", "x",
                "--base-carry-out", "b", "--days", "100"],
               ["settle", "--qflux", "q", "--output-dir", "x",
                "--base-carry-out", "b", "--eval-days", "4000"]]
        for argv in bad:
            with self.assertRaises(SystemExit):
                validate_args(parse_args(argv))


if __name__ == "__main__":
    unittest.main()
