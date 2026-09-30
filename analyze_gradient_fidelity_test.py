"""Tests for the pre-registered Experiment 1 analysis (synthetic data only)."""

import unittest

import numpy as np

from analyze_gradient_fidelity import (
    analyze,
    angle_deg,
    choose_w_star,
    finite_differences,
    primary_outcome,
    projection_ratio,
    tail_means,
    verdict,
)
from run_gradient_fidelity import run_labels

K = 3
HORIZONS = [30, 60]
TAIL = 10
DELTA = 0.02
OBJECTIVES = ["T0", "T1", "T2", "LAND"]
TRUTH = np.array([-0.8, -0.5, -0.3])


def _synthetic(windows, estimator_for, fd_noise=0.005, n_ic=6, k_fd=4, k_g=2,
               seed=0):
    """Build synthetic Experiment 1 outputs.

    Daily series are constant in time, with central differences equal to
    TRUTH (+ noise); Jacobians come from ``estimator_for(w, rng)``.
    """
    rng = np.random.default_rng(seed)
    labels = run_labels(K, zero_run=True)
    n_days = max(HORIZONS)
    fd_series = np.zeros((n_ic, k_fd, len(labels), n_days, len(OBJECTIVES)))
    for i in range(n_ic):
        for m in range(k_fd):
            base = rng.normal(0.0, 0.01, len(OBJECTIVES))
            fd_series[i, m, labels.index("zero")] = base
            fd_series[i, m, labels.index("center")] = base - 0.1
            for k in range(K):
                for sign, name in ((1, "plus"), (-1, "minus")):
                    noise = rng.normal(0.0, fd_noise, len(OBJECTIVES))
                    fd_series[i, m, labels.index(f"{name}_{k}")] = (
                        base - 0.1 + sign * DELTA * TRUTH[k] + noise)
    jac = np.zeros((n_ic, k_g, len(windows), len(HORIZONS), len(OBJECTIVES),
                    K))
    for wi, w in enumerate(windows):
        for i in range(n_ic):
            for m in range(k_g):
                for hi in range(len(HORIZONS)):
                    for oi in range(len(OBJECTIVES)):
                        jac[i, m, wi, hi, oi] = estimator_for(w, rng)
    meta = {"config": {"tail_days": TAIL, "delta": DELTA},
            "run_labels": labels, "horizons": HORIZONS, "windows": windows,
            "objective_names": OBJECTIVES, "finished_ics": n_ic}
    return {"fd_series": fd_series, "jacobians": jac}, meta


def _good(rng):
    return TRUTH + rng.normal(0.0, 0.05, K)


def _bad(rng):
    return rng.normal(0.0, 1.0, K)


class BasicsTest(unittest.TestCase):
    def test_angle_and_ratio(self):
        self.assertAlmostEqual(float(angle_deg([1, 0], [0, 1])), 90.0)
        self.assertAlmostEqual(float(angle_deg([1, 1], [2, 2])), 0.0,
                               places=5)
        self.assertAlmostEqual(float(projection_ratio([2, 0], [1, 0])), 2.0)

    def test_tail_means_and_finite_differences_recover_truth(self):
        arrays, meta = _synthetic([1, 0], lambda w, rng: _good(rng),
                                  fd_noise=0.0)
        tm = tail_means(arrays["fd_series"], HORIZONS, TAIL)
        fd = finite_differences(tm, meta["run_labels"], DELTA, K)
        np.testing.assert_allclose(fd[..., 0, :].mean(axis=(0, 1))[0], TRUTH,
                                   rtol=1e-9)

    def test_verdict_rules(self):
        self.assertEqual(verdict(False, [1, 2], [0.9, 1.1], 5),
                         "unresolved_truth")
        self.assertEqual(verdict(True, [3, 12], [0.8, 1.2], 20), "useful")
        self.assertEqual(verdict(True, [25, 40], [0.8, 1.2], 20), "failed")
        self.assertEqual(verdict(True, [3, 12], [1.5, 2.0], 20), "failed")
        self.assertEqual(verdict(True, [3, 12], [0.8, 1.2], 60), "failed")
        self.assertEqual(verdict(True, [10, 30], [0.8, 1.2], 20),
                         "inconclusive")

    def test_w_star_tie_goes_to_larger_window(self):
        cells = [(1, "useful", 10.0), (7, "useful", 11.5),
                 (14, "useful", 20.0), (30, "failed", 5.0)]
        self.assertEqual(choose_w_star(cells), 7)
        self.assertIsNone(choose_w_star([(1, "failed", 5.0)]))

    def test_outcome_letters(self):
        useful = [(1, "useful", 10.0)]
        self.assertEqual(primary_outcome(False, useful, "failed"), "U")
        self.assertEqual(primary_outcome(True, useful, "failed"), "A")
        self.assertEqual(primary_outcome(True, useful, "inconclusive"), "A'")
        self.assertEqual(primary_outcome(True, useful, "useful"), "B")
        self.assertEqual(primary_outcome(True, [(1, "failed", 60.0)],
                                         "failed"), "C")


class EndToEndTest(unittest.TestCase):
    def _run(self, estimator_for, **kw):
        arrays, meta = _synthetic([1, 7, 0], estimator_for, **kw)
        return analyze(arrays, meta, n_boot=200, seed=1)

    def test_truncation_rescues(self):
        res = self._run(lambda w, rng: _bad(rng) if w == 0 else _good(rng))
        self.assertEqual(res["primary"]["outcome"], "A")
        self.assertIn(res["primary"]["w_star"], (1, 7))
        self.assertEqual(res["primary"]["full_bptt_verdict"], "failed")

    def test_full_bptt_already_fine(self):
        res = self._run(lambda w, rng: _good(rng))
        self.assertEqual(res["primary"]["outcome"], "B")

    def test_nothing_works(self):
        res = self._run(lambda w, rng: _bad(rng))
        self.assertEqual(res["primary"]["outcome"], "C")
        self.assertIsNone(res["primary"]["w_star"])

    def test_unresolved_truth(self):
        res = self._run(lambda w, rng: _good(rng), fd_noise=5.0)
        self.assertEqual(res["primary"]["outcome"], "U")
        self.assertEqual(res["land_prediction_confirmed"], "untestable")

    def test_operating_point_and_land_flag_are_reported(self):
        res = self._run(lambda w, rng: _bad(rng) if w == 1 else _good(rng))
        self.assertAlmostEqual(res["operating_point_response"]["T0@60"]
                               ["mean"], -0.1, places=6)
        self.assertTrue(res["land_prediction_confirmed"])


if __name__ == "__main__":
    unittest.main()
