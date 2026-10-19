"""Tests for analyze_exp3e_pilot (Experiment 3e's pilot rules)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_exp3e_pilot import (
    detection_terms,
    detection_z,
    kappa,
    summarize,
    z_null_sd,
)

IL, K, N_REPLANS, MEMBERS = 12, 5, 12, 3


def write_run(folder, idx, rng, noise_sd, signal=0.0):
    """One synthetic oracle run: a summary and its innovations file."""
    folder.mkdir(parents=True, exist_ok=True)
    w = rng.uniform(0.5, 1.5, IL)
    w /= w.sum()
    sens = rng.uniform(0.0, 0.02, (MEMBERS, N_REPLANS, IL, K))
    belief = np.ones((MEMBERS, N_REPLANS, K))
    noise = noise_sd * rng.standard_normal((MEMBERS, N_REPLANS, IL))
    predicted = rng.standard_normal((MEMBERS, N_REPLANS, IL))
    # With signal > 0 the world responds more strongly than the forecast assumed.
    observed = predicted + noise + signal * sens.sum(-1)
    np.savez_compressed(folder / f"ic{idx}.innovations.npz",
                        observed=observed, predicted=predicted,
                        sensitivity=sens, belief=belief,
                        profile_weights=w, day=14 * np.arange(1, N_REPLANS + 1),
                        true_efficacy=np.ones(K))
    logs = []
    for m in range(MEMBERS):
        log = [{"segment": 0, "background_error": {"atm[0]": 2.0}}]
        for j in range(N_REPLANS):
            r = observed[m, j] - predicted[m, j]
            log.append({"segment": j + 1,
                        "innovation_ms": float(np.sum(w * r * r)),
                        "background_error": {"atm[0]": 1.0}})
        logs.append(log)
    (folder / f"ic{idx}.json").write_text(json.dumps({
        "planner_logs": logs,
        "objective": {"controlled": {"zonal_pattern": 0.003}}}))


class PilotRulesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(3)

    def tearDown(self):
        self.tmp.cleanup()

    def level(self, name, noise_sd, n_runs=40):
        folder = self.root / name / "plan_oracle"
        for i in range(n_runs):
            write_run(folder, 100 + i, self.rng, noise_sd)
        return sorted(folder.glob("ic*.innovations.npz"))

    def test_the_statistic_is_standard_normal_without_an_effect(self):
        paths = self.level("a10", noise_sd=0.01)
        scale, n = kappa(paths)
        self.assertEqual(n, 40 * MEMBERS * N_REPLANS)
        # Independent white noise: Var(g'Wr) = s^2 sum(W^2 g^2), so kappa is
        # of the order of s times the root of a typical weight, and the
        # statistic built from pure noise has unit spread.
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(z_null_sd(paths, scale), 1.0, delta=0.1)

    def test_kappa_scales_with_the_noise(self):
        small = kappa(self.level("a10", noise_sd=0.01))[0]
        big = kappa(self.level("a30", noise_sd=0.04))[0]
        self.assertAlmostEqual(big / small, 4.0, delta=0.4)

    def test_a_real_effect_drives_the_statistic_up(self):
        folder = self.root / "sig" / "plan_oracle"
        write_run(folder, 1, self.rng, noise_sd=0.01, signal=2.0)
        scale = kappa(self.level("a10", noise_sd=0.01))[0]
        u, v, _ = detection_terms(folder / "ic1.innovations.npz")
        z = detection_z(u, v, scale)
        self.assertEqual(z.shape, (MEMBERS, N_REPLANS))
        self.assertTrue(np.all(z[:, -1] > 5.0))

    def test_summary_applies_every_rule_per_level(self):
        self.level("a10", noise_sd=0.01, n_runs=4)
        self.level("a100", noise_sd=0.05, n_runs=4)
        out = summarize(self.root)["levels"]
        self.assertEqual(sorted(out), ["a10", "a100"])
        self.assertGreater(out["a100"]["noise_k"], 3 * out["a10"]["noise_k"])
        self.assertEqual(out["a10"]["runs"], 4)
        self.assertEqual(out["a10"]["drift_day0"], {"atm[0]": 2.0})
        self.assertEqual(out["a10"]["drift_later"], {"atm[0]": 1.0})
        self.assertAlmostEqual(out["a10"]["oracle_zonal_pattern_mean"], 0.003)


if __name__ == "__main__":
    unittest.main()
