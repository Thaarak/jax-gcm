"""Tests for the Experiment 3b pilot's fixed rules (analyze_exp3b_pilot.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import analyze_exp3b_pilot as px
from jcm.mcb.hidden_strength import hidden_strength

IX, IL, DAYS, M = 6, 4, 182, 3
LATS = np.deg2rad(np.array([-45.0, -15.0, 15.0, 45.0]))


def _pilot(tmp: Path, scale: dict, member_noise=0.05, seed=0):
    """16 pilot states; each arm's error is ``scale * base`` plus member noise."""
    rng = np.random.default_rng(seed)
    ocean = np.ones((IX, IL))
    ocean[0] = 0.0
    refs = tmp / "refs"
    refs.mkdir()
    np.savez(refs / "grid.npz", ocean_mask=ocean, land_mask=1.0 - ocean,
             latitudes_rad=LATS, longitudes_rad=np.linspace(0, 6, IX))
    counts = ocean.sum(axis=0)
    for macro in range(16, 48, 2):
        idx = 100 * macro + 1
        normal = 290.0 + np.zeros((DAYS, IX, IL))
        np.savez(refs / f"ic{idx:04d}_references.npz", normal_sst=normal,
                 normal_land_temperature=normal, normal_precipitation=normal)
        base = (0.1 + 0.03 * rng.normal()) * (1.0 + LATS[None, :] ** 2)
        for arm in px.PILOT_ARMS:
            d = tmp / "runs" / arm
            d.mkdir(parents=True, exist_ok=True)
            members = [normal + scale.get(arm, 1.0) * base[None]
                       + member_noise * rng.normal(size=(1, 1, IL))
                       for _ in range(M)]
            mean = np.mean(members, axis=0)
            profiles = np.stack([(x * ocean).sum(axis=-2) / counts
                                 for x in members])
            np.savez(d / f"ic{idx:04d}.fields.npz", sst=mean,
                     land_temperature=mean, precipitation=mean,
                     member_zonal_sst=profiles)
            summary = {"effort": 0.02, "members": M, "seconds": 1.0,
                       "true_efficacy": hidden_strength(macro, 1).tolist(),
                       "schedules": [[[0.03] * 5] * 13] * M}
            if arm == "plan_oracle":
                summary["planner_logs"] = [
                    [{"efficacy_belief": [1.0] * 5}]
                    + [{"efficacy_belief": [1.0] * 5,
                        "innovation_ms": 1e-4 * (1 + 0.1 * k)}
                       for k in range(12)]] * M
            (d / f"ic{idx:04d}.json").write_text(json.dumps(summary))
    return tmp / "runs", refs


class PilotRulesTest(unittest.TestCase):
    SCALE = {"uncontrolled": 1.0, "fixed": 0.8, "pi": 0.45, "pi_slow": 0.47,
             "adaptive": 0.5, "plan_naive": 0.5, "plan_oracle": 0.28,
             "plan_learn": 0.31, "plan_learn_cautious": 0.36,
             "plan_learn_global": 0.33}

    def test_noise_rule_is_the_root_median_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, _ = _pilot(Path(tmp), self.SCALE)
            got = px.noise_rule(runs)
        ms = 1e-4 * (1 + 0.1 * np.arange(12))
        self.assertAlmostEqual(got["noise_k"], np.sqrt(np.median(ms)))
        self.assertEqual(got["n"], 16 * M * 12)

    def test_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _pilot(Path(tmp), self.SCALE)
            with redirect_stdout(io.StringIO()):
                px.main(["decide", "--runs-dir", str(runs),
                         "--references-dir", str(refs)])
            out = json.loads((runs / "decisions.json").read_text())
        chosen = out["chosen"]
        # The variants are worse, so the defaults stay.
        self.assertEqual((chosen["learner"], chosen["pi"]),
                         ("plan_learn", "pi"))
        self.assertFalse(any(s["switch"] for s in out["switches"].values()))
        self.assertIn(chosen["members"], px.MEMBER_CHOICES)
        mdes = [out["power"][str(m)]["mde"] for m in px.MEMBER_CHOICES]
        self.assertTrue(mdes[0] >= mdes[1] >= mdes[2])
        self.assertLess(out["pilot_effects"]["H1"]["rel"], 0.0)
        self.assertEqual(sorted(map(int, out["noise_by_members"])),
                         [1, 2, 3])

    def test_a_clearly_better_variant_replaces_its_default(self):
        scale = dict(self.SCALE, pi_slow=0.30)
        with tempfile.TemporaryDirectory() as tmp:
            runs, refs = _pilot(Path(tmp), scale, member_noise=0.002)
            out = px.decide(runs, refs)
        self.assertTrue(out["switches"]["pi_slow"]["switch"])
        self.assertEqual(out["chosen"]["pi"], "pi_slow")

    def test_noise_fit_and_mde(self):
        fit = px.fit_noise({1: 3e-6, 2: 1.5e-6, 3: 1e-6})
        self.assertAlmostEqual(fit["A"], 0.0, places=12)
        self.assertAlmostEqual(fit["B"], 3e-6, places=12)
        fit = px.fit_noise({1: 1e-6, 2: 1e-6, 3: 1e-6})
        self.assertAlmostEqual(fit["A"], 1e-6, places=12)
        self.assertGreater(px.minimum_detectable_effect(4e-6),
                           px.minimum_detectable_effect(1e-6))


if __name__ == "__main__":
    unittest.main()
