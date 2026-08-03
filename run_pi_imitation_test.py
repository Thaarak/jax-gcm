"""Tests for PI-imitation distillation (Tier-2b, Amendment 4).

The decisive check is closed-loop: an imitation-distilled MLP rolled out in
the FakeCoupler under eta != 1 must track the PI policy's dSST to ~1-2 mK —
i.e. the distillation transfers the LAW, not just pointwise outputs.
"""

import argparse
import unittest

import jax.numpy as jnp
import numpy as np

from jcm.mcb.coupled_controller import (
    CoupledControllerConfig,
    create_coupled_step_fn,
    evaluate_coupled_policy,
)
from jcm.mcb.coupled_features import (
    CoupledFeatureConfig,
    compute_baseline_trajectory,
)
from jcm.mcb.coupled_train_test import WORKFLOW, FakeCoupler, make_fake_carry
from jcm.mcb.policy import MCBPolicyMLP
from jcm.physics.speedy.speedy_coords import get_speedy_coords

from run_confirmatory_eval import make_pi_policy_fn
from run_pi_imitation import distill, sample_features


def _distill_args(**overrides):
    base = dict(
        target_cooling=-0.02, max_perturbation=0.15, seed=52,
        steps=4000, batch=512, learning_rate=3e-3,
        f0_range=(-0.05, 0.01), acceptance=0.004,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class DistillFidelityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coords = get_speedy_coords()
        cls.shape = cls.coords.horizontal.nodal_shape
        # Non-saturating pattern: PI compensates BOTH eta directions here.
        cls.pattern = np.full(cls.shape, 0.1, dtype=np.float32)
        cls.args = _distill_args()
        cls.params, cls.report = distill(cls.pattern, cls.args, log_every=750)
        cls.policy = MCBPolicyMLP(
            output_shape=cls.shape, hidden_dims=(256, 256),
            max_perturbation=cls.args.max_perturbation,
        )

    def test_fidelity_accepted(self):
        self.assertTrue(self.report["accepted"],
                        msg=f"fidelity report: {self.report}")
        self.assertLess(self.report["mean_abs_err"], self.args.acceptance)

    def test_gain_law_at_real_operating_point(self):
        """Regression for the first campaign failure: the net must track PI
        at the MEASURED real-model feature values (f11=-1.83, f12=-4.23),
        not just near zero, where the old sampling collapsed it to gain~1."""
        from run_pi_imitation import F11_ANCHOR, F12_ANCHOR
        pi_fn = make_pi_policy_fn()
        pi_params = {"pattern": jnp.asarray(self.pattern),
                     "target": self.args.target_cooling}
        for f0, tfrac in ((-0.005, 0.5), (-0.015, 0.5), (-0.008, 0.75)):
            feats = np.zeros(13, dtype=np.float32)
            feats[0] = f0
            feats[10] = tfrac
            feats[11] = F11_ANCHOR
            feats[12] = F12_ANCHOR
            feats = jnp.asarray(feats)
            nn = self.policy.apply(self.params, feats)
            pi = jnp.clip(pi_fn(pi_params, feats), 0.0,
                          self.args.max_perturbation)
            self.assertLess(
                float(jnp.mean(jnp.abs(nn - pi))), 0.010,
                msg=f"real operating point f0={f0} tfrac={tfrac}: gap "
                    f"{float(jnp.mean(jnp.abs(nn - pi))):.4f}")

    def test_gain_law_pointwise(self):
        """NN output tracks pattern x gain across the (f0, tfrac) plane."""
        pi_fn = make_pi_policy_fn()
        pi_params = {"pattern": jnp.asarray(self.pattern),
                     "target": self.args.target_cooling}
        rng = np.random.default_rng(7)
        for tfrac in (0.25, 0.5, 0.75):
            for f0 in (-0.002, -0.01, -0.02):
                feats = sample_features(rng, 1, 13,
                                        self.args.f0_range)[0]
                feats[0] = f0
                feats[10] = tfrac
                feats = jnp.asarray(feats, dtype=jnp.float32)
                nn = self.policy.apply(self.params, feats)
                pi = jnp.clip(pi_fn(pi_params, feats), 0.0,
                              self.args.max_perturbation)
                # 0.010 bound: the (tfrac=0.25, moderate-f0) corner is the
                # steepest cliff of the clipped-gain surface; the decisive
                # equivalence criteria are the grid acceptance (0.004 mean)
                # and the closed-loop 3 mK bound below.
                self.assertLess(
                    float(jnp.mean(jnp.abs(nn - pi))), 0.010,
                    msg=f"tfrac={tfrac} f0={f0}: mean gap "
                        f"{float(jnp.mean(jnp.abs(nn - pi))):.4f}")

    def test_closed_loop_tracks_pi(self):
        """Rolled out under eta != 1, imitation ~= PI to a few mK (FakeCoupler)."""
        coupler = FakeCoupler()
        config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4,
            target_cooling=self.args.target_cooling,
            feature_config=CoupledFeatureConfig(include_absolute_sst=True),
            max_perturbation=self.args.max_perturbation,
            use_checkpointing=True,
        )
        pi_config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4,
            target_cooling=self.args.target_cooling,
            feature_config=CoupledFeatureConfig(),  # fc11 for the PI arm
            max_perturbation=self.args.max_perturbation,
            use_checkpointing=True,
        )
        step_fn = create_coupled_step_fn(coupler, WORKFLOW)
        carry = make_fake_carry(self.coords, 288.0)
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=4, coords=self.coords)
        pi_fn = make_pi_policy_fn()
        pi_params = {"pattern": jnp.asarray(self.pattern),
                     "target": self.args.target_cooling}

        def run(policy_fn, params, cfg, eta):
            return evaluate_coupled_policy(
                coupler=coupler, workflow=WORKFLOW, policy_fn=policy_fn,
                policy_params=params, initial_carry=carry,
                baseline_trajectory=baseline, coords=self.coords,
                ocean_mask=jnp.ones(self.shape), config=cfg,
                tail_mean_days=0, efficacy=eta,
            )["metrics"]["final_sst_change"]

        for eta in (0.8, 1.0, 1.25):
            d_pi = run(pi_fn, pi_params, pi_config, eta)
            d_nn = run(self.policy.apply, self.params, config, eta)
            self.assertLess(
                abs(d_nn - d_pi), 0.003,
                msg=f"eta={eta}: imitation {d_nn:+.4f} vs PI {d_pi:+.4f}")
            # And both must beat the uncompensated miss at eta != 1.
            if eta != 1.0:
                static_miss = abs(self.args.target_cooling * (eta - 1.0))
                self.assertLess(abs(d_nn - self.args.target_cooling),
                                static_miss)


if __name__ == "__main__":
    unittest.main()
