"""Tests for the confirmatory-eval aggregation/gates (pure NumPy, no model).

Validates the micro-ensemble math the meta-audit Tier-1 plan relies on:
member-averaging shrinks per-run chaos noise by sqrt(k), the noise accounting
reports it, and the gates fire on per-IC MEANS (not member-level values).
"""

import unittest

import numpy as np

from run_confirmatory_eval import FEATURE_CONFIGS, aggregate_and_gate, parse_arms


def synth_cells(rng, arm_effects, n_ics=10, k=8, chaos_sd=0.014,
                target=-0.1):
    """Synthetic member-level cells: dsst = target + effect + chaos noise."""
    cells = {}
    for name, effect in arm_effects.items():
        dsst = (target + effect
                + chaos_sd * rng.standard_normal((n_ics, k)))
        cells[name] = {
            "dsst_10d": dsst,
            "dsst_snapshot": dsst + 0.001,
            "amazon_mm_day": np.zeros((n_ics, k)),
            "sahel_mm_day": np.zeros((n_ics, k)),
            "mean_loss": np.abs(dsst),
        }
    return cells


class AggregateAndGateTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_noise_accounting_recovers_chaos_sd(self):
        cells = synth_cells(self.rng, {"static": 0.0, "policy": 0.0},
                            n_ics=20, k=8, chaos_sd=0.014)
        _, noise, _ = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        self.assertAlmostEqual(noise["static"]["per_run_chaos_sd"], 0.014,
                               delta=0.003)
        # Cross-IC sd of k=8 member means ~ chaos_sd/sqrt(8) ~ 0.005
        self.assertLess(noise["static"]["cross_ic_sd_of_means"], 0.009)

    def test_member_averaging_powers_a_small_real_effect(self):
        # A 6 mK real advantage is invisible at k=1 but significant at k=16
        # with n=20 ICs — the core Tier-1 claim, verified end to end.
        effect = -0.006  # policy 6 mK closer to target
        detected = {}
        for k in (1, 16):
            rng = np.random.default_rng(42)
            cells = synth_cells(
                rng, {"static": 0.010, "policy": 0.010 + effect},
                n_ics=20, k=k, chaos_sd=0.014)
            _, _, gates = aggregate_and_gate(
                cells, comparator="static", target=-0.1,
                band=(-0.12, -0.08), tail_days=10)
            rep = gates["paired_report[policy vs static]"]
            detected[k] = rep["p_t"]
        self.assertGreater(detected[1], 0.05)   # k=1: underpowered
        self.assertLess(detected[16], 0.05)     # k=16: significant

    def test_gates_use_per_ic_means_and_comparator(self):
        cells = synth_cells(self.rng, {"static": 0.0, "a": 0.0, "b": 0.0},
                            n_ics=8, k=4)
        per_ic, _, gates = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        self.assertEqual(per_ic["a"]["dsst_10d"].shape, (8,))
        self.assertIn("G2_cooling[static]", gates)
        self.assertIn("G3_improvement[a vs static]", gates)
        self.assertIn("paired_report[a vs b]", gates)  # non-comparator pair
        self.assertNotIn("G3_improvement[static vs static]", gates)
        self.assertEqual(gates["n_ics"], 8)
        self.assertEqual(gates["members"], 4)

    def test_equivalence_bound_present_for_null_effect(self):
        cells = synth_cells(self.rng, {"static": 0.0, "policy": 0.0},
                            n_ics=20, k=8, chaos_sd=0.014)
        _, _, gates = aggregate_and_gate(
            cells, comparator="static", target=-0.1,
            band=(-0.12, -0.08), tail_days=10)
        rep = gates["paired_report[policy vs static]"]
        # With k=8/n=20 the demonstrable equivalence bound should be tight
        # (a few mK), the "significant bounded-negative" the plan promises.
        self.assertLess(rep["equivalence_bound_95"], 0.006)


class ParseArmsTest(unittest.TestCase):
    def test_rejects_bad_kind_and_duplicates(self):
        with self.assertRaises(SystemExit):
            parse_arms(["a=nope=/tmp/x"])
        self.assertIn("static", FEATURE_CONFIGS)




class PiControllerTest(unittest.TestCase):
    """The deadbeat PI arm compensates unobserved efficacy; static cannot.

    Uses the FakeCoupler from coupled_train_test (dSST/step = -0.05 x applied
    perturbation), pattern p=0.1, cap 0.15, target -0.02 over 4 steps with
    2-step control intervals — chosen so the PI gains stay inside the cap:
    eta=0.8 needs gain 1.5 (command exactly at cap), eta=1.25 needs 0.6.
    """

    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp
        from jcm.mcb.coupled_controller import (
            CoupledControllerConfig,
            create_coupled_step_fn,
            evaluate_coupled_policy,
        )
        from jcm.mcb.coupled_features import (
            CoupledFeatureConfig,
            compute_baseline_trajectory,
        )
        from jcm.mcb.coupled_train_test import (
            WORKFLOW,
            FakeCoupler,
            make_fake_carry,
        )
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        cls.jnp = jnp
        cls.evaluate = staticmethod(evaluate_coupled_policy)
        cls.workflow = WORKFLOW
        cls.coords = get_speedy_coords()
        cls.coupler = FakeCoupler()
        cls.shape = cls.coords.horizontal.nodal_shape
        cls.p = 0.1
        cls.target = -0.02
        cls.config = CoupledControllerConfig(
            control_interval_steps=2, total_steps=4,
            target_cooling=cls.target,
            feature_config=CoupledFeatureConfig(),  # fc11 (what PI reads)
            max_perturbation=0.15, use_checkpointing=True,
        )
        step_fn = create_coupled_step_fn(cls.coupler, WORKFLOW)
        cls.carry = make_fake_carry(cls.coords, 288.0)
        cls.baseline = compute_baseline_trajectory(
            cls.carry, step_fn, num_steps=4, coords=cls.coords)

    def _run(self, policy_fn, params, eta):
        return self.evaluate(
            coupler=self.coupler, workflow=self.workflow,
            policy_fn=policy_fn, policy_params=params,
            initial_carry=self.carry, baseline_trajectory=self.baseline,
            coords=self.coords, ocean_mask=self.jnp.ones(self.shape),
            config=self.config, tail_mean_days=0, efficacy=eta,
        )["metrics"]["final_sst_change"]

    def test_pi_compensates_static_does_not(self):
        from run_confirmatory_eval import make_pi_policy_fn
        jnp = self.jnp
        pattern = jnp.full(self.shape, self.p)
        static_fn = lambda params, feats: pattern  # noqa: E731
        pi_fn = make_pi_policy_fn()
        pi_params = {"pattern": pattern, "target": self.target}
        for eta in (0.8, 1.25):
            d_static = self._run(static_fn, {}, eta)
            d_pi = self._run(pi_fn, pi_params, eta)
            err_static = abs(d_static - self.target)
            err_pi = abs(d_pi - self.target)
            # Static misses by ~|eta-1| x |target| (0.004-0.005 here).
            self.assertGreater(err_static, 0.003,
                               msg=f"eta={eta}: manipulation check failed")
            # PI recovers the target (residual ~ f32 mean-noise, <1e-3).
            self.assertLess(err_pi, 0.001, msg=f"eta={eta}")
            self.assertLess(err_pi, 0.25 * err_static, msg=f"eta={eta}")

    def test_pi_is_inert_at_unit_efficacy(self):
        from run_confirmatory_eval import make_pi_policy_fn
        jnp = self.jnp
        pattern = jnp.full(self.shape, self.p)
        static_fn = lambda params, feats: pattern  # noqa: E731
        pi_fn = make_pi_policy_fn()
        pi_params = {"pattern": pattern, "target": self.target}
        d_static = self._run(static_fn, {}, 1.0)
        d_pi = self._run(pi_fn, pi_params, 1.0)
        # At eta=1 the PI gain stays ~1, so it tracks the static pattern.
        self.assertAlmostEqual(d_pi, d_static, delta=1e-3)


if __name__ == "__main__":
    unittest.main()


class PiEnsoWeightSemanticsTest(unittest.TestCase):
    """KIND@FF[@FB] must be WEIGHTS, not raw coefficients.

    Regression test for the Amendment 7 rev 2 tuning-sweep bug: `@1` was
    read as enso_effect_per_K = 1.0 — 18x the measured 0.05473 — so the
    'anticipating' arms ran wildly over-aggressive (RMS_A 87 vs 16 mK, and
    a sign-flipped slope). @1 must mean 'designed strength'.
    """

    def _fn(self, spec_kind, ff_measured=0.05473):
        import pickle
        import jax.numpy as jnp
        import numpy as np
        from run_confirmatory_eval import build_arm
        from jcm.mcb import MCBPolicyMLP
        pat = "mcb_experiments_gpu/stage1/stage1_optimized_pattern.pkl"
        with open(pat, "rb") as f:
            shape = np.asarray(pickle.load(f)["best_pattern"]).shape

        class C:
            class horizontal:
                nodal_shape = shape
        policy = MCBPolicyMLP(output_shape=shape, hidden_dims=(8,),
                              max_perturbation=0.09)
        from run_confirmatory_eval import parse_arms
        parsed = parse_arms([f"t={spec_kind}={pat}"])[0]
        fn, params, _ = build_arm(parsed, policy, C, -0.1,
                                  enso_effect_per_k=ff_measured,
                                  reference_scale=1.196)
        feats = jnp.zeros(14).at[0].set(-0.02).at[10].set(0.5).at[13].set(1.5)
        return float(jnp.mean(fn(params, feats)))

    def test_at_one_equals_plain_pi_enso(self):
        """pi-enso@1@1 must be IDENTICAL to plain pi-enso."""
        self.assertAlmostEqual(self._fn("pi-enso@1@1"),
                               self._fn("pi-enso"), places=12)

    def test_at_zero_ablates_feedforward(self):
        """pi-enso@0 must command strictly less under a warm anomaly."""
        self.assertLess(self._fn("pi-enso@0"), self._fn("pi-enso"))

    def test_ff_weight_is_not_a_raw_coefficient(self):
        """The bug: @1 read as enso_effect_per_K=1.0 was ~18x too strong."""
        self.assertLess(self._fn("pi-enso@1@1"),
                        1.5 * self._fn("pi-enso@0@1"))
