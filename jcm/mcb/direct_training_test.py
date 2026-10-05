"""Tests for direct training through the simulation (jcm.mcb.direct_training).

The toy is the test world's small linear model with two bands. The key test
is that the closed loop written inside JAX matches ``run_episode`` with a
``StudentPolicy``: same settings, same temperatures.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.direct_training import (
    Budget,
    DirectConfig,
    TrainingEpisode,
    eki_update,
    episode_objective,
    episode_residuals,
    make_closed_loop_rollout,
    make_episode_loss,
    target_segment_means,
    train_eki,
    train_gradient,
)
from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.scores import area_weights
from jcm.mcb.student import (
    StudentConfig,
    StudentPolicy,
    band_observation_weights,
    init_student,
)
from jcm.mcb.test_world import make_segment_fn, make_warming, run_episode
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

SEG_DAYS, N_SEG = 3, 4
CFG = StudentConfig(k=2, hidden=4, cap=5.0, anomaly_scale_k=0.5)


class _Toy(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.obs_w = band_observation_weights(LATS, self.patterns)
        self.weights = area_weights(LATS, OCEAN)
        self.carry = _carry()
        self.q_base = self.carry["ocn"]["forcing"].q_flux
        heat = jnp.asarray(OCEAN, jnp.float32)
        n_days = SEG_DAYS * N_SEG
        long_seg = make_segment_fn(_toy_step, self.patterns, n_days,
                                   fields_fn=_toy_fields)
        _, f = long_seg(self.carry, jnp.zeros(2), jnp.asarray(1.0),
                        self.q_base, make_warming(self.carry, heat))
        self.target_daily = np.asarray(f["sst"], np.float64)
        self.target_means = target_segment_means(self.target_daily,
                                                 SEG_DAYS, N_SEG)
        self.warming = make_warming(self.carry, heat, step_wm2=0.4)
        self.rollout = make_closed_loop_rollout(_toy_step, self.patterns,
                                                self.obs_w, CFG, SEG_DAYS,
                                                N_SEG)
        # Random weights, so the settings really depend on what is seen.
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        p = init_student(k1, CFG)
        p["w2"] = 2.0 * jax.random.normal(k2, p["w2"].shape)
        self.params = p

    def episode(self, efficacy=1.0):
        return TrainingEpisode(self.carry, self.target_means,
                               jnp.asarray(efficacy, jnp.float32),
                               self.q_base, self.warming)


class ClosedLoopTest(_Toy):
    def test_matches_run_episode_with_a_student_policy(self):
        for eff in (1.0, 0.5):
            means, actions = jax.jit(self.rollout)(
                self.params, self.carry, self.target_means,
                jnp.asarray(eff), self.q_base, self.warming,
                jnp.asarray(NO_TRUNCATION_DAYS))
            policy = StudentPolicy(self.params, CFG, self.target_daily,
                                   self.obs_w, SEG_DAYS)
            seg = make_segment_fn(_toy_step, self.patterns, SEG_DAYS,
                                  fields_fn=_toy_fields)
            ep = run_episode(self.carry, seg, policy, N_SEG, SEG_DAYS, 2,
                             self.warming, efficacy=eff)
            np.testing.assert_allclose(np.asarray(actions), ep.amplitudes,
                                       rtol=1e-4, atol=1e-5)
            sst = ep.fields["sst"].reshape(N_SEG, SEG_DAYS, 4, 3).mean(1)
            np.testing.assert_allclose(np.asarray(means),
                                       sst - MAP_REFERENCE_K, atol=2e-4)
            self.assertGreater(np.ptp(ep.amplitudes[:, 0]), 1e-3)

    def test_the_snip_changes_gradients_not_values(self):
        losses, grads = [], []
        for method in ("bptt", "snipped"):
            cfg = DirectConfig(method=method, window_days=2)
            loss = make_episode_loss(self.rollout, self.weights, cfg)
            v, g = jax.value_and_grad(loss)(self.params, self.episode())
            losses.append(float(v))
            grads.append(np.concatenate([np.ravel(x) for x in
                                         jax.tree_util.tree_leaves(g)]))
        self.assertEqual(losses[0], losses[1])
        self.assertGreater(np.linalg.norm(grads[0] - grads[1]), 0.0)

    def test_gradient_matches_finite_differences(self):
        cfg = DirectConfig(method="bptt", mu=0.01, lam=0.05)
        loss = jax.jit(make_episode_loss(self.rollout, self.weights, cfg))
        ep = self.episode()
        g = jax.grad(loss)(self.params, ep)
        direction = jax.tree_util.tree_map(
            lambda x: jax.random.normal(jax.random.PRNGKey(9), x.shape), g)
        exact = sum(float(jnp.sum(a * b)) for a, b in zip(
            jax.tree_util.tree_leaves(g),
            jax.tree_util.tree_leaves(direction)))
        h = 1e-2

        def shifted(sign):
            p = jax.tree_util.tree_map(lambda x, d: x + sign * h * d,
                                       self.params, direction)
            return float(loss(p, ep))

        fd = (shifted(1.0) - shifted(-1.0)) / (2 * h)
        self.assertAlmostEqual(exact, fd, delta=0.05 * abs(fd) + 1e-6)


class ObjectiveTest(_Toy):
    def test_residuals_square_to_the_objective(self):
        rng = np.random.default_rng(1)
        means = jnp.asarray(rng.normal(size=(N_SEG, 4, 3)), jnp.float32)
        actions = jnp.asarray(rng.uniform(0, 1, (N_SEG, 2)), jnp.float32)
        for first in (0, 2):
            j = episode_objective(means, actions, self.target_means,
                                  self.weights, 1.0, 0.5, 0.02, 0.1, first)
            r = episode_residuals(means, actions, self.target_means,
                                  self.weights, 1.0, 0.5, 0.02, 0.1, first)
            self.assertAlmostEqual(float(jnp.sum(r ** 2)), float(j),
                                   delta=1e-5 * float(j))
        with self.assertRaises(ValueError):
            episode_objective(means, actions, self.target_means,
                              self.weights, first_segment=N_SEG)

    def test_config_and_budget(self):
        with self.assertRaises(ValueError):
            DirectConfig(method="evolution").validate()
        self.assertEqual(DirectConfig(method="bptt").window,
                         NO_TRUNCATION_DAYS)
        self.assertEqual(DirectConfig().window, 14)
        b = Budget(2.0)
        b.charge(1.5)
        self.assertFalse(b.exhausted)
        b.charge(0.5)
        self.assertTrue(b.exhausted)
        with self.assertRaises(ValueError):
            Budget(0.0)


class TrainingTest(_Toy):
    def test_gradient_training_lowers_the_objective(self):
        eps = [self.episode(1.0), self.episode(0.7)]
        for method in ("bptt", "snipped"):
            cfg = DirectConfig(method=method, window_days=2,
                               learning_rate=0.05)
            params, log = train_gradient(self.rollout, self.weights,
                                         init_student(jax.random.PRNGKey(5),
                                                      CFG),
                                         eps, cfg, Budget(1e9), max_steps=40)
            self.assertEqual(len(log), 40)
            first = np.mean([r["objective"] for r in log[:2]])
            last = np.mean([r["objective"] for r in log[-2:]])
            self.assertLess(last, 0.5 * first, msg=method)

    def test_budget_stops_training(self):
        cfg = DirectConfig(method="snipped", window_days=2)
        _, log = train_gradient(self.rollout, self.weights, self.params,
                                [self.episode()], cfg, Budget(1e-9))
        self.assertEqual(len(log), 1)

    def test_eki_update_solves_a_linear_problem(self):
        rng = np.random.default_rng(2)
        a = jnp.asarray(rng.normal(size=(30, 4)))
        truth = jnp.asarray([0.5, -1.0, 2.0, 0.3])
        theta = jnp.asarray(rng.normal(size=(20, 4)))
        key = jax.random.PRNGKey(0)
        start = float(jnp.mean(jnp.sum((theta @ a.T - a @ truth) ** 2, 1)))
        for _ in range(30):
            key, sub = jax.random.split(key)
            theta = eki_update(theta, theta @ a.T - a @ truth, sub, 0.5)
        end = float(jnp.sum((theta.mean(0) @ a.T - a @ truth) ** 2))
        self.assertLess(end, 1e-3 * start)

    def test_eki_keeps_a_small_ensemble_alive_on_long_residuals(self):
        # Map-sized residuals (D >> J) must not collapse the ensemble in one
        # step, or the method stops exploring after its first iteration.
        rng = np.random.default_rng(3)
        a = jnp.asarray(rng.normal(size=(5000, 4)))
        theta = jnp.asarray(rng.normal(size=(8, 4)))
        new = eki_update(theta, theta @ a.T - 1.0, jax.random.PRNGKey(1), 1.0)
        ratio = float(jnp.mean(jnp.std(new, 0)) / jnp.mean(jnp.std(theta, 0)))
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 1.0)

    def test_eki_training_lowers_the_objective(self):
        cfg = DirectConfig(method="eki", ensemble=16, init_spread=0.5,
                           noise_scale=0.5)
        loss = jax.jit(make_episode_loss(self.rollout, self.weights, cfg))
        p0 = init_student(jax.random.PRNGKey(6), CFG)
        ep = self.episode()
        params, log = train_eki(self.rollout, self.weights, p0, [ep], cfg,
                                Budget(1e9), max_iterations=15)
        self.assertEqual(len(log), 15)
        self.assertLess(float(loss(params, ep)), 0.5 * float(loss(p0, ep)))


if __name__ == "__main__":
    unittest.main()
