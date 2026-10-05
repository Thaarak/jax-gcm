"""Tests for the student network (jcm.mcb.student)."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.student import (
    StudentConfig,
    StudentPolicy,
    band_observation_weights,
    day_of_year,
    features,
    init_student,
    load_student,
    n_parameters,
    observe_bands,
    save_student,
    student_apply,
)
from jcm.mcb.test_world import EpisodeState
from jcm.mcb.test_world_test import LATS, OCEAN, _carry

CFG = StudentConfig()


class ShapeTest(unittest.TestCase):
    def test_about_a_hundred_numbers(self):
        self.assertEqual(CFG.n_inputs, 12)
        self.assertEqual(n_parameters(CFG), 113)
        params = init_student(jax.random.PRNGKey(0), CFG)
        self.assertEqual(sum(np.size(v) for v in params.values()), 113)
        self.assertEqual(StudentConfig(use_season=False).n_inputs, 10)

    def test_outputs_start_near_the_initial_fraction_and_stay_bounded(self):
        params = init_student(jax.random.PRNGKey(1), CFG)
        a0 = student_apply(params, jnp.zeros(CFG.n_inputs), CFG)
        np.testing.assert_allclose(a0, CFG.init_fraction * CFG.cap,
                                   rtol=1e-6)
        x = 10.0 * jax.random.normal(jax.random.PRNGKey(2),
                                     (50, CFG.n_inputs))
        a = np.asarray(student_apply(params, x, CFG))
        self.assertEqual(a.shape, (50, CFG.k))
        self.assertTrue(np.all(a > 0.0) and np.all(a < CFG.cap))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            init_student(jax.random.PRNGKey(0), StudentConfig(init_fraction=1))


class FeatureTest(unittest.TestCase):
    def test_band_weights_average_a_uniform_anomaly(self):
        patterns = gaussian_band_patterns(LATS, OCEAN, centers_deg=(30.0,
                                                                   -30.0),
                                          width_deg=15.0)
        w = band_observation_weights(LATS, patterns)
        np.testing.assert_allclose(w.sum(axis=(1, 2)), 1.0)
        bands = observe_bands(jnp.full((4, 3), 0.3), w)
        np.testing.assert_allclose(bands, 0.3, rtol=1e-6)

    def test_feature_layout(self):
        x = features(jnp.array([0.1, -0.2, 0.0, 0.05, 0.3]), 91.3125,
                     jnp.full(5, 0.03), CFG)
        np.testing.assert_allclose(x[:5], [1.0, -2.0, 0.0, 0.5, 3.0],
                                   rtol=1e-6)
        np.testing.assert_allclose(x[5:7], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(x[7:], 0.2, rtol=1e-6)

    def test_day_of_year_wraps(self):
        self.assertAlmostEqual(float(day_of_year(400 * 86400.0, CFG)),
                               400.0 - 365.25, places=3)


class SaveLoadTest(unittest.TestCase):
    def test_round_trip(self):
        params = init_student(jax.random.PRNGKey(3), CFG)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.npz"
            save_student(path, params, CFG, method="dagger", round=2)
            loaded, cfg, meta = load_student(path)
        self.assertEqual(cfg, CFG)
        self.assertEqual(meta, {"method": "dagger", "round": 2})
        for k in params:
            np.testing.assert_array_equal(np.asarray(params[k]),
                                          np.asarray(loaded[k]))


class PolicyTest(unittest.TestCase):
    def test_observes_the_last_segment_against_the_same_days(self):
        cfg = StudentConfig(k=2, cap=5.0)
        patterns = gaussian_band_patterns(LATS, OCEAN, centers_deg=(30.0,
                                                                   -30.0),
                                          width_deg=15.0)
        w = band_observation_weights(LATS, patterns)
        target = 290.0 + 0.01 * np.arange(12)[:, None, None] + np.zeros(
            (12, 4, 3))
        params = init_student(jax.random.PRNGKey(4), cfg)
        policy = StudentPolicy(params, cfg, target, w, segment_days=4)
        carry = _carry()
        first = EpisodeState(segment=0, day=0, carry=carry,
                             previous=np.zeros(2), last_fields=None)
        x0 = policy.observation(first)
        np.testing.assert_allclose(x0[:2], 0.0)
        anomaly = np.random.default_rng(5).normal(size=(4, 3)) * 0.1
        later = EpisodeState(segment=2, day=8, carry=carry,
                             previous=np.array([1.0, 2.0]),
                             last_fields={"sst": target[4:8] + anomaly})
        x = policy.observation(later)
        expected = np.tensordot(w, anomaly, axes=([1, 2], [0, 1])) / 0.1
        np.testing.assert_allclose(x[:2], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(x[-2:], [0.2, 0.4], rtol=1e-6)
        a = policy(later)
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
