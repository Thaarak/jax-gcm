"""Tests for the copying loop (jcm.mcb.imitation).

On the test world's toy, a privileged teacher knows each episode's hidden
spraying strength and the student does not. The last test uses the real
receding-horizon planner as that teacher.
"""

import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.imitation import (
    CopyingRecorder,
    ImitationConfig,
    ImitationData,
    fit_student,
    run_dagger,
    teacher_share,
)
from jcm.mcb.planner import Planner, PlannerConfig
from jcm.mcb.scores import area_weights
from jcm.mcb.student import (
    StudentConfig,
    StudentPolicy,
    band_observation_weights,
    init_student,
    student_apply,
)
from jcm.mcb.test_world import (
    EpisodeState,
    make_segment_fn,
    make_warming,
    run_episode,
)
from jcm.mcb.test_world_test import LATS, OCEAN, _carry, _toy_fields, _toy_step

SEG_DAYS, N_SEG = 3, 4
CFG = StudentConfig(k=2, hidden=4, cap=5.0, anomaly_scale_k=0.5)


class Spec(NamedTuple):
    efficacy: float


class _Toy(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.obs_w = band_observation_weights(LATS, self.patterns)
        self.carry = _carry()
        self.q_base = self.carry["ocn"]["forcing"].q_flux
        heat = jnp.asarray(OCEAN, jnp.float32)
        days = SEG_DAYS * N_SEG + 6
        long_seg = make_segment_fn(_toy_step, self.patterns, days,
                                   fields_fn=_toy_fields)
        _, f = long_seg(self.carry, jnp.zeros(2), jnp.asarray(1.0),
                        self.q_base, make_warming(self.carry, heat))
        self.target = np.asarray(f["sst"], np.float64)
        self.warming = make_warming(self.carry, heat, step_wm2=0.4)
        self.seg = make_segment_fn(_toy_step, self.patterns, SEG_DAYS,
                                   fields_fn=_toy_fields)

    def run_fn(self, spec, policy):
        return run_episode(self.carry, self.seg, policy, N_SEG, SEG_DAYS, 2,
                           self.warming, efficacy=spec.efficacy)

    def make_student(self, params, spec=None):
        return StudentPolicy(params, CFG, self.target, self.obs_w, SEG_DAYS)

    def make_teacher(self, spec):
        """Return a privileged rule: brighten more when warm, divided by the strength."""
        observer = self.make_student(init_student(jax.random.PRNGKey(0),
                                                  CFG))

        def teacher(state):
            bands = observer.band_anomalies(state)
            return np.clip((1.0 + 4.0 * bands) / spec.efficacy, 0.0, 5.0)

        return teacher


class DataTest(unittest.TestCase):
    def test_teacher_share(self):
        self.assertEqual(teacher_share(0, 0.0), 1.0)
        self.assertEqual(teacher_share(1, 0.0), 0.0)
        self.assertAlmostEqual(teacher_share(2, 0.5), 0.25)

    def test_save_and_load(self):
        data = ImitationData()
        data.add([1.0, 2.0], [0.1], round=0, student=[0.2])
        data.add([3.0, 4.0], [0.3], round=1, student=[0.4])
        with tempfile.TemporaryDirectory() as tmp:
            data.save(Path(tmp) / "d.npz")
            back = ImitationData.load(Path(tmp) / "d.npz")
        np.testing.assert_array_equal(back.inputs, data.inputs)
        np.testing.assert_array_equal(back.targets, data.targets)
        self.assertEqual(back.meta, data.meta)

    def test_config_checks(self):
        with self.assertRaises(ValueError):
            ImitationConfig(beta0=2.0).validate()

    def test_fit_learns_a_reachable_mapping(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, CFG.n_inputs)).astype(np.float32)
        truth = init_student(jax.random.PRNGKey(11), CFG)
        truth["w2"] = 3.0 * truth["w2"] + 1.0
        y = np.asarray(student_apply(truth, x, CFG))
        p0 = init_student(jax.random.PRNGKey(12), CFG)
        icfg = ImitationConfig(fit_steps=1500, learning_rate=2e-2,
                               weight_decay=0.0)
        _, losses = fit_student(p0, x, y, CFG, icfg)
        self.assertLess(losses[-1], 0.05 * losses[0])


class RecorderTest(_Toy):
    def test_records_every_state_and_obeys_the_mixing(self):
        params = init_student(jax.random.PRNGKey(1), CFG)
        for beta, teacher_drives in ((1.0, True), (0.0, False)):
            data = ImitationData()
            rec = CopyingRecorder(self.make_student(params),
                                  self.make_teacher(Spec(0.8)), beta,
                                  np.random.default_rng(0), data,
                                  {"round": 0})
            ep = self.run_fn(Spec(0.8), rec)
            self.assertEqual(len(data), N_SEG)
            executed = data.targets if teacher_drives else np.stack(
                [m["student"] for m in data.meta])
            np.testing.assert_allclose(ep.amplitudes, executed, rtol=1e-6)
            self.assertTrue(all(m["teacher_drove"] == teacher_drives
                                for m in data.meta))


class DaggerTest(_Toy):
    def test_the_student_learns_to_copy_the_privileged_teacher(self):
        specs = [Spec(0.7), Spec(1.0), Spec(1.4)]
        icfg = ImitationConfig(rounds=4, fit_steps=800, learning_rate=2e-2)
        params, data, history = run_dagger(
            specs, self.run_fn, self.make_teacher, self.make_student,
            init_student(jax.random.PRNGKey(2), CFG), CFG, icfg, log_fn=None)
        self.assertEqual(len(data), 4 * len(specs) * N_SEG)
        self.assertEqual([h["teacher_share"] for h in history],
                         [1.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(m["teacher_drove"] for m in data.meta
                            if m["round"] == 0))
        self.assertLess(history[-1]["student_error"],
                        0.8 * history[0]["student_error"])

    def test_a_spent_budget_stops_the_loop(self):
        from jcm.mcb.direct_training import Budget
        budget = Budget(1e-9)
        _, data, history = run_dagger(
            [Spec(1.0), Spec(0.5)], self.run_fn, self.make_teacher,
            self.make_student, init_student(jax.random.PRNGKey(3), CFG), CFG,
            ImitationConfig(rounds=3, fit_steps=10), budget=budget,
            log_fn=None)
        self.assertEqual(len(history), 1)
        self.assertEqual(len(data), N_SEG)


class PlannerTeacherTest(_Toy):
    def test_the_planner_told_the_strength_is_a_teacher(self):
        weights = area_weights(LATS, OCEAN)
        planners = []

        def make_teacher(spec):
            planner = Planner(_toy_step, self.patterns, weights, self.target,
                              self.q_base, self.warming, spec.efficacy,
                              PlannerConfig(lookahead_days=6,
                                            window_days=NO_TRUNCATION_DAYS,
                                            copies=1, copy_amp=0.0, cap=5.0,
                                            optimizer="gauss_newton",
                                            iterations=1))
            planners.append(planner)
            return planner

        _, data, history = run_dagger(
            [Spec(0.6)], self.run_fn, make_teacher, self.make_student,
            init_student(jax.random.PRNGKey(4), CFG), CFG,
            ImitationConfig(rounds=1, fit_steps=50), log_fn=None)
        self.assertEqual(len(data), N_SEG)
        finals = np.array([r["final"] for r in planners[0].log])
        np.testing.assert_allclose(data.targets, finals, rtol=1e-6)
        self.assertTrue(np.all(data.targets >= 0.0)
                        and np.all(data.targets <= 5.0))
        # Told a weaker spraying, the teacher asks for more brightening.
        strong = Planner(_toy_step, self.patterns, weights, self.target,
                         self.q_base, self.warming, 1.2,
                         planners[0].cfg)
        first = EpisodeState(0, 0, self.carry, np.zeros(2, np.float32),
                             None)
        self.assertGreater(float(np.sum(planners[0](first))),
                           float(np.sum(strong(first))))


if __name__ == "__main__":
    unittest.main()
