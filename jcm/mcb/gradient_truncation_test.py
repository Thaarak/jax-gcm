"""Tests for atmosphere-truncated gradients (jcm.mcb.gradient_truncation).

The toy system copies the coupler's real data flow for the workflow
["coupling", "atm", "ocn"]:

  coupling: ocn.forcing.flux <- atm.derived.flux   (the PREVIOUS day's flux)
            atm.forcing.sst  <- ocn.state.sst
  atm:      x' = r*x + u + c*sst_atm               (r > 1: chaos-like growth)
            derived.flux = x' + d*u                (direct actuator path)
  ocn:      sst' = sst + ocn.forcing.flux ; sim_time += dt

The actuator ``u`` lives in ``atm.derived.mcb_perturbation`` exactly as in
jax-esm's JCM wrapper. JAX's reverse-mode gradient of the final SST is checked
against an independent forward-mode tangent recursion that applies the
intended cut by hand (tangent of x zeroed at cut steps; the tangents of
derived flux, atm-forcing SST and the ocean untouched). The damped variant
is checked the same way (tangent of x multiplied by the decay factor at the
start of every step).
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.gradient_truncation import (
    NO_TRUNCATION_DAYS,
    atmosphere_decay_factor,
    damp_atmosphere_gradient,
    scale_tangent,
    stop_atmosphere_gradient,
    truncation_cut,
    wrap_step_fn_with_atm_truncation,
)

DT = 86400.0
R, C, D = 1.5, 0.3, 0.7
T_STEPS = 10
START_DAY = 5


@jax.tree_util.register_pytree_node_class
class _FakeOceanState:
    """Minimal stand-in for the slab OceanState (sim_time + SST + copy)."""

    def __init__(self, sim_time, sst):
        self.sim_time = sim_time
        self.sea_surface_temperature = sst

    def copy(self, updates):
        return _FakeOceanState(
            updates.get("sim_time", self.sim_time),
            updates.get("sea_surface_temperature",
                        self.sea_surface_temperature),
        )

    def tree_flatten(self):
        return (self.sim_time, self.sea_surface_temperature), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _toy_step_fn(carry, step_idx):
    """One coupling step of the toy system (see module docstring)."""
    atm, ocn = dict(carry["atm"]), dict(carry["ocn"])
    # coupling mapper
    ocn_forcing = {"flux": atm["derived"]["flux"]}
    atm_forcing = {"sst": ocn["state"].sea_surface_temperature}
    # atmosphere step
    u = atm["derived"]["mcb_perturbation"]
    x_new = R * atm["state"]["x"] + u + C * atm_forcing["sst"]
    derived = {"flux": x_new + D * u, "mcb_perturbation": u}
    new_atm = {"state": {"x": x_new}, "derived": derived,
               "forcing": atm_forcing}
    # ocean step
    state = ocn["state"]
    new_state = state.copy({
        "sea_surface_temperature": (state.sea_surface_temperature
                                    + ocn_forcing["flux"]),
        "sim_time": state.sim_time + DT,
    })
    new_ocn = {"state": new_state, "forcing": ocn_forcing}
    return {"atm": new_atm, "ocn": new_ocn}, None


def _initial_carry(u, start_day=START_DAY):
    return {
        "atm": {"state": {"x": jnp.asarray(0.2)},
                "derived": {"flux": jnp.asarray(0.0),
                            "mcb_perturbation": u},
                "forcing": {"sst": jnp.asarray(0.0)}},
        "ocn": {"state": _FakeOceanState(jnp.asarray(start_day * DT),
                                         jnp.asarray(0.0)),
                "forcing": {"flux": jnp.asarray(0.0)}},
    }


def _rollout(u, step_fn, n_steps=T_STEPS, start_day=START_DAY):
    carry = _initial_carry(u, start_day)

    def body(c, i):
        return step_fn(c, i)

    final, _ = jax.lax.scan(body, carry, jnp.arange(n_steps))
    return final


def _final_sst(u, window, t0=None, start_day=START_DAY, n_steps=T_STEPS,
               decay=None):
    step_fn = wrap_step_fn_with_atm_truncation(_toy_step_fn, window,
                                               t0_seconds=t0,
                                               atm_decay=decay)
    final = _rollout(u, step_fn, n_steps, start_day)
    return final["ocn"]["state"].sea_surface_temperature


def _reference_gradient(window, aligned_to_episode, start_day=START_DAY,
                        n_steps=T_STEPS, decay=None):
    """Forward-mode tangent recursion with the cut applied by hand (NumPy)."""
    dx = dflux_derived = dsst = dflux_ocn = dsst_atm = 0.0
    for t in range(n_steps):
        day = t if aligned_to_episode else start_day + t
        if decay is not None:
            dx = decay * dx               # damping: ONLY the atmospheric state
        if window is not None and day % window == 0:
            dx = 0.0                      # cut: ONLY the atmospheric state
        dflux_ocn = dflux_derived          # coupling mapper
        dsst_atm = dsst
        dx = R * dx + 1.0 + C * dsst_atm   # atm step (du/du = 1)
        dflux_derived = dx + D * 1.0
        dsst = dsst + dflux_ocn            # ocean step
    return dsst


class TruncationCutTest(unittest.TestCase):
    def test_absolute_alignment(self):
        cuts = [bool(truncation_cut(d * DT, 3)) for d in range(7)]
        self.assertEqual(cuts, [True, False, False, True, False, False, True])

    def test_episode_alignment(self):
        t0 = 5 * DT
        cuts = [bool(truncation_cut((5 + d) * DT, 3, t0_seconds=t0))
                for d in range(4)]
        self.assertEqual(cuts, [True, False, False, True])

    def test_window_one_cuts_every_day(self):
        self.assertTrue(all(bool(truncation_cut(d * DT, 1))
                            for d in range(5)))

    def test_rounds_float_clock(self):
        # A clock a fraction of a second off still lands on the right day.
        self.assertTrue(bool(truncation_cut(3 * DT + 0.4, 3)))
        self.assertTrue(bool(truncation_cut(3 * DT - 0.4, 3)))
        self.assertFalse(bool(truncation_cut(4 * DT - 0.4, 3)))


class ToyGradientTest(unittest.TestCase):
    def test_no_window_is_full_bptt(self):
        g_none = float(jax.grad(_final_sst)(0.1, None))
        g_plain = float(jax.grad(
            lambda u: _rollout(u, _toy_step_fn)["ocn"]["state"]
            .sea_surface_temperature)(0.1))
        self.assertAlmostEqual(g_none, g_plain, places=10)
        self.assertAlmostEqual(g_none, _reference_gradient(None, False),
                               places=4)

    def test_matches_reference_for_every_window(self):
        for window in (1, 2, 3, 4, 7, 10, 100):
            for episode in (False, True):
                t0 = START_DAY * DT if episode else None
                g = float(jax.grad(_final_sst)(0.1, window, t0))
                ref = _reference_gradient(window, episode)
                self.assertAlmostEqual(
                    g, ref, delta=1e-4 * max(1.0, abs(ref)),
                    msg=f"window={window} episode={episode}")

    def test_truncation_shrinks_the_amplified_gradient(self):
        full = abs(float(jax.grad(_final_sst)(0.1, None)))
        w1 = abs(float(jax.grad(_final_sst)(0.1, 1)))
        w3 = abs(float(jax.grad(_final_sst)(0.1, 3)))
        # r > 1 makes long atmospheric chains dominate; cutting them shrinks
        # the gradient monotonically, but the direct path keeps it non-zero.
        self.assertGreater(full, w3)
        self.assertGreater(w3, w1)
        self.assertGreater(w1, 0.0)

    def test_window_longer_than_rollout_equals_full(self):
        g_full = float(jax.grad(_final_sst)(0.1, None))
        g_long = float(jax.grad(_final_sst)(0.1, T_STEPS + 1,
                                            START_DAY * DT))
        self.assertAlmostEqual(g_full, g_long, places=10)

    def test_forward_values_are_bit_identical(self):
        plain = _rollout(0.1, _toy_step_fn)
        for window in (1, 3, None):
            wrapped = _rollout(
                0.1, wrap_step_fn_with_atm_truncation(_toy_step_fn, window))
            for a, b in zip(jax.tree.leaves(plain),
                            jax.tree.leaves(wrapped)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_traced_window_and_t0_compile_once(self):
        @jax.jit
        def grad_fn(u, window, t0):
            return jax.grad(_final_sst)(u, window, t0)

        t0 = jnp.asarray(START_DAY * DT)
        for window in (1, 2, 3, 7):
            traced = float(grad_fn(0.1, jnp.asarray(window), t0))
            ref = _reference_gradient(window, True)
            self.assertAlmostEqual(traced, ref,
                                   delta=1e-4 * max(1.0, abs(ref)))
        no_trunc = float(grad_fn(0.1, jnp.asarray(NO_TRUNCATION_DAYS), t0))
        self.assertAlmostEqual(no_trunc, _reference_gradient(None, True),
                               delta=1e-4 * abs(_reference_gradient(None,
                                                                    True)))

    def test_cut_follows_ocean_clock_not_scan_index(self):
        # Two 5-step "control intervals" (scan index restarts at 0) must give
        # the same gradient as one 10-step scan.
        def two_intervals(u, window):
            step_fn = wrap_step_fn_with_atm_truncation(_toy_step_fn, window)
            carry = _initial_carry(u)
            for _ in range(2):
                carry, _ = jax.lax.scan(step_fn, carry, jnp.arange(5))
            return carry["ocn"]["state"].sea_surface_temperature

        for window in (2, 3):
            one = float(jax.grad(_final_sst)(0.1, window))
            two = float(jax.grad(two_intervals)(0.1, window))
            # Same maths, different scan structure: float32 round-off only.
            self.assertAlmostEqual(one, two, delta=1e-6 * abs(one))


class StopAtmosphereGradientScopeTest(unittest.TestCase):
    def test_only_the_atmospheric_state_is_cut(self):
        carry = _initial_carry(jnp.asarray(0.1))

        def total(c, cut):
            out = stop_atmosphere_gradient(c, cut)
            return (out["atm"]["state"]["x"]
                    + out["atm"]["derived"]["flux"]
                    + out["atm"]["derived"]["mcb_perturbation"]
                    + out["atm"]["forcing"]["sst"]
                    + out["ocn"]["state"].sea_surface_temperature)

        g_cut = jax.grad(total)(carry, True)
        g_open = jax.grad(total)(carry, False)
        self.assertEqual(float(g_cut["atm"]["state"]["x"]), 0.0)
        self.assertEqual(float(g_open["atm"]["state"]["x"]), 1.0)
        for grads in (g_cut, g_open):
            self.assertEqual(float(grads["atm"]["derived"]["flux"]), 1.0)
            self.assertEqual(
                float(grads["atm"]["derived"]["mcb_perturbation"]), 1.0)
            self.assertEqual(float(grads["atm"]["forcing"]["sst"]), 1.0)
            self.assertEqual(
                float(grads["ocn"]["state"].sea_surface_temperature), 1.0)

    def test_does_not_mutate_input_carry(self):
        carry = _initial_carry(jnp.asarray(0.1))
        before = carry["atm"]["state"]
        stop_atmosphere_gradient(carry, True)
        self.assertIs(carry["atm"]["state"], before)

    def test_none_and_nonpositive_window_return_step_fn_unchanged(self):
        self.assertIs(wrap_step_fn_with_atm_truncation(_toy_step_fn, None),
                      _toy_step_fn)
        self.assertIs(wrap_step_fn_with_atm_truncation(_toy_step_fn, 0),
                      _toy_step_fn)
        # A decay alone still needs the wrapper.
        self.assertIsNot(wrap_step_fn_with_atm_truncation(
            _toy_step_fn, None, atm_decay=0.5), _toy_step_fn)


class DampedGradientTest(unittest.TestCase):
    """The exploratory damped estimator (Amendment 9 revision 0.1)."""

    def test_decay_factor(self):
        self.assertAlmostEqual(atmosphere_decay_factor(7.0),
                               math.exp(-1.0 / 7.0), places=12)
        self.assertAlmostEqual(atmosphere_decay_factor(3.0),
                               math.exp(-1.0 / 3.0), places=12)
        for bad in (0.0, -2.0):
            with self.assertRaises(ValueError):
                atmosphere_decay_factor(bad)

    def test_scale_tangent_is_identity_on_values(self):
        x = jnp.array([1.0, -2.0, 3.5])
        np.testing.assert_array_equal(np.asarray(scale_tangent(x, 0.25)),
                                      np.asarray(x))
        _, tangent = jax.jvp(lambda v: scale_tangent(v, 0.25), (x,),
                             (jnp.ones(3),))
        np.testing.assert_allclose(np.asarray(tangent), 0.25)
        _, vjp = jax.vjp(lambda v: scale_tangent(v, 0.25), x)
        np.testing.assert_allclose(np.asarray(vjp(jnp.ones(3))[0]), 0.25)

    def test_matches_reference_for_every_decay_and_window(self):
        for decay in (0.0, 0.25, 0.6, 0.9, 1.0):
            for window in (None, 3):
                g = float(jax.grad(_final_sst)(0.1, window, decay=decay))
                ref = _reference_gradient(window, False, decay=decay)
                self.assertAlmostEqual(
                    g, ref, delta=1e-4 * max(1.0, abs(ref)),
                    msg=f"decay={decay} window={window}")

    def test_decay_one_changes_nothing(self):
        for window in (None, 3):
            plain = float(jax.grad(_final_sst)(0.1, window))
            damped = float(jax.grad(_final_sst)(0.1, window, decay=1.0))
            self.assertEqual(plain, damped)

    def test_decay_zero_equals_window_one(self):
        zero = float(jax.grad(_final_sst)(0.1, None, decay=0.0))
        w1 = float(jax.grad(_final_sst)(0.1, 1))
        self.assertAlmostEqual(zero, w1, delta=1e-6 * max(1.0, abs(w1)))

    def test_damping_sits_between_full_and_window_one(self):
        full = abs(float(jax.grad(_final_sst)(0.1, None)))
        damped = abs(float(jax.grad(_final_sst)(0.1, None, decay=0.6)))
        w1 = abs(float(jax.grad(_final_sst)(0.1, 1)))
        self.assertGreater(full, damped)
        self.assertGreater(damped, w1)

    def test_forward_values_are_bit_identical(self):
        plain = _rollout(0.1, _toy_step_fn)
        for window in (None, 3):
            wrapped = _rollout(0.1, wrap_step_fn_with_atm_truncation(
                _toy_step_fn, window, atm_decay=0.5))
            for a, b in zip(jax.tree.leaves(plain),
                            jax.tree.leaves(wrapped)):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_forward_mode_agrees_with_reverse_mode(self):
        # The planner may use forward-mode Jacobians; both modes must damp.
        for window, decay in ((None, 0.6), (3, 0.6), (3, None)):
            rev = float(jax.grad(_final_sst)(0.1, window, decay=decay))
            fwd = float(jax.jacfwd(_final_sst)(0.1, window, decay=decay))
            self.assertAlmostEqual(rev, fwd, delta=1e-5 * max(1.0, abs(rev)))

    def test_traced_decay_compiles_once(self):
        @jax.jit
        def grad_fn(u, decay):
            return jax.grad(_final_sst)(u, jnp.asarray(NO_TRUNCATION_DAYS),
                                        decay=decay)

        for decay in (0.0, 0.5, 0.9):
            traced = float(grad_fn(0.1, jnp.asarray(decay)))
            ref = _reference_gradient(None, False, decay=decay)
            self.assertAlmostEqual(traced, ref,
                                   delta=1e-4 * max(1.0, abs(ref)))

    def test_only_float_leaves_of_the_atmospheric_state_are_damped(self):
        carry = _initial_carry(jnp.asarray(0.1))
        carry["atm"]["state"]["step"] = jnp.asarray(3, dtype=jnp.int32)

        def total(c):
            out = damp_atmosphere_gradient(c, 0.5)
            return (out["atm"]["state"]["x"]
                    + out["atm"]["derived"]["flux"]
                    + out["ocn"]["state"].sea_surface_temperature
                    + out["atm"]["state"]["step"].astype(jnp.float32))

        grads = jax.grad(total, allow_int=True)(carry)
        self.assertEqual(float(grads["atm"]["state"]["x"]), 0.5)
        self.assertEqual(float(grads["atm"]["derived"]["flux"]), 1.0)
        self.assertEqual(
            float(grads["ocn"]["state"].sea_surface_temperature), 1.0)


if __name__ == "__main__":
    unittest.main()
