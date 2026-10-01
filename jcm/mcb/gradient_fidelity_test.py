"""Tests for the Experiment 1 core (jcm.mcb.gradient_fidelity).

A small gridded toy with the real carry layout (atm state / derived fluxes /
actuator slot, slab ocean with sim_time, slab land) and the coupler's data
flow (previous-day flux handed to the ocean, then atmosphere, ocean, land).
The toy is LINEAR in the actuator, so central finite differences are exact
and must equal the full-BPTT Jacobian.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.band_basis import (
    gaussian_band_patterns,
    objective_weights,
    stack_objective_weights,
)
from jcm.mcb.gradient_fidelity import (
    apply_band_control,
    make_jacobian_fn,
    make_objective_fn,
    make_series_fn,
    rollout_objective_series,
    tail_mean,
)
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS

DT = 86400.0
IX, IL = 4, 3
LATS = np.deg2rad(np.array([-30.0, 0.0, 30.0]))
OCEAN = np.ones((IX, IL))
OCEAN[0, :] = 0.0
LAND = 1.0 - OCEAN
R_T, C_T, D_T, K_T = 0.6, 0.2, 0.8, 0.05
HORIZON, TAIL = 8, 3


@jax.tree_util.register_pytree_node_class
class _Ocean:
    def __init__(self, sim_time, sst):
        self.sim_time = sim_time
        self.sea_surface_temperature = sst

    def tree_flatten(self):
        return (self.sim_time, self.sea_surface_temperature), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class _Land:
    def __init__(self, temperature):
        self.land_surface_temperature = temperature

    def tree_flatten(self):
        return (self.land_surface_temperature,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _toy_step(carry, step_idx):
    atm, ocn, lnd = carry["atm"], carry["ocn"], carry["lnd"]
    ocn_flux = atm["derived"]["total_heat_flux"]        # coupling mapper
    lnd_flux = atm["derived"]["land_heat_flux"]
    sst_atm = ocn["state"].sea_surface_temperature
    m = atm["derived"]["mcb_perturbation"]
    x_new = R_T * atm["state"]["x"] + m + C_T * (sst_atm - 290.0)
    derived = {"total_heat_flux": x_new + D_T * m,
               "land_heat_flux": jnp.mean(x_new) * jnp.ones_like(x_new),
               "mcb_perturbation": m}
    new_ocn = _Ocean(ocn["state"].sim_time + DT,
                     ocn["state"].sea_surface_temperature - K_T * ocn_flux)
    new_lnd = _Land(lnd["state"].land_surface_temperature - K_T * lnd_flux)
    return {"atm": {"state": {"x": x_new}, "derived": derived,
                    "forcing": {"sst": sst_atm}},
            "ocn": {"state": new_ocn}, "lnd": {"state": new_lnd}}, None


def _carry(start_day=3):
    z = jnp.zeros((IX, IL))
    return {"atm": {"state": {"x": z + 0.1},
                    "derived": {"total_heat_flux": z, "land_heat_flux": z,
                                "mcb_perturbation": z},
                    "forcing": {"sst": z + 290.0}},
            "ocn": {"state": _Ocean(jnp.asarray(start_day * DT), z + 290.0)},
            "lnd": {"state": _Land(z + 280.0)}}


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.patterns = gaussian_band_patterns(LATS, OCEAN,
                                               centers_deg=(30.0, -30.0),
                                               width_deg=15.0)
        self.weights = stack_objective_weights(
            objective_weights(LATS, OCEAN, LAND))
        self.a0 = jnp.array([0.03, 0.03])
        self.carry = _carry()
        self.t0 = self.carry["ocn"]["state"].sim_time


class SeriesAndTailTest(_Fixture):
    def test_series_shape_and_tail_mean(self):
        series = make_series_fn(_toy_step, self.patterns, self.weights,
                                HORIZON)(self.a0, self.carry)
        self.assertEqual(series.shape, (HORIZON, 4))
        np.testing.assert_allclose(
            np.asarray(tail_mean(series, 6, 2)),
            np.asarray(series)[4:6].mean(axis=0), rtol=1e-6)

    def test_tail_mean_rejects_bad_windows(self):
        series = jnp.zeros((5, 4))
        for horizon, tail in ((6, 2), (3, 4), (3, 0)):
            with self.assertRaises(ValueError):
                tail_mean(series, horizon, tail)

    def test_rollout_matches_manual_scan(self):
        _, series = rollout_objective_series(
            apply_band_control(self.carry, self.a0, self.patterns),
            _toy_step, 4, self.weights)
        c = apply_band_control(self.carry, self.a0, self.patterns)
        for s in range(4):
            c, _ = _toy_step(c, s)
        sst = c["ocn"]["state"].sea_surface_temperature
        # objective_values reports T0 relative to its 288 K reference.
        t0_manual = float(jnp.sum(self.weights[0] * (sst - 288.0)))
        self.assertAlmostEqual(float(series[3, 0]), t0_manual, places=4)


class ApplyBandControlTest(_Fixture):
    def test_places_field_in_actuator_slot_without_mutation(self):
        before = self.carry["atm"]["derived"]["mcb_perturbation"]
        out = apply_band_control(self.carry, self.a0, self.patterns)
        self.assertIs(self.carry["atm"]["derived"]["mcb_perturbation"], before)
        np.testing.assert_allclose(
            np.asarray(out["atm"]["derived"]["mcb_perturbation"]),
            np.asarray(jnp.tensordot(self.a0, self.patterns, axes=1)))


class JacobianTest(_Fixture):
    def _central_fd(self, delta=5e-2):
        series_fn = make_series_fn(_toy_step, self.patterns, self.weights,
                                   HORIZON)
        cols = []
        for k in range(self.a0.size):
            e = jnp.zeros_like(self.a0).at[k].set(delta)
            plus = tail_mean(series_fn(self.a0 + e, self.carry), HORIZON,
                             TAIL)
            minus = tail_mean(series_fn(self.a0 - e, self.carry), HORIZON,
                              TAIL)
            cols.append((plus - minus) / (2 * delta))
        return np.stack([np.asarray(c) for c in cols], axis=1)

    def test_full_bptt_matches_finite_differences_on_linear_toy(self):
        jac = make_jacobian_fn(_toy_step, self.patterns, self.weights,
                               HORIZON, TAIL)
        full = np.asarray(jac(self.a0, self.carry,
                              jnp.asarray(NO_TRUNCATION_DAYS), self.t0))
        np.testing.assert_allclose(full, self._central_fd(), rtol=2e-3,
                                   atol=2e-6)

    def test_window_longer_than_horizon_equals_full(self):
        jac = make_jacobian_fn(_toy_step, self.patterns, self.weights,
                               HORIZON, TAIL)
        full = np.asarray(jac(self.a0, self.carry,
                              jnp.asarray(NO_TRUNCATION_DAYS), self.t0))
        long = np.asarray(jac(self.a0, self.carry,
                              jnp.asarray(HORIZON + 1), self.t0))
        np.testing.assert_allclose(long, full, rtol=1e-6)

    def test_short_window_changes_the_gradient_but_not_its_sign(self):
        jac = make_jacobian_fn(_toy_step, self.patterns, self.weights,
                               HORIZON, TAIL)
        full = np.asarray(jac(self.a0, self.carry,
                              jnp.asarray(NO_TRUNCATION_DAYS), self.t0))
        w1 = np.asarray(jac(self.a0, self.carry, jnp.asarray(1), self.t0))
        self.assertFalse(np.allclose(w1[0], full[0]))
        # T0 row: brightening cools the ocean either way.
        self.assertTrue(np.all(np.sign(w1[0]) == np.sign(full[0])))

    def test_objective_forward_value_independent_of_window(self):
        objective = make_objective_fn(_toy_step, self.patterns, self.weights,
                                      HORIZON, TAIL)
        values = [np.asarray(objective(self.a0, self.carry, w, self.t0))
                  for w in (jnp.asarray(1), jnp.asarray(3),
                            jnp.asarray(NO_TRUNCATION_DAYS))]
        values.append(np.asarray(objective(
            self.a0, self.carry, jnp.asarray(NO_TRUNCATION_DAYS), self.t0,
            jnp.asarray(0.5))))
        for v in values[1:]:
            np.testing.assert_array_equal(v, values[0])


class DampedJacobianTest(_Fixture):
    """Exploratory damped estimator (Amendment 9 revision 0.1)."""

    def _jac(self, window, decay=None):
        jac = make_jacobian_fn(_toy_step, self.patterns, self.weights,
                               HORIZON, TAIL)
        args = (self.a0, self.carry, jnp.asarray(window), self.t0)
        if decay is not None:
            args += (jnp.asarray(decay, dtype=jnp.float32),)
        return np.asarray(jac(*args))

    def test_decay_one_equals_the_registered_estimators(self):
        for window in (1, 3, NO_TRUNCATION_DAYS):
            np.testing.assert_array_equal(self._jac(window, 1.0),
                                          self._jac(window))

    def test_decay_zero_equals_window_one(self):
        np.testing.assert_allclose(self._jac(NO_TRUNCATION_DAYS, 0.0),
                                   self._jac(1), rtol=1e-6, atol=1e-9)

    def test_partial_decay_differs_from_both_ends(self):
        damped = self._jac(NO_TRUNCATION_DAYS, 0.5)
        self.assertFalse(np.allclose(damped[0], self._jac(1)[0]))
        self.assertFalse(np.allclose(damped[0],
                                     self._jac(NO_TRUNCATION_DAYS)[0]))


if __name__ == "__main__":
    unittest.main()
