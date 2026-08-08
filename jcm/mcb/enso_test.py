"""Tests for the ENSO pacemaker (jcm.mcb.enso)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.enso import (
    EnsoConfig,
    box_mean_weights,
    enso_amplitude,
    nino_pattern,
    wrap_step_fn_with_enso,
)
from jcm.physics.speedy.speedy_coords import get_speedy_coords


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


def _fake_step_fn(timestep=86400.0):
    """Ocean-only step: advances sim_time, leaves SST untouched."""
    def step_fn(carry, step_idx):
        state = carry["ocn"]["state"]
        new_state = state.copy({"sim_time": state.sim_time + timestep})
        return {"ocn": {"state": new_state}}, None
    return step_fn


def _carry(sst, t0=0.0):
    return {"ocn": {"state": _FakeOceanState(jnp.asarray(t0),
                                             jnp.asarray(sst))}}


class NinoPatternTest(unittest.TestCase):
    def setUp(self):
        self.grid = get_speedy_coords().horizontal

    def test_core_is_one_far_field_is_zero(self):
        pat = np.asarray(nino_pattern(self.grid))
        lats = np.rad2deg(np.asarray(self.grid.latitudes))
        lons = np.rad2deg(np.asarray(self.grid.longitudes))
        ix = int(np.argmin(np.abs(lons - 215.0)))   # box centre lon
        il = int(np.argmin(np.abs(lats - 0.0)))     # equator
        self.assertGreater(pat[ix, il], 0.99)
        ix_atl = int(np.argmin(np.abs(lons - 330.0)))  # Atlantic
        self.assertEqual(pat[ix_atl, il], 0.0)
        il_np = int(np.argmin(np.abs(lats - 60.0)))    # extratropics
        self.assertEqual(pat[ix, il_np], 0.0)

    def test_taper_is_monotone_and_bounded(self):
        pat = np.asarray(nino_pattern(self.grid))
        self.assertTrue(np.all(pat >= 0.0) and np.all(pat <= 1.0))
        lats = np.rad2deg(np.asarray(self.grid.latitudes))
        il = int(np.argmin(np.abs(lats - 0.0)))
        lons = np.rad2deg(np.asarray(self.grid.longitudes))
        order = np.argsort(lons)
        row = pat[order, il]
        # Along the equator the pattern must rise and fall exactly once.
        signs = np.sign(np.diff(row))
        changes = np.sum(np.abs(np.diff(signs[signs != 0])) > 0)
        self.assertLessEqual(changes, 2)

    def test_wraparound_box(self):
        cfg = EnsoConfig(lon_bounds=(350.0, 20.0))
        pat = np.asarray(nino_pattern(self.grid, cfg))
        lats = np.rad2deg(np.asarray(self.grid.latitudes))
        lons = np.rad2deg(np.asarray(self.grid.longitudes))
        il = int(np.argmin(np.abs(lats - 0.0)))
        ix0 = int(np.argmin(np.abs(lons - 0.0)))
        ix180 = int(np.argmin(np.abs(lons - 180.0)))
        self.assertGreater(pat[ix0, il], 0.99)
        self.assertEqual(pat[ix180, il], 0.0)


class EnsoAmplitudeTest(unittest.TestCase):
    def test_ramp_then_constant(self):
        cfg = EnsoConfig(amplitude=2.0, ramp_days=30.0, period_days=0.0)
        day = 86400.0
        self.assertAlmostEqual(
            float(enso_amplitude(15 * day, 0.0, cfg)), 1.0, places=6)
        self.assertAlmostEqual(
            float(enso_amplitude(30 * day, 0.0, cfg)), 2.0, places=6)
        self.assertAlmostEqual(
            float(enso_amplitude(300 * day, 0.0, cfg)), 2.0, places=6)

    def test_sinusoid_uses_sim_time_offset(self):
        cfg = EnsoConfig(amplitude=2.0, ramp_days=0.001, period_days=360.0,
                         phase0=jnp.pi / 2)  # cosine
        day = 86400.0
        t0 = 5000 * day  # arbitrary absolute origin — only t - t0 matters
        self.assertAlmostEqual(
            float(enso_amplitude(t0 + 0 * day, t0, cfg)), 0.0, places=3)
        a_quarter = float(enso_amplitude(t0 + 90 * day, t0, cfg))
        self.assertAlmostEqual(a_quarter, 0.0, places=3)
        a_peak = float(enso_amplitude(t0 + 1 * day, t0, cfg))
        self.assertGreater(a_peak, 1.9)


class WrapStepFnTest(unittest.TestCase):
    def setUp(self):
        self.grid = get_speedy_coords().horizontal
        self.pattern = nino_pattern(self.grid)
        self.ref = jnp.full(self.pattern.shape, 300.0)

    def test_relaxation_pins_commanded_anomaly(self):
        cfg = EnsoConfig(amplitude=2.0, ramp_days=0.001, relax_tau_days=5.0)
        step = wrap_step_fn_with_enso(
            _fake_step_fn(), self.pattern, cfg, self.ref, t0_seconds=0.0)
        carry = _carry(self.ref)
        for i in range(40):  # 40 days >> 5-day tau
            carry, _ = step(carry, i)
        sst = np.asarray(carry["ocn"]["state"].sea_surface_temperature)
        core = np.asarray(self.pattern) >= 0.999
        anom = sst - 300.0
        np.testing.assert_allclose(anom[core], 2.0, atol=0.02)
        far = np.asarray(self.pattern) == 0.0
        np.testing.assert_allclose(anom[far], 0.0, atol=1e-6)

    def test_phase_survives_scan_index_reset(self):
        # Two consecutive inner scans, each restarting step_idx at 0 (the
        # control-interval structure). Amplitude must keep ramping because
        # it reads sim_time, not the index.
        cfg = EnsoConfig(amplitude=2.0, ramp_days=40.0, relax_tau_days=0.01)
        step = wrap_step_fn_with_enso(
            _fake_step_fn(), self.pattern, cfg, self.ref, t0_seconds=0.0)
        carry = _carry(self.ref)

        def scan_interval(carry):
            def body(c, idx):
                new_c, _ = step(c, idx)
                return new_c, None
            out, _ = jax.lax.scan(body, carry, jnp.arange(10))
            return out

        carry = scan_interval(carry)   # days 1..10
        carry = scan_interval(carry)   # days 11..20 (idx resets to 0..9)
        sst = np.asarray(carry["ocn"]["state"].sea_surface_temperature)
        core = np.asarray(self.pattern) >= 0.999
        anom = float(np.mean(sst[core]) - 300.0)
        # With near-instant relaxation the anomaly equals A(day 20) = 1.0,
        # NOT A(day 10) = 0.5 (which an index-reset bug would produce).
        self.assertAlmostEqual(anom, 1.0, places=2)

    def test_trajectory_reference_indexed_by_day(self):
        cfg = EnsoConfig(amplitude=0.0, ramp_days=0.001,
                         relax_tau_days=0.01)
        ndays = 5
        ref_traj = jnp.stack(
            [self.ref + float(d) for d in range(ndays + 1)])
        step = wrap_step_fn_with_enso(
            _fake_step_fn(), self.pattern, cfg, ref_traj, t0_seconds=0.0)
        carry = _carry(self.ref)
        for i in range(3):
            carry, _ = step(carry, i)
        sst = np.asarray(carry["ocn"]["state"].sea_surface_temperature)
        core = np.asarray(self.pattern) >= 0.999
        # After 3 steps the reference is ref+3; near-instant relaxation
        # pins the core to it (amplitude 0).
        np.testing.assert_allclose(sst[core], 303.0, atol=0.01)

    def test_identical_wrapper_for_arm_and_baseline_cancels(self):
        cfg = EnsoConfig(amplitude=2.0, ramp_days=0.001, relax_tau_days=5.0)
        step = wrap_step_fn_with_enso(
            _fake_step_fn(), self.pattern, cfg, self.ref, t0_seconds=0.0)
        c1, c2 = _carry(self.ref), _carry(self.ref)
        for i in range(20):
            c1, _ = step(c1, i)
            c2, _ = step(c2, i)
        d = np.asarray(c1["ocn"]["state"].sea_surface_temperature
                       - c2["ocn"]["state"].sea_surface_temperature)
        np.testing.assert_allclose(d, 0.0, atol=0.0)

    def test_jit_and_grad_safe(self):
        cfg = EnsoConfig(amplitude=2.0, ramp_days=10.0, relax_tau_days=5.0)
        step = wrap_step_fn_with_enso(
            _fake_step_fn(), self.pattern, cfg, self.ref, t0_seconds=0.0)

        @jax.jit
        def run(sst0):
            carry = {"ocn": {"state": _FakeOceanState(jnp.asarray(0.0),
                                                      sst0)}}
            def body(c, idx):
                new_c, _ = step(c, idx)
                return new_c, None
            out, _ = jax.lax.scan(body, carry, jnp.arange(5))
            return jnp.mean(out["ocn"]["state"].sea_surface_temperature)

        g = jax.grad(run)(self.ref)
        self.assertFalse(bool(jnp.any(jnp.isnan(g))))


class BoxMeanWeightsTest(unittest.TestCase):
    def test_weights_normalized_and_core_only(self):
        grid = get_speedy_coords().horizontal
        pattern = nino_pattern(grid)
        area = jnp.ones(pattern.shape)
        w = box_mean_weights(grid, pattern, area)
        self.assertAlmostEqual(float(jnp.sum(w)), 1.0, places=5)
        self.assertTrue(bool(jnp.all(
            (w == 0.0) | (pattern >= 0.999))))


if __name__ == "__main__":
    unittest.main()
