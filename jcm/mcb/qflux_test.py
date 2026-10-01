"""Tests for the monthly slab-ocean Q-flux (jcm.mcb.qflux).

Fast tests use the real jax-esm slab ocean component on its own (no
atmosphere) plus small toys. The slow test runs the full coupled model for a
few days and checks that the new ocean class with a zero Q-flux reproduces
the old free slab bit for bit.
"""

import os
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import pytest
from jem.components.slab.slab_ocean_model.slab_ocean_model import (
    SlabOceanModel,
)

from jcm.mcb.qflux import (
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    MonthlyQfluxSlabOceanModel,
    calendar_month,
    fit_monthly_qflux,
    interpolate_monthly,
    load_qflux,
    monthly_weight_matrix,
    monthly_weights,
    ocean_heat_capacity,
    qflux_magnitude,
    save_qflux,
    set_qflux,
    wrap_step_fn_with_sst_restoring,
)

TERRAIN_NC = "jcm/data/bc/t30/clim/terrain.nc"
FORCING_NC = "jcm/data/bc/t30/clim/forcing.nc"
MONTH = SECONDS_PER_YEAR / 12.0


def _mid_month(m):
    return (m + 0.5) * MONTH


class MonthlyWeightsTest(unittest.TestCase):
    def test_mid_month_returns_that_month(self):
        clim = jnp.arange(12.0)
        for m in range(12):
            self.assertAlmostEqual(
                float(interpolate_monthly(clim, _mid_month(m))), m, places=3)

    def test_halfway_between_months_is_the_average(self):
        clim = jnp.arange(12.0)
        self.assertAlmostEqual(
            float(interpolate_monthly(clim, _mid_month(0) + 0.5 * MONTH)),
            0.5, places=3)

    def test_wraps_from_december_to_january(self):
        clim = jnp.arange(12.0)
        i0, i1, w = monthly_weights(0.0)       # 1 January: mid Dec -> mid Jan
        self.assertEqual((int(i0), int(i1)), (11, 0))
        self.assertAlmostEqual(float(w), 0.5, places=5)
        self.assertAlmostEqual(float(interpolate_monthly(clim, 0.0)), 5.5,
                               places=3)

    def test_periodic_in_the_year(self):
        clim = jnp.asarray(np.random.default_rng(0).normal(size=12))
        for t in (0.0, 0.3 * SECONDS_PER_YEAR, 0.9 * SECONDS_PER_YEAR):
            a = float(interpolate_monthly(clim, t))
            b = float(interpolate_monthly(clim, t + 7 * SECONDS_PER_YEAR))
            self.assertAlmostEqual(a, b, places=3)

    def test_numpy_matrix_matches_and_sums_to_one(self):
        rng = np.random.default_rng(1)
        clim = rng.normal(size=12)
        t = rng.uniform(0, 5 * SECONDS_PER_YEAR, 50)
        a = monthly_weight_matrix(t)
        np.testing.assert_allclose(a.sum(axis=1), 1.0)
        ref = [float(interpolate_monthly(jnp.asarray(clim), ti)) for ti in t]
        np.testing.assert_allclose(a @ clim, ref, atol=1e-3)

    def test_calendar_month(self):
        np.testing.assert_array_equal(
            calendar_month([0.0, _mid_month(5), _mid_month(11)]), [0, 5, 11])


class FitMonthlyQfluxTest(unittest.TestCase):
    def test_recovers_a_known_qflux_with_the_right_sign(self):
        rng = np.random.default_rng(2)
        q_true = rng.normal(0.0, 30.0, (4, 3, 12))
        t_mid = (np.arange(4 * 365) + 0.5) * SECONDS_PER_DAY
        a = monthly_weight_matrix(t_mid)
        # Restoring heat INTO the ocean is minus the upward Q-flux.
        heat = -(a @ q_true.reshape(-1, 12).T).reshape(len(t_mid), 4, 3)
        q_fit, rms = fit_monthly_qflux(heat, t_mid)
        np.testing.assert_allclose(q_fit, q_true, atol=1e-3)
        self.assertLess(float(rms.max()), 1e-6)

    def test_land_is_zeroed(self):
        t_mid = (np.arange(400) + 0.5) * SECONDS_PER_DAY
        heat = np.ones((400, 2, 2))
        mask = np.array([[1.0, 0.0], [1.0, 1.0]])
        q, _ = fit_monthly_qflux(heat, t_mid, ocean_mask=mask)
        np.testing.assert_array_equal(q[0, 1], 0.0)
        np.testing.assert_allclose(q[0, 0], -1.0, atol=1e-6)


class _SlabFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kw = dict(start_datetime=jdt.to_datetime("2000-01-01"),
                  timestep=SECONDS_PER_DAY, mask_file=TERRAIN_NC,
                  SST_clim_file=FORCING_NC)
        cls.old = SlabOceanModel(**kw)
        cls.new = MonthlyQfluxSlabOceanModel(**kw)
        cls.old_carry = cls.old.initialize()
        cls.new_carry = cls.new.initialize()
        cls.fns = {"old": jax.jit(cls.old.generate_step_function()),
                   "new": jax.jit(cls.new.generate_step_function())}
        shape = cls.old_carry["forcing"].total_heat_flux.shape
        cls.flux = jnp.asarray(np.random.default_rng(3).normal(0.0, 80.0,
                                                               shape),
                               dtype=jnp.float32)
        cls.ocean = np.asarray(
            cls.new.horizontal_grids["T"].bmask == cls.new.mask_value)

    def _run(self, which, carry, n=5, flux=None):
        flux = self.flux if flux is None else flux
        for i in range(n):
            carry = dict(carry)
            carry["forcing"] = carry["forcing"].copy(
                {"total_heat_flux": flux})
            carry, _ = self.fns[which](carry, float(i))
        return carry


class MonthlyQfluxSlabTest(_SlabFixture):
    def test_q_flux_field_is_monthly(self):
        self.assertEqual(self.new_carry["forcing"].q_flux.shape[-1], 12)

    def test_zero_qflux_is_bit_identical_to_the_free_slab(self):
        old = self._run("old", self.old_carry)
        new = self._run("new", self.new_carry)
        np.testing.assert_array_equal(
            np.asarray(old["state"].sea_surface_temperature),
            np.asarray(new["state"].sea_surface_temperature))
        np.testing.assert_array_equal(np.asarray(old["state"].sim_time),
                                      np.asarray(new["state"].sim_time))

    def test_constant_qflux_equals_an_extra_upward_flux(self):
        q0 = jnp.asarray(np.random.default_rng(4).normal(0.0, 20.0,
                                                         self.flux.shape),
                         dtype=jnp.float32)
        carry = dict(self.new_carry)
        carry["forcing"] = carry["forcing"].copy(
            {"q_flux": jnp.repeat(q0[:, :, None], 12, axis=2)})
        new = self._run("new", carry)
        old = self._run("old", self.old_carry,
                        flux=self.flux + jnp.where(self.ocean, q0, 0.0))
        np.testing.assert_allclose(
            np.asarray(new["state"].sea_surface_temperature),
            np.asarray(old["state"].sea_surface_temperature), rtol=0,
            atol=2e-4)

    def test_positive_qflux_cools_the_ocean_only(self):
        carry = dict(self.new_carry)
        carry["forcing"] = carry["forcing"].copy(
            {"q_flux": jnp.full(carry["forcing"].q_flux.shape, 10.0,
                                dtype=jnp.float32)})
        cooled = np.asarray(self._run("new", carry)["state"]
                            .sea_surface_temperature)
        free = np.asarray(self._run("new", self.new_carry)["state"]
                          .sea_surface_temperature)
        self.assertTrue(np.all(cooled[self.ocean] < free[self.ocean]))
        np.testing.assert_array_equal(cooled[~self.ocean], free[~self.ocean])
        # 10 W m-2 for 5 days through 40-60 m of water: 1e-3 to 2e-3 K.
        drop = (free - cooled)[self.ocean]
        cap = np.asarray(ocean_heat_capacity(
            self.new_carry["state"].mixed_layer_depth))[self.ocean]
        np.testing.assert_allclose(drop, 5 * SECONDS_PER_DAY * 10.0 / cap,
                                   rtol=0.02)

    def test_rejects_other_forcing_methods(self):
        with self.assertRaises(ValueError):
            MonthlyQfluxSlabOceanModel(
                start_datetime=jdt.to_datetime("2000-01-01"),
                timestep=SECONDS_PER_DAY, mask_file=TERRAIN_NC,
                SST_clim_file=FORCING_NC, forcing_method="relaxation")


class SetQfluxTest(_SlabFixture):
    def test_replaces_without_mutating_and_checks_shape(self):
        carry = {"ocn": self.new_carry}
        q = np.full(self.new_carry["forcing"].q_flux.shape, 3.0, np.float32)
        out = set_qflux(carry, q)
        self.assertEqual(qflux_magnitude(carry), 0.0)
        self.assertEqual(qflux_magnitude(out), 3.0)
        with self.assertRaises(ValueError):
            set_qflux(carry, np.zeros((2, 2, 12)))


@jax.tree_util.register_pytree_node_class
class _FakeOcean:
    def __init__(self, sim_time, sst):
        self.sim_time = sim_time
        self.sea_surface_temperature = sst

    def copy(self, updates):
        return _FakeOcean(updates.get("sim_time", self.sim_time),
                          updates.get("sea_surface_temperature",
                                      self.sea_surface_temperature))

    def tree_flatten(self):
        return (self.sim_time, self.sea_surface_temperature), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class RestoringTest(unittest.TestCase):
    def test_moves_sst_toward_the_climatology_and_reports_the_heat(self):
        def free_step(carry, i):
            ocn = carry["ocn"]
            state = ocn["state"].copy(
                {"sim_time": ocn["state"].sim_time + SECONDS_PER_DAY})
            return {"ocn": {"state": state}}, None

        clim = np.zeros((2, 1, 12), np.float32)
        clim[:, :, :] = 290.0
        mask = np.array([[1.0], [0.0]])
        cap = np.full((2, 1), 2.0e8)
        step = wrap_step_fn_with_sst_restoring(free_step, clim, mask, cap,
                                               tau_days=5.0)
        carry = {"ocn": {"state": _FakeOcean(jnp.asarray(0.0),
                                             jnp.array([[280.0], [288.15]]))}}
        out, heat = step(carry, 0)
        frac = 1.0 - np.exp(-1.0 / 5.0)
        sst = np.asarray(out["ocn"]["state"].sea_surface_temperature)
        self.assertAlmostEqual(float(sst[0, 0]), 280.0 + 10.0 * frac,
                               places=3)
        self.assertEqual(float(sst[1, 0]),                  # land untouched
                         float(carry["ocn"]["state"].sea_surface_temperature[1, 0]))
        self.assertAlmostEqual(float(heat[0, 0]),
                               2.0e8 * 10.0 * frac / SECONDS_PER_DAY,
                               delta=1.0)
        self.assertEqual(float(heat[1, 0]), 0.0)

    def test_rejects_nonpositive_timescale(self):
        with self.assertRaises(ValueError):
            wrap_step_fn_with_sst_restoring(lambda c, i: (c, None),
                                            np.zeros((1, 1, 12)),
                                            np.ones((1, 1)), np.ones((1, 1)),
                                            tau_days=0.0)


class NetcdfRoundTripTest(unittest.TestCase):
    def test_save_then_load(self):
        q = np.random.default_rng(5).normal(size=(4, 3, 12)).astype(
            np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.nc")
            save_qflux(path, q, lon_deg=np.arange(4) * 90.0,
                       lat_deg=[-30.0, 0.0, 30.0], attrs={"note": "test"})
            np.testing.assert_array_equal(load_qflux(path), q)


@pytest.mark.slow
class CoupledBitIdentityTest(unittest.TestCase):
    """The full coupled model: new ocean + zero Q-flux == the old free slab."""

    def test_three_coupled_days_are_identical(self):
        # The root conftest deletes the jcm modules after every test, but jem
        # and the run_* scripts keep the classes they imported first, and a
        # model built from both fails with "different tree structures".
        # Build it from one fresh import of everything that touches jcm.
        import sys
        for key in list(sys.modules):
            root = key.split(".")[0]
            if root in ("jcm", "jem") or (root.startswith("run_")
                                           and not root.endswith("_test")):
                del sys.modules[key]
        import run_coupled_training as rct
        from jcm.mcb.coupled_controller import create_coupled_step_fn
        from jcm.mcb.qflux import MonthlyQfluxSlabOceanModel as monthly_cls
        from jem.components.slab.slab_ocean_model.slab_ocean_model import (
            SlabOceanModel as slab_cls,
        )

        def sst_after(ocean_cls, n=3):
            with mock.patch.object(rct, "MonthlyQfluxSlabOceanModel",
                                   ocean_cls):
                coupler, _, _, _ = rct.setup_coupled_model(
                    jdt.to_datetime("2000-01-01"), jdt.to_timedelta(1, "day"),
                    realistic_terrain=True)
            carry = coupler.initialize()
            step = create_coupled_step_fn(coupler,
                                          rct.coupler_workflow(coupler))
            for i in range(n):
                carry, _ = step(carry, float(i))
            return (np.asarray(carry["ocn"]["state"].sea_surface_temperature),
                    np.asarray(carry["lnd"]["state"].land_surface_temperature))

        old_sst, old_land = sst_after(slab_cls)
        new_sst, new_land = sst_after(monthly_cls)
        np.testing.assert_array_equal(old_sst, new_sst)
        np.testing.assert_array_equal(old_land, new_land)


if __name__ == "__main__":
    unittest.main()
