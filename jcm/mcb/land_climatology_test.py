"""Tests for the land-model fix (jcm.mcb.land_climatology).

They use jax-esm's real slab land component on its own (no atmosphere), with
the T30 terrain and the monthly ``forcing.nc`` climatology the coupled model
reads.
"""

import os
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from jem.components.slab.slab_land_model.slab_land_model import SlabLandModel

from jcm.mcb.land_climatology import (
    LAND_CLIMATOLOGY_ENV,
    MonthlyClimatologySlabLandModel,
    check_land_mode,
    land_climatology_mode,
)
from jcm.mcb.qflux import SECONDS_PER_DAY, interpolate_monthly

TERRAIN_NC = "jcm/data/bc/t30/clim/terrain.nc"
FORCING_NC = "jcm/data/bc/t30/clim/forcing.nc"


class LandModeTest(unittest.TestCase):
    def test_default_is_the_old_lookup(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LAND_CLIMATOLOGY_ENV, None)
            self.assertEqual(land_climatology_mode(), "daily_index")

    def test_monthly_and_bad_values(self):
        with mock.patch.dict(os.environ, {LAND_CLIMATOLOGY_ENV: "monthly"}):
            self.assertEqual(land_climatology_mode(), "monthly")
        with mock.patch.dict(os.environ, {LAND_CLIMATOLOGY_ENV: "daily"}):
            with self.assertRaises(ValueError):
                land_climatology_mode()

    def test_states_from_the_other_land_model_are_refused(self):
        with mock.patch.dict(os.environ, {LAND_CLIMATOLOGY_ENV: "monthly"}):
            check_land_mode("monthly", "state")
            with self.assertRaises(SystemExit):
                check_land_mode(None, "an old state")   # made before the fix
            with self.assertRaises(SystemExit):
                check_land_mode("daily_index", "state")
        with mock.patch.dict(os.environ, {LAND_CLIMATOLOGY_ENV: "daily_index"}):
            check_land_mode(None, "an old state")
            with self.assertRaises(SystemExit):
                check_land_mode("monthly", "a fixed-land state")


class _LandFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kw = dict(start_datetime=jdt.to_datetime("2000-01-01"),
                  timestep=SECONDS_PER_DAY, mask_file=TERRAIN_NC,
                  land_clim_file=FORCING_NC)
        cls.old = SlabLandModel(**kw)
        cls.new = MonthlyClimatologySlabLandModel(**kw)
        cls.old_carry = cls.old.initialize()
        cls.new_carry = cls.new.initialize()
        cls.fns = {"old": jax.jit(cls.old.generate_step_function()),
                   "new": jax.jit(cls.new.generate_step_function())}
        cls.land = np.asarray(cls.new.bmask_l) > 0
        cls.offset = float(cls.new._compute_start_day_offset())

    def _on_climatology(self, model, carry, seconds):
        """``carry`` with land temperature set to the interpolated climatology."""
        clim = interpolate_monthly(model.stl_clim, self.offset + seconds)
        clim = jnp.where(model.bmask_l > 0, clim, 273.15 + 15.0)
        out = dict(carry)
        out["state"] = carry["state"].copy(
            {"land_surface_temperature": clim})
        return out

    def _run(self, which, carry, n):
        """Run ``n`` days with no surface heat flux; return the daily states."""
        zero = jnp.zeros_like(carry["forcing"].total_heat_flux)
        states = []
        for i in range(n):
            carry = dict(carry)
            carry["forcing"] = carry["forcing"].copy({"total_heat_flux": zero})
            carry, _ = self.fns[which](carry, float(i))
            states.append(carry["state"])
        return states


class MonthlyLandTest(_LandFixture):
    def test_with_no_flux_the_land_stays_on_the_monthly_climatology(self):
        carry = self._on_climatology(self.new, self.new_carry, 0.0)
        for day, state in enumerate(self._run("new", carry, 40), start=1):
            want = interpolate_monthly(self.new.stl_clim,
                                       self.offset + day * SECONDS_PER_DAY)
            np.testing.assert_allclose(
                np.asarray(state.land_surface_temperature)[self.land],
                np.asarray(want)[self.land], atol=2e-3)

    def test_snow_and_soil_follow_the_monthly_climatology(self):
        state = self._run("new", self.new_carry, 20)[-1]
        t = self.offset + 19 * SECONDS_PER_DAY      # start of the last step
        snow = jnp.minimum(1.0, interpolate_monthly(self.new.snowd_clim, t)
                           / self.new.sd2sc)
        np.testing.assert_allclose(np.asarray(state.snowc),
                                   np.asarray(snow), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(state.soilw),
            np.asarray(interpolate_monthly(self.new.soilw_clim, t)),
            atol=1e-5)

    def test_the_twelve_day_year_is_gone(self):
        """jax-esm's lookup repeats the year every 12 days; the fix does not."""
        series = {}
        for which, model, carry in (("old", self.old, self.old_carry),
                                    ("new", self.new, self.new_carry)):
            states = self._run(which, carry, 36)
            series[which] = np.array(
                [np.asarray(s.land_surface_temperature)[self.land].mean()
                 for s in states])
        old_jump = np.abs(np.diff(series["old"])).max()
        new_jump = np.abs(np.diff(series["new"])).max()
        # Mid-January to February warms the land mean by well under 0.2 K a
        # day; jax-esm's lookup jumps a whole month every day.
        self.assertLess(new_jump, 0.2)
        self.assertGreater(old_jump, 5 * new_jump)
        # jax-esm's land repeats after 12 days; the fixed land does not.
        old_repeat = np.abs(series["old"][24:36] - series["old"][12:24]).max()
        self.assertLess(old_repeat, 0.25 * np.ptp(series["old"]))

    def test_rejects_a_daily_climatology(self):
        model = MonthlyClimatologySlabLandModel(
            start_datetime=jdt.to_datetime("2000-01-01"),
            timestep=SECONDS_PER_DAY, mask_file=TERRAIN_NC,
            land_clim_file=FORCING_NC)
        model.initialize()
        model.stl_clim = jnp.zeros(model.stl_clim.shape[:2] + (365,))
        with self.assertRaises(ValueError):
            model.generate_step_function()


if __name__ == "__main__":
    unittest.main()
