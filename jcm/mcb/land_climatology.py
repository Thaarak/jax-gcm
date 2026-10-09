"""The slab land model with its monthly climatology read as months.

Why
---
jax-esm's ``SlabLandModel`` looks its climatology up with
``SlabModelBase._get_climatology_indices``, which returns ``floor(day) mod
cycle_length``. That is right for a daily climatology, but ``forcing.nc``
holds 12 monthly entries of ``stl``, ``snowc`` and ``soilw_am``, so the land
runs through a whole year every 12 model days (MCB_PROJECT_REPORT.md Part 26;
``upstream_reports/jax-esm-monthly-climatology``). Every experiment up to
Amendment 9 revision 2 ran with that defect.

The fix
-------
``MonthlyClimatologySlabLandModel`` reads the 12 entries as months, with the
same mid-month linear interpolation that ``jcm.mcb.qflux`` uses for the
ocean's Q-flux. Everything else is jax-esm's land step, written in the same
order: the temperature anomaly from the climatology at the start of the step
decays with ``cdland`` under the surface heat flux and is added back to the
climatology at the end of the step; snow cover and soil water are the
climatology at the start of the step.

Which land model a run uses
---------------------------
``JCM_LAND_CLIMATOLOGY`` selects it, so every driver picks it up without new
command-line plumbing:

* ``daily_index`` (the default): jax-esm's behaviour, which reproduces every
  earlier experiment;
* ``monthly``: the fixed land (Amendment 9 revision 3, Experiment 3d).

Starting states record the mode they were made with (``land_climatology`` in
their manifests), and the drivers refuse to run a state in the other mode.
"""

import os

import jax.numpy as jnp
from jem.components.slab.slab_land_model.slab_land_model import SlabLandModel
from jem.utils.bulk_op import stack_objects

from jcm.mcb.qflux import interpolate_monthly

LAND_CLIMATOLOGY_ENV = "JCM_LAND_CLIMATOLOGY"
LAND_MODES = ("daily_index", "monthly")
DEFAULT_LAND_MODE = "daily_index"


def land_climatology_mode() -> str:
    """Return the land model this process uses (from ``JCM_LAND_CLIMATOLOGY``)."""
    mode = os.environ.get(LAND_CLIMATOLOGY_ENV, DEFAULT_LAND_MODE)
    if mode not in LAND_MODES:
        raise ValueError(f"{LAND_CLIMATOLOGY_ENV} must be one of {LAND_MODES}, "
                         f"got {mode!r}")
    return mode


def check_land_mode(recorded, what: str):
    """Refuse a state or reference made with the other land model.

    ``recorded`` is the mode a manifest records; manifests written before the
    fix have none and were all made with ``daily_index``.
    """
    made = recorded or DEFAULT_LAND_MODE
    now = land_climatology_mode()
    if made != now:
        raise SystemExit(f"{what} was made with the {made!r} land model, but "
                         f"this run uses {now!r} (set {LAND_CLIMATOLOGY_ENV})")


class MonthlyClimatologySlabLandModel(SlabLandModel):
    """jax-esm's slab land, with the 12 climatology entries read as months."""

    def _create_step_function_body(self):
        if self.stl_clim.shape[2] != 12:
            raise ValueError("MonthlyClimatologySlabLandModel expects a "
                             f"12-month climatology, got "
                             f"{self.stl_clim.shape[2]} entries")
        start_offset = self._compute_start_day_offset()
        timestep = self.timestep

        def step_function(carry, t):
            state = carry["state"]
            forcing = carry["forcing"]
            t_beg = start_offset + state.sim_time
            stl_clim_beg = interpolate_monthly(self.stl_clim, t_beg)
            stl_clim_end = interpolate_monthly(self.stl_clim, t_beg + timestep)
            snowd_clim_beg = interpolate_monthly(self.snowd_clim, t_beg)
            soilw_clim_beg = interpolate_monthly(self.soilw_clim, t_beg)

            # jax-esm's land step (run_land_model in SPEEDY), unchanged.
            heatflx = - forcing.total_heat_flux
            t_anom = state.land_surface_temperature - stl_clim_beg
            new_t_anom = self.cdland * (t_anom + self.rhcapl * heatflx)
            new_t = new_t_anom + stl_clim_end
            new_t = jnp.where(self.bmask_l > 0, new_t, 273.15 + 15.0)
            new_state = state.copy({
                "sim_time": state.sim_time + timestep,
                "land_surface_temperature": new_t,
                "snowc": jnp.minimum(1.0, snowd_clim_beg / self.sd2sc),
                "soilw": soilw_clim_beg,
            })
            return (dict(state=new_state, forcing=forcing),
                    stack_objects([dict(state=new_state, forcing=forcing)]))

        return step_function
