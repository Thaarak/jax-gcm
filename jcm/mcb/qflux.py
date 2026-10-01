"""Monthly Q-flux for the slab ocean: diagnosis, application and storage.

Why
---
The slab ocean has no ocean heat transport. Left free, the settled control
climate ends up about 3.8 K colder than the observed SST climatology that
SPEEDY's boundary conditions were built around (MCB_PROJECT_REPORT.md Part
16.3). The standard slab-ocean fix is a Q-flux: a fixed, seasonally varying
heat source or sink in every ocean cell that stands in for the missing
transport.

How it is diagnosed
-------------------
A coupled run restores SST toward the monthly climatology (``forcing.nc``
``sst``) with a short timescale. The heat the restoring adds each day is the
heat the free slab lacks. Least squares on that daily heat gives 12 mid-month
values that, linearly interpolated in time, best reproduce it
(``fit_monthly_qflux``). Applied as a fixed flux, they hold the free slab near
the climatology without damping its response to cloud brightening. Over sea
ice the "SST" is the surface temperature the atmosphere sees (SPEEDY's
surface fluxes use it directly, with no separate ice temperature), so the
climatology is restored as it is, sub-freezing values included.

Conventions
-----------
* ``q_flux`` uses the slab ocean's sign: positive = upward (heat leaving the
  ocean), added to ``total_heat_flux`` (also upward-positive).
* 12 mid-month anchors on a mean year of 365.2425 days, linear interpolation,
  periodic. For the 2000-01-01 start this stays within about a day of the
  model's calendar for a century.

Why a subclass instead of jax-esm's "Qflux" mode
------------------------------------------------
jax-esm indexes the q_flux cycle by DAY (``floor(day) mod cycle_length``),
although the field's third dimension is called "month" and holds 12 entries.
A monthly Q-flux would therefore advance one month per day and repeat every 12
days. Its "relaxation" mode indexes the SST climatology the same way.
``MonthlyQfluxSlabOceanModel`` reads the 12 entries as months. With a zero
q_flux (every cold start and every carry saved before the Q-flux existed) it
is bit-identical to the free slab.
"""

import math

import jax.numpy as jnp
import numpy as np
from jem import constants
from jem.components.slab.slab_ocean_model.slab_ocean_model import (
    SlabOceanModel,
    default_land_surface_temperature,
)
from jem.utils.bulk_op import stack_objects

DAYS_PER_YEAR = 365.2425
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = DAYS_PER_YEAR * SECONDS_PER_DAY


# ------------------------------------------------------- time interpolation --

def monthly_weights(seconds_since_jan1):
    """Return indices and weight for interpolating between mid-month values.

    Args:
        seconds_since_jan1: Time since 1 January of the start year (seconds);
            scalar, may be traced.

    Returns:
        ``(i0, i1, w)``: the value is ``(1 - w) * clim[i0] + w * clim[i1]``.

    """
    year_frac = jnp.mod(seconds_since_jan1 / SECONDS_PER_YEAR, 1.0)
    pos = year_frac * 12.0 - 0.5
    lower = jnp.floor(pos)
    w = pos - lower
    i0 = jnp.mod(lower, 12.0).astype(jnp.int32)
    i1 = jnp.mod(i0 + 1, 12)
    return i0, i1, w


def interpolate_monthly(clim12, seconds_since_jan1):
    """Monthly climatology ``(..., 12)`` interpolated to one time."""
    i0, i1, w = monthly_weights(seconds_since_jan1)
    return (1.0 - w) * clim12[..., i0] + w * clim12[..., i1]


def monthly_weight_matrix(seconds_since_jan1) -> np.ndarray:
    """NumPy ``(n, 12)`` interpolation weights for an array of times."""
    t = np.asarray(seconds_since_jan1, dtype=np.float64)
    pos = np.mod(t / SECONDS_PER_YEAR, 1.0) * 12.0 - 0.5
    lower = np.floor(pos)
    w = pos - lower
    i0 = np.mod(lower, 12).astype(int)
    i1 = np.mod(i0 + 1, 12)
    a = np.zeros((t.size, 12))
    rows = np.arange(t.size)
    np.add.at(a, (rows, i0), 1.0 - w)
    np.add.at(a, (rows, i1), w)
    return a


def calendar_month(seconds_since_jan1) -> np.ndarray:
    """Month index 0..11 on the mean-year calendar (for diagnostics)."""
    t = np.asarray(seconds_since_jan1, dtype=np.float64)
    return np.minimum((np.mod(t / SECONDS_PER_YEAR, 1.0) * 12.0).astype(int),
                      11)


# --------------------------------------------------------- the slab ocean ---

class MonthlyQfluxSlabOceanModel(SlabOceanModel):
    """Free slab ocean plus a monthly Q-flux read from the carry.

    ``carry["forcing"].q_flux`` (``(lon, lat, 12)``, upward-positive W m-2)
    is interpolated to the middle of each step and added to the surface heat
    flux. Everything else is the free slab (``forcing_method`` None) of
    jax-esm, written in the same order so a zero Q-flux reproduces it bit for
    bit. The Q-flux lives in the carry, so it travels with every saved state.
    """

    def __init__(self, *args, **kwargs):
        """Build the slab exactly as jax-esm does; only a free slab is allowed."""
        method = kwargs.get("forcing_method")
        if method not in (None, "None"):
            raise ValueError("MonthlyQfluxSlabOceanModel is a free slab plus "
                             f"a Q-flux; forcing_method must be None, got "
                             f"{method!r}")
        super().__init__(*args, **kwargs)

    def _create_step_function_body(self):
        start_offset = self._compute_start_day_offset()
        ocn_idx = self.horizontal_grids["T"].bmask == self.mask_value
        timestep = self.timestep

        def step_function(carry, step):
            state = carry["state"]
            forcing = carry["forcing"]
            t_mid = start_offset + state.sim_time + 0.5 * timestep
            q_now = jnp.where(ocn_idx,
                              interpolate_monthly(forcing.q_flux, t_mid), 0.0)
            total_heat_flux = forcing.total_heat_flux + q_now

            new_sim_time = state.sim_time + timestep
            new_sea_surface_temperature = self.time_factor * (
                state.sea_surface_temperature
                + self.cd_factor * (- total_heat_flux)
            )
            new_sea_surface_temperature = jnp.where(
                ocn_idx, new_sea_surface_temperature,
                default_land_surface_temperature)
            new_state = state.copy({
                "sea_surface_temperature": new_sea_surface_temperature,
                "sim_time": new_sim_time,
            })
            predictions = dict(
                state=new_state,
                forcing=dict(total_heat_flux=total_heat_flux),
            )
            return dict(state=new_state, forcing=forcing), stack_objects(
                [predictions])

        return step_function


# ----------------------------------------------------------- carry helpers ---

def ocean_heat_capacity(mixed_layer_depth):
    """Areal heat capacity rho * c_p * h of the slab (J m-2 K-1)."""
    return (constants.ocean_density * constants.ocean_specific_heat_capacity
            * mixed_layer_depth)


def set_qflux(carry: dict, q_flux) -> dict:
    """Copy of a coupled ``carry`` with ``ocn.forcing.q_flux`` replaced."""
    forcing = carry["ocn"]["forcing"]
    q = jnp.asarray(q_flux, dtype=forcing.q_flux.dtype)
    if q.shape != forcing.q_flux.shape:
        raise ValueError(f"q_flux shape {q.shape} != carry's "
                         f"{forcing.q_flux.shape}")
    ocn = dict(carry["ocn"])
    ocn["forcing"] = forcing.copy({"q_flux": q})
    new_carry = dict(carry)
    new_carry["ocn"] = ocn
    return new_carry


def qflux_magnitude(carry: dict) -> float:
    """Largest |q_flux| in a coupled carry (0.0 means no Q-flux)."""
    return float(np.max(np.abs(np.asarray(carry["ocn"]["forcing"].q_flux))))


# ------------------------------------------------------------- diagnosis ---

def wrap_step_fn_with_sst_restoring(step_fn, sst_clim12, ocean_mask,
                                    heat_capacity, tau_days,
                                    start_offset_seconds=0.0,
                                    timestep_seconds=SECONDS_PER_DAY):
    """Restore SST toward a monthly climatology after every coupled step.

    After the free slab update, SST moves a fraction ``1 - exp(-dt / tau)`` of
    the way to the climatology interpolated to the end of the step (ocean
    cells only). The wrapped step returns the heat that restoring added,
    ``C * dSST / dt`` in W m-2 (positive = into the ocean), in place of the
    coupler's predictions.
    """
    if not tau_days > 0:
        raise ValueError(f"tau_days must be > 0, got {tau_days}")
    frac = 1.0 - math.exp(-timestep_seconds / (tau_days * SECONDS_PER_DAY))
    clim = jnp.asarray(sst_clim12)
    mask = jnp.asarray(ocean_mask) > 0.5
    cap = jnp.asarray(heat_capacity)

    def restored_step(carry, step_idx):
        new_carry, _ = step_fn(carry, step_idx)
        ocn = new_carry["ocn"]
        state = ocn["state"]
        target = interpolate_monthly(clim, start_offset_seconds
                                     + state.sim_time)
        sst = state.sea_surface_temperature
        delta = jnp.where(mask, frac * (target - sst), 0.0)
        heat = cap * delta / timestep_seconds
        new_ocn = dict(ocn)
        new_ocn["state"] = state.copy({"sea_surface_temperature": sst + delta})
        out = dict(new_carry)
        out["ocn"] = new_ocn
        return out, heat

    return restored_step


def fit_monthly_qflux(daily_heat, mid_step_seconds, ocean_mask=None):
    """Least-squares monthly Q-flux from daily restoring heat.

    Args:
        daily_heat: ``(n_days, ix, il)`` heat the restoring added (W m-2,
            positive into the ocean).
        mid_step_seconds: ``(n_days,)`` time of each step's middle, seconds
            since 1 January of the start year (the time the ocean model
            evaluates the Q-flux at).
        ocean_mask: Optional ``(ix, il)``; cells outside it get zero.

    Returns:
        ``(q_flux (ix, il, 12), residual_rms (ix, il))``. The Q-flux is
        upward-positive, i.e. minus the fitted heat input.

    """
    heat = np.asarray(daily_heat, dtype=np.float64)
    n, ix, il = heat.shape
    a = monthly_weight_matrix(mid_step_seconds)
    if a.shape[0] != n:
        raise ValueError(f"{a.shape[0]} times for {n} days of heat")
    flat = heat.reshape(n, ix * il)
    coef, *_ = np.linalg.lstsq(a, flat, rcond=None)
    resid = flat - a @ coef
    q = -coef.T.reshape(ix, il, 12)
    rms = np.sqrt(np.mean(resid ** 2, axis=0)).reshape(ix, il)
    if ocean_mask is not None:
        keep = np.asarray(ocean_mask) > 0.5
        q = np.where(keep[:, :, None], q, 0.0)
        rms = np.where(keep, rms, 0.0)
    return q.astype(np.float32), rms


# ------------------------------------------------------------------- I/O ---

def save_qflux(path, q_flux, lon_deg, lat_deg, attrs=None) -> None:
    """Write a ``(lon, lat, 12)`` Q-flux to NetCDF with its conventions."""
    import xarray as xr

    q = np.asarray(q_flux, dtype=np.float32)
    ds = xr.Dataset(
        {"q_flux": (("lon", "lat", "month"), q, {
            "units": "W m-2",
            "positive": "upward",
            "long_name": "slab-ocean Q-flux (heat leaving the ocean)",
            "time_interpolation": ("mid-month anchors on a 365.2425-day "
                                   "year, linear, periodic"),
        })},
        coords={"lon": np.asarray(lon_deg, dtype=np.float64),
                "lat": np.asarray(lat_deg, dtype=np.float64),
                "month": np.arange(1, 13)},
        attrs=dict(attrs or {}),
    )
    ds.to_netcdf(path)


def load_qflux(path) -> np.ndarray:
    """Read a Q-flux written by ``save_qflux`` as ``(lon, lat, 12)`` float32."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        q = ds["q_flux"].transpose("lon", "lat", "month").values
    return np.asarray(q, dtype=np.float32)
