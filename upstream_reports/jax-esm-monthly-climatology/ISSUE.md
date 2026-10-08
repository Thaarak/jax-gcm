# Slab land and ocean models read 12-month climatologies one month per day

*Draft for https://github.com/climate-analytics-lab/jax-esm/issues (not yet posted). Found in a
vendored copy of jax-esm 0.1.0 used by the JAX-GCM cloud-brightening project, 2026-09-30 (ocean) and
2026-10-07 (land).*

## Summary

`SlabModelBase._get_climatology_indices` returns `floor(day) mod cycle_length`, which is correct for
**daily** climatologies. Every slab component passes the climatology's length as `cycle_length`, so a
**12-entry monthly** climatology advances one month per model day and repeats the whole year every
12 days. Three code paths are affected:

| Component | Path | Monthly data it reads |
|---|---|---|
| Slab land model | every step (`slab_land_model.py`, around line 356) | `stl`, `snowd`/`snowc`, `soilw` from `land_clim_file` |
| Slab ocean model | `forcing_method="relaxation"` (around line 250) | `SST_clim` |
| Slab ocean model | `forcing_method="Qflux"` (around line 273) | `q_flux`, declared with a `"month"` dimension |

The relaxation path also computes the climatology's trend as `(end - beg) / 86400`, which assumes
consecutive entries are one day apart.

## Reproduction

`repro.py` (next to this file; CPU, a few seconds):

```
model day :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 ...
month used:  0  1  2  3  4  5  6  7  8  9 10 11  0  1  2 ...
calendar  :  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 ...
```

## What it does to a coupled run

These numbers come from JAX-GCM (T30) coupled to the slab land model (`land_clim_file` = the T30
`forcing.nc`, 12 monthly entries) and a slab ocean. They are measured from the normal-climate,
5-member mean of 240-day runs starting in late January:
- **Land temperature has a 12-day "year".** The land-temperature spectrum peaks at exactly 12.0
  days, with 7,600 times the median power.
  - The 12-day cycle's peak-to-peak averages 20 K over land and reaches 64 K.
  - Its amplitude matches the true annual range of the `stl` climatology cell by cell (r = 0.99).
  - At 60N 100E in January, land temperature swings from -40 C to +7 C and back every 12 days.
- **The atmosphere and ocean feel it.** At a 12-day period, spectral power exceeds the median by 211
  times for evaporation over land, 15 times for rainfall over land, and 7 times for sea-surface
  temperature, which the atmosphere passes on to the ocean.
- **Monthly ocean forcings** (`Qflux`, `relaxation`) cycle in the same way. We found this first,
  when a monthly Q-flux produced a strongly wrong base climate.

## Expected behaviour

A 12-entry climatology should follow the calendar. For a time `t` since 1 January, interpolate
between mid-month anchors:

```python
SECONDS_PER_YEAR = 365.2425 * 86400.0

def monthly_weights(seconds_since_jan1):
    pos = jnp.mod(seconds_since_jan1 / SECONDS_PER_YEAR, 1.0) * 12.0 - 0.5
    lower = jnp.floor(pos)
    w = pos - lower                        # weight of the next month
    i0 = jnp.mod(lower, 12).astype(int)
    return i0, jnp.mod(i0 + 1, 12), w

def interpolate_monthly(clim12, seconds_since_jan1):     # clim12: (..., 12)
    i0, i1, w = monthly_weights(seconds_since_jan1)
    return (1.0 - w) * clim12[..., i0] + w * clim12[..., i1]
```

This is what our subclass `MonthlyQfluxSlabOceanModel` does for the Q-flux. It is bit-identical to
the free slab with a zero Q-flux, and with our fitted Q-flux the coupled model settles within 0.26 K
of the ocean-mean SST of the `forcing.nc` climatology.

## Suggested fix

1. Make `_get_climatology_indices` (or a new helper) aware of the data's time resolution. With 12
   entries it should return the two bracketing months and a weight, as above. With 365 (or 366)
   entries it should keep today's daily indexing.
2. In the relaxation path, take the trend from the interpolated climatology (its change over the
   step divided by the step), not `(end - beg) / 86400`.
3. Add a regression test: with a 12-entry climatology, days 0 and 12 must give (nearly) the same
   January value, and day 45 must give mid-February.

## Workaround until then

Expand monthly files to 365 daily entries (with the interpolation above) before passing them in.
The existing daily indexing is then correct, up to leap days.

## Environment

- jax-esm 0.1.0 (vendored in https://github.com/Thaarak/jax-gcm, `jax-esm/`).
- JAX 0.10.x, on CPU (macOS) and GPU (NVIDIA GB10).
