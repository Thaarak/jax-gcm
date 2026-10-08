"""Minimal reproduction: jax-esm reads a 12-month climatology one month per DAY.

``SlabModelBase._get_climatology_indices`` returns ``floor(day) mod
cycle_length``. Every slab component passes the climatology's length as
``cycle_length``. A 12-entry (monthly) climatology therefore advances one
month per model day and repeats the year every 12 days.

The affected paths:
- the slab land model's ``stl``, ``snowd`` and ``soilw`` climatologies;
- the slab ocean model's ``Qflux`` and ``relaxation`` forcings.

Run from a checkout of jax-esm: ``python repro.py`` (CPU, a few seconds).
"""

import types

import numpy as np

from jem.components.slab.base import SlabModelBase

DAY = 86400.0
stub = types.SimpleNamespace(timestep=DAY)          # one-day coupling step
days = np.arange(40)
indices = [int(SlabModelBase._get_climatology_indices(stub, d * DAY, 0.0,
                                                      12)[0])
           for d in days]
print("model day :", " ".join(f"{d:2d}" for d in days))
print("month used:", " ".join(f"{i:2d}" for i in indices))


def expected_month(day_of_year):
    """Return the calendar month (0 = January) of a day of the year."""
    return min(int(day_of_year / 365.2425 * 12), 11)


print("calendar  :", " ".join(f"{expected_month(d):2d}" for d in days))
repeats = [d for d in days[1:] if indices[d] == indices[0]]
print(f"\nthe 12-month cycle restarts on model day {repeats[0]}; on the "
      f"calendar, December is reached on day 334, not day 11")
assert indices[:12] == list(range(12)), "behaviour changed: re-check"
