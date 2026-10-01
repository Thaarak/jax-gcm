"""Experiment 1 core: gradient estimators vs ensemble finite-difference truth.

For K band controls (``jcm.mcb.band_basis``) and a set of objectives, this
module provides the two measurements Experiment 1 compares:

* **Truth.** Forward-only rollouts at ``a0``, ``a0 + delta*e_k`` and
  ``a0 - delta*e_k``, each emitting the per-day objective series. Central
  differences of the tail means, averaged over many weather realizations,
  estimate the ensemble-mean (climate) sensitivity. Paired baselines cancel
  exactly in a central difference, so none are needed.
* **Estimators.** Reverse-mode Jacobians of the same tail-mean objectives
  through the atmosphere-truncated rollout
  (``jcm.mcb.gradient_truncation``), for any window W, including no
  truncation (ordinary backpropagation through time), and for the
  exploratory damped estimator (the atmosphere's memory fades instead of
  being cut; Amendment 9 revision 0.1).

Window length, episode start time and decay factor are traced arguments, so
one compiled Jacobian per horizon serves every estimator, initial condition
and member.

Maps (Amendment 9 revision 0.4, secondary outputs that the registered
analysis never reads):

* **Truth maps.** The same truth rollouts also return block means of the
  slab-ocean SST and slab-land temperature maps
  (``make_series_and_maps_fn``), so each band's brute-force response MAP
  comes from runs that are made anyway (``central_difference_maps``).
* **Map Jacobians.** A forward-mode pass through the same truncated rollout
  gives the derivative of every map cell with respect to each band
  (``make_map_jacobian_fn``): K tangents give the whole response map at once,
  where reverse mode would need one backward pass per cell.
* **Checks.** Every objective is a weighted sum of a map, so
  ``tail_objectives_from_maps`` must turn truth maps back into the tail means
  of the stored objective series, and forward-mode map Jacobians back into the
  registered reverse-mode Jacobians.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jcm.mcb.band_basis import (
    OCEAN_OBJECTIVES,
    band_perturbation,
    objective_values,
)
from jcm.mcb.gradient_truncation import wrap_step_fn_with_atm_truncation

# Maps are stored relative to the same reference as ``objective_values``.
# Subtracting it from ~290 K float32 values is exact, so the block averages
# add no rounding beyond the model's own float32 resolution.
MAP_REFERENCE_K = 288.0


def apply_band_control(carry: dict, amplitudes, patterns) -> dict:
    """Copy of ``carry`` with the band perturbation in the actuator slot.

    The perturbation goes where the coupled controller puts it
    (``carry["atm"]["derived"]["mcb_perturbation"]``); the JCM wrapper
    re-injects it into the shortwave cloud albedo every coupling step. The
    band patterns already carry the ocean mask.
    """
    derived = dict(carry["atm"]["derived"])
    derived["mcb_perturbation"] = band_perturbation(amplitudes, patterns)
    atm = dict(carry["atm"])
    atm["derived"] = derived
    new_carry = dict(carry)
    new_carry["atm"] = atm
    return new_carry


def _land_field(carry: dict, like):
    """Slab-land surface temperature, or zeros when there is no land model."""
    if "lnd" in carry:
        return carry["lnd"]["state"].land_surface_temperature
    return jnp.zeros_like(like)


def rollout_objective_series(carry: dict, step_fn: Callable, num_steps: int,
                             weight_stack):
    """Run ``num_steps`` coupling steps, emitting objectives after each step.

    Each step is checkpointed (only the carry is stored for the backward
    pass), like ``run_interval_final_carry``.

    Returns:
        ``(final_carry, series)`` with ``series`` of shape
        ``(num_steps, n_obj)``; row s holds the objectives after step s + 1.

    """
    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        sst = new_c["ocn"]["state"].sea_surface_temperature
        values = objective_values(sst, _land_field(new_c, sst), weight_stack)
        return new_c, values

    body = jax.checkpoint(body)
    return lax.scan(body, carry, jnp.arange(num_steps))


def tail_mean(series, horizon: int, tail_days: int):
    """Mean of rows ``[horizon - tail_days, horizon)`` of a daily series."""
    if not 0 < tail_days <= horizon <= series.shape[0]:
        raise ValueError(f"need 0 < tail_days ({tail_days}) <= horizon "
                         f"({horizon}) <= series length ({series.shape[0]})")
    return jnp.mean(series[horizon - tail_days:horizon], axis=0)


def make_series_fn(step_fn: Callable, patterns, weight_stack, num_steps: int):
    """Forward-only ``f(amplitudes, carry) -> (num_steps, n_obj)`` series."""
    def series_fn(amplitudes, carry):
        controlled = apply_band_control(carry, amplitudes, patterns)
        _, series = rollout_objective_series(controlled, step_fn, num_steps,
                                             weight_stack)
        return series

    return series_fn


def make_objective_fn(step_fn: Callable, patterns, weight_stack,
                      horizon: int, tail_days: int):
    """``J(amplitudes, carry, window, t0[, atm_decay]) -> (n_obj,)``.

    Returns the tail-mean objectives. ``window`` is the truncation window in
    days (use ``gradient_truncation.NO_TRUNCATION_DAYS`` for full BPTT);
    ``t0`` is the ocean ``sim_time`` the windows align to (the episode start
    for episode-aligned windows, 0.0 for absolute model days). The optional
    ``atm_decay`` damps the atmosphere's derivatives by that factor per
    coupling step (the exploratory damped estimator; 1.0 changes nothing).
    All three may be traced. Forward values depend on none of them.
    """
    if not 0 < tail_days <= horizon:
        raise ValueError(f"need 0 < tail_days ({tail_days}) <= horizon "
                         f"({horizon})")

    def objective(amplitudes, carry, window, t0, atm_decay=None):
        wrapped = wrap_step_fn_with_atm_truncation(step_fn, window,
                                                   t0_seconds=t0,
                                                   atm_decay=atm_decay)
        controlled = apply_band_control(carry, amplitudes, patterns)
        _, series = rollout_objective_series(controlled, wrapped, horizon,
                                             weight_stack)
        return jnp.mean(series[horizon - tail_days:], axis=0)

    return objective


def make_jacobian_fn(step_fn: Callable, patterns, weight_stack, horizon: int,
                     tail_days: int):
    """Jitted ``jac(amplitudes, carry, window, t0[, atm_decay]) -> (n_obj, K)``.

    Reverse mode, one backward pass per objective (vectorized). Compile once
    per horizon and call signature; window, t0, atm_decay, carry and
    amplitudes are all traced.
    """
    objective = make_objective_fn(step_fn, patterns, weight_stack, horizon,
                                  tail_days)
    return jax.jit(jax.jacrev(objective, argnums=0))


# --- Maps (Amendment 9 revision 0.4) ---------------------------------------

def block_means(daily, block_days: int):
    """Means over consecutive ``block_days`` rows of a daily series.

    ``daily`` has shape ``(n_days, ...)`` with ``n_days`` a multiple of
    ``block_days``; returns ``(n_days // block_days, ...)``.
    """
    n_days = daily.shape[0]
    if block_days < 1 or n_days % block_days:
        raise ValueError(f"block_days ({block_days}) must divide the number "
                         f"of days ({n_days})")
    blocks = daily.reshape((n_days // block_days, block_days)
                           + daily.shape[1:])
    return jnp.mean(blocks, axis=1)


def rollout_series_and_fields(carry: dict, step_fn: Callable, num_steps: int,
                              weight_stack):
    """``rollout_objective_series`` that also emits the daily surface maps.

    Returns:
        ``(final_carry, series, sst, land)``: the objective series
        ``(num_steps, n_obj)`` computed as in ``rollout_objective_series``,
        and the slab-ocean SST and slab-land temperature after every step,
        each ``(num_steps, ix, il)`` in kelvin relative to
        ``MAP_REFERENCE_K``.

    """
    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        sst = new_c["ocn"]["state"].sea_surface_temperature
        land = _land_field(new_c, sst)
        values = objective_values(sst, land, weight_stack)
        return new_c, (values, sst - MAP_REFERENCE_K, land - MAP_REFERENCE_K)

    body = jax.checkpoint(body)
    final, (series, sst, land) = lax.scan(body, carry, jnp.arange(num_steps))
    return final, series, sst, land


def make_series_and_maps_fn(step_fn: Callable, patterns, weight_stack,
                            num_steps: int, block_days: int):
    """Forward-only ``f(amplitudes, carry) -> (series, sst_blocks, land_blocks)``.

    The truth runs' function when maps are kept: ONE rollout gives the
    registered objective series ``(num_steps, n_obj)`` and the block-mean SST
    and land-temperature maps ``(num_steps // block_days, ix, il)``, relative
    to ``MAP_REFERENCE_K``.
    """
    if block_days < 1 or num_steps % block_days:
        raise ValueError(f"block_days ({block_days}) must divide num_steps "
                         f"({num_steps})")

    def series_and_maps_fn(amplitudes, carry):
        controlled = apply_band_control(carry, amplitudes, patterns)
        _, series, sst, land = rollout_series_and_fields(
            controlled, step_fn, num_steps, weight_stack)
        return (series, block_means(sst, block_days),
                block_means(land, block_days))

    return series_and_maps_fn


def make_map_jacobian_fn(step_fn: Callable, patterns, weight_stack,
                         num_steps: int, block_days: int):
    """Jitted forward-mode ``jac(amplitudes, carry, window, t0[, atm_decay])``.

    Returns ``(sst_jac, land_jac)``, each ``(K, num_steps // block_days, ix,
    il)``: the derivative of every block-mean map cell with respect to each
    band amplitude, through the same atmosphere-truncated rollout as
    ``make_jacobian_fn`` (same ``window``, ``t0`` and ``atm_decay``
    semantics, all traced). One call covers every horizon up to
    ``num_steps``.
    """
    if block_days < 1 or num_steps % block_days:
        raise ValueError(f"block_days ({block_days}) must divide num_steps "
                         f"({num_steps})")

    def maps(amplitudes, carry, window, t0, atm_decay=None):
        wrapped = wrap_step_fn_with_atm_truncation(step_fn, window,
                                                   t0_seconds=t0,
                                                   atm_decay=atm_decay)
        controlled = apply_band_control(carry, amplitudes, patterns)
        _, _, sst, land = rollout_series_and_fields(controlled, wrapped,
                                                    num_steps, weight_stack)
        return block_means(sst, block_days), block_means(land, block_days)

    jac = jax.jacfwd(maps, argnums=0)

    def band_first(*args):
        sst_jac, land_jac = jac(*args)
        return jnp.moveaxis(sst_jac, -1, 0), jnp.moveaxis(land_jac, -1, 0)

    return jax.jit(band_first)


def tail_objectives_from_maps(sst_blocks, land_blocks, weight_stack,
                              horizon: int, tail_days: int, block_days: int):
    """Tail-mean objectives implied by block-mean maps (host-side, float64).

    Args:
        sst_blocks, land_blocks: ``(..., n_blocks, ix, il)`` block means
            relative to ``MAP_REFERENCE_K``, or their derivatives.
        weight_stack: ``(n_obj, ix, il)`` objective weights.
        horizon, tail_days: the tail window ``[horizon - tail_days,
            horizon)`` in days; both must be multiples of ``block_days``.
        block_days: block length of the maps.

    Returns:
        ``(..., n_obj)`` in ``OBJECTIVE_NAMES`` order. On a truth run this
        reproduces the tail mean of its objective series; on map Jacobians
        ``(K, ...)`` it gives the objective Jacobian transposed, ``(K,
        n_obj)``.

    """
    if not 0 < tail_days <= horizon or tail_days % block_days \
            or horizon % block_days:
        raise ValueError(f"tail_days ({tail_days}) and horizon ({horizon}) "
                         f"must be multiples of block_days ({block_days}) "
                         f"with 0 < tail_days <= horizon")
    lo, hi = (horizon - tail_days) // block_days, horizon // block_days
    if hi > np.shape(sst_blocks)[-3]:
        raise ValueError(f"horizon {horizon} d is beyond the "
                         f"{np.shape(sst_blocks)[-3]} stored blocks")
    w = np.asarray(weight_stack, dtype=np.float64)
    n_ocean = len(OCEAN_OBJECTIVES)
    sst_tail = np.asarray(sst_blocks, np.float64)[..., lo:hi, :, :].mean(-3)
    land_tail = np.asarray(land_blocks, np.float64)[..., lo:hi, :, :].mean(-3)
    ocean = np.einsum("...xy,oxy->...o", sst_tail, w[:n_ocean])
    land = np.einsum("...xy,xy->...", land_tail, w[n_ocean])
    return np.concatenate([ocean, land[..., None]], axis=-1)


def central_difference_maps(blocks, labels: Sequence[str], delta: float):
    """Each band's brute-force response maps from the truth runs (host-side).

    Args:
        blocks: ``(..., n_labels, n_blocks, ix, il)`` truth block maps in the
            run order ``labels`` (``run_labels`` in run_gradient_fidelity.py).
        labels: run labels, containing ``plus_k`` and ``minus_k`` for every
            band k.
        delta: the central-difference half step.

    Returns:
        ``(..., K, n_blocks, ix, il)``: ``(plus_k - minus_k) / (2 delta)``,
        the counterpart of the forward-mode map Jacobians.

    """
    labels = list(labels)
    k_bands = sum(label.startswith("plus_") for label in labels)
    plus = [labels.index(f"plus_{k}") for k in range(k_bands)]
    minus = [labels.index(f"minus_{k}") for k in range(k_bands)]
    blocks = np.asarray(blocks, dtype=np.float64)
    return ((np.take(blocks, plus, axis=-4) - np.take(blocks, minus, axis=-4))
            / (2.0 * delta))
