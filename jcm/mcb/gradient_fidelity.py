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
  truncation (ordinary backpropagation through time).

Window length and episode start time are traced arguments, so one compiled
Jacobian per horizon serves every window, initial condition and member.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from jcm.mcb.band_basis import band_perturbation, objective_values
from jcm.mcb.gradient_truncation import wrap_step_fn_with_atm_truncation


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
    """``J(amplitudes, carry, window, t0) -> (n_obj,)`` tail-mean objectives.

    ``window`` is the truncation window in days (use
    ``gradient_truncation.NO_TRUNCATION_DAYS`` for full BPTT); ``t0`` is the
    ocean ``sim_time`` the windows align to (the episode start for
    episode-aligned windows, 0.0 for absolute model days). Both may be
    traced. Forward values do not depend on either.
    """
    if not 0 < tail_days <= horizon:
        raise ValueError(f"need 0 < tail_days ({tail_days}) <= horizon "
                         f"({horizon})")

    def objective(amplitudes, carry, window, t0):
        wrapped = wrap_step_fn_with_atm_truncation(step_fn, window,
                                                   t0_seconds=t0)
        controlled = apply_band_control(carry, amplitudes, patterns)
        _, series = rollout_objective_series(controlled, wrapped, horizon,
                                             weight_stack)
        return jnp.mean(series[horizon - tail_days:], axis=0)

    return objective


def make_jacobian_fn(step_fn: Callable, patterns, weight_stack, horizon: int,
                     tail_days: int):
    """Jitted ``jac(amplitudes, carry, window, t0) -> (n_obj, K)``.

    Reverse mode, one backward pass per objective (vectorized). Compile once
    per horizon; window, t0, carry and amplitudes are all traced.
    """
    objective = make_objective_fn(step_fn, patterns, weight_stack, horizon,
                                  tail_days)
    return jax.jit(jax.jacrev(objective, argnums=0))
