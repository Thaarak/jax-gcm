"""Atmosphere-truncated ("slow-manifold") gradients for the coupled model.

Reverse-mode gradients through the full coupled model stop tracking the true
sensitivity after a few weeks: the chaotic atmosphere amplifies every
perturbation, so a single-trajectory gradient grows like exp(lambda * T) while
the sensitivity of a time-averaged objective stays bounded (Lea et al. 2000;
Metz et al. 2021). Dubey, Abbot & Chattopadhyay (2026, arXiv:2609.12528)
measured the breakdown in JAX-GCM with prescribed SST: gradient vs
finite-difference correlation 0.91 / 0.87 / 0.49 at 7 / 14 / 28 days. Our own
Tier-2 policies, trained on 60-day coupled rollouts, never learned.

The slab ocean is not chaotic: it integrates surface fluxes and damps
anomalies. This module keeps the gradient path through the ocean (and land)
for the whole rollout, but cuts the path through the atmosphere's multi-day
memory every ``window_days`` days. Setting ``window_days = W`` keeps at most
W days of atmospheric history in any gradient path:

* W = None (or <= 0)  -> no truncation: ordinary backpropagation through time.
* W = 1               -> only the same-day atmospheric response survives.
* 1 < W < horizon     -> fast adjustment kept, long chaotic chains dropped.

What is cut, and what is deliberately NOT cut
---------------------------------------------
Only ``carry["atm"]["state"]`` (the spectral dynamical state, i.e. the
atmosphere's memory from one day to the next) passes through
``jax.lax.stop_gradient``. Left intact:

* ``carry["atm"]["derived"]`` — the previous day's surface fluxes and the MCB
  actuator field. The coupler copies ``derived.total_heat_flux`` into the
  ocean at the START of the next step (workflow ["coupling", "atm", "ocn",
  ...]), so cutting ``derived`` would sever the actuator -> flux -> ocean
  path, and at W = 1 the gradient would vanish entirely.
* ``carry["atm"]["forcing"]`` — SST and land temperature arriving FROM the
  slow components.
* ``carry["ocn"]`` / ``carry["lnd"]`` — the slow memory this estimator keeps.

``stop_gradient`` is the identity in the forward pass, so a truncated rollout
produces bit-identical forward values; only the backward pass changes.

Window alignment
----------------
A cut happens at the start of every coupling step whose day index ``d``
satisfies ``d % W == 0``, with ``d = round((sim_time - t0) / dt)`` read from
the OCEAN clock (``carry["ocn"]["state"].sim_time``). Never the scan index:
inner scans restart at 0 every control interval (the ENSO pacemaker's design
rule). With ``t0_seconds=None`` windows align to absolute model days, so one
compiled rollout serves every initial condition. Pass the episode's start
time to align windows to the episode instead. Both ``window_days`` and
``t0_seconds`` may be traced JAX scalars, so a single compiled function can
sweep window lengths and initial conditions without recompiling.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

# Window length standing in for "no truncation" when the window is a traced
# value (a Python None cannot be traced). Any rollout shorter than this is
# cut only at day 0, where nothing yet depends on the controls.
NO_TRUNCATION_DAYS = 1_000_000


def truncation_cut(
    sim_time,
    window_days,
    t0_seconds=None,
    coupling_timestep_seconds: float = 86400.0,
):
    """Whether the atmosphere's gradient is cut at the step starting now.

    Args:
        sim_time: Ocean clock (seconds) at the START of the coupling step.
        window_days: Truncation window W in days (int or traced int >= 1).
        t0_seconds: Episode start time, or None for absolute-day alignment.
        coupling_timestep_seconds: Coupling step length (86400 for daily).

    Returns:
        Boolean scalar: True when the day index is a multiple of W.

    """
    elapsed = sim_time if t0_seconds is None else sim_time - t0_seconds
    day = jnp.round(elapsed / coupling_timestep_seconds).astype(jnp.int32)
    window = jnp.asarray(window_days, dtype=jnp.int32)
    return jnp.mod(day, window) == 0


def stop_atmosphere_gradient(carry: dict, cut) -> dict:
    """Return a copy of ``carry`` whose atmospheric STATE is gradient-stopped.

    ``cut`` may be a traced boolean. Forward values are unchanged either way;
    when ``cut`` is True no cotangent flows back into the incoming
    ``carry["atm"]["state"]``. Everything else in the carry keeps its gradient
    (see the module docstring for why ``derived`` and ``forcing`` must).
    """
    def _maybe_stop(leaf):
        return jnp.where(cut, jax.lax.stop_gradient(leaf), leaf)

    atm = dict(carry["atm"])
    atm["state"] = jax.tree.map(_maybe_stop, atm["state"])
    new_carry = dict(carry)
    new_carry["atm"] = atm
    return new_carry


def wrap_step_fn_with_atm_truncation(
    step_fn: Callable,
    window_days: Optional[Any],
    t0_seconds=None,
    coupling_timestep_seconds: float = 86400.0,
) -> Callable:
    """Wrap a coupler step function so gradients forget old atmosphere.

    Args:
        step_fn: Coupler step function ``(carry, step_idx) -> (carry, preds)``.
        window_days: Truncation window W in days. None (or a Python int <= 0)
            returns ``step_fn`` unchanged (full BPTT). A traced value is
            allowed; use ``NO_TRUNCATION_DAYS`` for "no truncation" then.
        t0_seconds: Episode start (ocean ``sim_time``) for episode-aligned
            windows, or None for absolute-day alignment. May be traced.
        coupling_timestep_seconds: Coupling step length in seconds.

    Returns:
        A step function with the same signature, bit-identical forward
        values, and truncated reverse-mode gradients.

    """
    if window_days is None:
        return step_fn
    if isinstance(window_days, int) and window_days <= 0:
        return step_fn

    def truncated_step_fn(carry, step_idx):
        cut = truncation_cut(
            carry["ocn"]["state"].sim_time,
            window_days,
            t0_seconds=t0_seconds,
            coupling_timestep_seconds=coupling_timestep_seconds,
        )
        return step_fn(stop_atmosphere_gradient(carry, cut), step_idx)

    return truncated_step_fn
