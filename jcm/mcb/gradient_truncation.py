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

Damped alternative (exploratory, Amendment 9 revision 0.1)
----------------------------------------------------------
Instead of cutting the atmosphere's memory every W days, ``atm_decay`` makes
it fade: at the start of every coupling step the tangents (forward mode) and
cotangents (reverse mode) of ``carry["atm"]["state"]`` are multiplied by
``atm_decay``, so k days of atmospheric history carry weight
``atm_decay ** k``. With ``atm_decay = exp(-dt / tau)`` the memory has an
e-folding time of tau. This is a constant-rate analogue of the adjoint damping
Sugiura et al. (2008, JGR 113 C10017) used to keep a coupled 4D-Var adjoint
stable over 9-month windows (theirs adapts to the size of the sensitivity).
It tames the chaos only if 1/tau exceeds the growth rate of the gradient
noise. ``atm_decay = 1`` changes nothing and ``atm_decay = 0`` equals W = 1.
Forward values stay bit-identical, and the factor may be traced.
"""

import math
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


def atmosphere_decay_factor(efold_days: float,
                            coupling_timestep_seconds: float = 86400.0
                            ) -> float:
    """Per-step factor ``exp(-dt / tau)`` for an e-folding time of tau days."""
    if not efold_days > 0:
        raise ValueError(f"efold_days must be > 0, got {efold_days}")
    return math.exp(-coupling_timestep_seconds / (efold_days * 86400.0))


@jax.custom_jvp
def scale_tangent(x, factor):
    """Identity on values; multiplies tangents and cotangents by ``factor``.

    The JVP rule is linear in the tangent, so JAX transposes it for reverse
    mode: the same function damps forward-mode and reverse-mode derivatives.
    ``factor`` itself receives no derivative.
    """
    return x


@scale_tangent.defjvp
def _scale_tangent_jvp(primals, tangents):
    x, factor = primals
    x_dot, _ = tangents
    return x, factor * x_dot


def damp_atmosphere_gradient(carry: dict, factor) -> dict:
    """Return a copy of ``carry`` whose atmospheric STATE derivatives fade.

    Only ``carry["atm"]["state"]`` is affected, for the same reasons as in
    ``stop_atmosphere_gradient``. Non-float leaves are passed through, since
    they carry no derivative. Forward values are unchanged; ``factor`` may be
    traced.
    """
    def _damp(leaf):
        if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact):
            return scale_tangent(leaf, factor)
        return leaf

    atm = dict(carry["atm"])
    atm["state"] = jax.tree.map(_damp, atm["state"])
    new_carry = dict(carry)
    new_carry["atm"] = atm
    return new_carry


def wrap_step_fn_with_atm_truncation(
    step_fn: Callable,
    window_days: Optional[Any],
    t0_seconds=None,
    coupling_timestep_seconds: float = 86400.0,
    atm_decay: Optional[Any] = None,
) -> Callable:
    """Wrap a coupler step function so gradients forget old atmosphere.

    Args:
        step_fn: Coupler step function ``(carry, step_idx) -> (carry, preds)``.
        window_days: Truncation window W in days. None (or a Python int <= 0)
            means no truncation (full BPTT). A traced value is allowed; use
            ``NO_TRUNCATION_DAYS`` for "no truncation" then.
        t0_seconds: Episode start (ocean ``sim_time``) for episode-aligned
            windows, or None for absolute-day alignment. May be traced.
        coupling_timestep_seconds: Coupling step length in seconds.
        atm_decay: Optional per-step factor that damps the atmospheric state's
            derivatives at the start of every coupling step (see the module
            docstring); applied before the cut. None skips it. May be traced.

    Returns:
        A step function with the same signature, bit-identical forward
        values, and truncated and/or damped gradients. ``step_fn`` itself when
        there is neither a cut nor a decay.

    """
    no_cut = window_days is None or (isinstance(window_days, int)
                                     and window_days <= 0)
    if no_cut and atm_decay is None:
        return step_fn

    def truncated_step_fn(carry, step_idx):
        if atm_decay is not None:
            carry = damp_atmosphere_gradient(carry, atm_decay)
        if not no_cut:
            cut = truncation_cut(
                carry["ocn"]["state"].sim_time,
                window_days,
                t0_seconds=t0_seconds,
                coupling_timestep_seconds=coupling_timestep_seconds,
            )
            carry = stop_atmosphere_gradient(carry, cut)
        return step_fn(carry, step_idx)

    return truncated_step_fn
