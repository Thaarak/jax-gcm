"""The fixed-pattern ladder (MCB_PROJECT_REPORT.md Part 18 step 18).

Dubey et al.'s four fixed opponents for the planners of Experiment 3a. Each
rung changes one thing from the one before, so the gaps between rungs show
where a planner's win comes from:

1. ``uniform_cancel``: the same setting in every band, sized so the
   ocean-mean SST over the scoring window matches the normal climate. It
   tests *how much* to brighten.
2. ``uniform_effort``: the same setting in every band, at the planner's
   effort. It tests *where* to brighten, at equal effort.
3. ``planner_average``: the planner's own time-mean pattern, held fixed. It
   tests *when* the pattern changes.
4. ``linear_response``: the classical design, the best fixed setting under a
   linear model built from brute-force runs that brighten one band at a
   time. It tests *how the design was found*: one-knob-at-a-time model runs
   versus the planner's gradients.

Every design is a fixed band setting, run with
``jcm.mcb.test_world.constant_policy``.

**The linear model.** Over a window, the time-mean SST error of a fixed
setting ``a`` is ``d(a) = d_warm + sum_j a_j R_j``.
- ``d_warm`` is the uncontrolled warmed run's error against the normal
  climate.
- ``R_j`` is the response map per unit of band ``j``
  (``response_maps``).

The error is affine in ``a``, so Dubey et al.'s objective is an exact sum of
squares, ``||r0 + R a||^2`` (``linear_system``), and the best design inside
``[0, cap]`` is a bounded least-squares problem. Designs pooled over several
training states minimize the mean of their objectives. Any response maps can
be used, so the same solver also serves a design from a model-free guess.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear

from jcm.mcb.planner import gauss_newton_residuals
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA, effort
from jcm.mcb.test_world import BRIGHTENING_CAP

RUNGS = ("uniform_cancel", "uniform_effort", "planner_average",
         "linear_response")


def response_maps(base, runs, delta: float) -> np.ndarray:
    """Return per-unit response maps from one-sided step runs.

    Args:
        base: ``(..., ix, il)`` the run without brightening (the warmed run).
        runs: ``(K, ..., ix, il)`` run ``j`` holds band ``j`` at ``delta``
            and the others at zero, with the same members as ``base``.
        delta: the step size (albedo units).

    Returns:
        ``(K, ..., ix, il)``, ``(runs_j - base) / delta``.

    """
    if not delta > 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    runs = np.asarray(runs, np.float64)
    return (runs - np.asarray(base, np.float64)[None]) / delta


def _as_states(warm_errors, responses) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(S, ix, il)`` errors and ``(S, K, ix, il)`` responses."""
    w = np.asarray(warm_errors, np.float64)
    r = np.asarray(responses, np.float64)
    if w.ndim == 2:
        w, r = w[None], r[None]
    if (w.ndim != 3 or r.ndim != 4 or r.shape[0] != w.shape[0]
            or r.shape[2:] != w.shape[1:]):
        raise ValueError(f"need errors (S, ix, il) and responses "
                         f"(S, K, ix, il); got {w.shape} and {r.shape}")
    return w, r


def linear_system(warm_errors, responses, weights,
                  alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                  mu: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(r0, R)`` with ``||r0 + R a||^2`` = the mean objective over states.

    Each state's objective is ``alpha <d>^2 + beta Var(d) + mu |a|^2`` of
    its predicted error ``d(a)`` (``jcm.mcb.scores.segment_objective``
    without the movement term, which is zero for a fixed design).
    ``warm_errors`` is ``(ix, il)`` or ``(S, ix, il)``; ``responses`` is
    ``(K, ix, il)`` or ``(S, K, ix, il)``.
    """
    w_err, resp = _as_states(warm_errors, responses)
    n_states, k = resp.shape[:2]
    zeros = np.zeros(k)
    blocks_r, blocks_big = [], []
    for e, maps in zip(w_err, resp):
        r, big = gauss_newton_residuals(e, np.moveaxis(maps, 0, -1), weights,
                                        zeros, zeros, alpha, beta, 0.0, 0.0)
        blocks_r.append(r)
        blocks_big.append(big)
    scale = 1.0 / np.sqrt(n_states)
    r0 = np.concatenate(blocks_r) * scale
    big_r = np.concatenate(blocks_big) * scale
    if mu > 0.0:
        r0 = np.concatenate([r0, np.zeros(k)])
        big_r = np.concatenate([big_r, np.sqrt(mu) * np.eye(k)])
    return r0, big_r


def predicted_objective(amplitudes, warm_errors, responses, weights,
                        alpha: float = PATTERN_ALPHA,
                        beta: float = PATTERN_BETA, mu: float = 0.0) -> float:
    """Return the linear model's mean objective for a fixed setting."""
    r0, big_r = linear_system(warm_errors, responses, weights, alpha, beta, mu)
    r = r0 + big_r @ np.asarray(amplitudes, np.float64)
    return float(r @ r)


def linear_response_design(warm_errors, responses, weights,
                           alpha: float = PATTERN_ALPHA,
                           beta: float = PATTERN_BETA, mu: float = 0.0,
                           cap: float = BRIGHTENING_CAP) -> np.ndarray:
    """Return rung 4: the best fixed setting in ``[0, cap]`` under the linear model.

    It solves the bounded least-squares problem exactly (``scipy``'s BVLS),
    pooled over the states given.
    """
    r0, big_r = linear_system(warm_errors, responses, weights, alpha, beta, mu)
    sol = lsq_linear(big_r, -r0, bounds=(0.0, cap), method="bvls")
    return np.clip(sol.x, 0.0, cap)


def uniform_to_cancel(warm_errors, responses, weights,
                      cap: float = BRIGHTENING_CAP) -> Tuple[np.ndarray, dict]:
    """Return rung 1: the uniform setting that cancels the mean warming.

    The setting ``u`` in every band minimizes the mean over states of
    ``<d_warm + u sum_j R_j>^2``. That is the ocean-mean bias under the
    linear model, so the result cancels it exactly unless ``u`` has to be
    clipped to ``[0, cap]``.
    """
    w_err, resp = _as_states(warm_errors, responses)
    w = np.asarray(weights, np.float64)
    bias = np.sum(w * w_err, axis=(-2, -1))                     # (S,)
    per_unit = np.sum(w * resp.sum(axis=1), axis=(-2, -1))      # (S,)
    denom = float(per_unit @ per_unit)
    if denom == 0.0:
        raise ValueError("the bands have no effect on the ocean mean")
    u = -float(bias @ per_unit) / denom
    clipped = not 0.0 <= u <= cap
    u = float(np.clip(u, 0.0, cap))
    return np.full(resp.shape[1], u), {"uniform": u, "clipped": clipped}


def uniform_at_effort(target_effort: float, unit_profiles, sphere_weights,
                      cap: float = BRIGHTENING_CAP) -> Tuple[np.ndarray, dict]:
    """Return rung 2: the uniform setting with a given effort.

    Effort is Dubey et al.'s ``<|sum_j g_j a_j|>`` over the sphere
    (``jcm.mcb.scores.effort``). The profiles are non-negative, so it is
    linear in a uniform setting.
    """
    if target_effort < 0.0:
        raise ValueError("target_effort must be >= 0")
    k = np.shape(unit_profiles)[0]
    per_unit = effort(np.ones((1, k)), unit_profiles, sphere_weights)
    u = target_effort / per_unit
    clipped = u > cap
    u = float(min(u, cap))
    return np.full(k, u), {"uniform": u, "clipped": bool(clipped),
                           "effort_per_unit": float(per_unit)}


def planner_average(schedules, segment_days: int, start_day: int = 0,
                    end_day: Optional[int] = None) -> np.ndarray:
    """Return rung 3: the planner's time-mean setting over ``[start_day, end_day)``.

    ``schedules`` is ``(n_segments, K)`` or ``(members, n_segments, K)``.
    Every day counts equally, and members are averaged.
    """
    s = np.asarray(schedules, np.float64)
    if s.ndim == 2:
        s = s[None]
    daily = np.repeat(s, segment_days, axis=1)
    end = daily.shape[1] if end_day is None else end_day
    if not 0 <= start_day < end <= daily.shape[1]:
        raise ValueError(f"need 0 <= start ({start_day}) < end ({end}) <= "
                         f"{daily.shape[1]} days")
    return daily[:, start_day:end].mean(axis=(0, 1))


def ladder_designs(warm_errors, responses, weights, unit_profiles,
                   sphere_weights, planner_schedules: Optional[Sequence] = None,
                   planner_effort: Optional[float] = None,
                   segment_days: int = 14, alpha: float = PATTERN_ALPHA,
                   beta: float = PATTERN_BETA, mu: float = 0.0,
                   cap: float = BRIGHTENING_CAP) -> Dict[str, dict]:
    """Return every rung's setting and its predicted objective.

    Rungs 1 and 4 come from the response maps (training states). Rungs 2 and
    3 come from a planner run and are skipped without one.
    """
    out = {}
    a, info = uniform_to_cancel(warm_errors, responses, weights, cap)
    out["uniform_cancel"] = {"amplitudes": a, **info}
    if planner_effort is not None:
        a, info = uniform_at_effort(planner_effort, unit_profiles,
                                    sphere_weights, cap)
        out["uniform_effort"] = {"amplitudes": a, **info}
    if planner_schedules is not None:
        out["planner_average"] = {"amplitudes": planner_average(
            planner_schedules, segment_days)}
    out["linear_response"] = {"amplitudes": linear_response_design(
        warm_errors, responses, weights, alpha, beta, mu, cap)}
    for rung in out.values():
        rung["predicted_objective"] = predicted_objective(
            rung["amplitudes"], warm_errors, responses, weights, alpha, beta,
            mu)
        rung["amplitudes"] = np.asarray(rung["amplitudes"]).tolist()
    return out
