"""Fixed-pattern design for Experiment 2 (Amendment 9 revision 1.1).

MCB_PROJECT_REPORT.md Part 18 steps 25-27 and Part 25. Every method chooses
ONE band setting, held for a whole episode, that cancels a growing warming
over the scoring window. All of them minimize the same objective: Dubey et
al.'s pattern objective of the window-mean SST error's ocean zonal-mean
profile, plus ``mu |a|^2``, pooled over the training states, inside
``[0, cap]``. For a fixed setting this objective is exactly the bounded
least-squares problem of ``jcm.mcb.ladder`` once each state's window-mean
map is linear in the setting. The methods differ only in where that linear
model comes from:

* ``gauss_newton_design`` (the gradient). Each iteration runs one weather
  sample per training state, with the forward-mode Jacobian of the
  window-mean SST map (``make_window_jacobian_fn``; the atmosphere snipped
  every W days, or not at all). It then solves the bounded least-squares
  problem of the model linearized at the current setting. The forward
  errors are exact, so repeating the step corrects a Jacobian that is
  merely too small. The 14-day snip undercounts the response by about 30%
  at 120 days (Experiment 1). One step therefore overshoots by about 1/0.7,
  as the 120-day planner did in Experiment 3a, while k steps leave about
  |1 - 1/c|^k of the error.
* ``brute_force_design``. Response maps come from one-band step runs
  averaged over weather samples. This is the classical linear-response
  design, rung 4 of the ladder.
* ``sunlight_design``. A guess that uses no run of the brightening: Duncan's
  back-of-envelope (MCB_PROJECT_REPORT.md Part 20.2). Each band cools a cell
  at the rate (sunlight at the top of the atmosphere x the model's own
  cloud cover x the band's profile) / (the slab's heat capacity), and the
  warming heats it at (the warming flux) / (the slab's heat capacity), both
  accumulated over the episode without feedbacks.
* ``ladder.uniform_to_cancel``: one setting in every band, sized to cancel
  the mean warming under the brute-force model.

The knob layouts are ``b5`` (Dubey et al.'s five bands) and ``b13``
(thirteen narrow bands, 60S to 60N every 10 degrees).
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jcm.mcb.band_basis import (
    DEFAULT_BAND_CENTERS_DEG,
    DEFAULT_BAND_WIDTH_DEG,
    gaussian_band_patterns,
)
from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K, apply_band_control
from jcm.mcb.gradient_truncation import wrap_step_fn_with_atm_truncation
from jcm.mcb.ladder import linear_response_design, predicted_objective
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA, zonal_projection
from jcm.mcb.test_world import BRIGHTENING_CAP, wrap_step_fn_with_warming

# Knob layouts: (band centres in degrees, north to south; Gaussian width).
BAND_LAYOUTS: Dict[str, Tuple[Tuple[float, ...], float]] = {
    "b5": (tuple(DEFAULT_BAND_CENTERS_DEG), DEFAULT_BAND_WIDTH_DEG),
    "b13": (tuple(float(c) for c in range(60, -61, -10)), 5.0),
}
EPISODE_DAYS = 182
SCORE_WINDOW = (98, 182)

# The sunlight guess's physics (the model's own slab constants; jem).
SOLAR_CONSTANT = 1361.0          # W m-2
OCEAN_DENSITY = 1025.0           # kg m-3 (jem.constants)
OCEAN_HEAT_CAPACITY = 3985.0     # J kg-1 K-1 (jem.constants)
MIXED_LAYER_MIN = 40.0           # m, at the equator (slab default)
MIXED_LAYER_MAX = 60.0           # m, at the poles (slab default)
YEAR_DAYS = 365.2425
SECONDS_PER_DAY = 86400.0


# --- Knob layouts ------------------------------------------------------------------

def band_patterns(layout: str, latitudes_rad, ocean_mask) -> jnp.ndarray:
    """Return a layout's ocean-masked band patterns, ``(K, ix, il)``."""
    centers, width = BAND_LAYOUTS[layout]
    return gaussian_band_patterns(latitudes_rad, ocean_mask, centers, width)


def unit_profiles(layout: str, latitudes_rad, shape) -> np.ndarray:
    """Return a layout's band profiles before the ocean mask (for the effort)."""
    centers, width = BAND_LAYOUTS[layout]
    return np.asarray(gaussian_band_patterns(latitudes_rad, np.ones(shape),
                                             centers, width))


# --- The gradient ---------------------------------------------------------------------

def make_window_sst_fn(step_fn: Callable, patterns, num_days: int,
                       window: Tuple[int, int] = SCORE_WINDOW) -> Callable:
    """Return the window-mean SST map as a function of a fixed setting.

    ``f(amplitudes, carry, efficacy, q_base, warming, snip, atm_decay=None)``
    holds ``efficacy * amplitudes`` for ``num_days`` warmed coupled days and
    returns the mean over days ``[start, end)`` of the SST minus
    ``MAP_REFERENCE_K``, ``(ix, il)``. Day ``d`` is the state after ``d + 1``
    steps, as in ``run_episode``. The atmosphere's derivatives are cut every
    ``snip`` days from the run's start (``NO_TRUNCATION_DAYS``: never).
    Forward values do not depend on ``snip``.
    """
    start, end = window
    if not 0 <= start < end <= num_days:
        raise ValueError(f"need 0 <= start ({start}) < end ({end}) <= "
                         f"num_days ({num_days})")

    def window_mean(amplitudes, carry, efficacy, q_base, warming, snip,
                    atm_decay=None):
        stepper = wrap_step_fn_with_warming(step_fn, q_base, warming)
        stepper = wrap_step_fn_with_atm_truncation(
            stepper, snip, t0_seconds=carry["ocn"]["state"].sim_time,
            atm_decay=atm_decay)
        controlled = apply_band_control(carry, efficacy * amplitudes,
                                        patterns)

        def body(c, step_idx):
            c, _ = stepper(c, step_idx)
            return c, c["ocn"]["state"].sea_surface_temperature - \
                MAP_REFERENCE_K

        _, daily = lax.scan(jax.checkpoint(body), controlled,
                            jnp.arange(num_days))
        return jnp.mean(daily[start:end], axis=0)

    return window_mean


def make_window_jacobian_fn(step_fn: Callable, patterns, num_days: int,
                            window: Tuple[int, int] = SCORE_WINDOW
                            ) -> Callable:
    """Return the jitted forward-mode Jacobian of the window-mean SST map.

    ``jac(amplitudes, carry, efficacy, q_base, warming, snip)`` returns
    ``(jacobian (ix, il, K), map (ix, il))`` from ONE forward pass with K
    tangents. Its cost grows with K but needs no weather ensemble.
    """
    window_mean = make_window_sst_fn(step_fn, patterns, num_days, window)

    def twice(a, carry, efficacy, q_base, warming, snip):
        out = window_mean(a, carry, efficacy, q_base, warming, snip)
        return out, out

    return jax.jit(jax.jacfwd(twice, has_aux=True))


# --- One pooled linear solve, shared by every method ---------------------------------

def zonal_design(warm_errors, responses, ocean_mask, weights,
                 alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                 mu: float = 0.0, cap: float = BRIGHTENING_CAP
                 ) -> Tuple[np.ndarray, float]:
    """Solve the pooled zonal-mean design problem exactly.

    Args:
        warm_errors: ``(S, ix, il)`` each state's error with no brightening
            under the linear model (K).
        responses: ``(S, K, ix, il)`` each state's response per unit of
            each band (K per unit albedo).
        ocean_mask: ``(ix, il)`` 0/1.
        weights: ``(ix, il)`` ocean area weights.
        alpha, beta, mu: the objective's weights.
        cap: the largest setting.

    Returns:
        The setting in ``[0, cap]`` and its predicted objective. Both maps
        are projected on their ocean zonal means first, so only the
        latitude profile is scored.

    """
    mask = np.asarray(ocean_mask, np.float64)
    warm_z = zonal_projection(np.asarray(warm_errors, np.float64), mask)
    resp_z = zonal_projection(np.asarray(responses, np.float64), mask)
    a = linear_response_design(warm_z, resp_z, weights, alpha, beta, mu, cap)
    return a, predicted_objective(a, warm_z, resp_z, weights, alpha, beta, mu)


def zonal_objective(error_map, ocean_mask, weights,
                    alpha: float = PATTERN_ALPHA,
                    beta: float = PATTERN_BETA) -> float:
    """Return ``J_zonal`` of one window-mean error map (no penalties)."""
    from jcm.mcb.scores import pattern_objective
    mask = np.asarray(ocean_mask, np.float64)
    z = zonal_projection(np.asarray(error_map, np.float64), mask)
    return float(pattern_objective(z, np.asarray(weights, np.float64), alpha,
                                   beta))


# --- The gradient design ------------------------------------------------------------

def gauss_newton_design(evaluate: Callable, targets, ocean_mask, weights,
                        k: int, iterations: int,
                        alpha: float = PATTERN_ALPHA,
                        beta: float = PATTERN_BETA, mu: float = 0.0,
                        cap: float = BRIGHTENING_CAP,
                        start: Optional[Sequence[float]] = None
                        ) -> Tuple[np.ndarray, List[dict]]:
    """Iterate pooled Gauss-Newton steps from ``start`` (default zero).

    Args:
        evaluate: ``evaluate(state, amplitudes)`` returns that training
            state's window-mean map and its Jacobian, ``((ix, il),
            (ix, il, K))``, relative to the same reference as ``targets``.
        targets: ``(S, ix, il)`` the normal climate's window-mean maps.
        ocean_mask, weights: as in ``zonal_design``.
        k: number of bands.
        iterations: number of steps.
        alpha, beta, mu, cap: as in ``zonal_design``.
        start: the first linearization point (default all zeros).

    Returns:
        The last iterate and one log record per iteration. A record holds
        the point, the pooled training objective there (from the exact
        forward maps), the states used and the new point.
        - A state whose map or Jacobian is not finite sits out that step.
        - If every state is out, the point is kept.

    """
    a = np.zeros(k) if start is None else np.asarray(start, np.float64)
    targets = np.asarray(targets, np.float64)
    log: List[dict] = []
    for it in range(iterations):
        warm, resp, used, objectives = [], [], [], []
        for s in range(targets.shape[0]):
            mean_map, jac = evaluate(s, a)
            mean_map = np.asarray(mean_map, np.float64)
            jac = np.moveaxis(np.asarray(jac, np.float64), -1, 0)  # (K,..)
            if not (np.all(np.isfinite(mean_map))
                    and np.all(np.isfinite(jac))):
                continue
            error = mean_map - targets[s]
            objectives.append(zonal_objective(error, ocean_mask, weights,
                                              alpha, beta))
            warm.append(error - np.tensordot(a, jac, axes=1))
            resp.append(jac)
            used.append(s)
        record = {"iteration": it, "point": a.tolist(), "states_used": used,
                  "training_objective": (float(np.mean(objectives))
                                         if objectives else None)}
        if used:
            a, predicted = zonal_design(np.stack(warm), np.stack(resp),
                                        ocean_mask, weights, alpha, beta, mu,
                                        cap)
            record["predicted_objective"] = predicted
        record["new_point"] = a.tolist()
        log.append(record)
    return a, log


# --- Brute force ----------------------------------------------------------------------

def brute_force_design(warmed_members, band_members, targets, delta: float,
                       ocean_mask, weights, alpha: float = PATTERN_ALPHA,
                       beta: float = PATTERN_BETA, mu: float = 0.0,
                       cap: float = BRIGHTENING_CAP,
                       members: Optional[int] = None) -> Tuple[np.ndarray,
                                                               dict]:
    """Return the classical linear-response design from one-band step runs.

    Args:
        warmed_members: ``(S, M, ix, il)`` window-mean maps of the warmed
            runs with no brightening, per weather sample.
        band_members: ``(S, K, M, ix, il)`` the same with band ``k`` held
            at ``delta``, sharing the warmed runs' sample seeds.
        targets: ``(S, ix, il)`` the normal climate's window-mean maps.
        delta: the step (albedo units).
        ocean_mask, weights, alpha, beta, mu, cap: as in ``zonal_design``.
        members: use only the first ``members`` samples (default all).

    Returns:
        The setting, and a dict with the predicted objective and the
        uniform setting that cancels the mean warming (ladder rung 1).

    """
    from jcm.mcb.ladder import response_maps, uniform_to_cancel

    warmed_members = np.asarray(warmed_members, np.float64)
    band_members = np.asarray(band_members, np.float64)
    m = warmed_members.shape[1] if members is None else int(members)
    if not 1 <= m <= warmed_members.shape[1]:
        raise ValueError(f"members must lie in [1, {warmed_members.shape[1]}]")
    warmed = warmed_members[:, :m].mean(axis=1)                 # (S, ix, il)
    runs = band_members[:, :, :m].mean(axis=2)                  # (S, K, ...)
    resp = np.stack([response_maps(w, r, delta) for w, r in
                     zip(warmed, runs)])                         # (S, K, ...)
    warm = warmed - np.asarray(targets, np.float64)
    a, predicted = zonal_design(warm, resp, ocean_mask, weights, alpha, beta,
                                mu, cap)
    mask = np.asarray(ocean_mask, np.float64)
    u, info = uniform_to_cancel(zonal_projection(warm, mask),
                                zonal_projection(resp, mask), weights, cap)
    return a, {"predicted_objective": predicted, "members": m,
               "uniform_cancel": u.tolist(), "uniform_info": info}


def equal_cost_members(iterations: int, seconds_per_jacobian_run: float,
                       k: int, seconds_per_forward_run: float,
                       max_members: int = 5) -> int:
    """Return the samples brute force can afford for the gradient's model time.

    The gradient design costs ``iterations`` Jacobian runs per training
    state. Brute force costs ``k + 1`` forward runs (one per band plus the
    warmed run) per state and per sample. The result is clipped to
    ``[1, max_members]``.
    """
    if min(iterations, k, max_members) < 1 or min(
            seconds_per_jacobian_run, seconds_per_forward_run) <= 0.0:
        raise ValueError("iterations, k, max_members >= 1 and positive "
                         "timings are required")
    affordable = (iterations * seconds_per_jacobian_run
                  / ((k + 1) * seconds_per_forward_run))
    return int(min(max(np.floor(affordable), 1), max_members))


# --- The sunlight guess -----------------------------------------------------------------

def toa_insolation(latitudes_rad, day_of_year):
    """Return the daily-mean top-of-atmosphere insolation (W m-2).

    The formula is the standard one, with declination
    ``-23.44 deg * cos(2 pi (day + 10) / 365.2425)`` and no eccentricity.
    Arguments broadcast.
    """
    phi = np.asarray(latitudes_rad, np.float64)
    dec = np.radians(-23.44) * np.cos(2.0 * np.pi
                                      * (np.asarray(day_of_year) + 10.0)
                                      / YEAR_DAYS)
    h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0))
    return SOLAR_CONSTANT / np.pi * (h0 * np.sin(phi) * np.sin(dec)
                                     + np.cos(phi) * np.cos(dec) * np.sin(h0))


def slab_heat_capacity(latitudes_rad):
    """Return the slab's heat capacity per area (J m-2 K-1), as jem builds it."""
    depth = (MIXED_LAYER_MAX + (MIXED_LAYER_MIN - MIXED_LAYER_MAX)
             * np.cos(np.asarray(latitudes_rad, np.float64)) ** 3)
    return OCEAN_DENSITY * OCEAN_HEAT_CAPACITY * depth


def accumulated_window_mean(daily_flux, window: Tuple[int, int]):
    """Return the window mean of a flux accumulated from day 0 (J m-2).

    ``daily_flux`` is ``(n_days, ...)`` in W m-2. Row ``d`` is the mean
    over day ``d + 1``, and the state after day ``d + 1`` has absorbed
    rows ``0..d``, as in ``run_episode``.
    """
    flux = np.asarray(daily_flux, np.float64)
    start, end = window
    if not 0 <= start < end <= flux.shape[0]:
        raise ValueError("the window must lie inside the flux's days")
    accumulated = np.cumsum(flux, axis=0) * SECONDS_PER_DAY
    return accumulated[start:end].mean(axis=0)


def sunlight_guess_maps(latitudes_rad, patterns, cloud_daily,
                        start_day_of_year: float,
                        window: Tuple[int, int] = SCORE_WINDOW):
    """Return the guess's response maps per unit of each band, ``(K, ix, il)``.

    Holding band ``k`` at 1 reflects ``insolation x cloud cover x profile``
    more sunlight, which cools the slab at that rate over its heat
    capacity, with no feedbacks. ``cloud_daily`` is ``(n_days, ix, il)``;
    its row ``d`` is day ``d + 1`` of the episode, which starts on
    ``start_day_of_year``.
    """
    cloud = np.asarray(cloud_daily, np.float64)
    days = start_day_of_year + np.arange(cloud.shape[0]) + 0.5
    lat = np.broadcast_to(np.asarray(latitudes_rad, np.float64)[None, :],
                          cloud.shape[1:])
    sun = np.stack([toa_insolation(lat, d) for d in days])        # (n,ix,il)
    reflected = accumulated_window_mean(sun * cloud, window)        # J m-2
    cooling = reflected / slab_heat_capacity(lat)                   # K
    return -np.asarray(patterns, np.float64) * cooling[None]


def sunlight_guess_warming(latitudes_rad, ocean_mask, num_days: int,
                           step_wm2: float, ramp_wm2_per_day: float,
                           window: Tuple[int, int] = SCORE_WINDOW):
    """Return the guess's uncontrolled warming map (K), with no feedbacks.

    The flux is evaluated at mid-day, as ``wrap_step_fn_with_warming`` does.
    """
    mask = np.asarray(ocean_mask, np.float64)
    flux = step_wm2 + ramp_wm2_per_day * (np.arange(num_days) + 0.5)
    lat = np.broadcast_to(np.asarray(latitudes_rad, np.float64)[None, :],
                          mask.shape)
    heat = accumulated_window_mean(flux[:, None, None] * mask[None], window)
    return heat / slab_heat_capacity(lat) * mask


def sunlight_design(latitudes_rad, ocean_mask, patterns, cloud_daily,
                    start_day_of_year: float, weights, num_days: int,
                    step_wm2: float, ramp_wm2_per_day: float,
                    window: Tuple[int, int] = SCORE_WINDOW,
                    alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                    mu: float = 0.0, cap: float = BRIGHTENING_CAP
                    ) -> Tuple[np.ndarray, dict]:
    """Return the guess's design: its own warming and responses, solved exactly."""
    resp = sunlight_guess_maps(latitudes_rad, patterns, cloud_daily,
                               start_day_of_year, window)
    warm = sunlight_guess_warming(latitudes_rad, ocean_mask, num_days,
                                  step_wm2, ramp_wm2_per_day, window)
    a, predicted = zonal_design(warm[None], resp[None], ocean_mask, weights,
                                alpha, beta, mu, cap)
    return a, {"predicted_objective": predicted,
               "guess_mean_warming_K": float(np.sum(np.asarray(weights)
                                                    * warm))}
