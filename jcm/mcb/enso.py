"""ENSO pacemaker: relaxation-imposed Nino3.4 SST anomalies.

Implements the Molteni/Kucharski/Farneti (2024, WCD 5:293) style pacemaker
for the coupled slab model: inside a tapered Nino3.4 mask, sea surface
temperature is relaxed each coupling step toward a commanded anomaly on top
of a reference trajectory. Relaxation (not heat-flux forcing) is used
deliberately: a slab ocean under flux forcing can amplify or self-generate
El Nino-like variability with uncontrolled amplitude (Dommenget 2010, GRL),
whereas relaxation pins the realized anomaly to the command.

Design rules inherited from the 2026-08-03 verification of this experiment:
  * The anomaly phase is driven by ``sim_time`` from the ocean state, never
    by a scan index — inner scan indices reset every control interval and
    would silently phase-reset the ENSO.
  * The same wrapped step function must be used for a policy rollout AND its
    paired no-MCB baseline, so the imposed ENSO cancels exactly in paired
    anomalies and MCB attribution stays clean.
  * The reference the pacemaker relaxes toward is a pre-computed no-ENSO
    control trajectory from the same initial state (or a constant field),
    so the commanded anomaly is measured against the model's own evolution.

The pacemaker is pure JAX (jit/scan/grad-safe).
"""

from typing import NamedTuple, Tuple

import jax.numpy as jnp
import numpy as np


# Nino3.4 box (single source of truth — the pacemaker, the realized index,
# and the controller's absolute-box feature must agree on it).
NINO34_LAT_BOUNDS: Tuple[float, float] = (-5.0, 5.0)
NINO34_LON_BOUNDS: Tuple[float, float] = (190.0, 240.0)   # 170W-120W


class EnsoConfig(NamedTuple):
    """Configuration for the ENSO pacemaker.

    Attributes:
        amplitude: Peak commanded Nino3.4 SST anomaly (K). Positive =
            El Nino. The realized box anomaly tracks this closely for
            relax_tau_days << the slab's flux-damping timescale.
        period_days: Sinusoid period in days. 0 disables the oscillation:
            the anomaly holds at ``amplitude`` after the ramp (step
            response — the scoping-run mode).
        ramp_days: Linear ramp-in applied to the anomaly over the first
            ramp_days of the episode, avoiding an initialization shock.
        relax_tau_days: Pacemaker relaxation e-folding timescale (days).
        phase0: Sinusoid phase (radians) at episode start.
        lat_bounds: Core box latitude bounds in degrees (Nino3.4: -5..5).
        lon_bounds: Core box longitude bounds in degrees, 0-360 convention
            (Nino3.4: 190..240, i.e. 170W-120W). Wraparound handled.
        taper_lat_deg: Cosine taper width outside the lat core (degrees).
        taper_lon_deg: Cosine taper width outside the lon core (degrees).

    """

    amplitude: float = 2.0
    period_days: float = 0.0
    ramp_days: float = 30.0
    relax_tau_days: float = 5.0
    phase0: float = 0.0
    lat_bounds: Tuple[float, float] = NINO34_LAT_BOUNDS
    lon_bounds: Tuple[float, float] = NINO34_LON_BOUNDS
    taper_lat_deg: float = 3.0
    taper_lon_deg: float = 10.0


def _axis_weight(x_deg, lo, hi, taper):
    """1 inside [lo, hi], cosine-tapered to 0 over ``taper`` degrees outside.

    ``x_deg`` is recentred on the interval midpoint modulo 360 first, so a
    longitude interval crossing 0/360 works unchanged; latitudes never wrap
    but the recentring is harmless for them.
    """
    span = (hi - lo) % 360.0            # wraparound-safe (lo > hi allowed)
    half = 0.5 * span
    center = lo + half
    dx = jnp.abs((x_deg - center + 180.0) % 360.0 - 180.0)
    u = jnp.clip((half + taper - dx) / taper, 0.0, 1.0)
    return 0.5 * (1.0 - jnp.cos(jnp.pi * u))


def nino_pattern(grid, config: EnsoConfig = EnsoConfig()) -> jnp.ndarray:
    """Tapered Nino-box relaxation-strength pattern, shape (ix, il).

    1.0 in the core box, cosine-tapered to 0 over the configured widths.
    Used as a spatially varying relaxation strength, so the realized anomaly
    approaches the full commanded amplitude in the core and fades smoothly
    at the edges (no grid-scale SST fronts).
    """
    lats_deg = jnp.rad2deg(grid.latitudes)
    lons_deg = jnp.rad2deg(grid.longitudes)
    lon_grid, lat_grid = jnp.meshgrid(lons_deg, lats_deg, indexing="ij")
    w_lat = _axis_weight(lat_grid, config.lat_bounds[0], config.lat_bounds[1],
                         config.taper_lat_deg)
    w_lon = _axis_weight(lon_grid, config.lon_bounds[0], config.lon_bounds[1],
                         config.taper_lon_deg)
    return (w_lat * w_lon).astype(jnp.float32)


def enso_amplitude(sim_time, t0_seconds, config: EnsoConfig) -> jnp.ndarray:
    """Commanded anomaly A(t) in K at absolute ocean ``sim_time`` (seconds).

    Driven by sim_time minus the episode start time — NOT by any scan index.
    """
    dt_days = (sim_time - t0_seconds) / 86400.0
    ramp = jnp.clip(dt_days / jnp.maximum(config.ramp_days, 1e-9), 0.0, 1.0)
    if config.period_days > 0.0:
        osc = jnp.sin(2.0 * jnp.pi * dt_days / config.period_days
                      + config.phase0)
    else:
        osc = 1.0
    return config.amplitude * ramp * osc


def wrap_step_fn_with_enso(
    step_fn,
    pattern: jnp.ndarray,
    config: EnsoConfig,
    ref_sst,
    t0_seconds: float,
    coupling_timestep_seconds: float = 86400.0,
):
    """Wrap a coupler step function with the ENSO pacemaker nudge.

    After each inner step, ocean SST is relaxed toward
    ``ref + A(sim_time)`` with per-cell strength ``pattern``:

        sst <- sst + alpha * pattern * (ref + A - sst),
        alpha = 1 - exp(-coupling_timestep / tau)

    Args:
        step_fn: Coupler step function (carry, step_idx) -> (carry, preds).
        pattern: (ix, il) relaxation-strength pattern from nino_pattern().
        config: EnsoConfig (amplitude profile + timescale).
        ref_sst: Reference SST the anomaly is imposed on top of. Either a
            constant (ix, il) field, or a (T+1, ix, il) daily trajectory
            (e.g. CoupledBaselineTrajectory.sst from a no-ENSO control run
            started from the same carry) indexed by whole days since t0.
        t0_seconds: Ocean ``sim_time`` at episode start (read it from the
            initial carry: carry["ocn"]["state"].sim_time).
        coupling_timestep_seconds: Ocean step length (86400 for daily
            coupling).

    Returns:
        A step function with the same (carry, step_idx) -> (carry, preds)
        signature. Use it for BOTH the policy rollout and its paired
        baseline so the imposed ENSO cancels in every paired anomaly.

    """
    alpha = 1.0 - jnp.exp(-coupling_timestep_seconds
                          / (config.relax_tau_days * 86400.0))
    ref_arr = jnp.asarray(ref_sst)
    trajectory_ref = ref_arr.ndim == 3

    def enso_step_fn(carry, step_idx):
        new_carry, preds = step_fn(carry, step_idx)
        state = new_carry["ocn"]["state"]
        sst = state.sea_surface_temperature
        sim_time = state.sim_time
        if trajectory_ref:
            day = jnp.round((sim_time - t0_seconds)
                            / coupling_timestep_seconds).astype(jnp.int32)
            ref = ref_arr[jnp.clip(day, 0, ref_arr.shape[0] - 1)]
        else:
            ref = ref_arr
        amp = enso_amplitude(sim_time, t0_seconds, config)
        nudged = sst + alpha * pattern * (ref + amp - sst)
        new_carry = dict(new_carry)
        new_carry["ocn"] = dict(new_carry["ocn"])
        new_carry["ocn"]["state"] = state.copy(
            {"sea_surface_temperature": nudged})
        return new_carry, preds

    return enso_step_fn


def tail_reference_scale(days: int, tail_days: int) -> float:
    """Ramp-reference correction for a tail-averaged scoring metric.

    A deadbeat law that drives the INSTANTANEOUS anomaly along a linear ramp
    to ``target`` at day ``days`` is scored by a metric that averages the
    final ``tail_days``. A perfect tracker of that ramp therefore scores
    ``target * f`` with f = mean(t/days) over the tail window — a structural
    miss of (1-f)*|target| that no control authority can remove (measured in
    the Amendment-6 campaign: predicted +16.4 mK, observed +16.6 mK).

    Returns 1/f, the factor by which the law's reference must be scaled so
    that a perfect tracker scores exactly ``target``.
    """
    t = np.arange(days - tail_days + 1, days + 1, dtype=float)
    return float(1.0 / np.mean(t / days))


def make_enso_pi_policy_fn(
    enso_effect_per_K: float = 0.0525,
    gain_max: float = 4.0,
    ff_weight: float = 1.0,
    reference_scale: float = 1.0,
    dsst_idx: int = 0,
    time_idx: int = 10,
    nino_idx: int = 13,
):
    """Classical feedforward + proportional controller for the ENSO task.

    The honest hand-designed yardstick (Kravitz/MacMartin / Lee et al. 2025
    style: feedforward on the observed disturbance plus feedback on the
    tracking error), stateless per control interval. Features are paired
    against the NO-ENSO control baseline, so:

      features[dsst_idx] = realized global dSST (MCB effect + ENSO effect),
      features[time_idx] = time fraction in [0, 1],
      features[nino_idx] = realized Nino3.4 box anomaly ~ A(t).

    Command = pattern * gain with

      gain = clip(1 + err / (|target| * (1 - min(tfrac, 0.9)))       # FB
                    + ff_weight * enso_effect_per_K * nino / |target|,  # FF
                  0, gain_max)
      err  = realized - target * tfrac        # ramp on-track reference

    The feedback term is the Tier-2 deadbeat structure (close the remaining
    error in the remaining time; per-unit-gain final response = target,
    because the rescaled static pattern is calibrated to deliver the target
    at gain 1). The feedforward term pre-compensates the expected
    horizon-matched ENSO effect; ``enso_effect_per_K`` MUST be measured on
    the same variable the metric scores (Amendment-6 post-mortem: a value
    measured on atmospheric GMST under-scaled the ocean-dSST response by
    1.36x and crippled the open-loop control arm). The proportional term
    cleans up timing mismatch between the ENSO and MCB response lags.

    ``reference_scale`` (see tail_reference_scale) stretches the internal
    ramp reference so a perfect tracker scores ``target`` on a
    tail-AVERAGED metric instead of structurally undershooting it.

    Note the actuator FLOOR: gain is clipped below at 0, so a cold (La Nina)
    anomaly larger than |target| / enso_effect_per_K cannot be compensated
    even by spraying nothing. Amendment 7 sizes its amplitude range to sit
    exactly at that floor.

    params: {"pattern": (ix, il) rescaled static pattern, "target": float}
    """
    def pi_policy_fn(params, features):
        pattern = params["pattern"]
        target = params["target"] * reference_scale
        realized = features[dsst_idx]
        tfrac = features[time_idx]
        nino = features[nino_idx]
        abs_target = jnp.abs(target)
        err = realized - target * tfrac                  # >0 = too warm
        g_fb = err / (abs_target * (1.0 - jnp.minimum(tfrac, 0.9)))
        g_ff = ff_weight * enso_effect_per_K * nino / abs_target
        gain = jnp.clip(1.0 + g_fb + g_ff, 0.0, gain_max)
        return pattern * gain

    return pi_policy_fn


def make_enso_ff_mean_policy_fn(
    amp_mean: float,
    enso_effect_per_K: float = 0.0525,
    ramp_days: float = 30.0,
    horizon_days: float = 180.0,
    gain_max: float = 4.0,
    time_idx: int = 10,
):
    """Open-loop mean-feedforward schedule (the fair non-feedback control).

    Compensates the EXPECTED disturbance — the mean of the hidden amplitude
    distribution — on the standard pacemaker ramp, reading only the time
    feature. This is the Kravitz et al. (2014)-style "mis-specified
    feedforward": correct on average, wrong for every actual draw, and
    structurally unable to adapt. Distinguishes "feedback works" from "any
    schedule works" (per-episode randomized amplitudes are what a fixed
    schedule cannot track).

    params: {"pattern": (ix, il) rescaled static pattern, "target": float}
    """
    def ff_policy_fn(params, features):
        pattern = params["pattern"]
        target = params["target"]
        tfrac = features[time_idx]
        nino_expected = amp_mean * jnp.minimum(
            1.0, tfrac * horizon_days / ramp_days)
        gain = jnp.clip(
            1.0 + enso_effect_per_K * nino_expected / jnp.abs(target),
            0.0, gain_max)
        return pattern * gain

    return ff_policy_fn


def box_mean_weights(grid, pattern: jnp.ndarray,
                     area_weights: jnp.ndarray,
                     core_threshold: float = 0.999) -> jnp.ndarray:
    """Build normalized area weights over the pattern CORE.

    Defines the realized Nino3.4 index: only cells at full relaxation
    strength count, so the index is not diluted by the taper skirt.
    """
    core = (pattern >= core_threshold).astype(jnp.float32)
    w = core * area_weights
    return w / jnp.maximum(jnp.sum(w), 1e-12)
