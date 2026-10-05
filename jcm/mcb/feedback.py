"""Classical feedback controllers (MCB_PROJECT_REPORT.md Part 18 step 19).

Both are ``run_episode`` policies that sense the way a real system could: the
ocean temperature averaged over the last segment (two weeks), against the
normal climate over the same days.

* ``IndexPIController`` (GLENS-style). It holds three numbers on target,
  ``T0`` (the ocean mean), ``T1`` (north minus south) and ``T2`` (equator
  minus poles), with proportional and true integral action on each. The
  integral is a running sum of past errors, which removes steady misses. A
  sensitivity matrix maps the three demands onto the five bands, inside
  ``[0, cap]``, and anti-windup keeps the integral from piling up while the
  bands are saturated. This follows the stratospheric-aerosol feedback of
  Kravitz et al. (2017) and MacMartin et al. (2017), GLENS.
* ``AdaptiveController``. This is the Tier 2 adaptive law of Parts 8-15,
  kept for continuity. It scales one fixed pattern by one gain, estimates
  how strongly the spraying works from realized versus expected cooling,
  divides by that estimate, and closes the gap. Tier 2 measured realized
  cooling against a same-weather twin and aimed at a fixed cooling. Neither
  exists in the test world, so here realized cooling is measured against
  the model's forecast of the uncontrolled warmed run (``forecast``
  sensing), and the target is to cancel that forecast warming.

**The plant model behind both** (Part 18; measured in Experiment 1): over
months, the slab ocean integrates the brightening. Holding band ``j`` at
``a_j`` changes index ``i`` at a nearly steady rate ``S_ij a_j`` (K per day).
``sensitivity_rates`` fits ``S`` from response series. Gains then follow from
a chosen closed-loop time scale (``pi_gains``) instead of a tuning sweep.
"""

import dataclasses
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import lsq_linear

from jcm.mcb.band_basis import OCEAN_OBJECTIVES
from jcm.mcb.test_world import BRIGHTENING_CAP, EpisodeState


# --- Sensing ---------------------------------------------------------------

def last_segment_means(state: EpisodeState, daily, segment_days: int,
                       field: str = "sst"):
    """Return the last segment's mean of a field and of a reference.

    Args:
        state: the policy's ``EpisodeState``. At segment 0 there is no last
            segment yet, and the function returns None.
        daily: ``(n_days, ix, il)`` reference (normal climate or forecast),
            aligned with the episode's daily fields (index ``d`` = end of
            day ``d + 1``).
        segment_days: days per segment.
        field: which of ``state.last_fields`` to average.

    Returns:
        ``(observed, reference)`` mean maps over days ``[day - segment_days,
        day)``, or None at segment 0.

    """
    if state.segment == 0 or state.last_fields is None:
        return None
    start, end = state.day - segment_days, state.day
    if start < 0 or end > np.shape(daily)[0]:
        raise ValueError(f"days [{start}, {end}) are outside the "
                         f"{np.shape(daily)[0]} reference days")
    observed = np.asarray(state.last_fields[field], np.float64).mean(axis=0)
    reference = np.asarray(daily[start:end], np.float64).mean(axis=0)
    return observed, reference


def ocean_indices(anomaly_map, weight_stack) -> np.ndarray:
    """Return ``(T0, T1, T2)`` of an SST anomaly map (``jcm.mcb.band_basis``)."""
    w = np.asarray(weight_stack, np.float64)[:len(OCEAN_OBJECTIVES)]
    return np.tensordot(w, np.asarray(anomaly_map, np.float64),
                        axes=([1, 2], [0, 1]))


def segment_index_means(daily, weight_stack, segment_days: int,
                        reference_daily=None) -> np.ndarray:
    """Return ``(n_segments, 3)`` segment means of ``T0..T2``.

    With ``reference_daily`` the indices are of ``daily - reference_daily``
    (an anomaly, computed before summing to keep float32 precision).
    """
    x = np.asarray(daily, np.float64)
    if reference_daily is not None:
        x = x - np.asarray(reference_daily, np.float64)[:x.shape[0]]
    n = x.shape[0] // segment_days
    seg = x[:n * segment_days].reshape((n, segment_days) + x.shape[1:])
    return np.stack([ocean_indices(m, weight_stack) for m in seg.mean(axis=1)])


# --- The plant model ---------------------------------------------------------

def sensitivity_rates(responses, days: Optional[Sequence[int]] = None
                      ) -> np.ndarray:
    """Fit index rates ``S`` (K per day per unit albedo) to response series.

    Args:
        responses: ``(n_days, n_indices, K)``, the change of each index per
            unit of each band held from day 0; entry ``d`` is the end of
            day ``d + 1``.
        days: which entries to fit (default all).

    Returns:
        ``(n_indices, K)`` slopes of a line through the origin, which is
        the integrator model's rate.

    """
    r = np.asarray(responses, np.float64)
    idx = np.arange(r.shape[0]) if days is None else np.asarray(days)
    t = (idx + 1.0)[:, None, None]
    return np.sum(t * r[idx], axis=0) / np.sum(t[:, 0, 0] ** 2)


def pi_gains(closed_loop_days: float, damping: float = 1.0
             ) -> Tuple[float, float]:
    """Return ``(kp, ki)`` for an integrator plant.

    For ``de/dt = -v + disturbance`` with ``v = kp e + ki * integral(e)``,
    the closed loop is ``e'' + kp e' + ki e = disturbance'``. A natural
    frequency of ``1 / closed_loop_days`` and the given damping give
    ``kp = 2 damping / tau`` (per day) and ``ki = 1 / tau^2`` (per day^2).
    """
    if not closed_loop_days > 0.0 or not damping > 0.0:
        raise ValueError("closed_loop_days and damping must be > 0")
    return 2.0 * damping / closed_loop_days, 1.0 / closed_loop_days ** 2


def bands_for_rates(demand, sensitivity, cap: float = BRIGHTENING_CAP,
                    ridge: float = 1e-6) -> np.ndarray:
    """Return band settings in ``[0, cap]`` whose index rates best meet a demand.

    ``demand`` (n,) is the cooling rate wanted on each index (K per day), so
    the bands should give ``sensitivity @ a = -demand``. A small ridge
    (relative to ``||S||^2``) picks the smallest of several equal fits,
    since five bands can meet three demands in many ways.
    """
    s = np.atleast_2d(np.asarray(sensitivity, np.float64))
    k = s.shape[1]
    lam = np.sqrt(ridge * np.sum(s ** 2) / k)
    big = np.concatenate([s, lam * np.eye(k)])
    rhs = np.concatenate([-np.asarray(demand, np.float64), np.zeros(k)])
    return np.clip(lsq_linear(big, rhs, bounds=(0.0, cap),
                              method="bvls").x, 0.0, cap)


# --- GLENS-style controller ------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PIConfig:
    """Settings of ``IndexPIController``.

    ``closed_loop_days`` sets how fast errors are corrected (``pi_gains``).
    Revision 1 freezes it. The default of three segments keeps the loop
    well inside its stability limit with two-week updates and a
    two-week-mean sensor. ``indices`` chooses which of T0, T1 and T2 are
    held. With ``feedforward`` the controller also cancels the forecast's
    warming rate, as GLENS adds a feedforward from the known scenario.
    """

    closed_loop_days: float = 42.0
    damping: float = 1.0
    segment_days: int = 14
    cap: float = BRIGHTENING_CAP
    ridge: float = 1e-6
    indices: Tuple[str, ...] = OCEAN_OBJECTIVES
    anti_windup: bool = True
    feedforward: bool = False

    def validate(self):
        """Raise ValueError on settings the controller cannot use."""
        if not self.indices or not set(self.indices) <= set(OCEAN_OBJECTIVES):
            raise ValueError(f"indices must be a non-empty subset of "
                             f"{OCEAN_OBJECTIVES}")
        if self.segment_days < 1 or not self.cap > 0.0:
            raise ValueError("segment_days >= 1 and cap > 0 are required")
        pi_gains(self.closed_loop_days, self.damping)


class IndexPIController:
    """Proportional-integral control of T0, T1 and T2, usable as a policy.

    Args:
        sensitivity: ``(3, K)`` index rates per unit band (K per day), for
            T0, T1 and T2 (``sensitivity_rates``).
        weight_stack: ``(>= 3, ix, il)`` index weights
            (``jcm.mcb.band_basis.stack_objective_weights``).
        target_sst_daily: ``(n_days, ix, il)`` the normal climate's SST.
        config: a ``PIConfig``.
        forecast_sst_daily: the forecast of the uncontrolled warmed run (its
            SST), needed only with ``feedforward``.

    Every call logs the measured errors, the integral, the demand, the
    realized rates and the settings in ``self.log``. A controller holds its
    integral, so use a new one per episode.

    """

    def __init__(self, sensitivity, weight_stack, target_sst_daily,
                 config: PIConfig = PIConfig(), forecast_sst_daily=None):
        """Store the plant model and set the gains."""
        config.validate()
        if config.feedforward and forecast_sst_daily is None:
            raise ValueError("feedforward needs forecast_sst_daily")
        self.cfg = config
        self.rows = [OCEAN_OBJECTIVES.index(n) for n in config.indices]
        self.sensitivity = np.asarray(sensitivity, np.float64)[self.rows]
        self.weight_stack = np.asarray(weight_stack, np.float64)
        self.target = np.asarray(target_sst_daily, np.float64)
        self.kp, self.ki = pi_gains(config.closed_loop_days, config.damping)
        self.forecast_indices = None
        if config.feedforward:
            self.forecast_indices = segment_index_means(
                forecast_sst_daily, self.weight_stack, config.segment_days,
                self.target)[:, self.rows]
        self.integral = np.zeros(len(self.rows))
        self.log = []

    def measure(self, state: EpisodeState) -> np.ndarray:
        """Return the held indices' errors over the last segment (zero at first)."""
        means = last_segment_means(state, self.target, self.cfg.segment_days)
        if means is None:
            return np.zeros(len(self.rows))
        return ocean_indices(means[0] - means[1],
                             self.weight_stack)[self.rows]

    def feedforward_rate(self, segment: int) -> np.ndarray:
        """Return the forecast's warming rate of the indices over a segment."""
        if self.forecast_indices is None:
            return np.zeros(len(self.rows))
        w = self.forecast_indices
        if segment >= w.shape[0]:
            raise ValueError(f"the forecast covers {w.shape[0]} segments")
        before = w[segment - 1] if segment > 0 else np.zeros_like(w[0])
        return (w[segment] - before) / self.cfg.segment_days

    def step(self, error, feedforward=None) -> np.ndarray:
        """Advance the controller by one segment and return the band settings.

        ``error`` holds the held indices' measured errors (K); the
        integral grows by ``error * segment_days``.
        """
        e = np.asarray(error, np.float64)
        ff = np.zeros_like(e) if feedforward is None else np.asarray(
            feedforward, np.float64)
        self.integral = self.integral + e * self.cfg.segment_days
        demand = ff + self.kp * e + self.ki * self.integral
        a = bands_for_rates(demand, self.sensitivity, self.cfg.cap,
                            self.cfg.ridge)
        realized = -self.sensitivity @ a
        if self.cfg.anti_windup:
            # Back-calculation: reset the integral to what the bands could
            # deliver, so saturation does not wind it up.
            self.integral = self.integral + (realized - demand) / self.ki
        self.log.append({"error": e.tolist(), "integral": self.integral.tolist(),
                         "demand": demand.tolist(),
                         "realized": realized.tolist(), "settings": a.tolist()})
        return a

    def __call__(self, state: EpisodeState) -> np.ndarray:
        a = self.step(self.measure(state), self.feedforward_rate(state.segment))
        self.log[-1].update(segment=state.segment, day=state.day)
        return a.astype(np.float32)


# --- The Tier 2 adaptive law, ported ----------------------------------------------

@dataclasses.dataclass(frozen=True)
class AdaptiveConfig:
    """Settings of ``AdaptiveController``.

    ``eta_clip`` bounds the strength estimate, as in Tier 2. The estimate
    stays at 1 until the expected cooling exceeds ``min_expected_k``,
    about the sensor's weather noise. ``relaxation`` is the fraction of the
    gap to the gain that would close it in one segment (1 = deadbeat).
    """

    segment_days: int = 14
    cap: float = BRIGHTENING_CAP
    eta_clip: Tuple[float, float] = (0.25, 4.0)
    min_expected_k: float = 0.02
    relaxation: float = 0.5

    def validate(self):
        """Raise ValueError on settings the controller cannot use."""
        lo, hi = self.eta_clip
        if not 0.0 < lo <= 1.0 <= hi:
            raise ValueError("eta_clip must satisfy 0 < low <= 1 <= high")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")
        if self.segment_days < 1 or not self.cap > 0.0:
            raise ValueError("segment_days >= 1 and cap > 0 are required")


class AdaptiveController:
    """One fixed pattern scaled by an efficacy-compensating gain (T0 only).

    Args:
        pattern: ``(K,)`` non-negative band pattern, scaled so its largest
            band is 1. The settings are ``gain * pattern``.
        t0_rates: ``(K,)`` T0 rate per unit of each band (K per day).
        weight_stack: index weights (T0 is used).
        target_sst_daily: the normal climate's SST, ``(n_days, ix, il)``.
        forecast_sst_daily: the forecast of the uncontrolled warmed run.
        config: an ``AdaptiveConfig``.

    The internal model: T0's anomaly is the forecast warming minus
    ``eta * s * integral(gain)``, where ``s`` is the pattern's cooling rate
    per unit gain and ``eta`` the unknown strength. A new controller is
    needed per episode.

    """

    def __init__(self, pattern, t0_rates, weight_stack, target_sst_daily,
                 forecast_sst_daily, config: AdaptiveConfig = AdaptiveConfig()):
        """Normalize the pattern and precompute the forecast warming."""
        config.validate()
        p = np.asarray(pattern, np.float64)
        if np.any(p < 0.0) or not p.max() > 0.0:
            raise ValueError("pattern must be non-negative and not all zero")
        self.cfg = config
        self.pattern = p / p.max()
        self.rate = -float(np.asarray(t0_rates, np.float64) @ self.pattern)
        if not self.rate > 0.0:
            raise ValueError("the pattern must cool the ocean mean")
        self.g_max = config.cap
        self.target = np.asarray(target_sst_daily, np.float64)
        self.warming = segment_index_means(
            forecast_sst_daily, weight_stack, config.segment_days,
            self.target)[:, 0]
        self.weight_stack = np.asarray(weight_stack, np.float64)
        self.gains = []
        self.eta_hat = 1.0
        self.log = []

    def step(self, segment: int, anomaly: Optional[float]) -> float:
        """Return the next gain from the last segment's T0 anomaly (K).

        ``anomaly`` is None at segment 0. Realized cooling is the forecast
        warming minus the measured anomaly; expected cooling is ``s`` times
        the integrated gain, averaged over the last segment.
        """
        cfg, d, s = self.cfg, self.cfg.segment_days, self.rate
        if segment >= self.warming.shape[0]:
            raise ValueError(f"the forecast covers {self.warming.shape[0]} "
                             f"segments")
        g_prev = self.gains[-1] if self.gains else 0.0
        realized = expected = 0.0
        if segment > 0 and anomaly is not None:
            realized = self.warming[segment - 1] - anomaly
            expected = s * d * (sum(self.gains[:-1]) + 0.5 * g_prev)
            if expected > cfg.min_expected_k:
                self.eta_hat = float(np.clip(realized / expected,
                                             *cfg.eta_clip))
        # Predicted cooling over the next segment, re-anchored on what was
        # realized: realized + eta s (g_prev d/2 + g d/2). Set it to the
        # forecast warming there.
        effective = self.eta_hat * s * 0.5 * d
        deadbeat = (self.warming[segment] - realized) / effective - g_prev
        gain = g_prev + cfg.relaxation * (deadbeat - g_prev)
        gain = float(np.clip(gain, 0.0, self.g_max))
        self.gains.append(gain)
        self.log.append({"segment": segment, "anomaly": anomaly,
                         "realized": realized, "expected": expected,
                         "eta_hat": self.eta_hat, "gain": gain})
        return gain

    def __call__(self, state: EpisodeState) -> np.ndarray:
        means = last_segment_means(state, self.target, self.cfg.segment_days)
        anomaly = None if means is None else float(
            ocean_indices(means[0] - means[1], self.weight_stack)[0])
        gain = self.step(state.segment, anomaly)
        return np.clip(gain * self.pattern, 0.0,
                       self.cfg.cap).astype(np.float32)
