"""Training the student straight through the simulation (Part 18 step 22).

MCB_PROJECT_REPORT.md Part 18 step 22: the same tiny network as the copying
route (``jcm.mcb.student``), trained three ways at the same GPU budget:

* ``bptt``: ordinary backpropagation through the whole closed-loop episode;
* ``snipped``: the same, with the atmosphere's derivatives cut every
  ``W* = 14`` days (Experiment 1's validated gradient);
* ``eki``: ensemble Kalman inversion, the climate community's standard
  gradient-free method for tuning chaotic models. An ensemble of parameter
  vectors moves toward lower misfit using only forward runs.

This re-tests the project's founding bet, which failed in Part 9, now with a
small network and possibly trustworthy gradients.

**The closed loop** (``make_closed_loop_rollout``) is ``run_episode`` written
inside JAX. Before each segment the student sees what ``StudentPolicy`` would
see in the same state, chooses the five settings, and the warmed coupled
model runs the segment with ``efficacy * settings``. Its forward values match
``run_episode`` with a ``StudentPolicy`` (tested). Gradients flow through the
student's inputs too, so a setting's later effect on what the student sees
is learned.

**The objective** (``episode_objective``) is the planner's map objective
averaged over the episode's segments, with the effort and movement
penalties. Each segment's error is that segment's mean SST minus the normal
climate's. ``episode_residuals`` returns the same objective as a sum of
squares, which is what ensemble Kalman inversion needs.

**Budgets.** Each method stops when its ``Budget`` of wall-clock GPU seconds
is spent. The copying route's budget includes its teacher.
"""

import dataclasses
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.flatten_util import ravel_pytree

from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K, apply_band_control
from jcm.mcb.gradient_truncation import (
    NO_TRUNCATION_DAYS,
    wrap_step_fn_with_atm_truncation,
)
from jcm.mcb.planner import W_STAR_DAYS
from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    pattern_objective,
    weighted_mean,
)
from jcm.mcb.student import (
    Params,
    StudentConfig,
    day_of_year,
    features,
    observe_bands,
    student_apply,
)
from jcm.mcb.test_world import Warming, wrap_step_fn_with_warming

METHODS = ("bptt", "snipped", "eki")


def target_segment_means(target_sst_daily, segment_days: int,
                         n_segments: int) -> jnp.ndarray:
    """Return ``(n_segments, ix, il)`` segment means of a daily SST, minus 288 K."""
    t = np.asarray(target_sst_daily, np.float64)[:n_segments * segment_days]
    if t.shape[0] < n_segments * segment_days:
        raise ValueError(f"the target covers {t.shape[0]} days, the episode "
                         f"needs {n_segments * segment_days}")
    means = t.reshape((n_segments, segment_days) + t.shape[1:]).mean(axis=1)
    return jnp.asarray(means - MAP_REFERENCE_K, jnp.float32)


def make_closed_loop_rollout(step_fn: Callable, patterns, observation_weights,
                             cfg: StudentConfig, segment_days: int,
                             n_segments: int) -> Callable:
    """Return the student's episode as one differentiable function.

    ``rollout(params, carry, target_means, efficacy, q_base, warming,
    window, atm_decay=None)`` returns:
    - ``(n_segments, ix, il)``, each segment's mean SST minus
      ``MAP_REFERENCE_K``;
    - ``(n_segments, K)``, the settings.

    ``target_means`` comes from ``target_segment_means``. ``window`` cuts the
    atmosphere's derivatives every that many days from the episode start
    (``NO_TRUNCATION_DAYS`` = none). Forward values do not depend on it.
    """
    if segment_days < 1 or n_segments < 1:
        raise ValueError("segment_days and n_segments must be >= 1")
    obs_w = jnp.asarray(observation_weights, jnp.float32)

    def rollout(params, carry, target_means, efficacy, q_base, warming,
                window, atm_decay=None):
        stepper = wrap_step_fn_with_warming(step_fn, q_base, warming)
        stepper = wrap_step_fn_with_atm_truncation(
            stepper, window, t0_seconds=carry["ocn"]["state"].sim_time,
            atm_decay=atm_decay)

        def day(c, step_idx):
            c, _ = stepper(c, step_idx)
            return c, c["ocn"]["state"].sea_surface_temperature - \
                MAP_REFERENCE_K

        def segment(state, n):
            c, previous, last_mean = state
            anomaly = last_mean - target_means[jnp.maximum(n - 1, 0)]
            bands = jnp.where(n > 0, observe_bands(anomaly, obs_w), 0.0)
            doy = day_of_year(c["ocn"]["state"].sim_time, cfg)
            a = student_apply(params, features(bands, doy, previous, cfg),
                              cfg)
            controlled = apply_band_control(c, efficacy * a, patterns)
            c, daily = lax.scan(jax.checkpoint(day), controlled,
                                jnp.arange(segment_days))
            mean = jnp.mean(daily, axis=0)
            return (c, a, mean), (mean, a)

        init = (carry, jnp.zeros(cfg.k, jnp.float32),
                jnp.zeros_like(target_means[0]))
        _, (means, actions) = lax.scan(segment, init, jnp.arange(n_segments))
        return means, actions

    return rollout


def _segment_slice(n: int, first_segment: int) -> slice:
    if not 0 <= first_segment < n:
        raise ValueError(f"first_segment must lie in [0, {n})")
    return slice(first_segment, n)


def episode_objective(segment_means, actions, target_means, weights,
                      alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                      mu: float = 0.0, lam: float = 0.0,
                      first_segment: int = 0):
    """Return the mean over segments of the map objective plus penalties.

    The movement penalty counts the change into every scored segment, the
    first one measured from zero.
    """
    used = _segment_slice(actions.shape[0], first_segment)
    errors = segment_means - target_means
    pattern = jax.vmap(lambda e: pattern_objective(e, weights, alpha, beta))(
        errors[used])
    steps = jnp.diff(actions, axis=0, prepend=jnp.zeros_like(actions[:1]))
    return (jnp.mean(pattern) + mu * jnp.mean(jnp.sum(actions[used] ** 2, 1))
            + lam * jnp.mean(jnp.sum(steps[used] ** 2, 1)))


def episode_residuals(segment_means, actions, target_means, weights,
                      alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                      mu: float = 0.0, lam: float = 0.0,
                      first_segment: int = 0):
    """Return a vector whose squared norm is ``episode_objective``."""
    used = _segment_slice(actions.shape[0], first_segment)
    n = actions.shape[0] - first_segment
    scale = 1.0 / jnp.sqrt(n)
    errors = (segment_means - target_means)[used]
    means = jax.vmap(lambda e: weighted_mean(e, weights))(errors)
    sw = jnp.sqrt(weights)
    spread = jnp.sqrt(beta) * sw[None] * (errors - means[:, None, None])
    steps = jnp.diff(actions, axis=0, prepend=jnp.zeros_like(actions[:1]))
    return scale * jnp.concatenate([
        jnp.sqrt(alpha) * means, spread.reshape(-1),
        jnp.sqrt(mu) * actions[used].reshape(-1),
        jnp.sqrt(lam) * steps[used].reshape(-1)])


class TrainingEpisode(NamedTuple):
    """Everything one training episode needs (the efficacy is hidden truth)."""

    carry: dict
    target_means: jnp.ndarray
    efficacy: jnp.ndarray
    q_base: jnp.ndarray
    warming: Warming


@dataclasses.dataclass(frozen=True)
class DirectConfig:
    """How a direct method trains. Revision 1 freezes the values."""

    method: str = "snipped"
    window_days: int = W_STAR_DAYS
    learning_rate: float = 1e-2
    clip_norm: float = 1.0
    alpha: float = PATTERN_ALPHA
    beta: float = PATTERN_BETA
    mu: float = 0.0
    lam: float = 0.0
    first_segment: int = 0
    ensemble: int = 32
    init_spread: float = 0.1
    noise_scale: float = 1.0
    seed: int = 0

    @property
    def window(self) -> int:
        """The snip: none for plain backpropagation."""
        return NO_TRUNCATION_DAYS if self.method == "bptt" else \
            self.window_days

    def validate(self):
        """Raise ValueError on settings the trainers cannot use."""
        checks = [
            (self.method in METHODS, f"method must be one of {METHODS}"),
            (self.window_days >= 1, "window_days must be >= 1"),
            (self.learning_rate > 0.0 and self.clip_norm > 0.0,
             "learning_rate and clip_norm must be > 0"),
            (min(self.alpha, self.beta, self.mu, self.lam) >= 0.0,
             "objective weights must be >= 0"),
            (self.ensemble >= 2, "ensemble must be >= 2"),
            (self.init_spread > 0.0 and self.noise_scale > 0.0,
             "init_spread and noise_scale must be > 0"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


class Budget:
    """Wall-clock seconds a method may spend; every timed call is charged."""

    def __init__(self, seconds: float):
        """Start an empty account with a limit of ``seconds``."""
        if not seconds > 0.0:
            raise ValueError("a budget must be > 0 seconds")
        self.limit = float(seconds)
        self.spent = 0.0

    def charge(self, seconds: float):
        """Add ``seconds`` to the account."""
        self.spent += float(seconds)

    @property
    def exhausted(self) -> bool:
        """Whether the account is spent."""
        return self.spent >= self.limit


def _timed(fn, *args):
    t0 = time.time()
    out = jax.block_until_ready(fn(*args))
    return out, time.time() - t0


def make_episode_loss(rollout: Callable, weights, cfg: DirectConfig
                      ) -> Callable:
    """Return ``loss(params, episode)``: the student's episode objective."""
    w = jnp.asarray(weights, jnp.float32)
    window = jnp.asarray(cfg.window)

    def loss(params, ep: TrainingEpisode):
        means, actions = rollout(params, ep.carry, ep.target_means,
                                 ep.efficacy, ep.q_base, ep.warming, window)
        return episode_objective(means, actions, ep.target_means, w,
                                 cfg.alpha, cfg.beta, cfg.mu, cfg.lam,
                                 cfg.first_segment)

    return loss


def train_gradient(rollout: Callable, weights, params0: Params,
                   episodes: Sequence[TrainingEpisode], cfg: DirectConfig,
                   budget: Budget, max_steps: Optional[int] = None):
    """Train by gradient descent (Adam, clipped) through the closed loop.

    One episode per step, cycling through ``episodes``. ``cfg.method``
    picks plain or snipped backpropagation. The function stops when the
    budget is spent (or after ``max_steps``) and returns the final
    parameters and a log of every step.
    """
    cfg.validate()
    if cfg.method == "eki":
        raise ValueError("use train_eki for ensemble Kalman inversion")
    grad_fn = jax.jit(jax.value_and_grad(make_episode_loss(rollout, weights,
                                                           cfg)))
    opt = optax.chain(optax.clip_by_global_norm(cfg.clip_norm),
                      optax.adam(cfg.learning_rate))
    params, state = params0, opt.init(params0)
    log: List[Dict] = []
    step = 0
    while not budget.exhausted and (max_steps is None or step < max_steps):
        ep = episodes[step % len(episodes)]
        (value, grads), seconds = _timed(grad_fn, params, ep)
        budget.charge(seconds)
        norm = float(optax.tree.norm(grads))
        if not np.isfinite(float(value)) or not np.isfinite(norm):
            log.append({"step": step, "nonfinite": True, "seconds": seconds})
            break
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        log.append({"step": step, "episode": step % len(episodes),
                    "objective": float(value), "grad_norm": norm,
                    "seconds": round(seconds, 3)})
        step += 1
    return params, log


def eki_update(theta, residuals, key, noise_scale: float = 1.0):
    """Return one ensemble Kalman inversion step toward zero residuals.

    Args:
        theta: ``(J, P)`` the ensemble of parameter vectors.
        residuals: ``(J, D)`` each member's residual vector (data minus
            prediction is ``-residual``, with data zero).
        key: PRNG key for the perturbed observations.
        noise_scale: the observation noise ``Gamma = gamma I`` relative to
            the ensemble's typical variance along its own directions in data
            space (the mean nonzero eigenvalue of its residual covariance).
            At 1, each update moves about halfway, like a Levenberg-Marquardt
            damping. Scaling by the variance per component instead would
            make ``gamma`` far too small when the residual vector is much
            longer than the ensemble. The real model's residuals are
            map-sized, and the ensemble would then collapse to a point in one
            step.

    Returns:
        ``(J, P)`` the updated ensemble. The Kalman gain is applied in the
        ensemble's own space (Woodbury identity), so it costs ``O(J^2 D)``
        however long the residual vector is.

    """
    theta = jnp.asarray(theta)
    g = jnp.asarray(residuals)
    j = g.shape[0]
    dth = theta - theta.mean(axis=0)
    dg = g - g.mean(axis=0)
    gamma = noise_scale * jnp.sum(dg ** 2) / (j * (j - 1)) + 1e-30
    innovation = jnp.sqrt(gamma) * jax.random.normal(key, g.shape) - g
    small = j * gamma * jnp.eye(j) + dg @ dg.T
    z = (innovation - (dg.T @ jnp.linalg.solve(small, dg @ innovation.T)).T
         ) / gamma
    return theta + (dth.T @ (dg @ z.T)).T / j


def train_eki(rollout: Callable, weights, params0: Params,
              episodes: Sequence[TrainingEpisode], cfg: DirectConfig,
              budget: Budget, max_iterations: Optional[int] = None):
    """Train by ensemble Kalman inversion with forward runs only.

    The ensemble starts as ``params0`` plus Gaussian spread ``init_spread``.
    Each iteration runs every member side by side on one episode (cycling
    through ``episodes``), all from the same weather. It then moves the
    ensemble toward zero residuals. The estimate is the ensemble mean.
    """
    cfg.validate()
    w = jnp.asarray(weights, jnp.float32)
    flat0, unravel = ravel_pytree(params0)
    key = jax.random.PRNGKey(cfg.seed)
    key, sub = jax.random.split(key)
    theta = flat0[None] + cfg.init_spread * jax.random.normal(
        sub, (cfg.ensemble, flat0.size))
    window = jnp.asarray(NO_TRUNCATION_DAYS)

    def residual(flat, ep):
        means, actions = rollout(unravel(flat), ep.carry, ep.target_means,
                                 ep.efficacy, ep.q_base, ep.warming, window)
        return episode_residuals(means, actions, ep.target_means, w,
                                 cfg.alpha, cfg.beta, cfg.mu, cfg.lam,
                                 cfg.first_segment)

    batched = jax.jit(jax.vmap(residual, in_axes=(0, None)))
    log: List[Dict] = []
    it = 0
    while not budget.exhausted and (max_iterations is None
                                    or it < max_iterations):
        ep = episodes[it % len(episodes)]
        g, seconds = _timed(batched, theta, ep)
        budget.charge(seconds)
        objectives = np.asarray(jnp.sum(g ** 2, axis=1))
        if not np.all(np.isfinite(objectives)):
            log.append({"iteration": it, "nonfinite": True,
                        "seconds": seconds})
            break
        key, sub = jax.random.split(key)
        theta = eki_update(theta, g, sub, cfg.noise_scale)
        log.append({"iteration": it, "episode": it % len(episodes),
                    "objective_mean": float(objectives.mean()),
                    "objective_min": float(objectives.min()),
                    "spread": float(jnp.mean(jnp.std(theta, axis=0))),
                    "seconds": round(seconds, 3)})
        it += 1
    return unravel(theta.mean(axis=0)), log
