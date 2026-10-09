"""The receding-horizon planner (MCB_PROJECT_REPORT.md Part 18 steps 13-15).

At the start of every segment (14 days) the planner starts from wherever the
simulated climate actually is. It chooses the five band settings that
minimize the map objective of ``jcm.mcb.scores`` over a look-ahead (60 days),
with the settings held for the whole look-ahead. It applies them for one
segment, then plans again.

* *The gradient.* It is the one Experiment 1 validated: the atmosphere's
  derivatives snipped every 14 days (W* = 14, Amendment 9 outcome A').
* *Copies.* Every gradient or Jacobian is averaged over ``copies`` slightly
  nudged copies of the current state (Dubey et al. use three). With
  ``batch_copies`` the copies run side by side in one vmapped call. On the
  GPU this halves Adam's cost and leaves Gauss-Newton's unchanged
  (step 17, ``run_planner_cost.py``).
  - *Over a few days* the two modes agree to float32 round-off (about 1e-3
    on the real model; exactly on the toy).
  - *Over two weeks or more* chaos amplifies that round-off, so on the GPU
    each mode samples different weather, as forward and backward mode did
    in Experiment 1. The results are then the same statistically, not
    bitwise.

Two optimizers, as Part 18 plans; the pilot (step 23) picks one:

* ``adam``: Dubey et al.'s 15 Adam steps at learning rate 0.1, on a smooth
  bounded parameterization ``a = cap * sigmoid(z)``, with a reverse-mode
  gradient.
* ``gauss_newton``: the objective is a sum of squares. Forward mode gives the
  whole look-ahead SST map's response to each band in one pass (K
  tangents), and each step solves a K x K linear system and then clips to
  ``[0, cap]``.

Safeguards (step 14):

* Every band stays inside ``[0, cap]``, through the smooth transform (Adam)
  or a box clip after each linear solve (Gauss-Newton). No knob can drift
  past a limit where its gradient is zero.
* The gradient's noise-to-signal ratio across the copies is logged at every
  re-plan, so a gradient that has faded into noise is seen, not silently
  used.
* The optimizer's LAST iterate is applied, never the best-looking one,
  which would reward luck among noisy evaluations.

The objective scores the ocean map, or with ``representation="zonal"``
(Gauss-Newton only) its latitude profile: the error map and every band's
response map are projected onto their ocean zonal means before the residuals
are formed. The step-23 pilot chose the zonal one (MCB_PROJECT_REPORT.md
Part 22).

What the planner knows (``sensing``):

* ``exact`` (Experiments 3a-3b): it forecasts from the true current state,
  weather included.
* ``ocean`` (Experiment 3c): it sees the ocean but not the weather. It
  forecasts from the true ocean plus its *background*: the atmosphere and
  land at the end of its own previous forecast, or on day 0 a state it is
  given. The forecast is otherwise unchanged, so it costs nothing extra.

The comparison planners of step 15 are presets: ``short14`` is a 14-day
look-ahead with exact backpropagation (Dubey et al.'s setting), and
``bptt60`` is 60 days without the snip.
"""

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA, zonal_projection
from jcm.mcb.strength_estimator import EstimatorConfig, StrengthEstimator
from jcm.mcb.test_world import (
    BRIGHTENING_CAP,
    EpisodeState,
    Warming,
    make_lookahead_objective,
    make_lookahead_sst_fn,
    perturb_member,
)

# Experiment 1's registered choice of window (outcome A', 2026-10-01).
W_STAR_DAYS = 14
OPTIMIZERS = ("adam", "gauss_newton")
# What the objective scores: the ocean map, or its latitude (zonal-mean)
# profile, which the step-23 pilot chose (MCB_PROJECT_REPORT.md Part 22).
REPRESENTATIONS = ("map", "zonal")
# What the planner knows of the current state (Experiment 3c).
SENSING = ("exact", "ocean")
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    """How the planner plans. The defaults are Part 18's starting values.

    The pilot (step 23) fixes the final values: the look-ahead, the
    optimizer, its iterations and step size, the number of copies, and the
    penalty weights ``mu`` and ``lam`` (in albedo units, so Dubey et al.'s
    kelvin-based weights do not carry over).
    """

    lookahead_days: int = 60
    window_days: int = W_STAR_DAYS
    copies: int = 3
    copy_amp: float = 0.001
    copy_seed0: int = 97000
    batch_copies: bool = False
    optimizer: str = "adam"
    iterations: int = 15
    learning_rate: float = 0.1
    alpha: float = PATTERN_ALPHA
    beta: float = PATTERN_BETA
    mu: float = 0.0
    lam: float = 0.0
    cap: float = BRIGHTENING_CAP
    first_guess: float = 0.5
    edge_margin: float = 0.02
    gn_damping: float = 1e-3
    representation: str = "map"
    sensing: str = "exact"

    def validate(self):
        """Raise ValueError on settings the planner cannot use."""
        checks = [
            (self.lookahead_days >= 1, "lookahead_days must be >= 1"),
            (self.window_days >= 1, "window_days must be >= 1"),
            (self.copies >= 1, "copies must be >= 1"),
            (self.copy_amp >= 0.0, "copy_amp must be >= 0"),
            (self.optimizer in OPTIMIZERS,
             f"optimizer must be one of {OPTIMIZERS}"),
            (self.iterations >= 1, "iterations must be >= 1"),
            (self.learning_rate > 0.0, "learning_rate must be > 0"),
            (min(self.alpha, self.beta, self.mu, self.lam) >= 0.0,
             "objective weights must be >= 0"),
            (self.cap > 0.0, "cap must be > 0"),
            (0.0 < self.first_guess < 1.0, "first_guess must lie in (0, 1)"),
            (0.0 < self.edge_margin < 0.5,
             "edge_margin must lie in (0, 0.5)"),
            (self.gn_damping >= 0.0, "gn_damping must be >= 0"),
            (self.representation in REPRESENTATIONS,
             f"representation must be one of {REPRESENTATIONS}"),
            (self.representation == "map" or self.optimizer == "gauss_newton",
             "the zonal representation is implemented for Gauss-Newton"),
            (self.sensing in SENSING, f"sensing must be one of {SENSING}"),
            (self.sensing == "exact" or self.optimizer == "gauss_newton",
             "ocean-only sensing is implemented for Gauss-Newton"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


PRESETS: Dict[str, PlannerConfig] = {
    "snipped60": PlannerConfig(),
    "short14": PlannerConfig(lookahead_days=14,
                             window_days=NO_TRUNCATION_DAYS),
    "bptt60": PlannerConfig(window_days=NO_TRUNCATION_DAYS),
}


def to_amplitudes(z, cap: float):
    """Map unconstrained ``z`` smoothly into ``(0, cap)``."""
    return cap * jax.nn.sigmoid(z)


def to_logits(amplitudes, cap: float, margin: float) -> np.ndarray:
    """Return the inverse of ``to_amplitudes``, keeping ``margin * cap`` from the bounds.

    The margin stops a warm start from sitting where the sigmoid is flat.
    """
    p = np.clip(np.asarray(amplitudes, np.float64) / cap, margin,
                1.0 - margin)
    return np.log(p / (1.0 - p))


def stack_carries(carries: Sequence[dict]) -> dict:
    """Stack copies of a carry along a new leading axis (for ``jax.vmap``)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *carries)


def noise_to_signal(gradients: Sequence[np.ndarray]) -> Optional[float]:
    """Return ``||std over copies|| / ||mean||``; None with fewer than two copies."""
    g = np.asarray(gradients, np.float64)
    if g.shape[0] < 2:
        return None
    norm = float(np.linalg.norm(g.mean(axis=0)))
    spread = float(np.linalg.norm(g.std(axis=0, ddof=1)))
    return spread / norm if norm > 0.0 else float("inf")


def gauss_newton_residuals(error, jacobian, weights, amplitudes, previous,
                           alpha: float, beta: float, mu: float, lam: float):
    """Residuals ``r`` and their Jacobian ``R`` with ``||r||^2`` = the objective.

    Args:
        error: ``(ix, il)`` look-ahead time-mean SST error (K).
        jacobian: ``(ix, il, K)`` its response to each band.
        weights: ``(ix, il)`` ocean area weights summing to one.
        amplitudes, previous: ``(K,)`` the current and previous settings.
        alpha, beta, mu, lam: the objective's weights.

    Returns:
        ``r`` and ``R``. ``||r||^2 = alpha <e>_w^2 + beta Var_w(e)
        + mu |a|^2 + lam |a - a_prev|^2``, and ``R = dr/da``, exact in the
        response map.

    """
    w = np.asarray(weights, np.float64)
    e = np.asarray(error, np.float64)
    m_jac = np.asarray(jacobian, np.float64)
    a = np.asarray(amplitudes, np.float64)
    p = np.asarray(previous, np.float64)
    k = a.size
    cells = w > 0.0
    sw = np.sqrt(w[cells])
    mean_e = float(np.sum(w * e))
    mean_jac = np.tensordot(w, m_jac, axes=([0, 1], [0, 1]))       # (K,)
    r = np.concatenate([[np.sqrt(alpha) * mean_e],
                        np.sqrt(beta) * sw * (e[cells] - mean_e),
                        np.sqrt(mu) * a, np.sqrt(lam) * (a - p)])
    big_r = np.concatenate([np.sqrt(alpha) * mean_jac[None, :],
                            np.sqrt(beta) * sw[:, None]
                            * (m_jac[cells] - mean_jac[None, :]),
                            np.sqrt(mu) * np.eye(k), np.sqrt(lam) * np.eye(k)])
    return r, big_r


class Planner:
    """A receding-horizon planner, usable as a ``run_episode`` policy.

    Args:
        step_fn: the coupled step function.
        patterns: ``(K, ix, il)`` band patterns (ocean-masked).
        ocean_weights: ``(ix, il)`` ocean area weights for the objective.
        target_sst_daily: ``(n_days, ix, il)`` the normal-climate SST (K),
            day 0 being the episode's first day; it must reach the last
            re-plan's look-ahead.
        q_base: the episode's base Q-flux.
        warming: the warming the planner believes in (the truth in
            Experiment 3a).
        efficacy: the spraying strength the planner believes in.
        config: a ``PlannerConfig``.
        seed_offset: separates the copy seeds of different episodes.
        background: with ocean-only sensing, the state whose atmosphere and
            land the planner assumes on day 0 (every part but the ocean is
            used).

    Every call logs a record in ``self.log``: the start and final settings,
    the objective at each iterate, and the noise-to-signal ratio across the
    copies at the first iterate.

    """

    def __init__(self, step_fn: Callable, patterns, ocean_weights,
                 target_sst_daily, q_base, warming: Warming,
                 efficacy: float = 1.0,
                 config: PlannerConfig = PlannerConfig(),
                 seed_offset: int = 0, background: Optional[dict] = None):
        """Build the jitted gradient (Adam) or Jacobian (Gauss-Newton)."""
        config.validate()
        if config.sensing == "ocean" and background is None:
            raise ValueError("ocean-only sensing needs a background: the "
                             "atmosphere and land assumed on day 0")
        self.cfg = config
        self.background = background
        self._forecast_final = None
        self.k = int(np.shape(patterns)[0])
        self.weights = np.asarray(ocean_weights, np.float64)
        self.target = np.asarray(target_sst_daily, np.float64)
        self.seed_offset = int(seed_offset)
        self.log: List[dict] = []
        # The strength the planner believes in, one value per band. It is an
        # argument of the jitted call, not a constant baked into it, so a
        # learning planner can change it between re-plans without
        # recompiling.
        self.efficacy = np.broadcast_to(
            np.asarray(efficacy, np.float32), (self.k,)).copy()
        if not np.all(self.efficacy > 0.0):
            raise ValueError("the believed strength must be > 0 in every band")
        # Per-latitude ocean weights, for comparing forecast and observed
        # latitude profiles.
        lat_w = self.weights.sum(axis=0)
        self.profile_weights = lat_w / lat_w.sum()
        self.prediction: Optional[dict] = None
        self.innovation: Optional[dict] = None
        window = jnp.asarray(config.window_days)
        if config.optimizer == "adam":
            objective = make_lookahead_objective(
                step_fn, patterns, ocean_weights, config.lookahead_days,
                config.alpha, config.beta, config.mu, config.lam)

            def objective_of_logits(z, carry, target_mean, previous, eff):
                return objective(to_amplitudes(z, config.cap), carry,
                                 target_mean, previous, eff, q_base, warming,
                                 window)

            evaluate = jax.value_and_grad(objective_of_logits)
            axes = (None, 0, None, None, None)
        else:
            ocean_only = config.sensing == "ocean"
            mean_sst = make_lookahead_sst_fn(step_fn, patterns,
                                             config.lookahead_days,
                                             return_final=ocean_only)

            def map_twice(a, carry, eff):
                out = mean_sst(a, carry, eff, q_base, warming, window)
                if ocean_only:       # also keep where the forecast ends
                    out, final = out
                    return out, (out, final)
                return out, out

            # (jacobian (ix, il, K), map (ix, il)) from ONE forward pass.
            evaluate = jax.jacfwd(map_twice, has_aux=True)
            axes = (None, 0, None)
        # Side by side, every copy goes through one vmapped call; otherwise
        # the copies run one after another through the same jitted function.
        self._evaluate = jax.jit(jax.vmap(evaluate, in_axes=axes)
                                 if config.batch_copies else evaluate)

    def target_mean(self, day: int) -> np.ndarray:
        """Return the normal climate's time-mean SST over the look-ahead from ``day``."""
        end = day + self.cfg.lookahead_days
        if day < 0 or end > self.target.shape[0]:
            raise ValueError(f"the look-ahead [{day}, {end}) runs past the "
                             f"{self.target.shape[0]} target days")
        return self.target[day:end].mean(axis=0).astype(np.float32)

    def copies_of(self, carry: dict, day: int) -> List[dict]:
        """Return the nudged copies of the current state (seeds unique per day)."""
        base = (self.cfg.copy_seed0 + 1_000_000 * self.seed_offset
                + 1_000 * day)
        return [perturb_member(carry, base + c, self.cfg.copy_amp)
                for c in range(self.cfg.copies)]

    def prepare_copies(self, carry: dict, day: int):
        """Return the copies as ``evaluate_copies`` takes them.

        This is a list when they run one after another, and one stacked
        carry when they run side by side.
        """
        copies = self.copies_of(carry, day)
        return stack_carries(copies) if self.cfg.batch_copies else copies

    def evaluate_copies(self, x, copies, target_mean=None, previous=None):
        """Return the planner's expensive call, for every copy.

        For Adam, at logits ``x``, it returns per-copy objective values
        ``(C,)`` and gradients ``(C, K)``. For Gauss-Newton, at amplitudes
        ``x``, it returns per-copy Jacobians ``(C, ix, il, K)`` and look-ahead
        mean-SST maps ``(C, ix, il)``. ``copies`` comes from
        ``prepare_copies``. A re-plan is ``iterations`` of these calls plus
        some cheap host arithmetic, which is what step 17 times.
        """
        x = jnp.asarray(x, jnp.float32)
        extra = ((jnp.asarray(target_mean), jnp.asarray(previous))
                 if self.cfg.optimizer == "adam" else ())
        eff = jnp.asarray(self.efficacy, jnp.float32)
        ocean_only = self.cfg.sensing == "ocean"
        if self.cfg.batch_copies:
            first, second = self._evaluate(x, copies, *extra, eff)
            if ocean_only:
                second, finals = second
                self._forecast_final = jax.tree_util.tree_map(
                    lambda v: v[0], finals)
            return (np.asarray(first, np.float64),
                    np.asarray(second, np.float64))
        outs = [self._evaluate(x, c, *extra, eff) for c in copies]
        if ocean_only:              # the first copy's forecast is kept
            self._forecast_final = outs[0][1][1]
            outs = [(o[0], o[1][0]) for o in outs]
        return (np.stack([np.asarray(o[0], np.float64) for o in outs]),
                np.stack([np.asarray(o[1], np.float64) for o in outs]))

    def profile(self, field):
        """Return the ocean latitude profile ``(..., il)`` of ``field`` (..., ix, il)."""
        mask = (self.weights > 0.0).astype(np.float64)
        counts = np.maximum(mask.sum(axis=0), 1.0)
        return (np.asarray(field, np.float64) * mask).sum(axis=-2) / counts

    def estimate(self, state: EpisodeState) -> dict:
        """Return the state the planner forecasts from.

        With exact sensing it is the true state. With ocean-only sensing it
        is the true ocean plus the background's atmosphere and land, which
        must be at the same time as the true state (the look-ahead must
        equal the segment).
        """
        if self.cfg.sensing == "exact":
            return state.carry
        now = float(state.carry["ocn"]["state"].sim_time)
        guess = float(self.background["ocn"]["state"].sim_time)
        if abs(now - guess) > 1.0:
            raise ValueError(f"the background is at {guess:.0f} s but the "
                             f"state at {now:.0f} s: with ocean-only sensing "
                             "the look-ahead must equal the segment")
        carry = dict(self.background)
        carry["ocn"] = state.carry["ocn"]
        return carry

    def observe(self, state: EpisodeState) -> Optional[dict]:
        """Compare the last re-plan's forecast with what happened.

        With Gauss-Newton every re-plan forecasts the look-ahead's mean SST
        and how it would change with each band's strength (from the Jacobian
        it plans with). When the look-ahead equals the segment that has just
        been run, the forecast and the observed latitude profiles are
        directly comparable. Returns (and keeps in ``self.innovation``) the
        observed and forecast profiles, the forecast's sensitivity to the
        strengths and the strength it assumed, or None when there is nothing
        to compare.
        """
        self.innovation = None
        pred = self.prediction
        if pred is None or state.last_fields is None:
            return None
        if state.day - pred["day"] != self.cfg.lookahead_days:
            return None
        sst = np.asarray(state.last_fields["sst"], np.float64).mean(axis=0)
        self.innovation = {
            "day": state.day,
            "observed": self.profile(sst - MAP_REFERENCE_K),
            "predicted": pred["profile"],
            "sensitivity": pred["strength_jacobian"],
            "belief": pred["efficacy"]}
        return self.innovation

    def __call__(self, state: EpisodeState) -> np.ndarray:
        self.observe(state)
        return self.plan(state)

    def plan(self, state: EpisodeState) -> np.ndarray:
        """Choose the next segment's settings from the current state."""
        t0 = time.time()
        cfg = self.cfg
        previous = np.asarray(state.previous, np.float32)
        start = (np.full(self.k, cfg.first_guess * cfg.cap, np.float32)
                 if state.segment == 0 else previous)
        copies = self.prepare_copies(self.estimate(state), state.day)
        target_mean = self.target_mean(state.day)
        if cfg.optimizer == "adam":
            final, record = self._adam(start, copies, target_mean, previous)
        else:
            final, record, last = self._gauss_newton(start, copies,
                                                     target_mean, previous)
            self._remember_forecast(state.day, final, *last)
            if cfg.sensing == "ocean":
                # The next re-plan starts from where this forecast ends.
                self.background = self._forecast_final
        if self.innovation is not None:
            r = self.innovation["observed"] - self.innovation["predicted"]
            record["innovation_ms"] = float(np.sum(self.profile_weights
                                                   * r ** 2))
        record.update(segment=state.segment, day=state.day,
                      optimizer=cfg.optimizer, start=start.tolist(),
                      final=final.tolist(),
                      efficacy_belief=self.efficacy.tolist(),
                      seconds=round(time.time() - t0, 2))
        self.log.append(record)
        return final

    def _remember_forecast(self, day, final, a_eval, jac, mean_map):
        """Keep the forecast for the settings actually applied.

        The look-ahead mean SST at ``final`` is linearized from the last
        evaluation at ``a_eval``. Because the bands enter as ``strength *
        settings``, the forecast's derivative with respect to band k's
        strength is ``(a_k / e_k)`` times its derivative with respect to
        the setting.
        """
        a = np.asarray(final, np.float64)
        predicted = mean_map + jac @ (a - np.asarray(a_eval, np.float64))
        jac_profile = self.profile(np.moveaxis(jac, -1, 0)).T     # (il, K)
        self.prediction = {
            "day": day, "profile": self.profile(predicted),
            "strength_jacobian": jac_profile * (a / self.efficacy)[None, :],
            "efficacy": self.efficacy.astype(np.float64).copy()}

    def _zonal(self, field):
        """Project ``field`` (..., ix, il) onto its ocean latitude profile."""
        return zonal_projection(np.asarray(field, np.float64),
                                self.weights > 0.0)

    def _adam(self, start, copies, target_mean, previous):
        cfg = self.cfg
        z = to_logits(start, cfg.cap, cfg.edge_margin)
        m1, m2 = np.zeros_like(z), np.zeros_like(z)
        b1, b2 = _ADAM_BETAS
        values, nts = [], None
        for it in range(cfg.iterations):
            vals, grads = self.evaluate_copies(z, copies, target_mean,
                                               previous)
            if it == 0:
                nts = noise_to_signal(list(grads))
            values.append(float(np.mean(vals)))
            g = grads.mean(axis=0)
            m1 = b1 * m1 + (1.0 - b1) * g
            m2 = b2 * m2 + (1.0 - b2) * g * g
            step = (m1 / (1.0 - b1 ** (it + 1))) / (
                np.sqrt(m2 / (1.0 - b2 ** (it + 1))) + _ADAM_EPS)
            z = z - cfg.learning_rate * step
        final = np.asarray(to_amplitudes(jnp.asarray(z), cfg.cap), np.float32)
        return final, {"objective": values, "noise_to_signal": nts}

    def _gauss_newton(self, start, copies, target_mean, previous):
        cfg = self.cfg
        a = np.clip(np.asarray(start, np.float64), 0.0, cfg.cap)
        target_anomaly = (np.asarray(target_mean, np.float64)
                          - MAP_REFERENCE_K)
        values, nts = [], None
        for it in range(cfg.iterations):
            a_eval = a.copy()
            jacs, maps = self.evaluate_copies(a, copies)
            normal = np.zeros((self.k, self.k))
            rhs = np.zeros(self.k)
            total, grads = 0.0, []
            for jac, mean_map in zip(jacs, maps):
                error = mean_map - target_anomaly
                if cfg.representation == "zonal":
                    error, jac = self._zonal(error), np.moveaxis(
                        self._zonal(np.moveaxis(jac, -1, 0)), 0, -1)
                r, big_r = gauss_newton_residuals(
                    error, jac, self.weights, a,
                    previous, cfg.alpha, cfg.beta, cfg.mu, cfg.lam)
                normal += big_r.T @ big_r
                rhs += big_r.T @ r
                total += float(r @ r)
                grads.append(2.0 * big_r.T @ r)
            n = len(grads)
            normal, rhs = normal / n, rhs / n
            if it == 0:
                nts = noise_to_signal(grads)
            values.append(total / n)
            damping = cfg.gn_damping * np.trace(normal) / self.k
            delta = np.linalg.solve(normal + damping * np.eye(self.k), -rhs)
            a = np.clip(a + delta, 0.0, cfg.cap)
        # The last evaluation, averaged over copies: the forecast that
        # ``observe`` checks at the next re-plan.
        last = (a_eval, jacs.mean(axis=0), maps.mean(axis=0))
        return a.astype(np.float32), {"objective": values,
                                      "noise_to_signal": nts}, last


class LearningPlanner(Planner):
    """A Gauss-Newton planner that learns how strongly each band works.

    Experiment 3b (MCB_PROJECT_REPORT.md Part 24). It starts from nominal
    strength 1 and, before every re-plan, compares its last forecast with
    what happened (``Planner.observe``) and updates its belief with a
    ``StrengthEstimator``. It never sees the true strength: only the
    simulated world applies that. The look-ahead must equal the segment
    length, so that each forecast covers exactly the segment that follows.

    Args:
        estimator: an ``EstimatorConfig``; every other argument is
            ``Planner``'s (the initial belief is always nominal).

    """

    def __init__(self, step_fn: Callable, patterns, ocean_weights,
                 target_sst_daily, q_base, warming: Warming,
                 estimator: EstimatorConfig,
                 config: PlannerConfig = PlannerConfig(),
                 seed_offset: int = 0, background: Optional[dict] = None):
        """Build the planner at nominal strength and the estimator."""
        if config.optimizer != "gauss_newton":
            raise ValueError("learning needs the Gauss-Newton planner, whose "
                             "Jacobian gives the forecast's sensitivity")
        super().__init__(step_fn, patterns, ocean_weights, target_sst_daily,
                         q_base, warming, 1.0, config, seed_offset,
                         background=background)
        self.estimator = StrengthEstimator(self.k, estimator)
        self.efficacy = self.estimator.strength().astype(np.float32)

    def __call__(self, state: EpisodeState) -> np.ndarray:
        if self.prediction is not None and state.last_fields is not None \
                and state.day - self.prediction["day"] != \
                self.cfg.lookahead_days:
            raise ValueError("the learning planner's look-ahead must equal "
                             "the segment length")
        innovation = self.observe(state)
        if innovation is not None:
            self.efficacy = self.estimator.update(
                innovation["observed"] - innovation["predicted"],
                innovation["sensitivity"], innovation["belief"],
                self.profile_weights).astype(np.float32)
        return self.plan(state)
