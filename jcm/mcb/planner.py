"""The receding-horizon planner (MCB_PROJECT_REPORT.md Part 18 steps 13-15).

At the start of every segment (14 days) the planner starts from wherever the
simulated climate actually is. It chooses the five band settings that
minimize the map objective of ``jcm.mcb.scores`` over a look-ahead (60 days),
with the settings held for the whole look-ahead. It applies them for one
segment, then plans again.

* *The gradient.* It is the one Experiment 1 validated: the atmosphere's
  derivatives snipped every 14 days (W* = 14, Amendment 9 outcome A').
* *Copies.* Every gradient or Jacobian is averaged over ``copies`` slightly
  nudged copies of the current state (Dubey et al. use three).

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
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA
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

    Every call logs a record in ``self.log``: the start and final settings,
    the objective at each iterate, and the noise-to-signal ratio across the
    copies at the first iterate.

    """

    def __init__(self, step_fn: Callable, patterns, ocean_weights,
                 target_sst_daily, q_base, warming: Warming,
                 efficacy: float = 1.0,
                 config: PlannerConfig = PlannerConfig(),
                 seed_offset: int = 0):
        """Build the jitted gradient (Adam) or Jacobian (Gauss-Newton)."""
        config.validate()
        self.cfg = config
        self.k = int(np.shape(patterns)[0])
        self.weights = np.asarray(ocean_weights, np.float64)
        self.target = np.asarray(target_sst_daily, np.float64)
        self.seed_offset = int(seed_offset)
        self.log: List[dict] = []
        window = jnp.asarray(config.window_days)
        eff = jnp.asarray(efficacy, jnp.float32)
        if config.optimizer == "adam":
            objective = make_lookahead_objective(
                step_fn, patterns, ocean_weights, config.lookahead_days,
                config.alpha, config.beta, config.mu, config.lam)

            def objective_of_logits(z, carry, target_mean, previous):
                return objective(to_amplitudes(z, config.cap), carry,
                                 target_mean, previous, eff, q_base, warming,
                                 window)

            self._value_and_grad = jax.jit(jax.value_and_grad(
                objective_of_logits))
        else:
            mean_sst = make_lookahead_sst_fn(step_fn, patterns,
                                             config.lookahead_days)

            def map_twice(a, carry):
                out = mean_sst(a, carry, eff, q_base, warming, window)
                return out, out

            # (jacobian (ix, il, K), map (ix, il)) from ONE forward pass.
            self._jacobian_and_map = jax.jit(jax.jacfwd(map_twice,
                                                        has_aux=True))

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

    def __call__(self, state: EpisodeState) -> np.ndarray:
        t0 = time.time()
        cfg = self.cfg
        previous = np.asarray(state.previous, np.float32)
        start = (np.full(self.k, cfg.first_guess * cfg.cap, np.float32)
                 if state.segment == 0 else previous)
        copies = self.copies_of(state.carry, state.day)
        target_mean = self.target_mean(state.day)
        if cfg.optimizer == "adam":
            final, record = self._adam(start, copies, target_mean, previous)
        else:
            final, record = self._gauss_newton(start, copies, target_mean,
                                               previous)
        record.update(segment=state.segment, day=state.day,
                      optimizer=cfg.optimizer, start=start.tolist(),
                      final=final.tolist(),
                      seconds=round(time.time() - t0, 2))
        self.log.append(record)
        return final

    def _adam(self, start, copies, target_mean, previous):
        cfg = self.cfg
        z = to_logits(start, cfg.cap, cfg.edge_margin)
        m1, m2 = np.zeros_like(z), np.zeros_like(z)
        b1, b2 = _ADAM_BETAS
        tm, prev = jnp.asarray(target_mean), jnp.asarray(previous)
        values, nts = [], None
        for it in range(cfg.iterations):
            vals, grads = [], []
            for carry in copies:
                val, g = self._value_and_grad(jnp.asarray(z, jnp.float32),
                                              carry, tm, prev)
                vals.append(float(val))
                grads.append(np.asarray(g, np.float64))
            if it == 0:
                nts = noise_to_signal(grads)
            values.append(float(np.mean(vals)))
            g = np.mean(grads, axis=0)
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
            normal = np.zeros((self.k, self.k))
            rhs = np.zeros(self.k)
            total, grads = 0.0, []
            for carry in copies:
                jac, mean_map = self._jacobian_and_map(
                    jnp.asarray(a, jnp.float32), carry)
                r, big_r = gauss_newton_residuals(
                    np.asarray(mean_map, np.float64) - target_anomaly,
                    np.asarray(jac), self.weights, a, previous, cfg.alpha,
                    cfg.beta, cfg.mu, cfg.lam)
                normal += big_r.T @ big_r
                rhs += big_r.T @ r
                total += float(r @ r)
                grads.append(2.0 * big_r.T @ r)
            n = len(copies)
            normal, rhs = normal / n, rhs / n
            if it == 0:
                nts = noise_to_signal(grads)
            values.append(total / n)
            damping = cfg.gn_damping * np.trace(normal) / self.k
            delta = np.linalg.solve(normal + damping * np.eye(self.k), -rhs)
            a = np.clip(a + delta, 0.0, cfg.cap)
        return a.astype(np.float32), {"objective": values,
                                      "noise_to_signal": nts}
