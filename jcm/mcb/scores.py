"""Objective and scores of the Experiment 2-3 test world.

MCB_PROJECT_REPORT.md Part 18 step 12. This follows Dubey, Abbot &
Chattopadhyay (2026, arXiv:2609.12528), moved from their land temperature to
this study's ocean temperature, which the brightening acts on and which
Experiment 1 validated:

* **Pattern objective** (their Eq. 1). With ``d`` the time-mean error map
  (controlled minus target) over a weighted domain, area weights ``w`` that
  sum to one, and band amplitudes ``a``:

      J = alpha <d>_w^2 + beta Var_w(d)
          + mu sum_j a_j^2 + lam sum_j (a_j - a_prev_j)^2.

  ``alpha = beta = 1`` gives the weighted mean-square error, and ``beta = 0``
  keeps only the bias. Their pattern objective uses ``alpha = 1``,
  ``beta = 0.5``, ``mu = 0.01`` and ``lam = 0.1``. Those penalty weights were
  set for cooling in kelvin, not for albedo, so ours are set in the pilot.
* **Gain** (their Eq. 3). On time-mean fields over a weighted domain,
  ``G = <(x_warm - x_tgt)^2>_w / <(x_ctrl - x_tgt)^2>_w``: how many times
  smaller the mean-square error is than in the uncontrolled warmed run.
  ``G = 1`` means no improvement.
* **Effort** (their Eq. 4). The area-weighted global mean and time mean of
  ``|sum_j g_j a_j(t)|``, with the unit band profiles ``g_j`` taken BEFORE
  the ocean mask. It measures what a strategy asks for, in albedo units here.

All functions are pure and differentiable where it makes sense (the
objective terms), so planners can use them inside ``jax.grad``.
"""

from typing import Dict, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

# Dubey et al.'s pattern objective weights (their Eq. 1).
PATTERN_ALPHA = 1.0
PATTERN_BETA = 0.5

# Where each scored variable's gain is measured. Ocean temperature is the
# objective's own variable; the others are never seen by the objective.
SCORE_DOMAINS = {
    "sst": ("ocean",),
    "land_temperature": ("land",),
    "precipitation": ("land", "global"),
    "evaporation": ("land", "global"),
}


def area_weights(latitudes_rad, mask) -> jnp.ndarray:
    """Cos-latitude weights on ``mask`` (ix, il), normalized to sum to one."""
    mask = jnp.asarray(mask, dtype=jnp.float32)
    lat = jnp.broadcast_to(jnp.asarray(latitudes_rad)[None, :], mask.shape)
    w = jnp.cos(lat) * mask
    return w / jnp.sum(w)


def weighted_mean(field, weights):
    """Weighted mean over the last two (grid) axes."""
    return jnp.sum(field * weights, axis=(-2, -1))


def weighted_variance(field, weights):
    """Weighted spatial variance over the last two (grid) axes."""
    mean = weighted_mean(field, weights)
    return weighted_mean((field - mean[..., None, None]) ** 2, weights)


def pattern_objective(error_map, weights, alpha: float = PATTERN_ALPHA,
                      beta: float = PATTERN_BETA):
    """``alpha <d>_w^2 + beta Var_w(d)`` for a time-mean error map ``d``."""
    return (alpha * weighted_mean(error_map, weights) ** 2
            + beta * weighted_variance(error_map, weights))


def amplitude_penalty(amplitudes, mu: float):
    """``mu * sum_j a_j^2``."""
    return mu * jnp.sum(jnp.asarray(amplitudes) ** 2)


def movement_penalty(amplitudes, previous, lam: float):
    """``lam * sum_j (a_j - a_prev_j)^2``."""
    return lam * jnp.sum((jnp.asarray(amplitudes) - jnp.asarray(previous))
                         ** 2)


def segment_objective(error_map, weights, amplitudes, previous,
                      alpha: float = PATTERN_ALPHA,
                      beta: float = PATTERN_BETA, mu: float = 0.0,
                      lam: float = 0.0):
    """Return the full Dubey-style objective of one segment (a scalar)."""
    return (pattern_objective(error_map, weights, alpha, beta)
            + amplitude_penalty(amplitudes, mu)
            + movement_penalty(amplitudes, previous, lam))


def objective_terms(error_map, weights, amplitudes, previous,
                    alpha: float = PATTERN_ALPHA, beta: float = PATTERN_BETA,
                    mu: float = 0.0, lam: float = 0.0) -> Dict[str, float]:
    """Each term of ``segment_objective`` as a float, for logging."""
    bias = float(weighted_mean(error_map, weights))
    terms = {
        "bias_sq": alpha * bias ** 2,
        "variance": beta * float(weighted_variance(error_map, weights)),
        "amplitude": float(amplitude_penalty(amplitudes, mu)),
        "movement": float(movement_penalty(amplitudes, previous, lam)),
    }
    terms["total"] = sum(terms.values())
    terms["bias_K"] = bias
    return terms


def gain(controlled, warmed, target, weights) -> float:
    """Dubey et al.'s gain on time-mean maps (inf if the control is perfect).

    ``<(warmed - target)^2>_w / <(controlled - target)^2>_w``.
    """
    w = np.asarray(weights, dtype=np.float64)
    num = float(np.sum(w * (np.asarray(warmed, np.float64)
                            - np.asarray(target, np.float64)) ** 2))
    den = float(np.sum(w * (np.asarray(controlled, np.float64)
                            - np.asarray(target, np.float64)) ** 2))
    return num / den if den > 0 else float("inf")


def effort(amplitude_series, unit_profiles, sphere_weights) -> float:
    """Dubey et al.'s effort: mean over time of ``<|sum_j g_j a_j(t)|>_A``.

    Args:
        amplitude_series: ``(T, K)`` band amplitudes in force on each day
            (or each equal-length interval).
        unit_profiles: ``(K, ix, il)`` unit-amplitude band profiles BEFORE
            the ocean mask (``gaussian_band_patterns`` with a mask of ones).
        sphere_weights: ``(ix, il)`` area weights over the whole sphere.

    """
    a = np.asarray(amplitude_series, dtype=np.float64)
    g = np.asarray(unit_profiles, dtype=np.float64)
    w = np.asarray(sphere_weights, dtype=np.float64)
    summed = np.tensordot(a, g, axes=([1], [0]))          # (T, ix, il)
    return float(np.mean(np.sum(w * np.abs(summed), axis=(-2, -1))))


def restoration_scores(controlled: Mapping[str, np.ndarray],
                       warmed: Mapping[str, np.ndarray],
                       target: Mapping[str, np.ndarray],
                       domain_weights: Mapping[str, np.ndarray],
                       domains: Mapping[str, Sequence[str]] = None
                       ) -> Dict[str, float]:
    """Gains of every scored variable on its domains, from time-mean maps.

    Returns ``{"<variable>@<domain>": G}`` for each pair in ``domains``
    (default ``SCORE_DOMAINS``) whose variable is in all three inputs.
    """
    domains = SCORE_DOMAINS if domains is None else domains
    out = {}
    for var, doms in domains.items():
        if var not in controlled or var not in warmed or var not in target:
            continue
        for dom in doms:
            out[f"{var}@{dom}"] = gain(controlled[var], warmed[var],
                                       target[var], domain_weights[dom])
    return out
