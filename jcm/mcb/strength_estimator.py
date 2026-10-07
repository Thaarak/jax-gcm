"""Learning how strongly the spraying works (Experiment 3b, Part 24).

A planner forecasts the next segment's ocean latitude profile and, from the
Jacobian it already plans with, how that forecast would change with each
band's strength. After the segment it compares forecast and observation.
Because the brightening enters as ``strength * setting``, the miss is,
to first order,

    observed - forecast = G (e - e_believed) + noise,

with ``G`` the forecast's sensitivity to the strengths. So
``z = (observed - forecast) + G e_believed`` is a linear measurement of the
true strengths ``e``. ``StrengthEstimator`` accumulates these measurements
over segments by regularized least squares (equivalently, the posterior mean
under a Gaussian prior):

    e_hat = argmin  sum_s ||z_s - G_s e||^2_W / noise^2 + ||e - 1||^2 / prior^2

where ``W`` holds the latitude profile's ocean-area weights (summing to one),
so ``noise`` is the root-mean-square of a whole profile's forecast miss when
the strength is known. Experiment 3b sets ``noise`` from the pilot's oracle
runs by a fixed rule, not by tuning. Treating a whole profile as about one
observation is conservative, because neighbouring latitudes are correlated.

``mode="bands"`` learns one strength per band; ``mode="global"`` learns one
factor shared by all bands. No extra model runs are needed: everything comes
from the forecast the planner makes anyway.
"""

import dataclasses
from typing import Tuple

import numpy as np

MODES = ("bands", "global")


@dataclasses.dataclass(frozen=True)
class EstimatorConfig:
    """Settings of ``StrengthEstimator``.

    ``noise_k`` is the forecast miss's typical size (K) when the strength is
    known; ``prior_sd`` the prior's standard deviation around nominal
    strength 1; ``bounds`` the range the estimate is clipped to.
    """

    noise_k: float
    mode: str = "bands"
    prior_sd: float = 1.0
    bounds: Tuple[float, float] = (0.2, 5.0)

    def validate(self):
        """Raise ValueError on settings the estimator cannot use."""
        lo, hi = self.bounds
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not self.noise_k > 0.0 or not self.prior_sd > 0.0:
            raise ValueError("noise_k and prior_sd must be > 0")
        if not 0.0 < lo < 1.0 < hi:
            raise ValueError("bounds must satisfy 0 < low < 1 < high")


class StrengthEstimator:
    """Regularized least squares for the per-band strength, updated each segment."""

    def __init__(self, k_bands: int, config: EstimatorConfig):
        """Start at the prior: nominal strength 1."""
        config.validate()
        self.cfg = config
        self.basis = (np.eye(k_bands) if config.mode == "bands"
                      else np.ones((k_bands, 1)))
        p = self.basis.shape[1]
        self.theta0 = np.ones(p)
        self.normal = np.eye(p) / config.prior_sd ** 2
        self.rhs = self.theta0 / config.prior_sd ** 2
        self.theta = self.theta0.copy()
        self.log = []

    def strength(self) -> np.ndarray:
        """Return the current estimate of every band's strength, clipped."""
        return np.clip(self.basis @ self.theta, *self.cfg.bounds)

    def update(self, miss, sensitivity, belief, weights) -> np.ndarray:
        """Add one segment's measurement and return the new estimate.

        Args:
            miss: ``(n,)`` observed minus forecast latitude profile (K).
            sensitivity: ``(n, K)`` the forecast's derivative with respect
                to each band's strength, at ``belief``.
            belief: ``(K,)`` the strengths the forecast assumed.
            weights: ``(n,)`` non-negative profile weights summing to one.

        """
        r = np.asarray(miss, np.float64)
        g = np.asarray(sensitivity, np.float64)
        w = np.asarray(weights, np.float64) / self.cfg.noise_k ** 2
        z = r + g @ np.asarray(belief, np.float64)
        a = g @ self.basis
        self.normal = self.normal + a.T @ (w[:, None] * a)
        self.rhs = self.rhs + a.T @ (w * z)
        self.theta = np.linalg.solve(self.normal, self.rhs)
        estimate = self.strength()
        self.log.append({"miss_ms": float(np.sum(np.asarray(weights) * r ** 2)),
                         "theta": self.theta.tolist(),
                         "strength": estimate.tolist()})
        return estimate
