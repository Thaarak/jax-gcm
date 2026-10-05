"""The student network (MCB_PROJECT_REPORT.md Part 18 step 20).

A tiny controller, 113 adjustable numbers with the defaults, that sees only
what a real system could measure and outputs the five band settings.

* *Inputs (12).*
  - The ocean temperature anomaly against the normal climate under each
    band, averaged over the last segment (5). It is zero before the first
    measurement and scaled by ``anomaly_scale_k``.
  - The season, as the sine and cosine of the day of year (2).
  - Its own previous settings, over the cap (5).
* *Network.* One hidden layer of 6 tanh units.
* *Outputs.* ``cap * sigmoid(.)`` per band, so every setting stays inside
  ``[0, cap]``.

The previous settings make integral action learnable (new setting = old
setting + a correction). The season tells the student when sunlight makes a
band strong or weak. If every episode starts on the same date, the season
is also the time since the start, so ``use_season`` can switch it off.

The same functions build the features inside JAX, for direct training
through the simulation (``jcm.mcb.direct_training``), and in
``StudentPolicy``, a ``run_episode`` policy used by the copying loop. Both
give the same features from the same state.
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jcm.mcb.test_world import BRIGHTENING_CAP, SECONDS_PER_DAY, EpisodeState

Params = Dict[str, jnp.ndarray]


@dataclasses.dataclass(frozen=True)
class StudentConfig:
    """The student's shape and its fixed input scaling."""

    k: int = 5
    hidden: int = 6
    anomaly_scale_k: float = 0.1
    use_season: bool = True
    cap: float = BRIGHTENING_CAP
    init_fraction: float = 0.2
    year_days: float = 365.25
    start_offset_s: float = 0.0

    @property
    def n_inputs(self) -> int:
        """Band anomalies, the season (optional) and the previous settings."""
        return 2 * self.k + (2 if self.use_season else 0)

    def validate(self):
        """Raise ValueError on settings the student cannot use."""
        if self.k < 1 or self.hidden < 1:
            raise ValueError("k and hidden must be >= 1")
        if not 0.0 < self.init_fraction < 1.0:
            raise ValueError("init_fraction must lie in (0, 1)")
        if not self.anomaly_scale_k > 0.0 or not self.cap > 0.0:
            raise ValueError("anomaly_scale_k and cap must be > 0")


def n_parameters(cfg: StudentConfig) -> int:
    """Return the number of adjustable numbers."""
    return cfg.n_inputs * cfg.hidden + cfg.hidden + cfg.hidden * cfg.k + cfg.k


def band_observation_weights(latitudes_rad, patterns) -> np.ndarray:
    """Return ``(K, ix, il)`` weights for the mean anomaly under each band.

    The weight of a cell is its area (cos latitude) times the band's
    (ocean-masked) profile, normalized per band.
    """
    p = np.asarray(patterns, np.float64)
    area = np.cos(np.asarray(latitudes_rad, np.float64))[None, None, :]
    w = p * area
    return w / w.sum(axis=(1, 2), keepdims=True)


def observe_bands(anomaly_map, observation_weights):
    """Return the ``(K,)`` band-mean anomalies of a map (works in JAX)."""
    return jnp.tensordot(jnp.asarray(observation_weights, jnp.float32),
                         anomaly_map, axes=([1, 2], [0, 1]))


def day_of_year(sim_time_s, cfg: StudentConfig):
    """Return the day of year of a model time in seconds (works in JAX)."""
    return jnp.mod((sim_time_s + cfg.start_offset_s) / SECONDS_PER_DAY,
                   cfg.year_days)


def features(band_anomalies, doy, previous, cfg: StudentConfig):
    """Return the student's input vector (works in JAX)."""
    parts = [jnp.asarray(band_anomalies, jnp.float32) / cfg.anomaly_scale_k]
    if cfg.use_season:
        phase = 2.0 * jnp.pi * doy / cfg.year_days
        parts.append(jnp.stack([jnp.sin(phase), jnp.cos(phase)]))
    parts.append(jnp.asarray(previous, jnp.float32) / cfg.cap)
    return jnp.concatenate(parts).astype(jnp.float32)


def init_student(key, cfg: StudentConfig = StudentConfig()) -> Params:
    """Return initial parameters whose outputs start near ``init_fraction * cap``."""
    cfg.validate()
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (cfg.n_inputs, cfg.hidden)) / np.sqrt(
        cfg.n_inputs)
    w2 = 0.1 * jax.random.normal(k2, (cfg.hidden, cfg.k)) / np.sqrt(
        cfg.hidden)
    logit = np.log(cfg.init_fraction / (1.0 - cfg.init_fraction))
    return {"w1": w1.astype(jnp.float32),
            "b1": jnp.zeros(cfg.hidden, jnp.float32),
            "w2": w2.astype(jnp.float32),
            "b2": jnp.full(cfg.k, logit, jnp.float32)}


def student_apply(params: Params, x, cfg: StudentConfig):
    """Return the band settings in ``(0, cap)`` for inputs ``x`` (batched or not)."""
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    return cfg.cap * jax.nn.sigmoid(h @ params["w2"] + params["b2"])


def save_student(path, params: Params, cfg: StudentConfig, **meta):
    """Write the parameters and config to an ``.npz`` (no pickle)."""
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()},
             config=json.dumps(dataclasses.asdict(cfg)),
             meta=json.dumps(meta, default=str))


def load_student(path) -> Tuple[Params, StudentConfig, dict]:
    """Read what ``save_student`` wrote."""
    with np.load(Path(path), allow_pickle=False) as z:
        cfg = StudentConfig(**json.loads(str(z["config"])))
        params = {k: jnp.asarray(z[k]) for k in ("w1", "b1", "w2", "b2")}
        meta = json.loads(str(z["meta"]))
    return params, cfg, meta


class StudentPolicy:
    """A student as a ``run_episode`` policy.

    Args:
        params: the network's parameters.
        cfg: its ``StudentConfig``.
        target_sst_daily: ``(n_days, ix, il)`` the normal climate's SST,
            aligned with the episode's daily fields.
        observation_weights: ``band_observation_weights``.
        segment_days: days per segment.

    """

    def __init__(self, params: Params, cfg: StudentConfig, target_sst_daily,
                 observation_weights, segment_days: int):
        """Jit the network once."""
        self.params = params
        self.cfg = cfg
        self.target = np.asarray(target_sst_daily, np.float64)
        self.obs_w = np.asarray(observation_weights, np.float64)
        self.segment_days = segment_days
        self._apply = jax.jit(lambda p, x: student_apply(p, x, cfg))

    def band_anomalies(self, state: EpisodeState) -> np.ndarray:
        """Return the band-mean anomalies over the last segment (zero at first)."""
        if state.segment == 0 or state.last_fields is None:
            return np.zeros(self.cfg.k)
        start = state.day - self.segment_days
        anomaly = (np.asarray(state.last_fields["sst"], np.float64)
                   .mean(axis=0) - self.target[start:state.day].mean(axis=0))
        return np.tensordot(self.obs_w, anomaly, axes=([1, 2], [0, 1]))

    def observation(self, state: EpisodeState) -> np.ndarray:
        """Return the student's input vector for a state."""
        doy = day_of_year(float(state.carry["ocn"]["state"].sim_time),
                          self.cfg)
        return np.asarray(features(self.band_anomalies(state), doy,
                                   state.previous, self.cfg))

    def act(self, x) -> np.ndarray:
        """Return the settings for an input vector."""
        return np.asarray(self._apply(self.params, jnp.asarray(x)),
                          np.float32)

    def __call__(self, state: EpisodeState) -> np.ndarray:
        return self.act(self.observation(state))
