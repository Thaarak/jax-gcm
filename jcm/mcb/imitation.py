"""The copying loop, with a teacher that knows more (Part 18 step 21).

MCB_PROJECT_REPORT.md Part 18 step 21. The student (``jcm.mcb.student``)
learns to copy the receding-horizon planner with DAgger (Ross, Gordon &
Bagnell 2011):

1. *Round 0.* The teacher drives the training episodes, and every state it
   meets is recorded with the teacher's settings.
2. *Every later round.* The student drives, by default with no help.
   - In each state the student reaches, the teacher is asked what it would
     have done there.
   - The answers join the growing data set.
   - The student is fitted again to all of it.

Plain copying breaks down once the student's small errors take it where the
teacher never went; DAgger collects data exactly there.

The teacher is *privileged*: in Experiment 3b it is a planner told the
hidden spraying strength of each episode, while the student never is. The
student sees only its band anomalies, the season and its own previous
settings (``StudentPolicy``), so it has to learn to infer the strength from
how the ocean answers its settings. With probability ``beta`` a round lets
the teacher drive instead; ``beta`` is 1 in round 0 and ``beta0 ** round``
after, so 0 by default.

``run_dagger`` is generic. The caller supplies the episodes, a way to run
one with a policy, the teacher and the student policy, so the same loop runs
on the toy in the tests and on the coupled model in
``run_student_training.py``. Every second it spends counts toward the
copying route's GPU budget, the budget the direct methods then get
(``jcm.mcb.direct_training``).
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jcm.mcb.student import Params, StudentConfig, StudentPolicy, student_apply
from jcm.mcb.test_world import EpisodeState


@dataclasses.dataclass(frozen=True)
class ImitationConfig:
    """Settings of the copying loop. Revision 1 freezes the values."""

    rounds: int = 4
    beta0: float = 0.0
    fit_steps: int = 2000
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    warm_start: bool = True
    seed: int = 0

    def validate(self):
        """Raise ValueError on settings the loop cannot use."""
        if self.rounds < 1 or self.fit_steps < 1:
            raise ValueError("rounds and fit_steps must be >= 1")
        if not 0.0 <= self.beta0 <= 1.0:
            raise ValueError("beta0 must lie in [0, 1]")
        if not self.learning_rate > 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate > 0 and weight_decay >= 0")


def teacher_share(round_index: int, beta0: float) -> float:
    """Return the probability that the teacher drives in a round."""
    return 1.0 if round_index == 0 else beta0 ** round_index


class ImitationData:
    """The growing data set: student inputs, teacher settings and notes."""

    def __init__(self):
        """Start empty."""
        self.x: List[np.ndarray] = []
        self.y: List[np.ndarray] = []
        self.meta: List[dict] = []

    def __len__(self) -> int:
        """Return the number of recorded states."""
        return len(self.x)

    def add(self, x, y, **meta):
        """Record one state: the student's inputs and the teacher's settings."""
        self.x.append(np.asarray(x, np.float32))
        self.y.append(np.asarray(y, np.float32))
        self.meta.append(meta)

    @property
    def inputs(self) -> np.ndarray:
        """All inputs, ``(rows, n_inputs)``."""
        return np.stack(self.x)

    @property
    def targets(self) -> np.ndarray:
        """All teacher settings, ``(rows, K)``."""
        return np.stack(self.y)

    def save(self, path):
        """Write the data set to an ``.npz`` (no pickle)."""
        np.savez(path, inputs=self.inputs, targets=self.targets,
                 meta=json.dumps(self.meta, default=_jsonable))

    @classmethod
    def load(cls, path) -> "ImitationData":
        """Read what ``save`` wrote."""
        data = cls()
        with np.load(Path(path), allow_pickle=False) as z:
            for x, y, m in zip(z["inputs"], z["targets"],
                               json.loads(str(z["meta"]))):
                data.add(x, y, **m)
        return data


def _jsonable(value):
    return np.asarray(value).tolist()


def fit_student(params: Params, inputs, targets, cfg: StudentConfig,
                icfg: ImitationConfig) -> Tuple[Params, np.ndarray]:
    """Fit the student to the teacher's settings by full-batch AdamW.

    The loss is the mean squared difference of the settings divided by the
    cap. The function returns the fitted parameters and the loss after every
    step.
    """
    x = jnp.asarray(inputs, jnp.float32)
    y = jnp.asarray(targets, jnp.float32)
    opt = optax.adamw(icfg.learning_rate, weight_decay=icfg.weight_decay)

    def loss(p):
        return jnp.mean(((student_apply(p, x, cfg) - y) / cfg.cap) ** 2)

    @jax.jit
    def run(p):
        def body(carry, _):
            p, state = carry
            value, grads = jax.value_and_grad(loss)(p)
            updates, state = opt.update(grads, state, p)
            return (optax.apply_updates(p, updates), state), value

        (p, _), values = jax.lax.scan(body, (p, opt.init(p)), None,
                                      length=icfg.fit_steps)
        return p, values

    fitted, values = run(params)
    return fitted, np.asarray(values)


class CopyingRecorder:
    """A policy that asks the teacher in every state and records the answer.

    Args:
        student: the current ``StudentPolicy`` (its inputs are recorded).
        teacher: the privileged teacher, a policy.
        beta: the probability that the teacher drives this state.
        rng: a ``numpy`` random generator for that choice.
        data: the ``ImitationData`` to add to.
        meta: notes stored with every row (round, episode, ...).

    """

    def __init__(self, student: StudentPolicy, teacher: Callable,
                 beta: float, rng: np.random.Generator, data: ImitationData,
                 meta: dict):
        """Keep the pieces; nothing runs until the episode does."""
        self.student = student
        self.teacher = teacher
        self.beta = beta
        self.rng = rng
        self.data = data
        self.meta = dict(meta)

    def __call__(self, state: EpisodeState) -> np.ndarray:
        x = self.student.observation(state)
        label = np.asarray(self.teacher(state), np.float32)
        own = self.student.act(x)
        teacher_drives = bool(self.rng.random() < self.beta)
        self.data.add(x, label, **self.meta, segment=state.segment,
                      day=state.day, teacher_drove=teacher_drives,
                      student=own.tolist())
        return label if teacher_drives else own


def run_dagger(episodes: Sequence, run_fn: Callable, make_teacher: Callable,
               make_student: Callable, params0: Params, cfg: StudentConfig,
               icfg: ImitationConfig, budget=None,
               log_fn: Optional[Callable] = print):
    """Run the copying loop and return ``(params, data, history)``.

    Args:
        episodes: opaque episode specifications (a training state, a weather
            sample and a hidden strength each).
        run_fn: ``run_fn(spec, policy)`` runs one episode with ``policy``.
        make_teacher: ``make_teacher(spec)`` returns a fresh teacher policy
            that knows the spec's hidden strength.
        make_student: ``make_student(params, spec)`` returns a
            ``StudentPolicy``.
        params0: the student's starting parameters.
        cfg, icfg: the student's and the loop's settings.
        budget: an optional ``jcm.mcb.direct_training.Budget``. It is charged
            every episode and fit, and the loop stops when it is spent.
        log_fn: where progress lines go (None for silence).

    Returns:
        The final parameters, the data set, and one history record per round:
        - its rows;
        - the student's error against the teacher on the states of that
          round (on-policy once the student drives), as a fraction of the cap;
        - the fit's final loss;
        - the seconds it took.

    """
    icfg.validate()
    data = ImitationData()
    params = params0
    rng = np.random.default_rng(icfg.seed)
    history = []
    for r in range(icfg.rounds):
        beta = teacher_share(r, icfg.beta0)
        start_rows, t_round = len(data), time.time()
        for i, spec in enumerate(episodes):
            if budget is not None and budget.exhausted:
                break
            recorder = CopyingRecorder(make_student(params, spec),
                                       make_teacher(spec), beta, rng, data,
                                       {"round": r, "episode": i})
            t0 = time.time()
            run_fn(spec, recorder)
            if budget is not None:
                budget.charge(time.time() - t0)
        new = range(start_rows, len(data))
        if not new:
            break
        error = float(np.mean([np.mean(np.abs(np.asarray(data.meta[j]
                                                         ["student"])
                                              - data.y[j])) / cfg.cap
                               for j in new]))
        t0 = time.time()
        params, losses = fit_student(params if icfg.warm_start else params0,
                                     data.inputs, data.targets, cfg, icfg)
        if budget is not None:
            budget.charge(time.time() - t0)
        record = {"round": r, "teacher_share": beta, "rows": len(data),
                  "new_rows": len(new), "student_error": error,
                  "fit_loss": float(losses[-1]),
                  "seconds": round(time.time() - t_round, 1)}
        history.append(record)
        if log_fn is not None:
            log_fn(f"round {r}: teacher share {beta:.2f}, {len(data)} rows, "
                   f"student error {error:.3f} of the cap, fit loss "
                   f"{record['fit_loss']:.2e} [{record['seconds']}s]")
    return params, data, history
