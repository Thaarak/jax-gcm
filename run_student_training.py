#!/usr/bin/env python
"""Training the student on the coupled model: copying and direct training.

MCB_PROJECT_REPORT.md Part 18 steps 20-22 (``jcm.mcb.student``,
``jcm.mcb.imitation``, ``jcm.mcb.direct_training``). Training states only:
the IC directory's role must end in ``_train``.

*Episodes.* A training episode is a training state (an IC with its
references), a weather sample (a member seed of the references) and a hidden
spraying strength. Strengths are drawn log-uniformly from
``--efficacy-range`` with a fixed seed. Every method gets the same list of
episodes.

``dagger``
    The copying loop (step 21). The teacher is the receding-horizon planner,
    told each episode's hidden strength. The stage writes:
    - the student (``student_dagger.npz``);
    - the data set (``dagger_data.npz``);
    - a log (``dagger_log.json``), whose ``seconds_total`` is the GPU budget
      the direct methods get.
``direct``
    Direct training (step 22): ``--method bptt``, ``snipped`` or ``eki``,
    within ``--budget-seconds`` or the budget from ``--budget-from
    dagger_log.json``. Writes ``student_<method>.npz`` and a log.

Settings default to Part 18's starting values. Revision 1 of Amendment 9
freezes them, including the strength range, which is a placeholder here.

Example (CPU smoke):
    python run_student_training.py dagger --ic-dir <ics>/exp3_train \
        --ic-positions 0 --references-dir <refs>/exp3_train --segments 2 \
        --segment-days 2 --strengths-per-state 1 --rounds 2 \
        --lookahead-days 3 --copies 1 --optimizer gauss_newton \
        --iterations 1 --output-dir /tmp/students
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, NamedTuple

import numpy as np

from jcm.mcb.direct_training import METHODS, Budget, DirectConfig
from jcm.mcb.imitation import ImitationConfig
from jcm.mcb.planner import OPTIMIZERS, PRESETS, W_STAR_DAYS
from jcm.mcb.scores import PATTERN_ALPHA, PATTERN_BETA
from jcm.mcb.student import StudentConfig


class EpisodeSpec(NamedTuple):
    """One training episode: a state, a weather sample, a hidden strength."""

    ic_position: int
    ic_index: int
    member: int
    efficacy: float


def parse_args(argv=None):
    from run_test_world import add_warming_args

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp):
        sp.add_argument("--ic-dir", required=True,
                        help="A training role's IC directory (*_train).")
        sp.add_argument("--ic-positions", type=int, nargs="+",
                        required=True)
        sp.add_argument("--references-dir", required=True)
        sp.add_argument("--segments", type=int, default=13)
        sp.add_argument("--segment-days", type=int, default=14)
        sp.add_argument("--weather-samples", type=int, default=1,
                        help="Member seeds per state (member 0 = the IC).")
        sp.add_argument("--strengths-per-state", type=int, default=2,
                        help="Hidden strengths per state and sample.")
        sp.add_argument("--efficacy-range", type=float, nargs=2,
                        default=[0.5, 2.0], metavar=("LOW", "HIGH"),
                        help="Log-uniform hidden strengths (placeholder "
                             "until revision 1).")
        sp.add_argument("--episode-seed", type=int, default=0)
        sp.add_argument("--hidden", type=int, default=6)
        sp.add_argument("--no-season", dest="use_season",
                        action="store_false")
        sp.add_argument("--student-seed", type=int, default=0)
        sp.add_argument("--alpha", type=float, default=PATTERN_ALPHA)
        sp.add_argument("--beta", type=float, default=PATTERN_BETA)
        sp.add_argument("--mu", type=float, default=0.0)
        sp.add_argument("--lam", type=float, default=0.0)
        sp.add_argument("--output-dir", required=True)
        add_warming_args(sp)

    d = sub.add_parser("dagger", help="the copying loop (step 21)")
    common(d)
    d.add_argument("--rounds", type=int, default=4)
    d.add_argument("--beta0", type=float, default=0.0,
                   help="Teacher share after round 0 is beta0 ** round.")
    d.add_argument("--fit-steps", type=int, default=2000)
    d.add_argument("--fit-learning-rate", type=float, default=1e-2)
    d.add_argument("--preset", default="snipped60", choices=sorted(PRESETS))
    d.add_argument("--lookahead-days", type=int, default=None)
    d.add_argument("--window-days", type=int, default=None)
    d.add_argument("--copies", type=int, default=None)
    d.add_argument("--optimizer", choices=OPTIMIZERS, default=None)
    d.add_argument("--iterations", type=int, default=None)
    d.add_argument("--learning-rate", type=float, default=None)

    r = sub.add_parser("direct", help="direct training (step 22)")
    common(r)
    r.add_argument("--method", choices=METHODS, required=True)
    budget = r.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-seconds", type=float)
    budget.add_argument("--budget-from", help="A dagger_log.json.")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--window-days", type=int, default=W_STAR_DAYS)
    r.add_argument("--learning-rate", type=float, default=1e-2)
    r.add_argument("--clip-norm", type=float, default=1.0)
    r.add_argument("--ensemble", type=int, default=32)
    r.add_argument("--init-spread", type=float, default=0.1)
    r.add_argument("--noise-scale", type=float, default=1.0)
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the stages cannot use."""
    if args.segments < 1 or args.segment_days < 1:
        raise SystemExit("--segments and --segment-days must be >= 1")
    if args.weather_samples < 1 or args.strengths_per_state < 1:
        raise SystemExit("--weather-samples and --strengths-per-state >= 1")
    lo, hi = args.efficacy_range
    if not 0.0 < lo <= hi:
        raise SystemExit("--efficacy-range needs 0 < LOW <= HIGH")
    try:
        student_config(args).validate()
        if args.stage == "dagger":
            imitation_config(args).validate()
            teacher_config(args)
        else:
            direct_config(args).validate()
    except ValueError as err:
        raise SystemExit(str(err)) from err


def student_config(args) -> StudentConfig:
    """Return the student's shape from the arguments."""
    return StudentConfig(hidden=args.hidden, use_season=args.use_season)


def imitation_config(args) -> ImitationConfig:
    """Return the copying loop's settings from the arguments."""
    return ImitationConfig(rounds=args.rounds, beta0=args.beta0,
                           fit_steps=args.fit_steps,
                           learning_rate=args.fit_learning_rate,
                           seed=args.episode_seed)


def teacher_config(args):
    """Return the teacher planner's settings (a preset plus overrides)."""
    from run_test_world import planner_config
    return planner_config(args)


def direct_config(args) -> DirectConfig:
    """Return a direct method's settings from the arguments."""
    return DirectConfig(method=args.method, window_days=args.window_days,
                        learning_rate=args.learning_rate,
                        clip_norm=args.clip_norm, alpha=args.alpha,
                        beta=args.beta, mu=args.mu, lam=args.lam,
                        ensemble=args.ensemble, init_spread=args.init_spread,
                        noise_scale=args.noise_scale, seed=args.student_seed)


def budget_seconds(args) -> float:
    """Return the direct method's budget: given, or the copying route's."""
    if args.budget_seconds is not None:
        return args.budget_seconds
    with open(args.budget_from) as f:
        return float(json.load(f)["seconds_total"])


def make_specs(entries, weather_samples: int, strengths: int,
               efficacy_range, seed: int) -> List[EpisodeSpec]:
    """Return the training episodes, the same for every method.

    ``entries`` is ``[(position, manifest entry)]``.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.log(efficacy_range[0]), np.log(efficacy_range[1])
    specs = []
    for position, entry in entries:
        for member in range(weather_samples):
            for _ in range(strengths):
                specs.append(EpisodeSpec(position, int(entry["index"]),
                                         member,
                                         float(np.exp(rng.uniform(lo, hi)))))
    return specs


def check_training_role(role: str):
    """Refuse anything but a training role: students never see evaluation states."""
    if not role.endswith("_train"):
        raise SystemExit(f"role '{role}' is not a training role (*_train); "
                         f"students are trained on training states only")


def load_training_ics(ic_dir, positions, template):
    """Return ``(role, [(position, entry, carry)])`` for the chosen positions."""
    from run_gradient_fidelity import load_carry
    from run_stage5_training import START_DATE

    ic_dir = Path(ic_dir)
    with open(ic_dir / "manifest.json") as f:
        manifest = json.load(f)
    if manifest.get("start_date") != START_DATE:
        raise SystemExit(f"manifest start_date {manifest.get('start_date')} "
                         f"!= {START_DATE}")
    role = manifest.get("macro_role") or ic_dir.name
    check_training_role(role)
    out = []
    for pos in positions:
        if not 0 <= pos < len(manifest["ics"]):
            raise SystemExit(f"--ic-positions: {pos} is outside the "
                             f"manifest's {len(manifest['ics'])} ICs")
        entry = manifest["ics"][pos]
        out.append((pos, entry, load_carry(str(ic_dir / entry["carry_file"]),
                                           template)))
    return role, out


class World:
    """The coupled model, the training states and their references."""

    def __init__(self, args):
        """Build the model and load every chosen state and its references."""
        from run_test_world import build_model, check_warming_matches

        from jcm.mcb.scores import area_weights
        from jcm.mcb.student import band_observation_weights
        from jcm.mcb.test_world import make_segment_fn

        self.args = args
        self.m = build_model()
        self.role, ics = load_training_ics(args.ic_dir, args.ic_positions,
                                           self.m["template"])
        refs_dir = Path(args.references_dir)
        with open(refs_dir / "references_manifest.json") as f:
            self.ref_config = json.load(f)["config"]
        check_warming_matches(self.ref_config, args)
        n_days = args.segments * args.segment_days
        self.carries, self.targets, self.entries = {}, {}, []
        for pos, entry, carry in ics:
            refs = np.load(refs_dir / f"ic{entry['index']:04d}_references.npz",
                           allow_pickle=False)
            if refs["normal_sst"].shape[0] < n_days:
                raise SystemExit("the references are shorter than an episode")
            self.carries[pos] = carry
            self.targets[pos] = np.asarray(refs["normal_sst"])
            self.entries.append((pos, entry))
        self.k = int(np.shape(self.m["patterns"])[0])
        self.ocean_w = np.asarray(area_weights(self.m["lats"],
                                               self.m["ocean"]))
        self.obs_w = band_observation_weights(self.m["lats"],
                                              self.m["patterns"])
        self.segment_fn = make_segment_fn(self.m["step_fn"],
                                          self.m["patterns"],
                                          args.segment_days)

    def start(self, spec: EpisodeSpec):
        """Return the spec's starting carry and its warming."""
        from jcm.mcb.test_world import make_warming, member_seed, \
            perturb_member

        carry = self.carries[spec.ic_position]
        seed = member_seed(self.ref_config["member_seed0"], spec.ic_index,
                           spec.member)
        if seed is not None:
            carry = perturb_member(carry, seed, self.ref_config["member_amp"])
        warming = make_warming(carry, self.m["ocean"],
                               self.args.warming_step_wm2,
                               self.args.warming_ramp_wm2_per_day)
        return carry, warming

    def run(self, spec: EpisodeSpec, policy):
        """Run one episode with ``policy`` at the spec's hidden strength."""
        from jcm.mcb.test_world import run_episode

        carry, warming = self.start(spec)
        return run_episode(carry, self.segment_fn, policy, self.args.segments,
                           self.args.segment_days, self.k, warming,
                           efficacy=spec.efficacy)


def _write(out_dir: Path, name: str, payload: dict):
    from run_gradient_fidelity import git_provenance

    payload.update(git=git_provenance(), command=" ".join(sys.argv),
                   finished_utc=datetime.now(timezone.utc).isoformat())
    with open(out_dir / name, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"-> {out_dir / name}")


def stage_dagger(args):
    import jax

    from jcm.mcb.imitation import run_dagger
    from jcm.mcb.planner import Planner
    from jcm.mcb.student import StudentPolicy, init_student, save_student

    t_start = time.time()
    world = World(args)
    cfg, icfg, tcfg = student_config(args), imitation_config(args), \
        teacher_config(args)
    specs = make_specs(world.entries, args.weather_samples,
                       args.strengths_per_state, args.efficacy_range,
                       args.episode_seed)
    teachers = []

    def make_teacher(spec):
        carry, warming = world.start(spec)
        planner = Planner(world.m["step_fn"], world.m["patterns"],
                          world.ocean_w, world.targets[spec.ic_position],
                          carry["ocn"]["forcing"].q_flux, warming,
                          spec.efficacy, tcfg,
                          seed_offset=1000 + len(teachers))
        teachers.append(planner)
        return planner

    def make_student(params, spec):
        return StudentPolicy(params, cfg, world.targets[spec.ic_position],
                             world.obs_w, args.segment_days)

    params, data, history = run_dagger(
        specs, world.run, make_teacher, make_student,
        init_student(jax.random.PRNGKey(args.student_seed), cfg), cfg, icfg)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_student(out / "student_dagger.npz", params, cfg, method="dagger",
                 rounds=len(history), rows=len(data))
    data.save(out / "dagger_data.npz")
    teacher_s = sum(r["seconds"] for t in teachers for r in t.log)
    _write(out, "dagger_log.json", {
        "stage": "dagger", "role": world.role, "config": vars(args),
        "student": dataclasses.asdict(cfg),
        "imitation": dataclasses.asdict(icfg),
        "teacher": dataclasses.asdict(tcfg),
        "episodes": [s._asdict() for s in specs], "history": history,
        "teacher_seconds": round(teacher_s, 1),
        "seconds_total": round(time.time() - t_start, 1)})


def stage_direct(args):
    import jax

    from jcm.mcb.direct_training import (
        TrainingEpisode,
        make_closed_loop_rollout,
        target_segment_means,
        train_eki,
        train_gradient,
    )
    from jcm.mcb.student import init_student, save_student

    limit = budget_seconds(args)
    t_start = time.time()
    world = World(args)
    cfg, dcfg = student_config(args), direct_config(args)
    specs = make_specs(world.entries, args.weather_samples,
                       args.strengths_per_state, args.efficacy_range,
                       args.episode_seed)
    episodes = []
    for spec in specs:
        carry, warming = world.start(spec)
        episodes.append(TrainingEpisode(
            carry, target_segment_means(world.targets[spec.ic_position],
                                        args.segment_days, args.segments),
            np.float32(spec.efficacy), carry["ocn"]["forcing"].q_flux,
            warming))
    rollout = make_closed_loop_rollout(world.m["step_fn"], world.m["patterns"],
                                       world.obs_w, cfg, args.segment_days,
                                       args.segments)
    params0 = init_student(jax.random.PRNGKey(args.student_seed), cfg)
    budget = Budget(limit)
    if args.method == "eki":
        params, log = train_eki(rollout, world.ocean_w, params0, episodes,
                                dcfg, budget, args.max_steps)
    else:
        params, log = train_gradient(rollout, world.ocean_w, params0,
                                     episodes, dcfg, budget, args.max_steps)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_student(out / f"student_{args.method}.npz", params, cfg,
                 method=args.method, steps=len(log))
    _write(out, f"{args.method}_log.json", {
        "stage": "direct", "method": args.method, "role": world.role,
        "config": vars(args), "student": dataclasses.asdict(cfg),
        "direct": dataclasses.asdict(dcfg),
        "episodes": [s._asdict() for s in specs], "log": log,
        "budget_seconds": limit, "budget_spent": round(budget.spent, 1),
        "seconds_total": round(time.time() - t_start, 1)})


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    import jax
    print(f"JAX devices: {jax.devices()}")
    {"dagger": stage_dagger, "direct": stage_direct}[args.stage](args)


if __name__ == "__main__":
    main()

