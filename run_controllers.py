#!/usr/bin/env python
"""Experiment 3's classical opponents and trained students on the coupled model.

MCB_PROJECT_REPORT.md Part 18 steps 18-20 (``jcm.mcb.ladder``,
``jcm.mcb.feedback``, ``jcm.mcb.student``). Four stages:

``responses`` (training states only)
    From one IC, run each band alone at ``--delta`` from day 0, with the
    warming, averaged over the references' members and seeds. Sharing the
    seeds keeps the weather of each run close to the warmed reference's at
    first. Writes ``ic<index>_responses.npz`` with each run's block means of
    SST and its daily T0, T1, T2 and LAND (relative to 288 K). These give the
    linear-response design's maps (step 18) and the feedback controllers'
    sensitivities (step 19).
``ladder`` (no model)
    Designs the fixed-pattern ladder. Rungs 1 and 4 come from response
    files pooled over training states, rungs 2 and 3 from planner summaries
    (``run_test_world.py plan``). Writes the designs as JSON; each is then
    run with ``run_test_world.py episode --amplitudes``.
``feedback``
    One scored episode with the GLENS-style controller (``--controller pi``)
    or the ported Tier 2 adaptive law (``--controller adaptive``).
``student``
    One scored episode with a trained student (an ``.npz`` from
    ``run_student_training.py``).

The controllers' sensitivities come from response files
(``--sensitivity-from <files>``) or from Experiment 1's brute-force runs
(``--sensitivity-from experiment1``). The latter uses central differences of
T0-T2 at 0.03 +- 0.03 over its eight January starts, fitted over days 1-60.

Examples (CPU smoke):
    python run_controllers.py feedback --controller pi --ic-dir <ics> \
        --references <refs>/ic0002_references.npz --segments 2 \
        --segment-days 2 --members 1 --output /tmp/ctl/pi.json
    python run_controllers.py ladder --responses <dir>/ic*_responses.npz \
        --references-dir <refs> --window 98 182 --output /tmp/ctl/ladder.json
"""

import argparse
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jcm.mcb.band_basis import (
    OCEAN_OBJECTIVES,
    gaussian_band_patterns,
    objective_values,
    objective_weights,
    stack_objective_weights,
)
from jcm.mcb.feedback import (
    AdaptiveConfig,
    AdaptiveController,
    IndexPIController,
    PIConfig,
    sensitivity_rates,
)
from jcm.mcb.ladder import ladder_designs, response_maps
from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    area_weights,
    zonal_projection,
)
from jcm.mcb.student import (
    StudentPolicy,
    band_observation_weights,
    load_student,
)
from jcm.mcb.test_world import BRIGHTENING_CAP, constant_policy

EXPERIMENT1_NPZ = "mcb_experiments_gpu/exp1_gradient_fidelity.npz"
EXPERIMENT1_JSON = "mcb_experiments_gpu/exp1_gradient_fidelity.json"


def parse_args(argv=None):
    from run_test_world import add_run_args, add_warming_args

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("responses", help="one-band step runs (training)")
    r.add_argument("--ic-dir", required=True)
    r.add_argument("--ic-position", type=int, default=0)
    r.add_argument("--references", required=True)
    r.add_argument("--days", type=int, default=182)
    r.add_argument("--segment-days", type=int, default=14)
    r.add_argument("--block-days", type=int, default=7)
    r.add_argument("--delta", type=float, default=0.1,
                   help="Step size of each band (albedo units).")
    r.add_argument("--members", type=int, default=None)
    r.add_argument("--output-dir", required=True)
    add_warming_args(r)

    lad = sub.add_parser("ladder", help="design the fixed-pattern ladder")
    lad.add_argument("--responses", nargs="+", required=True,
                     help="Training states' ic<index>_responses.npz.")
    lad.add_argument("--references-dir", required=True)
    lad.add_argument("--window", type=int, nargs=2, required=True,
                     metavar=("START", "END"),
                     help="Scoring window [start, end) in days.")
    lad.add_argument("--planner-summaries", nargs="*", default=[],
                     help="run_test_world.py plan JSONs (rungs 2 and 3).")
    lad.add_argument("--alpha", type=float, default=PATTERN_ALPHA)
    lad.add_argument("--beta", type=float, default=PATTERN_BETA)
    lad.add_argument("--mu", type=float, default=0.0)
    lad.add_argument("--cap", type=float, default=BRIGHTENING_CAP)
    lad.add_argument("--representation", choices=["map", "zonal"],
                     default="map",
                     help="Design on the ocean map or on its zonal-mean "
                          "profile (the pilot's choice).")
    lad.add_argument("--output", required=True)

    fb = sub.add_parser("feedback", help="one scored feedback episode")
    add_run_args(fb)
    fb.add_argument("--controller", choices=["pi", "adaptive"],
                    required=True)
    fb.add_argument("--sensitivity-from", nargs="+",
                    default=["experiment1"],
                    help="'experiment1' or response files.")
    fb.add_argument("--sensitivity-references-dir", default=None,
                    help="Where the response files' (training) references "
                         "are (default: next to --references).")
    fb.add_argument("--fit-days", type=int, default=60)
    fb.add_argument("--closed-loop-days", type=float, default=42.0)
    fb.add_argument("--damping", type=float, default=1.0)
    fb.add_argument("--indices", nargs="+", default=list(OCEAN_OBJECTIVES),
                    choices=OCEAN_OBJECTIVES)
    fb.add_argument("--no-anti-windup", dest="anti_windup",
                    action="store_false")
    fb.add_argument("--feedforward", action="store_true",
                    help="PI: also cancel the forecast's warming rate.")
    fb.add_argument("--pattern", type=float, nargs="+", default=None,
                    help="Adaptive: the fixed pattern (default uniform).")
    fb.add_argument("--relaxation", type=float, default=0.5)

    st = sub.add_parser("student", help="one scored student episode")
    add_run_args(st)
    st.add_argument("--student", required=True, help="Student .npz.")
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the stages cannot use."""
    from run_test_world import validate_run_args

    if args.stage == "responses":
        if args.days < 1 or args.segment_days < 1 or args.block_days < 1:
            raise SystemExit("--days, --segment-days, --block-days >= 1")
        if args.days % args.segment_days or args.days % args.block_days:
            raise SystemExit("--segment-days and --block-days must divide "
                             "--days")
        if not 0.0 < args.delta <= BRIGHTENING_CAP:
            raise SystemExit(f"--delta must lie in (0, {BRIGHTENING_CAP}]")
    elif args.stage == "ladder":
        if not 0 <= args.window[0] < args.window[1]:
            raise SystemExit("need 0 <= window start < window end")
    else:
        validate_run_args(args)
        if args.stage == "feedback" and args.pattern is not None and (
                len(args.pattern) != 5 or min(args.pattern) < 0.0):
            raise SystemExit("--pattern needs 5 non-negative numbers")


# --- Sensitivities -------------------------------------------------------------

def experiment1_responses(npz_path=EXPERIMENT1_NPZ,
                          json_path=EXPERIMENT1_JSON) -> np.ndarray:
    """Return Experiment 1's ``(days, 3, K)`` T0-T2 responses per unit band.

    These are central differences of the brute-force runs, averaged over
    starts and members.
    """
    with open(json_path) as f:
        meta = json.load(f)
    labels = meta["run_labels"]
    delta = meta["config"]["delta"]
    with np.load(npz_path, allow_pickle=False) as z:
        fd = z["fd_series"]                 # (ic, member, run, day, obj)
    k = sum(1 for lab in labels if lab.startswith("plus_"))
    diffs = np.stack([(fd[:, :, labels.index(f"plus_{j}")]
                       - fd[:, :, labels.index(f"minus_{j}")]) / (2 * delta)
                      for j in range(k)], axis=-1)
    return diffs.mean(axis=(0, 1))[:, :len(OCEAN_OBJECTIVES)]


def daily_indices(sst_daily, land_daily, weight_stack) -> np.ndarray:
    """Return ``(days, 4)`` T0, T1, T2 and LAND relative to 288 K."""
    import jax.numpy as jnp

    ws = jnp.asarray(weight_stack)
    return np.stack([np.asarray(objective_values(jnp.asarray(s),
                                                 jnp.asarray(lt), ws))
                     for s, lt in zip(sst_daily, land_daily)])


def load_grid(references_dir):
    """Return the grid's latitudes, masks and index weights (no model)."""
    with np.load(Path(references_dir) / "grid.npz", allow_pickle=False) as g:
        lats = g["latitudes_rad"]
        ocean, land = g["ocean_mask"], g["land_mask"]
    ws = np.asarray(stack_objective_weights(objective_weights(lats, ocean,
                                                              land)))
    return lats, ocean, land, ws


def responses_from_files(paths, references_dir, n_days=None):
    """Return pooled ``(days, 3, K)`` T0-T2 responses from response files."""
    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            runs = z["indices"]                  # (K, days, 4)
        refs = np.load(Path(references_dir) / meta["references_file"],
                       allow_pickle=False)
        _, _, _, ws = load_grid(references_dir)
        days = runs.shape[1] if n_days is None else n_days
        warmed = daily_indices(refs["warmed_sst"][:days],
                               refs["warmed_land_temperature"][:days], ws)
        out.append((runs[:, :days, :3] - warmed[None, :, :3])
                   / meta["delta"])
    return np.moveaxis(np.mean(out, axis=0), 0, -1)     # (days, 3, K)


def controller_sensitivity(args, references_dir):
    """Return ``(3, K)`` index rates per unit band and where they came from."""
    if args.sensitivity_from == ["experiment1"]:
        series, source = experiment1_responses(), "experiment1"
    else:
        series = responses_from_files(
            args.sensitivity_from,
            getattr(args, "sensitivity_references_dir", None)
            or references_dir)
        source = [str(p) for p in args.sensitivity_from]
    fit = range(min(args.fit_days, series.shape[0]))
    return sensitivity_rates(series, fit), source


# --- Stages ---------------------------------------------------------------------

def stage_responses(args):
    from run_gradient_fidelity import git_provenance
    from run_test_world import (
        build_model,
        check_role_allowed,
        load_ic,
        load_references,
        run_members,
    )

    ref_path, ref_meta, _ = load_references(args, args.days)
    role = ref_meta.get("role", "")
    check_role_allowed(role, allow_eval=False)
    m = build_model()
    entry, carry = load_ic(m, args, ref_path)
    k = int(np.shape(m["patterns"])[0])
    ws = np.asarray(stack_objective_weights(objective_weights(
        m["lats"], m["ocean"], m["land"])))
    run_args = types.SimpleNamespace(
        members=args.members, segments=args.days // args.segment_days,
        segment_days=args.segment_days, efficacy=1.0,
        warming_step_wm2=args.warming_step_wm2,
        warming_ramp_wm2_per_day=args.warming_ramp_wm2_per_day)
    blocks, indices, t0 = [], [], time.time()
    for j in range(k):
        setting = args.delta * np.eye(k, dtype=np.float32)[j]
        fields, _, seeds, _ = run_members(
            m, run_args, entry, carry, ref_meta["config"],
            lambda member, c, warming: constant_policy(setting))
        sst = fields["sst"]
        blocks.append(sst.reshape((-1, args.block_days) + sst.shape[1:])
                      .mean(axis=1).astype(np.float32))
        indices.append(daily_indices(sst, fields["land_temperature"], ws))
        print(f"  band {j}: {len(seeds)} members x {args.days} d "
              f"[{time.time() - t0:.0f}s]", flush=True)
    meta = {"stage": "responses", "role": role, "ic": entry,
            "delta": args.delta, "days": args.days,
            "block_days": args.block_days, "member_seeds": seeds,
            "references_file": ref_path.name, "config": vars(args),
            "git": git_provenance(), "command": " ".join(sys.argv),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.time() - t0, 1)}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ic{entry['index']:04d}_responses.npz"
    np.savez_compressed(path, sst_blocks=np.stack(blocks),
                        indices=np.stack(indices).astype(np.float32),
                        meta=json.dumps(meta, default=str))
    print(f"-> {path}")


def window_inputs(response_path, references_dir, window):
    """Return one state's ``(warm error, response maps)`` over a window."""
    with np.load(response_path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        blocks = z["sst_blocks"]                    # (K, n_blocks, ix, il)
    b = meta["block_days"]
    start, end = window
    if start % b or end % b or end > meta["days"]:
        raise SystemExit(f"window {window} must fall on {b}-day blocks "
                         f"within {meta['days']} days")
    refs = np.load(Path(references_dir) / meta["references_file"],
                   allow_pickle=False)
    normal = np.asarray(refs["normal_sst"][start:end], np.float64).mean(0)
    warmed = np.asarray(refs["warmed_sst"][start:end], np.float64).mean(0)
    runs = np.asarray(blocks[:, start // b:end // b], np.float64).mean(1)
    return warmed - normal, response_maps(warmed, runs, meta["delta"])


def stage_ladder(args):
    lats, ocean, _, _ = load_grid(args.references_dir)
    weights = np.asarray(area_weights(lats, ocean))
    sphere = np.asarray(area_weights(lats, np.ones_like(ocean)))
    unit = np.asarray(gaussian_band_patterns(lats, np.ones_like(ocean)))
    pairs = [window_inputs(p, args.references_dir, args.window)
             for p in args.responses]
    warm = np.stack([w for w, _ in pairs])
    resp = np.stack([r for _, r in pairs])
    if args.representation == "zonal":
        # Score the latitude profile only (the pilot's choice, Part 22): the
        # linear model is projected, so the design is the zonal optimum.
        mask = np.asarray(ocean) > 0
        warm = zonal_projection(warm, mask)
        resp = zonal_projection(resp, mask)
    plans = []
    for path in args.planner_summaries:
        with open(path) as f:
            plans.append(json.load(f))
    pooled_plan = {}
    if plans:
        # Rungs 2 and 3 from every planner run together: all their members'
        # schedules, and their mean effort.
        days = {p["config"]["segment_days"] for p in plans}
        if len(days) != 1:
            raise SystemExit(f"planner runs mix segment lengths {days}")
        pooled_plan = dict(
            planner_schedules=np.concatenate(
                [np.asarray(p["schedules"], np.float64) for p in plans]),
            planner_effort=float(np.mean([p["effort"] for p in plans])),
            segment_days=days.pop())
    summary = {"stage": "ladder", "config": vars(args),
               "training_states": [str(p) for p in args.responses],
               "pooled": ladder_designs(warm, resp, weights, unit, sphere,
                                        alpha=args.alpha, beta=args.beta,
                                        mu=args.mu, cap=args.cap,
                                        **pooled_plan),
               "per_planner_run": {}}
    for path, plan in zip(args.planner_summaries, plans):
        designs = ladder_designs(
            warm, resp, weights, unit, sphere,
            planner_schedules=plan["schedules"],
            planner_effort=plan["effort"],
            segment_days=plan["config"]["segment_days"], alpha=args.alpha,
            beta=args.beta, mu=args.mu, cap=args.cap)
        summary["per_planner_run"][str(path)] = {
            k: designs[k] for k in ("uniform_effort", "planner_average")}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    for name, rung in summary["pooled"].items():
        print(f"{name:>16}: {np.round(rung['amplitudes'], 4).tolist()} "
              f"(predicted objective {rung['predicted_objective']:.3g})")
    print(f"-> {out}")


def _episode_inputs(args):
    from run_test_world import build_model, load_ic, load_references

    n_days = args.segments * args.segment_days
    ref_path, ref_meta, refs = load_references(args, n_days)
    m = build_model()
    entry, carry = load_ic(m, args, ref_path)
    return m, entry, carry, ref_path, ref_meta, refs


def _finish(args, stage, entry, ref_path, ref_meta, m, refs, mean_fields,
            schedules, seeds, t0, **extra):
    from run_gradient_fidelity import git_provenance
    from run_test_world import score_runs, write_summary

    summary = {"stage": stage, "config": vars(args), "ic": entry,
               "members": len(seeds), "member_seeds": seeds,
               "reference_members": ref_meta["config"]["members"],
               "schedules": [s.tolist() for s in schedules],
               **score_runs(m, args, refs, mean_fields, schedules), **extra,
               "references": str(ref_path), "git": git_provenance(),
               "command": " ".join(sys.argv),
               "seconds": round(time.time() - t0, 1)}
    write_summary(args, summary, mean_fields, schedules)


def stage_feedback(args):
    from run_test_world import run_members

    m, entry, carry, ref_path, ref_meta, refs = _episode_inputs(args)
    ws = np.asarray(stack_objective_weights(objective_weights(
        m["lats"], m["ocean"], m["land"])))
    rates, source = controller_sensitivity(args, Path(ref_path).parent)
    target, forecast = refs["normal_sst"], refs["warmed_sst"]
    pattern = np.ones(5) if args.pattern is None else np.asarray(args.pattern)

    def make_policy(member, c, warming):
        if args.controller == "pi":
            return IndexPIController(
                rates, ws, target, PIConfig(
                    closed_loop_days=args.closed_loop_days,
                    damping=args.damping, segment_days=args.segment_days,
                    indices=tuple(args.indices),
                    anti_windup=args.anti_windup,
                    feedforward=args.feedforward), forecast)
        return AdaptiveController(pattern, rates[0], ws, target, forecast,
                                  AdaptiveConfig(
                                      segment_days=args.segment_days,
                                      relaxation=args.relaxation))

    t0 = time.time()
    mean_fields, schedules, seeds, controllers = run_members(
        m, args, entry, carry, ref_meta["config"], make_policy)
    _finish(args, f"feedback-{args.controller}", entry, ref_path, ref_meta,
            m, refs, mean_fields, schedules, seeds, t0,
            sensitivity={"source": source, "rates": rates.tolist()},
            controller_logs=[c.log for c in controllers])


def stage_student(args):
    from run_test_world import run_members

    params, cfg, meta = load_student(args.student)
    m, entry, carry, ref_path, ref_meta, refs = _episode_inputs(args)
    obs_w = band_observation_weights(m["lats"], m["patterns"])

    def make_policy(member, c, warming):
        return StudentPolicy(params, cfg, refs["normal_sst"], obs_w,
                             args.segment_days)

    t0 = time.time()
    mean_fields, schedules, seeds, _ = run_members(
        m, args, entry, carry, ref_meta["config"], make_policy)
    _finish(args, "student", entry, ref_path, ref_meta, m, refs, mean_fields,
            schedules, seeds, t0, student=str(args.student),
            student_meta=meta)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    if args.stage != "ladder":
        import jax
        print(f"JAX devices: {jax.devices()}")
    {"responses": stage_responses, "ladder": stage_ladder,
     "feedback": stage_feedback, "student": stage_student}[args.stage](args)


if __name__ == "__main__":
    main()
