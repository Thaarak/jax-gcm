#!/usr/bin/env python
"""Experiment 2: designing one fixed brightening pattern (Amendment 9 revision 1.1, Part B).

MCB_PROJECT_REPORT.md Part 25. Design on the 8 ``exp2_train`` states; judge
on the 16 ``exp2_eval`` states, which nothing has touched. The test world is
revision 1's:
- a Q-flux ramp to 6 W m-2 at day 182;
- efficacy 1;
- the normal-climate 5-member mean as the target;
- the cap 0.15;
- scoring over days 98-182.

Every design is one setting held for the whole 182-day episode
(``jcm.mcb.design``).

Stages, in campaign order:

``references``
    For each state of a role, the normal and warmed runs, 5 members from the
    references' seeds (93000), 182 days. Writes ``ic<index>_references.npz``:
    - the daily member-mean fields (the keys revision 1's analysis reads);
    - each member's window-mean SST, for brute force;
    - the normal climate's daily cloud cover, for the sunlight guess.
    Evaluation roles need ``--allow-eval-roles``. The campaign passes it only
    after ``designs.json`` exists.
``responses`` (training)
    For a knob layout (``b5`` or ``b13``), every band alone at ``--delta``,
    in the same 5 members, with the warming. Writes each member's window-mean
    SST.
``timing`` (training)
    Times one 182-day forward run and one forward-mode Jacobian run per
    layout on the first training state. The times set brute force's
    equal-cost sample count.
``gradient`` (training)
    The Gauss-Newton design for a layout, with the atmosphere snipped every
    14 days (``--mode snipped``) or not at all (``bptt``). It takes 4
    iterations on member 0 of every training state, from zero, then runs
    one forward diagnostic per state at the design.
``design`` (no model)
    Builds every arm's setting from the training outputs and writes
    ``designs.json``.
``episode``
    Runs one arm's frozen setting on one state in the same 5 members.
    Writes ``ic<index>.json`` and ``ic<index>.fields.npz`` in revision 1's
    format, read by ``analyze_experiment2.py``.

No pickle is written. Every stage records its seconds, which go into the
cost comparison.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- Registered settings (revision 1.1) -------------------------------------------
EPISODE_DAYS = 182
SEGMENT_DAYS = 14
WINDOW = (98, 182)
MEMBERS = 5
MEMBER_SEED0 = 93000
MEMBER_AMP = 0.001
STEP_WM2 = 0.0
RAMP_WM2_PER_DAY = 6.0 / 182.0           # revision 1: 0.03296703 W m-2/day
DELTA = 0.1
GN_ITERATIONS = 4
SNIP_DAYS = 14                           # Experiment 1's W*
MU = 6.325e-4                            # revision 1 (the pilot's R7)
LAYOUTS = ("b5", "b13")
TRAIN_ROLE, EVAL_ROLE = "exp2_train", "exp2_eval"
# --smoke (laptop checks only): a 4-day episode scored on days 2-4. Every
# output records it, and analyze_experiment2.py refuses smoke runs.
SMOKE = False
SMOKE_DAYS, SMOKE_WINDOW, SMOKE_SEGMENT_DAYS = 4, (2, 4), 2
# arm -> (layout, method); "uncontrolled" holds every band at zero.
ARMS = {
    "uncontrolled": ("b5", "none"),
    "uniform": ("b5", "uniform"),
    "brute5": ("b5", "brute"),
    "brute5_eq": ("b5", "brute_eq"),
    "grad5": ("b5", "snipped"),
    "bptt5": ("b5", "bptt"),
    "sunlight5": ("b5", "sunlight"),
    "brute13": ("b13", "brute"),
    "brute13_eq": ("b13", "brute_eq"),
    "grad13": ("b13", "snipped"),
    "sunlight13": ("b13", "sunlight"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("references")
    r.add_argument("--ic-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--days", type=int, default=EPISODE_DAYS)
    r.add_argument("--members", type=int, default=MEMBERS)
    r.add_argument("--max-ics", type=int, default=None)
    r.add_argument("--allow-eval-roles", action="store_true")

    s = sub.add_parser("responses")
    s.add_argument("--ic-dir", required=True)
    s.add_argument("--layout", choices=LAYOUTS, required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--days", type=int, default=EPISODE_DAYS)
    s.add_argument("--members", type=int, default=MEMBERS)
    s.add_argument("--max-ics", type=int, default=None)

    t = sub.add_parser("timing")
    t.add_argument("--ic-dir", required=True)
    t.add_argument("--days", type=int, default=EPISODE_DAYS)
    t.add_argument("--repeats", type=int, default=2)
    t.add_argument("--output", required=True)

    g = sub.add_parser("gradient")
    g.add_argument("--ic-dir", required=True)
    g.add_argument("--references-dir", required=True)
    g.add_argument("--layout", choices=LAYOUTS, required=True)
    g.add_argument("--mode", choices=["snipped", "bptt"], required=True)
    g.add_argument("--iterations", type=int, default=GN_ITERATIONS)
    g.add_argument("--days", type=int, default=EPISODE_DAYS)
    g.add_argument("--max-ics", type=int, default=None)
    g.add_argument("--output", required=True)

    d = sub.add_parser("design")
    d.add_argument("--references-dir", required=True)
    d.add_argument("--responses-dir", required=True)
    d.add_argument("--timing", required=True)
    d.add_argument("--gradient-dir", required=True)
    d.add_argument("--output", required=True)

    e = sub.add_parser("episode")
    e.add_argument("--ic-dir", required=True)
    e.add_argument("--ic-position", type=int, required=True)
    e.add_argument("--references-dir", required=True)
    e.add_argument("--designs", required=True)
    e.add_argument("--arm", choices=sorted(ARMS), required=True)
    e.add_argument("--output-dir", required=True)
    e.add_argument("--segments", type=int,
                   default=EPISODE_DAYS // SEGMENT_DAYS)
    e.add_argument("--members", type=int, default=MEMBERS)
    e.add_argument("--allow-eval-roles", action="store_true")
    return p.parse_args(argv)


def apply_smoke(args):
    """Shrink every setting to the 4-day smoke episode (laptop checks only)."""
    global SMOKE, EPISODE_DAYS, WINDOW, SEGMENT_DAYS
    SMOKE, EPISODE_DAYS, WINDOW = True, SMOKE_DAYS, SMOKE_WINDOW
    SEGMENT_DAYS = SMOKE_SEGMENT_DAYS
    if hasattr(args, "days"):
        args.days = SMOKE_DAYS
    if hasattr(args, "segments"):
        args.segments = SMOKE_DAYS // SMOKE_SEGMENT_DAYS


def validate_args(args):
    """Fail fast on settings a stage cannot use."""
    days = getattr(args, "days", EPISODE_DAYS)
    if days < WINDOW[1]:
        raise SystemExit(f"--days must reach the scoring window's end "
                         f"({WINDOW[1]})")
    if getattr(args, "members", 1) < 1:
        raise SystemExit("--members must be >= 1")
    if args.stage == "gradient" and args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.stage == "episode" and args.segments * SEGMENT_DAYS < WINDOW[1]:
        raise SystemExit("the episode must reach the scoring window's end")


def check_role(role: str, allow_eval: bool, training_only: bool = False):
    """Accept only Experiment 2's roles; evaluation needs explicit permission."""
    if role not in (TRAIN_ROLE, EVAL_ROLE):
        raise SystemExit(f"role '{role}' is not an Experiment 2 role")
    if role == EVAL_ROLE and (training_only or not allow_eval):
        raise SystemExit(f"role '{role}' is evaluation: "
                         + ("this stage is training-only" if training_only
                            else "pass --allow-eval-roles once designs.json "
                                 "is frozen"))


def window_mean(daily, window=None):
    """Return the mean over days ``[start, end)`` (default: the scoring window)."""
    start, end = WINDOW if window is None else window
    return np.asarray(daily, np.float64)[start:end].mean(axis=0)


def sha256(path) -> str:
    """Return a file's SHA-256 checksum."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Model setup (as run_test_world.build_model; kept here so Experiment 2 does
# not depend on files other experiments change) ------------------------------------

def build_model():
    """Return the coupled model pieces every stage needs."""
    import jax_datetime as jdt

    from jcm.mcb.coupled_controller import create_coupled_step_fn
    from jcm.mcb.coupled_train import ocean_mask_from_coupler
    from run_coupled_training import coupler_workflow, setup_coupled_model
    from run_gradient_fidelity import land_mask_from_coupler
    from run_stage5_training import START_DATE

    coupler, coords, _, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    template = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                     jitted=True)
    shape = coords.horizontal.nodal_shape
    return {"coords": coords, "template": template, "step_fn": step_fn,
            "lats": coords.horizontal.latitudes,
            "lons": coords.horizontal.longitudes,
            "ocean": ocean_mask_from_coupler(coupler),
            "land": land_mask_from_coupler(coupler, shape), "shape": shape}


def load_role(ic_dir, template, max_ics=None, positions=None):
    """Return ``(role, [(position, entry, carry)])``; loads only what is asked."""
    from run_gradient_fidelity import load_carry
    from run_stage5_training import START_DATE

    ic_dir = Path(ic_dir)
    manifest = json.loads((ic_dir / "manifest.json").read_text())
    if manifest.get("start_date") != START_DATE:
        raise SystemExit("manifest start_date differs from the model's")
    role = manifest.get("macro_role") or ic_dir.name
    entries = list(enumerate(manifest["ics"]))
    if positions is not None:
        entries = [entries[p] for p in positions]
    if max_ics is not None:
        entries = entries[:max_ics]
    return role, [(p, e, load_carry(str(ic_dir / e["carry_file"]), template))
                  for p, e in entries]


def members_of(carry, ic_index: int, members: int):
    """Yield ``(member, seed, carry)`` for the references' weather samples."""
    from jcm.mcb.test_world import member_seed, perturb_member

    for m in range(members):
        seed = member_seed(MEMBER_SEED0, ic_index, m)
        yield m, seed, (carry if seed is None
                        else perturb_member(carry, seed, MEMBER_AMP))


def warmings(carry, ocean):
    """Return the normal (no warming) and warmed (ramp) ``Warming`` objects."""
    from jcm.mcb.test_world import make_warming

    return (make_warming(carry, ocean),
            make_warming(carry, ocean, STEP_WM2, RAMP_WM2_PER_DAY))


def fields_with_cloud(carry):
    """Return the test world's daily surface fields plus the total cloud cover."""
    from jcm.mcb.test_world import surface_fields

    out = surface_fields(carry)
    out["cloud_cover"] = (carry["atm"]["derived"]["physics"]
                          .shortwave_rad.cloudc)
    return out


def sst_only(carry):
    """Return only the SST: what the design runs need."""
    return {"sst": carry["ocn"]["state"].sea_surface_temperature}


# --- Stages ---------------------------------------------------------------------------

def stage_references(args):
    import jax.numpy as jnp

    from jcm.mcb.test_world import make_segment_fn
    from run_gradient_fidelity import git_provenance

    m = build_model()
    role, ics = load_role(args.ic_dir, m["template"], args.max_ics)
    check_role(role, args.allow_eval_roles)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "grid.npz",
                        ocean_mask=np.asarray(m["ocean"], np.float32),
                        land_mask=np.asarray(m["land"], np.float32),
                        latitudes_rad=np.asarray(m["lats"]),
                        longitudes_rad=np.asarray(m["lons"]))
    seg = make_segment_fn(m["step_fn"], np.zeros((5,) + m["shape"],
                                                 np.float32),
                          args.days, fields_fn=fields_with_cloud)
    zero, one = jnp.zeros(5, jnp.float32), jnp.asarray(1.0, jnp.float32)
    meta = {"stage": "references", "role": role, "config": vars(args),
            "experiment": "Amendment 9 revision 1.1, Part B",
            "warming": {"step_wm2": STEP_WM2,
                        "ramp_wm2_per_day": RAMP_WM2_PER_DAY},
            "window": list(WINDOW), "member_seed0": MEMBER_SEED0,
            "member_amp": MEMBER_AMP, "smoke": SMOKE, "git": git_provenance(),
            "command": " ".join(sys.argv), "ics": []}
    for _, entry, carry in ics:
        t0, arrays, seeds = time.time(), {}, []
        q_base = carry["ocn"]["forcing"].q_flux
        for name, which in (("normal", 0), ("warmed", 1)):
            total, window_maps = None, []
            for _, seed, c in members_of(carry, entry["index"], args.members):
                warming = warmings(c, m["ocean"])[which]
                _, f = seg(c, zero, one, q_base, warming)
                f = {k: np.asarray(v, np.float64) for k, v in f.items()}
                total = f if total is None else {k: total[k] + f[k]
                                                 for k in total}
                window_maps.append(window_mean(f["sst"]))
                if name == "normal":
                    seeds.append(seed)
            for k, v in total.items():
                if k == "cloud_cover" and name != "normal":
                    continue
                arrays[f"{name}_{k}"] = (v / args.members).astype(np.float32)
            arrays[f"{name}_members_window_sst"] = np.stack(
                window_maps).astype(np.float32)
        fname = f"ic{entry['index']:04d}_references.npz"
        np.savez_compressed(out / fname, **arrays)
        meta["ics"].append({"entry": entry, "file": fname,
                            "member_seeds": seeds,
                            "start_sim_time_s": float(
                                carry["ocn"]["state"].sim_time),
                            "seconds": round(time.time() - t0, 1)})
        (out / "references_manifest.json").write_text(json.dumps(meta,
                                                                 indent=2))
        print(f"  IC {entry['index']:>4}: 2 x {args.members} members x "
              f"{args.days} d [{time.time() - t0:.0f}s]", flush=True)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "references_manifest.json").write_text(json.dumps(meta, indent=2))


def stage_responses(args):
    import jax.numpy as jnp

    from jcm.mcb.design import band_patterns
    from jcm.mcb.test_world import make_segment_fn
    from run_gradient_fidelity import git_provenance

    m = build_model()
    role, ics = load_role(args.ic_dir, m["template"], args.max_ics)
    check_role(role, False, training_only=True)
    patterns = band_patterns(args.layout, m["lats"], m["ocean"])
    k = int(patterns.shape[0])
    seg = make_segment_fn(m["step_fn"], patterns, args.days,
                          fields_fn=sst_only)
    one = jnp.asarray(1.0, jnp.float32)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _, entry, carry in ics:
        t0 = time.time()
        q_base = carry["ocn"]["forcing"].q_flux
        maps = np.zeros((k, args.members) + tuple(m["shape"]), np.float32)
        for j in range(k):
            setting = jnp.asarray(DELTA * np.eye(k)[j], jnp.float32)
            for mem, _, c in members_of(carry, entry["index"], args.members):
                _, f = seg(c, setting, one, q_base, warmings(c, m["ocean"])[1])
                maps[j, mem] = window_mean(f["sst"])
        meta = {"layout": args.layout, "delta": DELTA, "k": k, "smoke": SMOKE,
                "members": args.members, "ic": entry,
                "seconds": round(time.time() - t0, 1),
                "git": git_provenance()}
        np.savez_compressed(
            out / f"ic{entry['index']:04d}_responses_{args.layout}.npz",
            band_members_window_sst=maps, meta=json.dumps(meta))
        print(f"  IC {entry['index']:>4}: {k} bands x {args.members} members "
              f"[{meta['seconds']:.0f}s]", flush=True)


def stage_timing(args):
    import jax
    import jax.numpy as jnp

    from jcm.mcb.design import band_patterns, make_window_jacobian_fn
    from jcm.mcb.test_world import make_segment_fn
    from run_gradient_fidelity import git_provenance

    m = build_model()
    role, ics = load_role(args.ic_dir, m["template"], positions=[0])
    check_role(role, False, training_only=True)
    _, entry, carry = ics[0]
    q_base = carry["ocn"]["forcing"].q_flux
    warmed = warmings(carry, m["ocean"])[1]
    one = jnp.asarray(1.0, jnp.float32)

    def timed(fn, *a):
        jax.block_until_ready(fn(*a))                 # compile
        times = []
        for _ in range(args.repeats):
            t0 = time.time()
            jax.block_until_ready(fn(*a))
            times.append(time.time() - t0)
        return float(np.median(times)), times

    result = {"ic": entry, "days": args.days, "repeats": args.repeats,
              "smoke": SMOKE,
              "devices": [str(d) for d in jax.devices()],
              "git": git_provenance()}
    seg = make_segment_fn(m["step_fn"], band_patterns("b5", m["lats"],
                                                      m["ocean"]),
                          args.days, fields_fn=sst_only)
    result["forward_s"], result["forward_runs_s"] = timed(
        seg, carry, jnp.zeros(5, jnp.float32), one, q_base, warmed)
    result["jacobian_s"] = {}
    for layout in LAYOUTS:
        pats = band_patterns(layout, m["lats"], m["ocean"])
        jac = make_window_jacobian_fn(m["step_fn"], pats, args.days, WINDOW)
        a = jnp.zeros(pats.shape[0], jnp.float32)
        result["jacobian_s"][layout], _ = timed(
            jac, a, carry, one, q_base, warmed, jnp.asarray(SNIP_DAYS))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"forward {result['forward_s']:.1f} s; Jacobian runs "
          f"{result['jacobian_s']} -> {args.output}")


def normal_targets(references_dir, entries):
    """Return the normal climate's window-mean SST maps minus 288 K, ``(S, ix, il)``."""
    from jcm.mcb.gradient_fidelity import MAP_REFERENCE_K

    out = []
    for e in entries:
        refs = np.load(Path(references_dir)
                       / f"ic{e['index']:04d}_references.npz")
        out.append(window_mean(refs["normal_sst"]) - MAP_REFERENCE_K)
    return np.stack(out)


def stage_gradient(args):
    import jax
    import jax.numpy as jnp

    from jcm.mcb.design import (
        band_patterns,
        gauss_newton_design,
        make_window_jacobian_fn,
        make_window_sst_fn,
        zonal_objective,
    )
    from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
    from jcm.mcb.scores import area_weights
    from jcm.mcb.test_world import BRIGHTENING_CAP
    from run_gradient_fidelity import git_provenance

    t_start = time.time()
    m = build_model()
    role, ics = load_role(args.ic_dir, m["template"], args.max_ics)
    check_role(role, False, training_only=True)
    patterns = band_patterns(args.layout, m["lats"], m["ocean"])
    k = int(patterns.shape[0])
    ocean = np.asarray(m["ocean"], np.float64)
    weights = np.asarray(area_weights(m["lats"], m["ocean"]), np.float64)
    targets = normal_targets(args.references_dir, [e for _, e, _ in ics])
    jac_fn = make_window_jacobian_fn(m["step_fn"], patterns, args.days,
                                     WINDOW)
    snip = jnp.asarray(SNIP_DAYS if args.mode == "snipped"
                       else NO_TRUNCATION_DAYS)
    one = jnp.asarray(1.0, jnp.float32)
    seconds = []

    def evaluate(s, a):
        _, _, carry = ics[s]
        t0 = time.time()
        jac, mean_map = jac_fn(jnp.asarray(a, jnp.float32), carry, one,
                               carry["ocn"]["forcing"].q_flux,
                               warmings(carry, m["ocean"])[1], snip)
        mean_map, jac = np.asarray(mean_map), np.asarray(jac)
        seconds.append(time.time() - t0)
        return mean_map, jac

    design, log = gauss_newton_design(evaluate, targets, ocean, weights, k,
                                      args.iterations, mu=MU,
                                      cap=BRIGHTENING_CAP)
    design_seconds = float(np.sum(seconds))
    # Diagnostic only (not part of the design's cost): the training
    # objective at the design, one forward run per state.
    forward = jax.jit(make_window_sst_fn(m["step_fn"], patterns, args.days,
                                         WINDOW))
    finals = []
    for s, (_, _, carry) in enumerate(ics):
        mean_map = forward(jnp.asarray(design, jnp.float32), carry, one,
                           carry["ocn"]["forcing"].q_flux,
                           warmings(carry, m["ocean"])[1], snip)
        finals.append(zonal_objective(np.asarray(mean_map) - targets[s],
                                      ocean, weights))
    result = {"layout": args.layout, "mode": args.mode, "k": k, "smoke": SMOKE,
              "iterations": args.iterations, "snip_days": (
                  SNIP_DAYS if args.mode == "snipped" else None),
              "design": design.tolist(), "log": log,
              "final_training_objective": float(np.mean(finals)),
              "jacobian_run_seconds": [round(x, 2) for x in seconds],
              "design_seconds": round(design_seconds, 1),
              "ics": [e for _, e, _ in ics],
              "total_seconds": round(time.time() - t_start, 1),
              "git": git_provenance(), "command": " ".join(sys.argv)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{args.layout} {args.mode}: design {np.round(design, 4).tolist()}"
          f" | training objective by iterate "
          f"{[r['training_objective'] for r in log]} -> final "
          f"{result['final_training_objective']:.4g} -> {args.output}")


def stage_design(args):
    """Every arm's setting from the training outputs (no model run)."""
    from jcm.mcb.design import (
        band_patterns,
        brute_force_design,
        equal_cost_members,
        sunlight_design,
    )
    from jcm.mcb.scores import area_weights
    from jcm.mcb.test_world import BRIGHTENING_CAP

    refs_dir = Path(args.references_dir)
    manifest = json.loads((refs_dir / "references_manifest.json").read_text())
    if manifest["role"] != TRAIN_ROLE:
        raise SystemExit("designs are built from the training references only")
    with np.load(refs_dir / "grid.npz") as g:
        lats = g["latitudes_rad"]
        ocean = np.asarray(g["ocean_mask"], np.float64)
    weights = np.asarray(area_weights(lats, ocean), np.float64)
    entries = [i["entry"] for i in manifest["ics"]]
    targets_k = []
    warmed, clouds = [], []
    for e in entries:
        refs = np.load(refs_dir / f"ic{e['index']:04d}_references.npz")
        targets_k.append(window_mean(refs["normal_sst"]))
        warmed.append(np.asarray(refs["warmed_members_window_sst"],
                                 np.float64))
        clouds.append(np.asarray(refs["normal_cloud_cover"], np.float64))
    targets_k, warmed = np.stack(targets_k), np.stack(warmed)
    timing = json.loads(Path(args.timing).read_text())
    if bool(manifest.get("smoke")) != SMOKE or bool(timing.get("smoke")) \
            != SMOKE:
        raise SystemExit("smoke and real inputs cannot be mixed")
    start_doy = float(np.mean([i["start_sim_time_s"] / 86400.0 % 365.2425
                               for i in manifest["ics"]]))
    cloud = np.mean(clouds, axis=0)
    arms, extras = {"uncontrolled": {"layout": "b5", "method": "none",
                                     "amplitudes": [0.0] * 5}}, {}
    for layout in LAYOUTS:
        band = np.stack([np.asarray(np.load(
            Path(args.responses_dir)
            / f"ic{e['index']:04d}_responses_{layout}.npz")
            ["band_members_window_sst"], np.float64) for e in entries])
        k = band.shape[1]
        suffix = layout[1:]
        a, info = brute_force_design(warmed, band, targets_k, DELTA, ocean,
                                     weights, mu=MU, cap=BRIGHTENING_CAP)
        arms[f"brute{suffix}"] = {"layout": layout, "method": "brute",
                                  "amplitudes": a.tolist(), **info}
        m_eq = equal_cost_members(GN_ITERATIONS,
                                  timing["jacobian_s"][layout], k,
                                  timing["forward_s"], MEMBERS)
        a_eq, info_eq = brute_force_design(warmed, band, targets_k, DELTA,
                                           ocean, weights, mu=MU,
                                           cap=BRIGHTENING_CAP, members=m_eq)
        arms[f"brute{suffix}_eq"] = {"layout": layout, "method": "brute_eq",
                                     "amplitudes": a_eq.tolist(), **info_eq}
        if layout == "b5":
            arms["uniform"] = {"layout": "b5", "method": "uniform",
                               "amplitudes": info["uniform_cancel"],
                               **info["uniform_info"]}
        pats = band_patterns(layout, lats, ocean)
        a_sun, info_sun = sunlight_design(
            lats, ocean, np.asarray(pats), cloud, start_doy, weights,
            EPISODE_DAYS, STEP_WM2, RAMP_WM2_PER_DAY, WINDOW, mu=MU,
            cap=BRIGHTENING_CAP)
        arms[f"sunlight{suffix}"] = {"layout": layout, "method": "sunlight",
                                     "amplitudes": a_sun.tolist(),
                                     **info_sun}
        for mode, name in (("snipped", f"grad{suffix}"),
                           ("bptt", f"bptt{suffix}")):
            path = Path(args.gradient_dir) / f"gradient_{layout}_{mode}.json"
            if name not in ARMS or not path.exists():
                continue
            g = json.loads(path.read_text())
            arms[name] = {"layout": layout, "method": mode,
                          "amplitudes": g["design"],
                          "final_training_objective":
                              g["final_training_objective"],
                          "design_seconds": g["design_seconds"]}
        extras[layout] = {
            "equal_cost_members": m_eq,
            "brute_cost_seconds_per_member": float(
                (k + 1) * len(entries) * timing["forward_s"]),
            "gradient_cost_seconds": float(
                GN_ITERATIONS * len(entries) * timing["jacobian_s"][layout])}
    missing = sorted(set(ARMS) - set(arms))
    if missing:
        raise SystemExit(f"arms without a design: {missing}")
    out = {"experiment": "Amendment 9 revision 1.1, Part B", "smoke": SMOKE,
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "arms": arms, "costs": extras, "timing": timing,
           "start_day_of_year": start_doy,
           "training_states": [e["index"] for e in entries]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    for name in ARMS:
        print(f"{name:>13}: {np.round(arms[name]['amplitudes'], 4).tolist()}")
    print(f"-> {args.output} (SHA-256 {sha256(args.output)})")


def stage_episode(args):
    from jcm.mcb.design import band_patterns, unit_profiles
    from jcm.mcb.scores import area_weights, effort
    from jcm.mcb.test_world import (
        constant_policy,
        make_segment_fn,
        run_episode,
        surface_fields,
    )
    from run_gradient_fidelity import git_provenance

    t_start = time.time()
    designs = json.loads(Path(args.designs).read_text())
    arm = designs["arms"][args.arm]
    m = build_model()
    role, ics = load_role(args.ic_dir, m["template"],
                          positions=[args.ic_position])
    check_role(role, args.allow_eval_roles)
    _, entry, carry = ics[0]
    refs_manifest = json.loads((Path(args.references_dir)
                                / "references_manifest.json").read_text())
    if refs_manifest["role"] != role:
        raise SystemExit("the references belong to another role")
    patterns = band_patterns(arm["layout"], m["lats"], m["ocean"])
    k = int(patterns.shape[0])
    a = np.asarray(arm["amplitudes"], np.float32)
    if a.shape != (k,):
        raise SystemExit(f"arm {args.arm} has {a.size} settings, layout "
                         f"{arm['layout']} has {k} bands")
    seg = make_segment_fn(m["step_fn"], patterns, SEGMENT_DAYS,
                          fields_fn=surface_fields)
    runs, seeds = [], []
    for _, seed, c in members_of(carry, entry["index"], args.members):
        ep = run_episode(c, seg, constant_policy(a), args.segments,
                         SEGMENT_DAYS, k, warmings(c, m["ocean"])[1], 1.0)
        runs.append(ep.fields)
        seeds.append(seed)
    mean = {key: np.mean([r[key] for r in runs], axis=0).astype(np.float32)
            for key in runs[0]}
    days = args.segments * SEGMENT_DAYS
    sphere = np.asarray(area_weights(m["lats"], np.ones(m["shape"])))
    summary = {"stage": "episode", "arm": args.arm, "layout": arm["layout"],
               "method": arm["method"], "amplitudes": a.tolist(),
               "effort": effort(np.repeat(a[None], days, axis=0),
                                unit_profiles(arm["layout"], m["lats"],
                                              m["shape"]), sphere),
               "ic": entry, "members": len(seeds), "member_seeds": seeds,
               "smoke": SMOKE, "designs_sha256": sha256(args.designs),
               "seconds": round(time.time() - t_start, 1),
               "git": git_provenance(), "command": " ".join(sys.argv)}
    out = Path(args.output_dir) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"ic{entry['index']:04d}"
    np.savez_compressed(str(stem) + ".fields.npz", **mean)
    Path(str(stem) + ".json").write_text(json.dumps(summary, indent=2))
    print(f"{args.arm} on IC {entry['index']}: effort {summary['effort']:.4f}"
          f" [{summary['seconds']:.0f}s] -> {stem}.json")


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    validate_args(args)
    if args.stage != "design":
        import jax
        print(f"JAX devices: {jax.devices()}")
    {"references": stage_references, "responses": stage_responses,
     "timing": stage_timing, "gradient": stage_gradient,
     "design": stage_design, "episode": stage_episode}[args.stage](args)


if __name__ == "__main__":
    main()
