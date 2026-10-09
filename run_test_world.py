#!/usr/bin/env python
"""The Experiment 2-3 test world: reference ensembles and scored episodes.

MCB_PROJECT_REPORT.md Part 18 steps 10-12 (``jcm/mcb/test_world.py`` and
``jcm/mcb/scores.py``). Two stages:

``references``
    For every IC in a Step-0 role directory, run ``--members`` member runs
    with no brightening for ``--days`` days. They run once without warming
    (the normal climate: the target) and once with the warming (the
    uncontrolled warmed run: the gain's reference).
    Writes ``<output-dir>/ic<index>_references.npz`` per IC (the ensemble-mean
    daily fields and each member's block means), ``grid.npz`` and
    ``references_manifest.json``. No pickle.
    Evaluation roles (``*_eval``) are refused unless ``--allow-eval-roles``
    is given. Their references are built only after revision 1 of
    Amendment 9 is frozen, so no design choice can be informed by them.

``episode`` / ``plan``
    Run one controlled episode from one IC with the warming, either with a
    fixed band setting (``episode``) or with the receding-horizon planner of
    ``jcm.mcb.planner`` (``plan``: Part 18 steps 13-15, presets
    ``snipped60``, ``short14``, ``bptt60``). Score it against that IC's
    references over ``[--score-start-day, --score-end-day)``. The controlled side is averaged
    over as many members as the references, started from the same member
    seeds, so both sides carry the same weather noise. The scores are:
    - the gains of SST (ocean), land temperature (land), and rainfall and
      evaporation (land and global);
    - the effort;
    - the map objective's terms, with the uncontrolled warmed run's terms for
      comparison.
    Writes a JSON summary.

The warming strength, run lengths, member counts and scoring window are set
by the pilot (Part 18 step 23) and frozen in revision 1; the defaults here
are placeholders for testing.

Examples (CPU smoke; GPU runs use the macro ICs of Step 0):
    python run_test_world.py references --ic-dir <ics>/exp1 --days 4 \
        --members 2 --member-block-days 2 --output-dir /tmp/tw/refs
    python run_test_world.py episode --ic-dir <ics>/exp1 \
        --references /tmp/tw/refs/ic0000_references.npz --uniform 0.05 \
        --segments 2 --segment-days 2 --output /tmp/tw/episode.json
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb.band_basis import gaussian_band_patterns
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.gradient_truncation import NO_TRUNCATION_DAYS
from jcm.mcb.land_climatology import check_land_mode, land_climatology_mode
from jcm.mcb.planner import (
    OPTIMIZERS,
    PRESETS,
    REPRESENTATIONS,
    SENSING,
    LearningPlanner,
    Planner,
    PlannerConfig,
)
from jcm.mcb.strength_estimator import MODES as LEARN_MODES
from jcm.mcb.strength_estimator import EstimatorConfig
from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    effort,
    objective_terms,
    pattern_objective,
    restoration_scores,
    zonal_projection,
)
from jcm.mcb.test_world import (
    BRIGHTENING_CAP,
    FIELD_NAMES,
    constant_policy,
    domain_weights,
    make_segment_fn,
    make_warming,
    member_seed,
    perturb_member,
    reference_ensemble,
    run_episode,
    time_means,
)
from run_coupled_training import coupler_workflow, setup_coupled_model
from run_gradient_fidelity import (
    git_provenance,
    land_mask_from_coupler,
    load_manifest_ics,
)
from run_stage5_training import START_DATE

K_BANDS = 5          # the default band layout (jcm.mcb.band_basis)


def add_warming_args(sp):
    """Add the warming options (shared with the other Experiment 3 runners)."""
    sp.add_argument("--warming-step-wm2", type=float, default=4.0,
                    help="Steady heat into the ocean, W m-2 "
                         "(placeholder until the pilot).")
    sp.add_argument("--warming-ramp-wm2-per-day", type=float,
                    default=0.0,
                    help="Growth of the heat input, W m-2 per day.")


def add_run_args(sp):
    """Add the options of one scored episode (shared with other runners)."""
    sp.add_argument("--ic-dir", required=True)
    sp.add_argument("--ic-position", type=int, default=0,
                    help="Position of the IC in the manifest's list.")
    sp.add_argument("--references", required=True,
                    help="That IC's ic<index>_references.npz.")
    sp.add_argument("--segments", type=int, required=True)
    sp.add_argument("--segment-days", type=int, default=14,
                    help="Days between decisions (re-plans).")
    sp.add_argument("--efficacy", type=float, nargs="+", default=[1.0],
                    help="True strength of the spraying (1 = nominal): one "
                         "value for every band, or one per band "
                         "(Experiment 3b). Only the simulated world sees "
                         "it.")
    sp.add_argument("--members", type=int, default=None,
                    help="Members to run and average before scoring "
                         "(default: as many as the references, from the "
                         "same member seeds, so both sides carry the "
                         "same weather noise).")
    sp.add_argument("--score-start-day", type=int, default=0)
    sp.add_argument("--score-end-day", type=int, default=None,
                    help="Default: the end of the episode.")
    sp.add_argument("--alpha", type=float, default=PATTERN_ALPHA)
    sp.add_argument("--beta", type=float, default=PATTERN_BETA)
    sp.add_argument("--mu", type=float, default=0.0)
    sp.add_argument("--lam", type=float, default=0.0)
    sp.add_argument("--output", required=True, help="JSON summary path.")
    sp.add_argument("--save-fields", action="store_true",
                    help="Also write the daily fields next to the JSON.")
    add_warming_args(sp)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("references", help="normal-climate and warmed "
                                          "reference ensembles")
    r.add_argument("--ic-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--split", default="all",
                   choices=["all", "train", "heldout"])
    r.add_argument("--max-ics", type=int, default=None)
    r.add_argument("--branches", type=int, nargs="+", default=None,
                   help="Only these weather branches (default: all), so "
                        "branch sets can run side by side into separate "
                        "output directories.")
    r.add_argument("--days", type=int, default=240,
                   help="Run length; an episode plus its last look-ahead.")
    r.add_argument("--members", type=int, default=5)
    r.add_argument("--member-seed0", type=int, default=93000)
    r.add_argument("--member-amp", type=float, default=0.001)
    r.add_argument("--member-block-days", type=int, default=5,
                   help="Store each member's block means (0 = do not).")
    r.add_argument("--no-warmed", dest="warmed", action="store_false",
                   help="Build only the normal climate.")
    r.add_argument("--allow-eval-roles", action="store_true",
                   help="Allow *_eval roles (only after revision 1).")
    add_warming_args(r)

    e = sub.add_parser("episode", help="one scored episode, fixed design")
    add_run_args(e)
    amp = e.add_mutually_exclusive_group(required=True)
    amp.add_argument("--amplitudes", type=float, nargs=K_BANDS)
    amp.add_argument("--uniform", type=float)

    pl = sub.add_parser("plan", help="one scored episode, driven by the "
                                     "receding-horizon planner")
    add_run_args(pl)
    pl.add_argument("--preset", default="snipped60", choices=sorted(PRESETS),
                    help="snipped60 (60-day look-ahead, W* = 14), short14 "
                         "(Dubey et al.'s 14 days, exact BPTT), bptt60.")
    pl.add_argument("--lookahead-days", type=int, default=None)
    pl.add_argument("--window-days", type=int, default=None,
                    help="Atmosphere snip in days; 0 = none (full BPTT).")
    pl.add_argument("--copies", type=int, default=None)
    pl.add_argument("--optimizer", choices=OPTIMIZERS, default=None)
    pl.add_argument("--iterations", type=int, default=None)
    pl.add_argument("--learning-rate", type=float, default=None)
    pl.add_argument("--representation", choices=REPRESENTATIONS,
                    default=None,
                    help="What the planner's objective scores: the ocean "
                         "map or its zonal-mean profile (the pilot's "
                         "choice; Gauss-Newton only).")
    pl.add_argument("--planner-efficacy", type=float, nargs="+",
                    default=None,
                    help="The spraying strength the planner believes in, one "
                         "value or one per band (default: the true one, as "
                         "in Experiment 3a: the oracle).")
    pl.add_argument("--learn-strength", choices=("off",) + LEARN_MODES,
                    default="off",
                    help="Learn the strength from forecast misses "
                         "(Experiment 3b): per band, or one shared factor. "
                         "Starts from nominal strength.")
    pl.add_argument("--learn-noise-k", type=float, default=None,
                    help="Typical forecast miss (K) when the strength is "
                         "known; set from the pilot's oracle runs.")
    pl.add_argument("--learn-prior-sd", type=float, default=1.0,
                    help="Prior standard deviation of the strength.")
    pl.add_argument("--sensing", choices=SENSING, default=None,
                    help="What the planner knows: the exact state "
                         "(Experiments 3a-3b), or only the ocean "
                         "(Experiment 3c), with the weather of another "
                         "branch of the same macro state on day 0 and of "
                         "its own forecasts after that.")
    return p.parse_args(argv)


def validate_args(args):
    """Fail fast on settings the stages cannot use."""
    if args.stage == "references":
        if args.days < 1 or args.members < 1:
            raise SystemExit("--days and --members must be >= 1")
        if args.member_block_days < 0 or (
                args.member_block_days
                and args.days % args.member_block_days):
            raise SystemExit("--member-block-days must be 0 or divide "
                             "--days")
    else:
        validate_run_args(args)
        if args.stage == "episode":
            a = episode_amplitudes(args)
            if np.any(a < 0.0) or np.any(a > BRIGHTENING_CAP):
                raise SystemExit(f"band amplitudes must lie in [0, "
                                 f"{BRIGHTENING_CAP}]")
        else:
            cfg = planner_config(args)
            if args.planner_efficacy is not None and \
                    np.any(strength_vector(args.planner_efficacy) <= 0.0):
                raise SystemExit("--planner-efficacy must be > 0")
            if args.learn_strength != "off":
                validate_learning(args, cfg)


def validate_learning(args, cfg):
    """Fail fast on a learning planner that could not learn."""
    if cfg.optimizer != "gauss_newton":
        raise SystemExit("--learn-strength needs --optimizer gauss_newton")
    if cfg.lookahead_days != args.segment_days:
        raise SystemExit("--learn-strength needs the look-ahead to equal "
                         "--segment-days")
    if args.planner_efficacy is not None:
        raise SystemExit("--learn-strength starts from nominal strength; "
                         "drop --planner-efficacy")
    if args.learn_noise_k is None:
        raise SystemExit("--learn-strength needs --learn-noise-k")
    try:
        estimator_config(args).validate()
    except ValueError as err:
        raise SystemExit(f"learner: {err}") from err


def estimator_config(args) -> EstimatorConfig:
    """Return the learner's settings from the command line."""
    return EstimatorConfig(noise_k=args.learn_noise_k,
                           mode=args.learn_strength,
                           prior_sd=args.learn_prior_sd)


def strength_vector(values, k_bands: int = K_BANDS) -> np.ndarray:
    """Return a strength as ``(k_bands,)``: one value for all bands, or one each."""
    v = np.atleast_1d(np.asarray(values, np.float32))
    if v.size == 1:
        return np.full(k_bands, v[0], np.float32)
    if v.size != k_bands:
        raise SystemExit(f"a strength needs 1 or {k_bands} values, got "
                         f"{v.size}")
    return v


def validate_run_args(args):
    """Fail fast on episode settings that cannot be run or scored."""
    if args.segments < 1 or args.segment_days < 1:
        raise SystemExit("--segments and --segment-days must be >= 1")
    n_days = args.segments * args.segment_days
    end = n_days if args.score_end_day is None else args.score_end_day
    if not 0 <= args.score_start_day < end <= n_days:
        raise SystemExit("need 0 <= --score-start-day < "
                         "--score-end-day <= segments * segment-days")
    if np.any(strength_vector(args.efficacy) < 0.0):
        raise SystemExit("--efficacy must be >= 0")
    if args.members is not None and args.members < 1:
        raise SystemExit("--members must be >= 1")


def episode_amplitudes(args) -> np.ndarray:
    """Return the fixed band setting of an episode, ``(K_BANDS,)``."""
    if args.amplitudes is not None:
        return np.asarray(args.amplitudes, np.float32)
    return np.full(K_BANDS, args.uniform, np.float32)


def check_role_allowed(role: str, allow_eval: bool):
    """Refuse evaluation roles unless explicitly allowed (revision 1 rule)."""
    if role.endswith("_eval") and not allow_eval:
        raise SystemExit(
            f"role '{role}' is an evaluation role: its references are built "
            f"only after revision 1 of Amendment 9 is frozen (pass "
            f"--allow-eval-roles then)")


def check_warming_matches(reference_config: dict, args):
    """Refuse an episode whose warming differs from the warmed reference."""
    for key in ("warming_step_wm2", "warming_ramp_wm2_per_day"):
        if not np.isclose(reference_config[key], getattr(args, key)):
            raise SystemExit(f"{key} {getattr(args, key)} differs from the "
                             f"references' {reference_config[key]}")


def build_model():
    """Return the coupled model pieces both stages need."""
    coupler, coords, _, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    # initialize() BEFORE building the step function: the slab land model
    # loads its climatology inside initialize().
    template = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                     jitted=True)
    shape = coords.horizontal.nodal_shape
    lats = coords.horizontal.latitudes
    ocean = ocean_mask_from_coupler(coupler)
    land = land_mask_from_coupler(coupler, shape)
    return {"coords": coords, "template": template, "step_fn": step_fn,
            "lats": lats, "ocean": ocean, "land": land,
            "patterns": gaussian_band_patterns(lats, ocean)}


def select_branches(ics, branches):
    """Keep the ``(entry, carry)`` pairs on the given weather branches (None: all)."""
    if branches is None:
        return list(ics)
    kept = [(e, c) for e, c in ics if e.get("branch") in branches]
    if not kept:
        raise SystemExit(f"no ICs on branches {branches}")
    return kept


def stage_references(args):
    m = build_model()
    manifest, ics = load_manifest_ics(args.ic_dir, args.split, args.max_ics,
                                      m["template"])
    check_land_mode(manifest.get("land_climatology"),
                    f"the states in {args.ic_dir}")
    ics = select_branches(ics, args.branches)
    role = manifest.get("macro_role") or Path(args.ic_dir).name
    check_role_allowed(role, args.allow_eval_roles)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "grid.npz",
                        ocean_mask=np.asarray(m["ocean"], np.float32),
                        land_mask=np.asarray(m["land"], np.float32),
                        latitudes_rad=np.asarray(m["lats"]),
                        longitudes_rad=np.asarray(
                            m["coords"].horizontal.longitudes))
    seg = make_segment_fn(m["step_fn"], m["patterns"], args.days)
    config = {k: v for k, v in vars(args).items()}
    meta = {"stage": "references", "role": role, "config": config,
            "land_climatology": land_climatology_mode(),
            "fields": list(FIELD_NAMES),
            "units": {"sst": "K", "land_temperature": "K",
                      "precipitation": "mm/day", "evaporation": "mm/day"},
            "note": ("Part 18 steps 10-11: normal climate (no warming, no "
                     "brightening) and uncontrolled warmed run, each the "
                     "mean of the members; member 0 is the IC itself"),
            "git": git_provenance(), "command": " ".join(sys.argv),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "ics": []}
    for entry, carry in ics:
        t0 = time.time()
        arrays, seeds = {}, None
        scenarios = [("normal", make_warming(carry, m["ocean"]))]
        if args.warmed:
            scenarios.append(("warmed", make_warming(
                carry, m["ocean"], args.warming_step_wm2,
                args.warming_ramp_wm2_per_day)))
        for name, warming in scenarios:
            ref = reference_ensemble(carry, seg, args.days, K_BANDS, warming,
                                     args.members, args.member_seed0,
                                     entry["index"], args.member_amp,
                                     args.member_block_days)
            arrays.update({f"{name}_{k}": v for k, v in ref["mean"].items()})
            if ref["member_blocks"] is not None:
                arrays.update({f"{name}_members_{k}": v
                               for k, v in ref["member_blocks"].items()})
            seeds = ref["seeds"]
        fname = f"ic{entry['index']:04d}_references.npz"
        np.savez_compressed(out / fname, **arrays)
        meta["ics"].append({"entry": entry, "file": fname,
                            "member_seeds": seeds,
                            "seconds": round(time.time() - t0, 1)})
        with open(out / "references_manifest.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  IC {entry['index']:>4}: {len(scenarios)} x {args.members} "
              f"members x {args.days} d -> {fname} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(out / "references_manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out / 'references_manifest.json'} ({len(ics)} ICs)")


def load_references(args, days_needed: int):
    """Load one IC's references and check they fit this run."""
    ref_path = Path(args.references)
    with open(ref_path.parent / "references_manifest.json") as f:
        ref_meta = json.load(f)
    check_warming_matches(ref_meta["config"], args)
    check_land_mode(ref_meta.get("land_climatology"),
                    f"the references in {ref_path.parent}")
    refs = dict(np.load(ref_path, allow_pickle=False))
    if refs["normal_sst"].shape[0] < days_needed:
        raise SystemExit(f"references cover {refs['normal_sst'].shape[0]} "
                         f"days, this run needs {days_needed}")
    if "warmed_sst" not in refs:
        raise SystemExit("references have no warmed run (built with "
                         "--no-warmed)")
    return ref_path, ref_meta, refs


def load_ics_at(m, args, ref_path):
    """Load the IC directory; return the IC at ``--ic-position`` and all ICs.

    Checks the land model the states were made with and that the references
    belong to the chosen IC.
    """
    manifest, ics = load_manifest_ics(args.ic_dir, "all", None,
                                      m["template"])
    check_land_mode(manifest.get("land_climatology"),
                    f"the states in {args.ic_dir}")
    entry, carry = ics[args.ic_position]
    expected = f"ic{entry['index']:04d}_references.npz"
    if ref_path.name != expected:
        raise SystemExit(f"IC {entry['index']} needs {expected}, got "
                         f"{ref_path.name}")
    return entry, carry, ics


def load_ic(m, args, ref_path):
    """Load the IC at ``--ic-position`` and check its references match it."""
    entry, carry, _ = load_ics_at(m, args, ref_path)
    return entry, carry


def background_for(entry, ics):
    """The state an ocean-only planner takes its day-0 weather from.

    It is the next weather branch of the same macro state (cyclically): the
    same ocean state and date, but weather the planner has never seen. Only
    its atmosphere and land are used (Experiment 3c).
    """
    same = sorted(((e["branch"], e, c) for e, c in ics
                   if e.get("macro_index") == entry.get("macro_index")),
                  key=lambda t: t[0])
    if len(same) < 2:
        raise SystemExit(f"IC {entry['index']}: ocean-only sensing needs "
                         "another weather branch of the same macro state")
    branches = [b for b, _, _ in same]
    _, e, c = same[(branches.index(entry["branch"]) + 1) % len(same)]
    return e, c


def run_members(m, args, entry, carry, ref_config, make_policy):
    """Run the controlled episode for every member and average the fields.

    Fair scoring: as many members as the references by default, started from
    the same member seeds, so both sides carry the same weather noise.
    ``make_policy(member, member_carry, warming)`` returns each member's
    policy. Returns the mean daily fields, each member's ``(n_segments, K)``
    settings, the member seeds, the policies, and each member's daily ocean
    latitude profile of SST ``(members, n_days, il)`` (small, and enough to
    measure how weather noise falls with more members).
    """
    n_members = (ref_config["members"] if args.members is None
                 else args.members)
    seg = make_segment_fn(m["step_fn"], m["patterns"], args.segment_days)
    runs, schedules, seeds, policies, profiles = [], [], [], [], []
    ocean = np.asarray(m["ocean"], np.float64)
    counts = np.maximum(ocean.sum(axis=0), 1.0)
    for member in range(n_members):
        seed = member_seed(ref_config["member_seed0"], entry["index"], member)
        c = carry if seed is None else perturb_member(
            carry, seed, ref_config["member_amp"])
        warming = make_warming(c, m["ocean"], args.warming_step_wm2,
                               args.warming_ramp_wm2_per_day)
        policy = make_policy(member, c, warming)
        ep = run_episode(c, seg, policy, args.segments, args.segment_days,
                         K_BANDS, warming, strength_vector(args.efficacy))
        runs.append(ep.fields)
        profiles.append(((np.asarray(ep.fields["sst"], np.float64) * ocean)
                         .sum(axis=-2) / counts).astype(np.float32))
        schedules.append(ep.amplitudes)
        seeds.append(seed)
        policies.append(policy)
    mean_fields = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
    return mean_fields, schedules, seeds, policies, np.stack(profiles)


def score_runs(m, args, refs, mean_fields, schedules):
    """Score the member-mean fields against the references.

    Returns the gains, the effort (mean over members), and the objective's
    terms. The bias and variance terms use the window-mean error map. The
    amplitude penalty uses the window-mean settings. The movement penalty is
    the mean over segment changes, so it is zero for a fixed design.
    """
    n_days = args.segments * args.segment_days
    start = args.score_start_day
    end = n_days if args.score_end_day is None else args.score_end_day
    ctrl = time_means(mean_fields, start, end)
    tgt = time_means({k: refs[f"normal_{k}"][:n_days] for k in FIELD_NAMES},
                     start, end)
    warm = time_means({k: refs[f"warmed_{k}"][:n_days] for k in FIELD_NAMES},
                      start, end)
    weights = {k: np.asarray(v) for k, v in
               domain_weights(m["lats"], m["ocean"], m["land"]).items()}
    unit = np.asarray(gaussian_band_patterns(m["lats"],
                                             np.ones_like(m["ocean"])))
    daily = [np.repeat(s, args.segment_days, axis=0)[start:end]
             for s in schedules]
    mean_a = np.mean([d.mean(axis=0) for d in daily], axis=0)
    movement = float(np.mean([args.lam * np.mean(np.sum(np.diff(s, axis=0)
                                                        ** 2, axis=1))
                              if len(s) > 1 else 0.0 for s in schedules]))
    terms = {}
    for name, fields, amps, move in (("controlled", ctrl, mean_a, movement),
                                     ("uncontrolled", warm,
                                      np.zeros_like(mean_a), 0.0)):
        t = objective_terms(jnp.asarray(fields["sst"] - tgt["sst"]),
                            jnp.asarray(weights["ocean"]), jnp.asarray(amps),
                            jnp.asarray(amps), args.alpha, args.beta,
                            args.mu, 0.0)
        t["movement"] = move
        t["total"] = (t["bias_sq"] + t["variance"] + t["amplitude"]
                      + t["movement"])
        # The zonal-mean profile's pattern objective (no penalties): the
        # representation the step-23 pilot chose for scoring.
        zonal_error = zonal_projection(
            np.asarray(fields["sst"] - tgt["sst"], np.float64),
            np.asarray(m["ocean"]) > 0)
        t["zonal_pattern"] = float(pattern_objective(
            zonal_error, np.asarray(weights["ocean"], np.float64),
            args.alpha, args.beta))
        terms[name] = t
    return {"score_window_days": [start, end],
            "gains": restoration_scores(ctrl, warm, tgt, weights),
            "effort": float(np.mean([effort(d, unit, weights["global"])
                                     for d in daily])),
            "objective": terms}


def write_summary(args, summary, mean_fields, schedules,
                  member_profiles=None):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.setdefault("land_climatology", land_climatology_mode())
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_fields:
        extra = ({} if member_profiles is None
                 else {"member_zonal_sst": np.asarray(member_profiles)})
        np.savez_compressed(out.with_suffix(".fields.npz"),
                            amplitudes=np.asarray(schedules), **extra,
                            **mean_fields)
    t = summary["objective"]
    print("gains:", {k: round(v, 3) for k, v in summary["gains"].items()})
    print(f"effort {summary['effort']:.4f} | objective controlled "
          f"{t['controlled']['total']:.4g} vs uncontrolled "
          f"{t['uncontrolled']['total']:.4g} -> {out}")


def stage_episode(args):
    n_days = args.segments * args.segment_days
    ref_path, ref_meta, refs = load_references(args, n_days)
    m = build_model()
    entry, carry = load_ic(m, args, ref_path)
    a = episode_amplitudes(args)
    t0 = time.time()
    mean_fields, schedules, seeds, _, profiles = run_members(
        m, args, entry, carry, ref_meta["config"],
        lambda member, c, warming: constant_policy(a))
    summary = {"stage": "episode", "config": vars(args), "ic": entry,
               "members": len(seeds), "member_seeds": seeds,
               "reference_members": ref_meta["config"]["members"],
               "amplitudes": a.tolist(),
               "true_efficacy": strength_vector(args.efficacy).tolist(),
               **score_runs(m, args, refs, mean_fields, schedules),
               "references": str(ref_path), "git": git_provenance(),
               "command": " ".join(sys.argv),
               "seconds": round(time.time() - t0, 1)}
    write_summary(args, summary, mean_fields, schedules, profiles)


def planner_config(args) -> PlannerConfig:
    """Return the preset with any command-line overrides, validated."""
    overrides = {"alpha": args.alpha, "beta": args.beta, "mu": args.mu,
                 "lam": args.lam}
    for name in ("lookahead_days", "copies", "optimizer", "iterations",
                 "learning_rate", "representation", "sensing"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.window_days is not None:
        overrides["window_days"] = (NO_TRUNCATION_DAYS if args.window_days == 0
                                    else args.window_days)
    cfg = dataclasses.replace(PRESETS[args.preset], **overrides)
    try:
        cfg.validate()
    except ValueError as err:
        raise SystemExit(f"planner: {err}") from err
    return cfg


def stage_plan(args):
    cfg = planner_config(args)
    if cfg.sensing == "ocean" and cfg.lookahead_days != args.segment_days:
        raise SystemExit("ocean-only sensing needs the look-ahead to equal "
                         "the segment (each forecast ends where the next "
                         "re-plan starts)")
    n_days = args.segments * args.segment_days
    last_lookahead_end = (args.segments - 1) * args.segment_days \
        + cfg.lookahead_days
    ref_path, ref_meta, refs = load_references(
        args, max(n_days, last_lookahead_end))
    m = build_model()
    entry, carry, ics = load_ics_at(m, args, ref_path)
    background_entry, background = ((None, None) if cfg.sensing == "exact"
                                    else background_for(entry, ics))
    del ics
    ocean_w = domain_weights(m["lats"], m["ocean"], m["land"])["ocean"]
    learning = args.learn_strength != "off"
    belief = strength_vector(args.efficacy if args.planner_efficacy is None
                             else args.planner_efficacy)

    def make_policy(member, c, warming):
        # A perfect-model planner (Experiment 3a): it knows the warming. Its
        # belief about the spraying strength is --planner-efficacy (the
        # true one by default: the oracle), or, with --learn-strength,
        # nominal at first and then learned from its forecast misses.
        common = (m["step_fn"], m["patterns"], ocean_w, refs["normal_sst"],
                  c["ocn"]["forcing"].q_flux, warming)
        # With ocean-only sensing it sees the true ocean only; its weather
        # is another branch's on day 0 and its own forecast's after that.
        if learning:
            return LearningPlanner(*common, estimator_config(args), cfg,
                                   seed_offset=100 * entry["index"] + member,
                                   background=background)
        return Planner(*common, belief, cfg,
                       seed_offset=100 * entry["index"] + member,
                       background=background)

    t0 = time.time()
    mean_fields, schedules, seeds, planners, profiles = run_members(
        m, args, entry, carry, ref_meta["config"], make_policy)
    learner = ({"config": dataclasses.asdict(estimator_config(args)),
                "logs": [p.estimator.log for p in planners]}
               if learning else None)
    summary = {"stage": "plan", "config": vars(args),
               "planner": dataclasses.asdict(cfg),
               "planner_efficacy": None if learning else belief.tolist(),
               "true_efficacy": strength_vector(args.efficacy).tolist(),
               "learner": learner, "ic": entry,
               "background_ic": (None if background_entry is None
                                 else background_entry["index"]),
               "members": len(seeds), "member_seeds": seeds,
               "reference_members": ref_meta["config"]["members"],
               "schedules": [s.tolist() for s in schedules],
               **score_runs(m, args, refs, mean_fields, schedules),
               "planner_logs": [p.log for p in planners],
               "references": str(ref_path), "git": git_provenance(),
               "command": " ".join(sys.argv),
               "seconds": round(time.time() - t0, 1)}
    write_summary(args, summary, mean_fields, schedules, profiles)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print(f"JAX devices: {jax.devices()}")
    {"references": stage_references, "episode": stage_episode,
     "plan": stage_plan}[args.stage](args)


if __name__ == "__main__":
    main()
