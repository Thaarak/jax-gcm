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

``episode``
    Run one controlled episode from one IC with a fixed band setting and the
    warming, and score it against that IC's references over
    ``[--score-start-day, --score-end-day)``. The controlled side is averaged
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
from jcm.mcb.scores import (
    PATTERN_ALPHA,
    PATTERN_BETA,
    effort,
    objective_terms,
    restoration_scores,
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def warming_args(sp):
        sp.add_argument("--warming-step-wm2", type=float, default=4.0,
                        help="Steady heat into the ocean, W m-2 "
                             "(placeholder until the pilot).")
        sp.add_argument("--warming-ramp-wm2-per-day", type=float,
                        default=0.0,
                        help="Growth of the heat input, W m-2 per day.")

    r = sub.add_parser("references", help="normal-climate and warmed "
                                          "reference ensembles")
    r.add_argument("--ic-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--split", default="all",
                   choices=["all", "train", "heldout"])
    r.add_argument("--max-ics", type=int, default=None)
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
    warming_args(r)

    e = sub.add_parser("episode", help="one scored controlled episode")
    e.add_argument("--ic-dir", required=True)
    e.add_argument("--ic-position", type=int, default=0,
                   help="Position of the IC in the manifest's list.")
    e.add_argument("--references", required=True,
                   help="That IC's ic<index>_references.npz.")
    amp = e.add_mutually_exclusive_group(required=True)
    amp.add_argument("--amplitudes", type=float, nargs=K_BANDS)
    amp.add_argument("--uniform", type=float)
    e.add_argument("--segments", type=int, required=True)
    e.add_argument("--segment-days", type=int, default=14)
    e.add_argument("--efficacy", type=float, default=1.0,
                   help="True strength of the spraying (1 = nominal).")
    e.add_argument("--members", type=int, default=None,
                   help="Members to run and average before scoring "
                        "(default: as many as the references, from the same "
                        "member seeds, so both sides carry the same weather "
                        "noise).")
    e.add_argument("--score-start-day", type=int, default=0)
    e.add_argument("--score-end-day", type=int, default=None,
                   help="Default: the end of the episode.")
    e.add_argument("--alpha", type=float, default=PATTERN_ALPHA)
    e.add_argument("--beta", type=float, default=PATTERN_BETA)
    e.add_argument("--mu", type=float, default=0.0)
    e.add_argument("--lam", type=float, default=0.0)
    e.add_argument("--output", required=True, help="JSON summary path.")
    e.add_argument("--save-fields", action="store_true",
                   help="Also write the daily fields next to the JSON.")
    warming_args(e)
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
        if args.segments < 1 or args.segment_days < 1:
            raise SystemExit("--segments and --segment-days must be >= 1")
        n_days = args.segments * args.segment_days
        end = n_days if args.score_end_day is None else args.score_end_day
        if not 0 <= args.score_start_day < end <= n_days:
            raise SystemExit("need 0 <= --score-start-day < "
                             "--score-end-day <= segments * segment-days")
        a = episode_amplitudes(args)
        if np.any(a < 0.0) or np.any(a > BRIGHTENING_CAP):
            raise SystemExit(f"band amplitudes must lie in [0, "
                             f"{BRIGHTENING_CAP}]")
        if args.efficacy < 0.0:
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


def stage_references(args):
    m = build_model()
    manifest, ics = load_manifest_ics(args.ic_dir, args.split, args.max_ics,
                                      m["template"])
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


def stage_episode(args):
    ref_path = Path(args.references)
    with open(ref_path.parent / "references_manifest.json") as f:
        ref_meta = json.load(f)
    check_warming_matches(ref_meta["config"], args)
    refs = dict(np.load(ref_path, allow_pickle=False))
    n_days = args.segments * args.segment_days
    if refs["normal_sst"].shape[0] < n_days:
        raise SystemExit(f"references cover {refs['normal_sst'].shape[0]} "
                         f"days, the episode needs {n_days}")
    if "warmed_sst" not in refs:
        raise SystemExit("references have no warmed run (built with "
                         "--no-warmed)")
    m = build_model()
    _, ics = load_manifest_ics(args.ic_dir, "all", None, m["template"])
    entry, carry = ics[args.ic_position]
    expected = f"ic{entry['index']:04d}_references.npz"
    if ref_path.name != expected:
        raise SystemExit(f"IC {entry['index']} needs {expected}, got "
                         f"{ref_path.name}")
    # Fair scoring: the controlled side is averaged over as many members as
    # the references, from the same seeds, so both carry the same noise.
    rc = ref_meta["config"]
    n_members = rc["members"] if args.members is None else args.members
    a = episode_amplitudes(args)
    seg = make_segment_fn(m["step_fn"], m["patterns"], args.segment_days)
    start = args.score_start_day
    end = n_days if args.score_end_day is None else args.score_end_day
    t0 = time.time()
    seeds, runs = [], []
    for member in range(n_members):
        seed = member_seed(rc["member_seed0"], entry["index"], member)
        c = carry if seed is None else perturb_member(carry, seed,
                                                      rc["member_amp"])
        warming = make_warming(c, m["ocean"], args.warming_step_wm2,
                               args.warming_ramp_wm2_per_day)
        ep = run_episode(c, seg, constant_policy(a), args.segments,
                         args.segment_days, K_BANDS, warming, args.efficacy)
        runs.append(ep.fields)
        seeds.append(seed)
    mean_fields = {k: np.mean([r[k] for r in runs], axis=0)
                   for k in runs[0]}
    ctrl = time_means(mean_fields, start, end)
    tgt = time_means({k: refs[f"normal_{k}"][:n_days] for k in FIELD_NAMES},
                     start, end)
    warm = time_means({k: refs[f"warmed_{k}"][:n_days] for k in FIELD_NAMES},
                      start, end)
    weights = {k: np.asarray(v) for k, v in
               domain_weights(m["lats"], m["ocean"], m["land"]).items()}
    unit = np.asarray(gaussian_band_patterns(m["lats"],
                                             np.ones_like(m["ocean"])))
    daily_a = np.repeat(ep.amplitudes, args.segment_days, axis=0)[start:end]
    # A fixed design never moves, so the movement penalty is zero; the
    # uncontrolled run applies no brightening at all.
    terms = {
        name: objective_terms(jnp.asarray(fields["sst"] - tgt["sst"]),
                              jnp.asarray(weights["ocean"]),
                              jnp.asarray(amps), jnp.asarray(amps),
                              args.alpha, args.beta, args.mu, args.lam)
        for name, fields, amps in (("controlled", ctrl, a),
                                   ("uncontrolled", warm, np.zeros_like(a)))}
    summary = {
        "stage": "episode", "config": vars(args), "ic": entry,
        "members": n_members, "member_seeds": seeds,
        "reference_members": rc["members"], "amplitudes": a.tolist(),
        "score_window_days": [start, end],
        "gains": restoration_scores(ctrl, warm, tgt, weights),
        "effort": effort(daily_a, unit, weights["global"]),
        "objective": terms,
        "references": str(ref_path), "git": git_provenance(),
        "command": " ".join(sys.argv), "seconds": round(time.time() - t0, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_fields:
        np.savez_compressed(out.with_suffix(".fields.npz"),
                            amplitudes=ep.amplitudes, **mean_fields)
    print("gains:", {k: round(v, 3) for k, v in summary["gains"].items()})
    print(f"effort {summary['effort']:.4f} | objective controlled "
          f"{terms['controlled']['total']:.4g} vs uncontrolled "
          f"{terms['uncontrolled']['total']:.4g} -> {out}")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    print(f"JAX devices: {jax.devices()}")
    if args.stage == "references":
        stage_references(args)
    else:
        stage_episode(args)


if __name__ == "__main__":
    main()
