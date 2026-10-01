#!/usr/bin/env python
"""Macro x micro starting states for the long-horizon gradient study.

Step 0 of PREREGISTRATION.md Amendment 9. Every earlier campaign branched its
initial conditions (ICs) from ONE equilibrated ocean state on one calendar
date, so its statistics generalize across weather, not across ocean states
(the "effective climate n = 1" limitation). The large CESM ensembles solve
the same problem with two levels: "macro" members started from different
ocean states of a control run, and "micro" members perturbed at round-off
level. Micro members are drawn at evaluation time by the harnesses
(``perturb_sst`` with a 0.001 K amplitude). This script builds the macro
level:

1. Continue the equilibrated control run and save the coupled state every
   ``--spacing-days`` (default 730 = two years, so every macro state sits in
   the same season; the slab ocean's anomaly e-folding time is ~60-570 days).
2. Branch each macro state into independent weather trajectories exactly as
   ``run_generate_ics_independent.py`` does: seeded SST perturbation,
   ``--decorr-days`` independent spin, paired no-MCB baseline.
3. Write one IC directory per registered role (``exp1``, ``exp2_train``,
   ``exp2_eval``, ``exp3_train``, ``exp3_eval``), each with a
   ``manifest.json`` that ``load_ics`` / ``run_confirmatory_eval.py`` /
   ``run_gradient_fidelity.py`` read unchanged.

The registered plan never reuses a (macro state, branch) pair, and every
evaluation role draws from macro states no training role touches.

Example (GPU; the Q-flux base carry of Amendment 9 revision 0.2):
    python run_generate_macro_ics.py \
        --base-carry mcb_experiments_gpu/equilibrated_qflux/base_carry.pkl \
        --require-qflux --output-root mcb_experiments_gpu/ics_macro

CPU smoke (plumbing only, cold start):
    python run_generate_macro_ics.py --base-carry cold --num-macro 2 \
        --spacing-days 2 --decorr-days 1 --plan-json smoke_plan.json \
        --output-root /tmp/ics_macro_smoke
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb import compute_baseline_trajectory, load_carry, save_carry
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    run_interval_final_carry,
)
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.qflux import qflux_magnitude
from jcm.mcb.state_features import compute_area_weights
from run_coupled_training import (
    TERRAIN_NC,
    coupler_workflow,
    setup_coupled_model,
)
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE

# Registered plan (Amendment 9). Macro states 0-7 serve Experiment 1 and all
# training/validation; macro states 8-15 are reserved for evaluation.
DEFAULT_PLAN = {
    "exp1": [
        {"macro": list(range(0, 8)), "branches": [0], "split": "heldout",
         "horizon": 120},
    ],
    "exp2_train": [
        {"macro": list(range(0, 8)), "branches": [1], "split": "train",
         "horizon": 60},
    ],
    "exp2_eval": [
        {"macro": list(range(8, 16)), "branches": [0, 1], "split": "heldout",
         "horizon": 60},
    ],
    "exp3_train": [
        {"macro": list(range(0, 8)), "branches": [2, 3], "split": "train",
         "horizon": 60},
        {"macro": list(range(0, 8)), "branches": [4], "split": "heldout",
         "horizon": 60},
    ],
    "exp3_eval": [
        {"macro": list(range(8, 16)), "branches": [2, 3, 4],
         "split": "heldout", "horizon": 60},
    ],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-carry", required=True,
                   help="Equilibrated carry (run_equilibrate.py), or 'cold' "
                        "for a plumbing-only smoke test from initialize().")
    p.add_argument("--num-macro", type=int, default=16)
    p.add_argument("--spacing-days", type=int, default=730)
    p.add_argument("--decorr-days", type=int, default=30)
    p.add_argument("--perturb-amp", type=float, default=0.05,
                   help="SST perturbation (K) seeding each weather branch.")
    p.add_argument("--seed0", type=int, default=12000,
                   help="Branch seed = seed0 + 100 * macro + branch.")
    p.add_argument("--plan-json", default=None,
                   help="Override the registered plan (smoke tests only).")
    p.add_argument("--output-root", required=True)
    p.add_argument("--no-save-macro-bases", dest="save_macro_bases",
                   action="store_false")
    p.add_argument("--require-qflux", action="store_true",
                   help="Refuse a base carry without a Q-flux (Amendment 9 "
                        "revision 0.2: Step 0 branches from the Q-flux "
                        "climate).")
    return p.parse_args(argv)


def validate_plan(plan, num_macro):
    """Reject plans that reuse a (macro, branch) pair or exceed num_macro.

    Returns the flat list of (role, macro, branch, split, horizon).
    """
    seen = {}
    flat = []
    for role, groups in plan.items():
        for group in groups:
            for m in group["macro"]:
                if not 0 <= m < num_macro:
                    raise SystemExit(f"{role}: macro {m} outside "
                                     f"[0, {num_macro})")
                for b in group["branches"]:
                    if (m, b) in seen:
                        raise SystemExit(
                            f"(macro {m}, branch {b}) used by both "
                            f"{seen[(m, b)]} and {role}")
                    seen[(m, b)] = role
                    flat.append((role, m, b, group["split"],
                                 int(group["horizon"])))
    return flat


def branch_seed(seed0, macro, branch):
    return seed0 + 100 * macro + branch


def ic_index(macro, branch):
    """Globally unique IC index (harnesses derive member seeds from it)."""
    return 100 * macro + branch


def macro_diagnostics(ssts, weights):
    """Ocean-mean SST per macro state and similarity between neighbours."""
    ssts = np.asarray(ssts, float)
    w = np.asarray(weights, float)
    means = [float(np.sum(s * w)) for s in ssts]
    anomalies = ssts - ssts.mean(axis=0)
    pairs = []
    for a, b in zip(anomalies[:-1], anomalies[1:]):
        wa, wb = a * np.sqrt(w), b * np.sqrt(w)
        corr = float(np.sum(wa * wb) / np.sqrt(np.sum(wa ** 2) *
                                               np.sum(wb ** 2)))
        rms = float(np.sqrt(np.sum(w * (a - b) ** 2)))
        pairs.append({"pattern_corr": corr, "rms_diff_K": rms})
    return {"ocean_mean_sst_K": means, "neighbour_similarity": pairs}


def main(argv=None):
    args = parse_args(argv)
    plan = DEFAULT_PLAN
    if args.plan_json:
        with open(args.plan_json) as f:
            plan = json.load(f)
    flat = validate_plan(plan, args.num_macro)
    t_start = time.time()

    coupler, coords, _, _ = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    wf = coupler_workflow(coupler)
    # initialize() BEFORE building the step function: the slab land model
    # loads its climatology (stl_clim) inside initialize().
    template = coupler.initialize()
    step_fn = create_coupled_step_fn(coupler, wf, jitted=True)
    base = (template if args.base_carry == "cold"
            else load_carry(args.base_carry, template))
    q_max = qflux_magnitude(base)
    print(f"base carry {args.base_carry}: max |Q-flux| {q_max:.1f} W m-2")
    if args.require_qflux and q_max == 0.0:
        raise SystemExit("--require-qflux: the base carry has no Q-flux "
                         "(settle it with run_qflux_base_climate.py)")
    ocean_w = compute_area_weights(coords) * ocean_mask_from_coupler(coupler)
    ocean_w = ocean_w / jnp.sum(ocean_w)

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"MACRO ICs: {args.num_macro} macro states every "
          f"{args.spacing_days} d | {len(flat)} branches over "
          f"{len(plan)} roles")
    print("=" * 72)

    # 1) Control continuation: macro state m = base advanced m * spacing.
    macro_states, macro_ssts, macro_entries = [], [], []
    carry = base
    for m in range(args.num_macro):
        if m > 0:
            t0 = time.time()
            carry = run_interval_final_carry(carry, step_fn,
                                             args.spacing_days)
            jax.block_until_ready(carry["ocn"]["state"].sea_surface_temperature)
            print(f"  macro {m:02d}: +{args.spacing_days} d "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        macro_states.append(carry)
        macro_ssts.append(np.asarray(
            carry["ocn"]["state"].sea_surface_temperature))
        entry = {"macro_index": m,
                 "day_offset": m * args.spacing_days,
                 "ocn_sim_time": float(carry["ocn"]["state"].sim_time)}
        if args.save_macro_bases:
            base_dir = root / "macro_bases"
            base_dir.mkdir(exist_ok=True)
            fname = f"macro_{m:02d}_carry.pkl"
            save_carry(carry, str(base_dir / fname))
            entry["carry_file"] = fname
        macro_entries.append(entry)

    diagnostics = macro_diagnostics(macro_ssts, ocean_w)
    for m, sst_mean in enumerate(diagnostics["ocean_mean_sst_K"]):
        print(f"  macro {m:02d}: ocean-mean SST {sst_mean:.3f} K")
    for m, pair in enumerate(diagnostics["neighbour_similarity"]):
        print(f"  macro {m:02d}->{m + 1:02d}: anomaly pattern corr "
              f"{pair['pattern_corr']:+.2f}, rms diff "
              f"{pair['rms_diff_K']:.3f} K")
    with open(root / "macro_bases_manifest.json", "w") as f:
        json.dump({"base_carry": args.base_carry,
                   "base_qflux_max_abs_wm2": q_max,
                   "spacing_days": args.spacing_days,
                   "num_macro": args.num_macro,
                   "macro_states": macro_entries,
                   "diagnostics": diagnostics}, f, indent=2)

    # 2) Weather branches per role.
    roles = {}
    for role, m, b, split, horizon in flat:
        role_dir = root / role
        role_dir.mkdir(exist_ok=True)
        seed = branch_seed(args.seed0, m, b)
        t0 = time.time()
        ic_carry = perturb_sst(macro_states[m], seed, args.perturb_amp)
        ic_carry = run_interval_final_carry(ic_carry, step_fn,
                                            args.decorr_days)
        stem = f"ic_m{m:02d}_b{b}_{split}_seed{seed}"
        save_carry(ic_carry, str(role_dir / f"{stem}_carry.pkl"))
        baseline = compute_baseline_trajectory(
            ic_carry, step_fn, num_steps=horizon, coords=coords)
        with open(role_dir / f"{stem}_baseline{horizon}d.pkl", "wb") as f:
            pickle.dump(jax.device_get({
                "sst": baseline.sst,
                "surface_temperature": baseline.surface_temperature,
                "precipitation": baseline.precipitation,
                "heat_flux": baseline.heat_flux,
            }), f)
        roles.setdefault(role, {"horizon": horizon, "ics": []})
        roles[role]["horizon"] = min(roles[role]["horizon"], horizon)
        roles[role]["ics"].append({
            "index": ic_index(m, b), "split": split, "seed": seed,
            "macro_index": m, "branch": b,
            "spinup_days": args.decorr_days,
            "carry_file": f"{stem}_carry.pkl",
            "baseline_file": f"{stem}_baseline{horizon}d.pkl",
        })
        print(f"  {role:>11}: macro {m:02d} branch {b} ({split}, seed "
              f"{seed}) [{time.time() - t0:.0f}s]", flush=True)

    for role, info in roles.items():
        ics = info["ics"]
        manifest = {
            "start_date": START_DATE,
            "coupling_timestep_days": 1,
            "horizon": info["horizon"],
            "num_train": sum(e["split"] == "train" for e in ics),
            "num_heldout": sum(e["split"] != "train" for e in ics),
            "decorr_days": args.decorr_days,
            "perturb_amp": args.perturb_amp,
            "independent_trajectories": True,
            "realistic_terrain": True,
            "terrain_source": TERRAIN_NC,
            "macro_role": role,
            "macro_spacing_days": args.spacing_days,
            "base_carry": args.base_carry,
            "seed0": args.seed0,
            "preregistration": "PREREGISTRATION.md Amendment 9",
            "ics": ics,
        }
        with open(root / role / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {root / role / 'manifest.json'} ({len(ics)} ICs)")
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
