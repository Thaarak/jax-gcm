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
evaluation role draws from macro states no other role touches
(``validate_plan`` enforces both). Training and Experiment 1 use the even
macro states and evaluation the odd ones (Amendment 9 revision 0.4).

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

# Registered plan (Amendment 9 revision 0.4). Even macro states serve
# Experiment 1 and all training/validation; odd ones are reserved for
# evaluation. Interleaving spreads both sides over the whole control run, so a
# slow wander of the ocean's mean state cannot separate training from
# evaluation, and it keeps the states within each side four years apart.
TRAIN_MACROS = list(range(0, 16, 2))
EVAL_MACROS = list(range(1, 16, 2))
DEFAULT_PLAN = {
    "exp1": [
        {"macro": TRAIN_MACROS, "branches": [0], "split": "heldout",
         "horizon": 120},
    ],
    "exp2_train": [
        {"macro": TRAIN_MACROS, "branches": [1], "split": "train",
         "horizon": 60},
    ],
    "exp2_eval": [
        {"macro": EVAL_MACROS, "branches": [0, 1], "split": "heldout",
         "horizon": 60},
    ],
    "exp3_train": [
        {"macro": TRAIN_MACROS, "branches": [2, 3], "split": "train",
         "horizon": 60},
        {"macro": TRAIN_MACROS, "branches": [4], "split": "heldout",
         "horizon": 60},
    ],
    "exp3_eval": [
        {"macro": EVAL_MACROS, "branches": [2, 3, 4], "split": "heldout",
         "horizon": 60},
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


def is_eval_role(role):
    """Tell evaluation roles (named ``*_eval``) from roles that train or tune."""
    return role.endswith("_eval")


def split_macros(flat):
    """Sorted macro states used by non-evaluation and by evaluation roles."""
    train = sorted({m for role, m, *_ in flat if not is_eval_role(role)})
    evaluation = sorted({m for role, m, *_ in flat if is_eval_role(role)})
    return train, evaluation


def validate_plan(plan, num_macro):
    """Reject plans that break the registered rules.

    A plan may not reuse a (macro, branch) pair, exceed ``num_macro``, or let
    an evaluation role share a macro state with any other role.

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
    train, evaluation = split_macros(flat)
    shared = sorted(set(train) & set(evaluation))
    if shared:
        raise SystemExit(f"macro states {shared} are used by both an "
                         f"evaluation role and another role")
    return flat


def branch_seed(seed0, macro, branch):
    return seed0 + 100 * macro + branch


def ic_index(macro, branch):
    """Globally unique IC index (harnesses derive member seeds from it)."""
    return 100 * macro + branch


def _similarity(a, b, w):
    """Area-weighted anomaly pattern correlation and RMS difference."""
    wa, wb = a * np.sqrt(w), b * np.sqrt(w)
    corr = float(np.sum(wa * wb) / np.sqrt(np.sum(wa ** 2) * np.sum(wb ** 2)))
    rms = float(np.sqrt(np.sum(w * (a - b) ** 2)))
    return {"pattern_corr": corr, "rms_diff_K": rms}


def macro_diagnostics(ssts, weights, train_macros=None, eval_macros=None,
                      spacing_days=None):
    """How different the macro states are, and whether the split is balanced.

    Reported, not gated (Amendment 9 and its revision 0.4):

    * ``ocean_mean_sst_K``: ocean-mean SST of each macro state.
    * ``neighbour_similarity``: states one spacing apart. With the interleaved
      plan these pairs straddle training and evaluation.
    * ``two_apart_similarity``: states two spacings apart, i.e. neighbours
      within the same side of the interleaved plan.
    * ``trend_K_per_decade`` (with ``spacing_days``): least-squares trend of
      the ocean-mean SST across the macro states, with its standard error.
    * ``split_balance`` (with both macro lists): mean ocean SST of the
      training-side and evaluation-side states and their difference.
    """
    ssts = np.asarray(ssts, float)
    w = np.asarray(weights, float)
    means = np.array([float(np.sum(s * w)) for s in ssts])
    anomalies = ssts - ssts.mean(axis=0)
    out = {
        "ocean_mean_sst_K": means.tolist(),
        "neighbour_similarity": [_similarity(a, b, w) for a, b in
                                 zip(anomalies[:-1], anomalies[1:])],
        "two_apart_similarity": [_similarity(a, b, w) for a, b in
                                 zip(anomalies[:-2], anomalies[2:])],
    }
    if spacing_days is not None and means.size >= 3:
        years = np.arange(means.size) * spacing_days / 365.2425
        design = np.stack([np.ones_like(years), years], axis=1)
        coef, *_ = np.linalg.lstsq(design, means, rcond=None)
        resid = means - design @ coef
        se = np.sqrt(np.sum(resid ** 2) / (means.size - 2)
                     / np.sum((years - years.mean()) ** 2))
        out["trend_K_per_decade"] = {"slope": float(10.0 * coef[1]),
                                     "se": float(10.0 * se)}
    if train_macros and eval_macros:
        train_mean = float(np.mean(means[list(train_macros)]))
        eval_mean = float(np.mean(means[list(eval_macros)]))
        out["split_balance"] = {"train_macros": list(train_macros),
                                "eval_macros": list(eval_macros),
                                "train_mean_K": train_mean,
                                "eval_mean_K": eval_mean,
                                "eval_minus_train_K": eval_mean - train_mean}
    return out


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

    train_side, eval_side = split_macros(flat)
    diagnostics = macro_diagnostics(macro_ssts, ocean_w, train_side,
                                    eval_side, args.spacing_days)
    for m, sst_mean in enumerate(diagnostics["ocean_mean_sst_K"]):
        side = ("eval" if m in eval_side else
                "train" if m in train_side else "unused")
        print(f"  macro {m:02d} ({side}): ocean-mean SST {sst_mean:.3f} K")
    for m, pair in enumerate(diagnostics["neighbour_similarity"]):
        print(f"  macro {m:02d}->{m + 1:02d}: anomaly pattern corr "
              f"{pair['pattern_corr']:+.2f}, rms diff "
              f"{pair['rms_diff_K']:.3f} K")
    if "trend_K_per_decade" in diagnostics:
        trend = diagnostics["trend_K_per_decade"]
        print(f"  trend across macro states: {trend['slope']:+.3f} "
              f"+/- {trend['se']:.3f} K per decade")
    if "split_balance" in diagnostics:
        bal = diagnostics["split_balance"]
        print(f"  split balance: train {bal['train_mean_K']:.3f} K, eval "
              f"{bal['eval_mean_K']:.3f} K (eval - train "
              f"{bal['eval_minus_train_K']:+.3f} K)")
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
            "preregistration": "PREREGISTRATION.md Amendment 9 "
                               "(plan: revision 0.4)",
            "ics": ics,
        }
        with open(root / role / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote {root / role / 'manifest.json'} ({len(ics)} ICs)")
    print(f"DONE in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
