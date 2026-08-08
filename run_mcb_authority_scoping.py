#!/usr/bin/env python
"""MCB authority scoping: maximum achievable cooling at the ENSO horizon.

Second calibration run for the ENSO experiment (companion to
run_enso_scoping.py; see the PREREGISTRATION.md pre-amendment note).
The ENSO design requires the actuator to deliver the -0.1 K target AND
cancel an ENSO GMST perturbation of up to ~+0.11 K (180 d) — roughly
double the Tier-1 static effect. This run measures whether the 0.09
albedo cap leaves that much authority:

  uniform-cap  : 0.09 everywhere on ocean — the actuator's hard ceiling
  uniform-half : 0.045 everywhere — dose linearity check
  static       : the Tier-1 optimized pattern (its 180-d effect also
                 calibrates the task: what an ENSO-blind deployment
                 delivers at the longer horizon)

Each arm runs against the member's paired no-MCB control. Exploratory
calibration: no gates, no confirmatory reuse.

Example (diya):
    python run_mcb_authority_scoping.py \
        --base-carry mcb_experiments_gpu/equilibrated/base_carry.pkl \
        --stage1 mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl \
        --days 180 --members 2 \
        --output mcb_experiments_gpu/mcb_authority_scoping.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from jax import lax

from jcm.mcb import load_carry
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.state_features import compute_area_weights

from run_coupled_training import coupler_workflow, setup_coupled_model
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-carry", required=True,
                   help="Equilibrated carry, or 'template' for a smoke test.")
    p.add_argument("--stage1", required=True,
                   help="Tier-1 optimized pattern pickle ('none' to skip).")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--tail-days", type=int, default=60)
    p.add_argument("--members", type=int, default=2)
    p.add_argument("--member-perturb-amp", type=float, default=0.001)
    p.add_argument("--member-seed0", type=int, default=89000)
    p.add_argument("--cap", type=float, default=0.09)
    p.add_argument("--output", required=True)
    return p.parse_args()


def run_with_constant_mcb(carry, field, step_fn, days, gmst_w):
    """Run ``days`` coupling steps with a fixed MCB field; daily GMST out."""
    carry = jax.tree_util.tree_map(lambda x: x, carry)
    carry["atm"]["derived"]["mcb_perturbation"] = field

    def body(c, i):
        nc, _ = step_fn(c, i)
        sst = nc["ocn"]["state"].sea_surface_temperature
        return nc, jnp.sum(sst * gmst_w)

    _, gmst = lax.scan(body, carry, jnp.arange(days))
    return np.asarray(gmst)


def main():
    args = parse_args()
    print("=" * 72)
    print("MCB AUTHORITY SCOPING (max cooling at the ENSO horizon)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")
    print(f"days {args.days} | tail {args.tail_days} | members "
          f"{args.members} | cap {args.cap}")

    start_datetime = jdt.to_datetime(START_DATE)
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, jdt.to_timedelta(1, "day"), realistic_terrain=True)
    template = coupler.initialize()
    if args.base_carry == "template":
        print("WARNING: template carry (unequilibrated) — smoke test only")
        base = template
    else:
        base = load_carry(args.base_carry, template)
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)

    area = compute_area_weights(coords)
    gmst_w = area * ocean_mask
    gmst_w = gmst_w / jnp.sum(gmst_w)
    shape = coords.horizontal.nodal_shape

    fields = {
        "uniform_cap": jnp.full(shape, args.cap) * ocean_mask,
        "uniform_half": jnp.full(shape, args.cap / 2.0) * ocean_mask,
    }
    if args.stage1 != "none":
        with open(args.stage1, "rb") as f:
            stage1 = pickle.load(f)
        fields["static"] = jnp.asarray(stage1["best_pattern"]) * ocean_mask

    zero = jnp.zeros(shape)
    tail = args.tail_days
    results = {"config": vars(args), "members": []}
    out_path = Path(args.output)
    t_start = time.time()

    for m in range(args.members):
        carry = base if m == 0 else perturb_sst(
            base, args.member_seed0 + m, args.member_perturb_amp)
        t0 = time.time()
        g_ctrl = run_with_constant_mcb(carry, zero, step_fn, args.days,
                                       gmst_w)
        member = {"index": m, "control_gmst": g_ctrl}
        print(f"member {m}: control done in {time.time() - t0:.0f}s",
              flush=True)
        for name, field in fields.items():
            t0 = time.time()
            g = run_with_constant_mcb(carry, field, step_fn, args.days,
                                      gmst_w)
            d = g - g_ctrl
            member[name] = {
                "gmst": g,
                "dgmst_tail": float(d[-tail:].mean()),
                "dgmst_final": float(d[-1]),
                "mean_forcing": float(jnp.sum(field * gmst_w)),
            }
            print(f"member {m}: {name:>12} done in {time.time() - t0:.0f}s | "
                  f"dGMST tail{tail} {member[name]['dgmst_tail']:+.4f} K "
                  f"(final {member[name]['dgmst_final']:+.4f})", flush=True)
        results["members"].append(member)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    print("\n" + "=" * 72)
    print(f"SUMMARY ({args.members} members, tail = final {tail} days)")
    print("=" * 72)
    summary = {}
    for name in fields:
        dg = np.array([mm[name]["dgmst_tail"]
                       for mm in results["members"]])
        summary[name] = {"dgmst_tail_mean": float(dg.mean()),
                         "dgmst_tail_sd": float(dg.std(ddof=1))
                         if len(dg) > 1 else float("nan")}
        print(f"{name:>12}: dGMST tail-mean {dg.mean():+.4f} K "
              f"(sd {summary[name]['dgmst_tail_sd']:.4f})")
    if {"uniform_cap", "uniform_half"} <= summary.keys():
        lin = (summary["uniform_cap"]["dgmst_tail_mean"]
               / max(2 * summary["uniform_half"]["dgmst_tail_mean"], -1e9))
        print(f"linearity (cap / 2*half): {lin:.2f} "
              f"(1.0 = perfectly linear dose response)")
        summary["linearity_cap_over_2half"] = float(lin)
    results["summary"] = summary
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDONE in {(time.time() - t_start) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
