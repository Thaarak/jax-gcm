#!/usr/bin/env python
"""Stage 5 IC generation: realistic-terrain spin-up snapshots + baselines.

Identical in structure to run_stage4_generate_ics.py, but runs the no-MCB
coupled model over REALISTIC EARTH TERRAIN (T30 climatology orography +
land-sea mask) instead of the aquaplanet. Land cells are pinned to a fixed
temperature by the slab ocean mask, so only ocean SST evolves.

Saves full coupled carries at spin-up days {0, k, 2k, ...} (k =
--spinup-interval). The first --num-train snapshots are training ICs; the next
--num-heldout are held out. Each IC gets its own paired --horizon-day no-MCB
baseline trajectory computed from that exact carry (paired anomalies cancel
natural drift exactly).

Stage 4 aquaplanet carries are NOT reusable here: they embed a terrain-free
atmosphere state. The manifest records terrain_source so the training/eval
loaders can assert the ICs were generated under realistic terrain.

Usage:
    python run_stage5_generate_ics.py --realistic-terrain \
        --output-dir mcb_experiments/stage5/ics
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.mcb import compute_baseline_trajectory, save_carry
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    run_interval_final_carry,
)

from run_coupled_training import setup_coupled_model, TERRAIN_NC

WORKFLOW = ["coupling", "atm", "ocn"]
START_DATE = "2000-01-01"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 IC generation")
    parser.add_argument("--realistic-terrain", action="store_true", default=True,
                        help="Use T30 realistic terrain (default; Stage 5). "
                             "Pass --no-realistic-terrain to fall back to "
                             "the aquaplanet.")
    parser.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                        action="store_false",
                        help="Fall back to the aquaplanet (Stage 4 behavior).")
    parser.add_argument("--spinup-interval", type=int, default=45,
                        help="Days of spin-up between consecutive IC snapshots")
    parser.add_argument("--num-train", type=int, default=4,
                        help="Number of training ICs")
    parser.add_argument("--num-heldout", type=int, default=2,
                        help="Number of held-out ICs")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Paired baseline trajectory length (days)")
    parser.add_argument("--output-dir", type=str,
                        default="mcb_experiments/stage5/ics")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_ics = args.num_train + args.num_heldout
    terrain_source = TERRAIN_NC if args.realistic_terrain else "aquaplanet"

    print("=" * 70)
    print("STAGE 5 IC GENERATION (realistic terrain)")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"  Terrain: {terrain_source}")
    print(f"  Spin-up interval: {args.spinup_interval} days")
    print(f"  ICs: {args.num_train} train + {args.num_heldout} held-out")
    print(f"  Baseline horizon: {args.horizon} days")

    start_datetime = jdt.to_datetime(START_DATE)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep,
        realistic_terrain=args.realistic_terrain,
    )

    # Sanity gate 1 (terrain activation): orography reaches the dynamics only
    # if truncated_orography is non-zero. Aquaplanet is identically zero.
    orog_max = float(jnp.max(jnp.abs(atm_model.truncated_orography)))
    land_fraction = float(jnp.mean(terrain.fmask))
    print(f"  |truncated_orography|_max = {orog_max:.3e}")
    print(f"  land fraction = {land_fraction:.3f}")
    if args.realistic_terrain:
        assert orog_max > 0.0, (
            "Realistic terrain requested but truncated_orography is zero — "
            "orography is NOT reaching the dynamics."
        )
        assert land_fraction > 0.0, "Realistic terrain has no land cells."

    print("\nInitializing coupled simulation...")
    carry = coupler.initialize()
    # Zero the MCB perturbation for the entire spin-up (no-MCB climate)
    carry = jax.tree_util.tree_map(lambda x: x, carry)
    existing = carry["atm"]["derived"]["mcb_perturbation"]
    carry["atm"]["derived"]["mcb_perturbation"] = jnp.zeros_like(existing)

    step_fn = create_coupled_step_fn(coupler, WORKFLOW, jitted=True)

    manifest = {
        "start_date": START_DATE,
        "coupling_timestep_days": 1,
        "spinup_interval": args.spinup_interval,
        "horizon": args.horizon,
        "num_train": args.num_train,
        "num_heldout": args.num_heldout,
        "realistic_terrain": bool(args.realistic_terrain),
        "terrain_source": terrain_source,
        "ics": [],
    }

    for i in range(num_ics):
        day = i * args.spinup_interval
        split = "train" if i < args.num_train else "heldout"

        if i > 0:
            print(f"\nSpinning up to day {day} "
                  f"(+{args.spinup_interval} days)...")
            t0 = time.time()
            carry = run_interval_final_carry(
                carry, step_fn, args.spinup_interval
            )
            print(f"  done [{time.time() - t0:.1f}s]")

        carry_file = f"ic_{i:02d}_day{day:03d}_carry.pkl"
        baseline_file = f"ic_{i:02d}_day{day:03d}_baseline{args.horizon}d.pkl"

        print(f"Saving IC {i} ({split}, spin-up day {day})...")
        save_carry(carry, str(output_dir / carry_file))

        print(f"  Computing paired {args.horizon}-day baseline...")
        t0 = time.time()
        trajectory = compute_baseline_trajectory(
            carry, step_fn, num_steps=args.horizon, coords=coords
        )
        with open(output_dir / baseline_file, "wb") as f:
            pickle.dump(jax.device_get({
                "sst": trajectory.sst,
                "surface_temperature": trajectory.surface_temperature,
                "precipitation": trajectory.precipitation,
                "heat_flux": trajectory.heat_flux,
            }), f)
        print(f"  done [{time.time() - t0:.1f}s]")

        manifest["ics"].append({
            "index": i,
            "spinup_days": day,
            "split": split,
            "carry_file": carry_file,
            "baseline_file": baseline_file,
        })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")
    print(f"All ICs saved to {output_dir}")


if __name__ == "__main__":
    main()
