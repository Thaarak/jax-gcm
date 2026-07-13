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
    parser.add_argument("--train-days", type=str, default=None,
                        help="Comma-separated train spin-up days, e.g. "
                             "'0,45,90,135,180,225'. When given, requires "
                             "--heldout-days and overrides the "
                             "--spinup-interval/--num-train/--num-heldout path.")
    parser.add_argument("--heldout-days", type=str, default=None,
                        help="Comma-separated held-out spin-up days, e.g. "
                             "'70,200'. Every held-out day must be strictly "
                             "bracketed inside (min(train), max(train)).")
    parser.add_argument("--horizon", type=int, default=60,
                        help="Paired baseline trajectory length (days)")
    parser.add_argument("--output-dir", type=str,
                        default="mcb_experiments/stage5/ics")
    return parser.parse_args()


def _parse_day_list(raw, flag):
    """Parse a comma-separated int list, rejecting non-int tokens."""
    days = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            days.append(int(tok))
        except ValueError:
            raise ValueError(
                f"{flag}: '{tok}' is not an integer (expected comma-separated "
                f"ints like '0,45,90')"
            )
    if not days:
        raise ValueError(f"{flag}: no days parsed from '{raw}'")
    return days


def build_schedule(args):
    """Resolve the spin-up schedule as a sorted list of (day, split).

    Explicit path (--train-days/--heldout-days): both required. Held-out days
    must be strictly bracketed inside the training-day hull so that held-out
    evaluation is interpolation, not extrapolation.

    Fallback path (neither flag): reproduces the legacy
    --spinup-interval/--num-train/--num-heldout behavior exactly.
    """
    explicit = args.train_days is not None or args.heldout_days is not None
    if explicit:
        if args.train_days is None or args.heldout_days is None:
            raise ValueError(
                "--train-days and --heldout-days must be given together."
            )
        train_days = _parse_day_list(args.train_days, "--train-days")
        heldout_days = _parse_day_list(args.heldout_days, "--heldout-days")

        all_days = train_days + heldout_days
        if any(d < 0 for d in all_days):
            raise ValueError("All days must be >= 0.")
        train_set = set(train_days)
        heldout_set = set(heldout_days)
        if len(train_set) != len(train_days):
            raise ValueError("--train-days contains duplicates.")
        if len(heldout_set) != len(heldout_days):
            raise ValueError("--heldout-days contains duplicates.")
        if len(train_set) < 2:
            raise ValueError("Need >= 2 distinct train days.")
        if train_set & heldout_set:
            raise ValueError(
                f"train and held-out day sets overlap: "
                f"{sorted(train_set & heldout_set)}"
            )
        lo, hi = min(train_set), max(train_set)
        for d in heldout_days:
            if not (lo < d < hi):
                raise ValueError(
                    f"held-out day {d} is not strictly bracketed inside "
                    f"({lo}, {hi}); this reintroduces the extrapolation bug."
                )

        schedule = (
            [(d, "train") for d in train_days]
            + [(d, "heldout") for d in heldout_days]
        )
        return sorted(schedule, key=lambda t: t[0])

    # Fallback path: legacy interval behavior.
    num_ics = args.num_train + args.num_heldout
    schedule = []
    for i in range(num_ics):
        day = i * args.spinup_interval
        split = "train" if i < args.num_train else "heldout"
        schedule.append((day, split))
    return schedule


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain_source = TERRAIN_NC if args.realistic_terrain else "aquaplanet"

    schedule = build_schedule(args)
    train_days = [d for d, s in schedule if s == "train"]
    heldout_days = [d for d, s in schedule if s == "heldout"]
    num_train = len(train_days)
    num_heldout = len(heldout_days)
    explicit = args.train_days is not None or args.heldout_days is not None

    print("=" * 70)
    print("STAGE 5 IC GENERATION (realistic terrain)")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"  Terrain: {terrain_source}")
    print(f"  Train days:    {train_days}")
    print(f"  Held-out days: {heldout_days}")
    print(f"  ICs: {num_train} train + {num_heldout} held-out")
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
        "spinup_interval": None if explicit else args.spinup_interval,
        "horizon": args.horizon,
        "num_train": num_train,
        "num_heldout": num_heldout,
        "realistic_terrain": bool(args.realistic_terrain),
        "terrain_source": terrain_source,
        "ics": [],
    }

    prev_day = 0
    for i, (day, split) in enumerate(schedule):
        delta = day - prev_day
        assert delta >= 0, f"non-monotonic schedule: day {day} < prev {prev_day}"

        if delta > 0:
            print(f"\nSpinning up to day {day} "
                  f"(+{delta} days)...")
            t0 = time.time()
            carry = run_interval_final_carry(carry, step_fn, delta)
            print(f"  done [{time.time() - t0:.1f}s]")
        prev_day = day

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
