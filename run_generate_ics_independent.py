"""Independent-trajectory IC generation (PREREGISTRATION.md section 2).

Branch the equilibrated coupled state (run_equilibrate.py) into N INDEPENDENT
trajectories: each branch gets a distinct tiny seeded SST perturbation, then is
spun --decorr-days on its own so the weather fully decorrelates (atmospheric
predictability is ~2 weeks, so a 30-day spin from a 0.05 K SST perturbation
yields independent weather). Held-out ICs are ENTIRE independent trajectories
that share no simulated days with any training window.

This replaces the old run_stage5_generate_ics scheme, which snapshotted ONE
spin-up trajectory at {0,45,90,...} days — correlated points whose 60-day
windows overlapped, so "held-out" tested reproduction on maximally-correlated
data, not generalization (audit R5 / experimental-design finding).

Usage (GPU on diya, after run_equilibrate.py):
    python run_generate_ics_independent.py \
        --base-carry mcb_experiments_gpu/equilibrated/base_carry.pkl \
        --num-train 10 --num-heldout 10 --decorr-days 30 --horizon 60 \
        --output-dir mcb_experiments_gpu/ics_independent
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax_datetime as jdt

from jcm.mcb import compute_baseline_trajectory, load_carry, save_carry
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    run_interval_final_carry,
)
from run_coupled_training import (
    TERRAIN_NC,
    coupler_workflow,
    setup_coupled_model,
)
from run_stage5_training import START_DATE


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-carry", required=True,
                   help="Equilibrated carry from run_equilibrate.py")
    p.add_argument("--num-train", type=int, default=10)
    p.add_argument("--num-heldout", type=int, default=10)
    p.add_argument("--decorr-days", type=int, default=30,
                   help="Independent spin per branch to decorrelate weather")
    p.add_argument("--horizon", type=int, default=60,
                   help="Paired no-MCB baseline length (days)")
    p.add_argument("--perturb-amp", type=float, default=0.05,
                   help="SST perturbation amplitude (K) seeding each branch")
    p.add_argument("--seed0", type=int, default=1000,
                   help="Base seed; branch i uses seed0 + i")
    p.add_argument("--output-dir", default="mcb_experiments/ics_independent")
    return p.parse_args()


def perturb_sst(carry, seed, amp):
    """Return a copy of the coupled carry with a tiny seeded SST perturbation.

    A small SST perturbation is amplified by atmospheric chaos over the
    subsequent decorrelation spin, producing an independent weather trajectory
    from the same equilibrated climate.
    """
    ocn_state = carry["ocn"]["state"]
    key = jax.random.PRNGKey(seed)
    sst = ocn_state.sea_surface_temperature
    noise = amp * jax.random.normal(key, sst.shape)
    new_carry = dict(carry)
    new_carry["ocn"] = dict(carry["ocn"])
    new_carry["ocn"]["state"] = ocn_state.copy(
        {"sea_surface_temperature": sst + noise})
    return new_carry


def main():
    args = parse_args()
    start = jdt.to_datetime(START_DATE)
    dt = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm = setup_coupled_model(
        start, dt, realistic_terrain=True)
    wf = coupler_workflow(coupler)
    template = coupler.initialize()
    base = load_carry(args.base_carry, template)
    step_fn = create_coupled_step_fn(coupler, wf, jitted=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_train + args.num_heldout
    print("=" * 72)
    print(f"INDEPENDENT ICs: {args.num_train} train + {args.num_heldout} "
          f"held-out | decorr {args.decorr_days}d | horizon {args.horizon}d")
    print("=" * 72)

    ics = []
    for i in range(n):
        split = "train" if i < args.num_train else "heldout"
        seed = args.seed0 + i
        carry = perturb_sst(base, seed, args.perturb_amp)
        # Spin this branch independently so its weather decorrelates.
        carry = run_interval_final_carry(carry, step_fn, args.decorr_days)
        carry_file = f"ic_{i:02d}_{split}_seed{seed}_carry.pkl"
        save_carry(carry, str(out / carry_file))
        # Paired no-MCB baseline for this exact IC, on the current code.
        baseline = compute_baseline_trajectory(
            carry, step_fn, num_steps=args.horizon, coords=coords)
        baseline_file = f"ic_{i:02d}_{split}_seed{seed}_baseline{args.horizon}d.pkl"
        # Save as the dict form load_baseline_trajectory (run_stage5_training)
        # expects — NOT the CoupledBaselineTrajectory object, which is not
        # subscriptable on load.
        with open(out / baseline_file, "wb") as f:
            pickle.dump(jax.device_get({
                "sst": baseline.sst,
                "surface_temperature": baseline.surface_temperature,
                "precipitation": baseline.precipitation,
                "heat_flux": baseline.heat_flux,
            }), f)
        ics.append({
            "index": i, "split": split, "seed": seed,
            "spinup_days": args.decorr_days,  # kept for load_ics logging compat
            "carry_file": carry_file, "baseline_file": baseline_file,
        })
        print(f"  IC {i:02d} ({split:>7}, seed {seed}): saved carry + "
              f"{args.horizon}d baseline")

    manifest = {
        "start_date": START_DATE,
        "coupling_timestep_days": 1,
        "horizon": args.horizon,
        "num_train": args.num_train,
        "num_heldout": args.num_heldout,
        "decorr_days": args.decorr_days,
        "perturb_amp": args.perturb_amp,
        "independent_trajectories": True,
        "realistic_terrain": True,
        "terrain_source": TERRAIN_NC,
        "ics": ics,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {n} independent ICs + manifest -> {out}")


if __name__ == "__main__":
    main()
