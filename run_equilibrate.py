"""P2 equilibration: spin the coupled terrain model to quasi-equilibrium.

The coupled model starts from an isothermal-rest atmosphere plus a climatology
ocean/land, so its first weeks-to-months are a spin-up transient. Nothing
measured on that transient means anything (the audit's R5: the pre-fix run
drifted +1.4 K/60d, ~14x the -0.1 K MCB signal). This script spins the coupled
system until the 60-day ocean drift is << the signal (target |drift| < 0.02
K/60d), then saves the equilibrated coupled carry. Independent-IC generation
branches + perturbs that base state.

Requires the R5 ocean fix (SST initialized from the T30 climatology, not the
~6 K-too-cold idealized field) and the R7a slab-land wiring, both already in
setup_coupled_model.

Usage (GPU on diya, vLLM stopped):
    python run_equilibrate.py --days 3650 --log-every 90 \
        --output mcb_experiments_gpu/equilibrated/base_carry.pkl

CPU smoke:
    python run_equilibrate.py --days 30 --log-every 10 --output /tmp/eq.pkl
"""

import argparse
import time
from pathlib import Path

import numpy as np
import jax_datetime as jdt

from jcm.mcb import save_carry
from run_coupled_training import setup_coupled_model, coupler_workflow
from run_stage5_training import START_DATE


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=3650,
                   help="Coupling days to spin up (default 10 years)")
    p.add_argument("--log-every", type=int, default=90,
                   help="Log drift + land-T range every N days")
    p.add_argument("--drift-target", type=float, default=0.02,
                   help="Target |60-day ocean drift| (K) for 'equilibrated'")
    p.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                   action="store_false", default=True)
    p.add_argument("--output",
                   default="mcb_experiments/equilibrated/base_carry.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    start = jdt.to_datetime(START_DATE)
    dt = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm = setup_coupled_model(
        start, dt, realistic_terrain=args.realistic_terrain)
    wf = coupler_workflow(coupler)
    print("=" * 72)
    print(f"EQUILIBRATION  workflow={wf}  days={args.days}")
    print("=" * 72)

    ocean = np.asarray(terrain.fmask) < 0.5
    land = np.asarray(terrain.fmask) > 0.5
    carry = coupler.initialize()
    step = coupler.generate_step_function(wf, jitted=True, verbose=False)

    def ocean_sst(c):
        return float(np.asarray(
            c["ocn"]["state"].sea_surface_temperature)[ocean].mean())

    prev_sst = ocean_sst(carry)
    print(f"init ocean SST mean = {prev_sst:.3f} K")
    t0 = time.time()
    last_drift = None
    for d in range(1, args.days + 1):
        carry, _ = step(carry, 0.0)
        if d % args.log_every == 0 or d == args.days:
            sst = ocean_sst(carry)
            # Drift rate over the last window, extrapolated to 60 days.
            drift60 = (sst - prev_sst) / args.log_every * 60.0
            last_drift = drift60
            T = np.asarray(carry["lnd"]["state"].land_surface_temperature) \
                if "lnd" in carry else None
            land_str = (f" | landT [{T[land].min():.1f},{T[land].max():.1f}]"
                        if T is not None else "")
            print(f"day {d:>5}: ocean SST {sst:.3f} K | "
                  f"60d-drift(last {args.log_every}d) {drift60:+.4f} K{land_str}")
            prev_sst = sst
    print(f"\n{args.days} days in {time.time() - t0:.0f}s")
    if last_drift is not None:
        ok = abs(last_drift) < args.drift_target
        print(f"final 60d-drift ~ {last_drift:+.4f} K  "
              f"(target |.| < {args.drift_target}: {'MET' if ok else 'NOT met'})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_carry(carry, str(out))
    print(f"saved equilibrated carry -> {out}")


if __name__ == "__main__":
    main()
