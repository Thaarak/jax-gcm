#!/usr/bin/env python
"""CO2 scoping: measure this model's warming response to a transient CO2 ramp.

The transient-CO2 path in jcm/physics/speedy/forcing.py has been dormant for
the whole project and has NEVER been exercised. Before any control experiment
can be designed around it, three numbers have to be measured on the registered
metric (final-tail-mean global-ocean SST):

  1. the warming per unit ramp rate, and whether it is linear in the rate;
  2. the SHAPE of the response — a ramp forcing against a slab of 60-570 day
     e-folding time does NOT produce a ramp response, so an "amplified rate
     over a short horizon" is not automatically equivalent to a realistic rate
     over a long one, and the distortion has to be quantified rather than
     assumed;
  3. the chaos floor at the horizon used, from control-member spread.

Mechanism, verified: ablco2 = ablco2_ref * exp(rate * 0.005 * dyears), feeding
the longwave absorptivity band (shortwave_radiation.py) -> transmissivity ->
radiative heating. `increase_co2` is declared bool but used as a multiplier, so
it is a continuous rate knob. co2_year_ref MUST be the start year: its 1950
default against a 2000 start applies a +28.4% INSTANT step instead of a ramp.

Exploratory calibration: no gates, no hypotheses, and its output is not reused
for any confirmatory claim.

Example (diya):
    python run_co2_scoping.py \
        --base-carry mcb_experiments_gpu/equilibrated/base_carry.pkl \
        --rates 0 5 10 20 --days 365 --members 3 \
        --output mcb_experiments_gpu/co2_scoping.pkl
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
    p.add_argument("--rates", type=float, nargs="+", default=[0, 5, 10, 20],
                   help="increase_co2 multipliers. 0 is the control and must "
                        "be included.")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--tail-days", type=int, default=60)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--member-perturb-amp", type=float, default=0.001)
    p.add_argument("--member-seed0", type=int, default=91000)
    p.add_argument("--co2-year-ref", type=int, default=2000)
    p.add_argument("--output", required=True)
    return p.parse_args()


def run_daily_gmst(carry, step_fn, days, weights):
    """Run `days` coupling steps with no MCB; return the daily ocean-mean SST."""
    c = jax.tree_util.tree_map(lambda x: x, carry)
    c["atm"]["derived"]["mcb_perturbation"] = jnp.zeros_like(
        c["atm"]["derived"]["mcb_perturbation"])

    def body(cc, i):
        nc, _ = step_fn(cc, i)
        sst = nc["ocn"]["state"].sea_surface_temperature
        return nc, jnp.sum(sst * weights)

    _, series = lax.scan(body, c, jnp.arange(days))
    return np.asarray(series)


def main():
    args = parse_args()
    assert 0.0 in args.rates, "rate 0 (the control) must be included"
    print("=" * 76)
    print("TRANSIENT-CO2 SCOPING (warming response, shape, and chaos floor)")
    print("=" * 76)
    print(f"JAX devices: {jax.devices()}")
    print(f"rates {args.rates} | days {args.days} | tail {args.tail_days} | "
          f"members {args.members} | co2_year_ref {args.co2_year_ref}")

    start = jdt.to_datetime(START_DATE)
    dt = jdt.to_timedelta(1, "day")
    results = {"config": vars(args), "start_date": START_DATE, "rates": {}}
    out_path = Path(args.output)
    t_all = time.time()

    # Each rate needs its own model: Parameters is a closure constant on the
    # physics object, so a rate change forces a recompile. That is why the
    # rate is a DESIGN constant rather than a per-episode draw.
    for rate in args.rates:
        t_rate = time.time()
        coupler, coords, terrain, atm = setup_coupled_model(
            start, dt, realistic_terrain=True, co2_rate=rate,
            co2_year_ref=args.co2_year_ref)
        template = coupler.initialize()
        base = (template if args.base_carry == "template"
                else load_carry(args.base_carry, template))
        ocean_mask = ocean_mask_from_coupler(coupler)
        step_fn = create_coupled_step_fn(coupler, coupler_workflow(coupler),
                                         jitted=True)
        w = compute_area_weights(coords) * ocean_mask
        w = w / jnp.sum(w)

        series = []
        for m in range(args.members):
            c = base if m == 0 else perturb_sst(
                base, args.member_seed0 + m, args.member_perturb_amp)
            t0 = time.time()
            series.append(run_daily_gmst(c, step_fn, args.days, w))
            print(f"  rate {rate:>5}: member {m} done in "
                  f"{time.time() - t0:.0f}s", flush=True)
        arr = np.stack(series)
        results["rates"][rate] = {"gmst_daily": arr}
        print(f"  rate {rate:>5}: all members in "
              f"{(time.time() - t_rate) / 60:.1f} min", flush=True)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    # ---- Summary ----
    tail = args.tail_days
    ctrl = results["rates"][0.0]["gmst_daily"]
    chaos = float(ctrl[:, -tail:].mean(axis=1).std(ddof=1)) \
        if args.members > 1 else float("nan")
    print("\n" + "=" * 76)
    print(f"SUMMARY (tail = final {tail} days of {args.days})")
    print("=" * 76)
    print(f"control chaos floor (member spread of the tail mean): "
          f"{chaos * 1000:.1f} mK")
    summary = {"chaos_floor_mK": chaos * 1000, "response": {}}
    print(f"\n{'rate':>6} {'dGMST tail':>11} {'/rate':>9} {'day 90':>9} "
          f"{'day 180':>9} {'day 365':>9}")
    for rate in sorted(results["rates"]):
        if rate == 0.0:
            continue
        arr = results["rates"][rate]["gmst_daily"]
        d = arr.mean(axis=0) - ctrl.mean(axis=0)
        tail_mean = float(d[-tail:].mean())
        marks = [float(d[min(k, args.days) - 1]) for k in (90, 180, 365)]
        summary["response"][rate] = {
            "dgmst_tail": tail_mean, "per_rate": tail_mean / rate,
            "day90": marks[0], "day180": marks[1], "day365": marks[2],
            "snr_vs_chaos": tail_mean / chaos if chaos else float("nan"),
        }
        print(f"{rate:>6} {tail_mean * 1000:+10.1f} "
              f"{tail_mean / rate * 1000:+8.2f} "
              + " ".join(f"{v * 1000:+8.1f}" for v in marks))
    print("\n  (dGMST in mK vs the rate-0 control; /rate tests LINEARITY —")
    print("   equal values across rates mean the response scales linearly)")
    nz = sorted(r for r in results["rates"] if r != 0.0)
    if len(nz) >= 2:
        pr = [summary["response"][r]["per_rate"] for r in nz]
        print(f"  linearity: per-rate response ranges "
              f"{min(pr) * 1000:+.2f} to {max(pr) * 1000:+.2f} mK/unit "
              f"-> {'LINEAR' if max(pr) - min(pr) < 0.25 * abs(np.mean(pr)) else 'NONLINEAR'}")
        top = nz[-1]
        s = summary["response"][top]
        if s["day180"]:
            print(f"  response SHAPE at rate {top}: day90/day365 = "
                  f"{s['day90'] / s['day365']:.2f}, day180/day365 = "
                  f"{s['day180'] / s['day365']:.2f}")
            print("   (a pure ramp response would give 0.25 and 0.50; larger "
                  "means the slab is already equilibrating)")
        print(f"  SNR at rate {top}: {s['snr_vs_chaos']:.1f}x the chaos floor")
    results["summary"] = summary
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDONE in {(time.time() - t_all) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
