#!/usr/bin/env python
"""ENSO scoping run: measure this model's GMST-per-Nino3.4 sensitivity.

The decisive design question for the ENSO feedback experiment (2026-08-03
verification): what global-mean ocean SST swing does an imposed Nino3.4
anomaly produce in THIS coupled slab model? No published number exists for
SPEEDY + slab; the observational reference is ~0.11 K GMST per K of Nino3.4
at ~3 months lag (Trenberth et al. 2002). If the model delivers < ~0.05 K
per 2 K of commanded anomaly, the ENSO experiment design collapses back
into the chaos noise floor and must be rethought.

This run is exploratory instrument calibration (like run_noise_floor.py),
not a gated experiment: it measures (a) the realized-vs-commanded Nino3.4
anomaly (pacemaker fidelity), (b) the GMST response trajectory and its lag,
(c) the El Nino / La Nina asymmetry, and (d) the long-horizon chaos noise
floor from control-member spread — all needed to freeze the real
experiment's design in a future preregistration amendment.

Protocol per member m (micro-ensemble, Tier-1 discipline):
  control : no ENSO, no MCB -> daily SST trajectory (also the pacemaker's
            relaxation reference, so the commanded anomaly is measured
            against the model's own paired evolution)
  elnino  : pacemaker at +amplitude (step profile with ramp-in)
  lanina  : pacemaker at -amplitude
No MCB anywhere. All runs of a member start from the same perturbed carry.

Example (diya):
    python run_enso_scoping.py \
        --base-carry mcb_experiments_gpu/equilibrated/base_carry.pkl \
        --days 365 --members 4 \
        --output mcb_experiments_gpu/enso_scoping.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb import load_carry
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_features import compute_baseline_trajectory
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.enso import (
    EnsoConfig,
    box_mean_weights,
    nino_pattern,
    wrap_step_fn_with_enso,
)
from jcm.mcb.state_features import compute_area_weights

from run_coupled_training import coupler_workflow, setup_coupled_model
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-carry", required=True,
                   help="Equilibrated carry from run_equilibrate.py, or the "
                        "literal 'template' for an UNEQUILIBRATED smoke test "
                        "(plumbing check only — sensitivities from a "
                        "drifting ocean are meaningless).")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--members", type=int, default=4,
                   help="Micro-ensemble members; member 0 unperturbed.")
    p.add_argument("--member-perturb-amp", type=float, default=0.001)
    p.add_argument("--member-seed0", type=int, default=88000)
    p.add_argument("--amplitude", type=float, default=2.0,
                   help="Commanded Nino3.4 anomaly (K); lanina uses -A.")
    p.add_argument("--tau-days", type=float, default=5.0)
    p.add_argument("--ramp-days", type=float, default=30.0)
    p.add_argument("--period-days", type=float, default=0.0,
                   help="0 = step profile (default for sensitivity).")
    p.add_argument("--tail-days", type=int, default=90,
                   help="Terminal window for sensitivity + response maps.")
    p.add_argument("--skip-lanina", action="store_true")
    p.add_argument("--output", required=True)
    return p.parse_args()


def daily_scalars(sst_traj, gmst_w, nino_w):
    """(T+1,) ocean-mean and Nino-core-mean SST from a daily trajectory."""
    gmst = jnp.einsum("txy,xy->t", sst_traj, gmst_w)
    nino = jnp.einsum("txy,xy->t", sst_traj, nino_w)
    return np.asarray(gmst), np.asarray(nino)


def main():
    args = parse_args()
    print("=" * 72)
    print("ENSO SCOPING RUN (GMST-per-Nino3.4 sensitivity + pacemaker "
          "fidelity)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")
    print(f"days {args.days} | members {args.members} | "
          f"A = +/-{args.amplitude} K | tau {args.tau_days} d | "
          f"ramp {args.ramp_days} d | period {args.period_days} d")

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
    pattern = nino_pattern(coords.horizontal) * ocean_mask
    nino_w = box_mean_weights(coords.horizontal, pattern, area)
    n_core = int(np.sum(np.asarray(pattern) >= 0.999))
    print(f"Nino3.4 pattern: {n_core} core cells, "
          f"{int(np.sum(np.asarray(pattern) > 0))} total (with taper)")

    t0 = float(base["ocn"]["state"].sim_time)
    arms = {"elnino": args.amplitude}
    if not args.skip_lanina:
        arms["lanina"] = -args.amplitude

    tail = args.tail_days
    results = {
        "config": vars(args), "t0_seconds": t0, "start_date": START_DATE,
        "pattern": np.asarray(pattern), "members": [],
    }
    out_path = Path(args.output)
    t_start = time.time()

    for m in range(args.members):
        if m == 0:
            carry = base
        else:
            carry = perturb_sst(base, args.member_seed0 + m,
                                args.member_perturb_amp)
        t_run = time.time()
        control = compute_baseline_trajectory(
            carry, step_fn, num_steps=args.days, coords=coords)
        g_c, n_c = daily_scalars(control.sst, gmst_w, nino_w)
        member = {"index": m,
                  "control": {"gmst": g_c, "nino": n_c}}
        print(f"member {m}: control done in {time.time() - t_run:.0f}s "
              f"(GMST day0 {g_c[0]:.3f} -> day{args.days} {g_c[-1]:.3f})",
              flush=True)

        for arm, amp in arms.items():
            cfg = EnsoConfig(amplitude=amp, period_days=args.period_days,
                             ramp_days=args.ramp_days,
                             relax_tau_days=args.tau_days)
            enso_step = wrap_step_fn_with_enso(
                step_fn, pattern, cfg, control.sst, t0)
            t_run = time.time()
            traj = compute_baseline_trajectory(
                carry, enso_step, num_steps=args.days, coords=coords)
            g_e, n_e = daily_scalars(traj.sst, gmst_w, nino_w)
            dsst_map = np.asarray(
                jnp.mean(traj.sst[-tail:] - control.sst[-tail:], axis=0))
            member[arm] = {
                "gmst": g_e, "nino": n_e,
                "commanded": amp,
                "realized_nino_tail": float(np.mean(
                    (n_e - n_c)[-tail:])),
                "dgmst_tail": float(np.mean((g_e - g_c)[-tail:])),
                "dsst_map_tail": dsst_map,
            }
            print(f"member {m}: {arm} done in {time.time() - t_run:.0f}s | "
                  f"realized Nino {member[arm]['realized_nino_tail']:+.3f} K "
                  f"(cmd {amp:+.1f}) | dGMST(tail{tail}) "
                  f"{member[arm]['dgmst_tail']:+.4f} K", flush=True)

        results["members"].append(member)
        with open(out_path, "wb") as f:   # crash tolerance
            pickle.dump(results, f)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"SUMMARY ({args.members} members, tail = final {tail} days)")
    print("=" * 72)
    ctrl_end = np.array([mm["control"]["gmst"][-tail:].mean()
                         for mm in results["members"]])
    chaos_sd = float(ctrl_end.std(ddof=1)) if args.members > 1 else float("nan")
    print(f"Control GMST tail-mean member spread (long-horizon chaos "
          f"floor): sd = {chaos_sd:.4f} K")
    summary = {"control_tail_sd": chaos_sd}
    for arm in arms:
        dg = np.array([mm[arm]["dgmst_tail"] for mm in results["members"]])
        rn = np.array([mm[arm]["realized_nino_tail"]
                       for mm in results["members"]])
        sens = dg.mean() / rn.mean() if abs(rn.mean()) > 1e-9 else float("nan")
        se = dg.std(ddof=1) / np.sqrt(len(dg)) if len(dg) > 1 else float("nan")
        print(f"{arm:>7}: realized Nino {rn.mean():+.3f} K | dGMST "
              f"{dg.mean():+.4f} +/- {se:.4f} K | sensitivity "
              f"{sens:+.4f} K/K (Trenberth obs ~0.11)")
        summary[arm] = {"realized_nino": float(rn.mean()),
                        "dgmst_mean": float(dg.mean()),
                        "dgmst_se": float(se),
                        "sensitivity_K_per_K": float(sens)}
    results["summary"] = summary
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDONE in {(time.time() - t_start) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
