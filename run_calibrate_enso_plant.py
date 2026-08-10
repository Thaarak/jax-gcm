#!/usr/bin/env python
"""Measure the ENSO plant constants IN METRIC SPACE (Amendment 7).

The Amendment-6 campaign built both its control law and its open-loop
comparator on constants measured with the wrong instrument: the ENSO
sensitivity came from atmospheric global-mean surface air temperature while
the registered metric scores global-OCEAN dSST (measured ratio 1.36x), and
the MCB authority came from a single base state. That mis-specification
crippled the open-loop arm and left every arm short of target
(MCB_META_AUDIT.md Addendum 5).

This run measures, on TRAIN ICs only, with micro-ensembles, on exactly the
registered metric (final-``tail-days`` mean global-ocean dSST vs a paired
no-ENSO no-MCB baseline):

  mu   = MCB authority per unit gain     (dSST from the pattern, ENSO off)
  s    = ENSO sensitivity                (dSST per K of Nino3.4, MCB off)

and reports the two derived design constants:

  rescale        = target / mu           (pattern scaling so gain 1 hits target)
  enso_effect_per_K = s                  (the control law's feedforward gain)
  A_floor        = |target| / s          (the cold amplitude at which the
                                          actuator floor binds: even spraying
                                          nothing cannot stop the overcooling)

Example (diya):
    python run_calibrate_enso_plant.py \
        --ic-dir mcb_experiments_gpu/ics_enso7 --split train \
        --stage1 mcb_experiments_gpu/stage1_v2/stage1_optimized_pattern.pkl \
        --days 180 --tail-days 60 --members 4 --probe-amp 1.4 \
        --output mcb_experiments_gpu/enso7_plant.json
"""

import argparse
import json
import pickle
import time

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
from jax import lax

from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_features import compute_baseline_trajectory
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.enso import EnsoConfig, nino_pattern, wrap_step_fn_with_enso
from jcm.mcb.state_features import compute_area_weights

from run_coupled_training import coupler_workflow, setup_coupled_model
from run_generate_ics_independent import perturb_sst
from run_stage5_training import START_DATE, load_ics


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "heldout"],
                   help="TRAIN only — calibration must not touch held-out ICs.")
    p.add_argument("--stage1", required=True)
    p.add_argument("--num-ics", type=int, default=6)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--tail-days", type=int, default=60)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--member-perturb-amp", type=float, default=0.001)
    p.add_argument("--member-seed0", type=int, default=78000)
    p.add_argument("--probe-amp", type=float, default=1.4,
                   help="Symmetric probe amplitude (+/-) for the ENSO slope.")
    p.add_argument("--target-cooling", type=float, default=-0.1)
    p.add_argument("--enso-ramp-days", type=float, default=30.0)
    p.add_argument("--enso-tau-days", type=float, default=5.0)
    p.add_argument("--output", required=True)
    return p.parse_args()


def tail_mean_dsst(carry, step_fn, days, tail, weights, mcb_field,
                   transform=None):
    """Score one rollout on the registered final-`tail`-day mean ocean SST."""
    fn = transform(step_fn) if transform is not None else step_fn
    c = jax.tree_util.tree_map(lambda x: x, carry)
    c["atm"]["derived"]["mcb_perturbation"] = mcb_field

    def body(cc, i):
        nc, _ = fn(cc, i)
        sst = nc["ocn"]["state"].sea_surface_temperature
        return nc, jnp.sum(sst * weights)

    _, series = lax.scan(body, c, jnp.arange(days))
    return np.asarray(series)[-tail:].mean()


def main():
    args = parse_args()
    print("=" * 76)
    print("ENSO PLANT CALIBRATION — IN METRIC SPACE (Amendment 7)")
    print("=" * 76)
    print(f"JAX devices: {jax.devices()}")

    coupler, coords, terrain, atm = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    template = coupler.initialize()
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)

    area = compute_area_weights(coords)
    w = area * ocean_mask
    w = w / jnp.sum(w)
    pattern_enso = nino_pattern(coords.horizontal) * ocean_mask
    with open(args.stage1, "rb") as f:
        mcb_pattern = jnp.asarray(pickle.load(f)["best_pattern"]) * ocean_mask
    zero = jnp.zeros(coords.horizontal.nodal_shape)

    _, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template, require_realistic_terrain=True)
    ics = ({"train": train_ics, "heldout": heldout_ics}[args.split]
           )[:args.num_ics]
    if not ics:
        raise SystemExit(f"No ICs in split '{args.split}'")

    mu_s, s_s = [], []
    t0 = time.time()
    for entry, carry, _ in ics:
        for m in range(args.members):
            c = carry if m == 0 else perturb_sst(
                carry, args.member_seed0 + 97 * entry["index"] + m,
                args.member_perturb_amp)
            base = compute_baseline_trajectory(c, step_fn, args.days, coords)
            base_tail = float(np.mean(
                [float(jnp.sum(base.sst[t] * w))
                 for t in range(args.days - args.tail_days + 1,
                                args.days + 1)]))

            # (1) MCB authority at gain 1, ENSO off.
            mcb = tail_mean_dsst(c, step_fn, args.days, args.tail_days, w,
                                 mcb_pattern) - base_tail
            mu_s.append(mcb)

            # (2) ENSO sensitivity, MCB off, symmetric +/- probe.
            resp = {}
            for sign in (+1.0, -1.0):
                cfg = EnsoConfig(amplitude=sign * args.probe_amp,
                                 ramp_days=args.enso_ramp_days,
                                 relax_tau_days=args.enso_tau_days)
                def tf(sf, _cfg=cfg, _ref=base.sst,
                       _t0=float(c["ocn"]["state"].sim_time)):
                    return wrap_step_fn_with_enso(sf, pattern_enso, _cfg,
                                                  _ref, _t0)
                resp[sign] = tail_mean_dsst(
                    c, step_fn, args.days, args.tail_days, w, zero,
                    transform=tf) - base_tail
            slope = (resp[+1.0] - resp[-1.0]) / (2.0 * args.probe_amp)
            s_s.append(slope)
            print(f"  IC {entry['index']:02d} m{m}: mu {mcb:+.4f} K | "
                  f"ENSO +{args.probe_amp} {resp[+1.0]:+.4f} / "
                  f"-{args.probe_amp} {resp[-1.0]:+.4f} -> s {slope:+.4f} K/K",
                  flush=True)

    mu = float(np.mean(mu_s))
    s = float(np.mean(s_s))
    n = len(mu_s)
    out = {
        "mu_mcb_per_unit_gain": mu,
        "mu_se": float(np.std(mu_s, ddof=1) / np.sqrt(n)),
        "enso_effect_per_K": s,
        "enso_se": float(np.std(s_s, ddof=1) / np.sqrt(n)),
        "rescale": float(args.target_cooling / mu),
        "amplitude_floor_K": float(abs(args.target_cooling) / s) if s else None,
        "n_rollouts": n, "days": args.days, "tail_days": args.tail_days,
        "split": args.split, "num_ics": len(ics), "members": args.members,
        "probe_amp": args.probe_amp, "metric": "final_tail_mean_ocean_dSST",
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 76)
    print(f"MCB authority mu = {mu:+.4f} +/- {out['mu_se']:.4f} K per unit "
          f"gain  ->  pattern rescale = {out['rescale']:.4f}")
    print(f"ENSO sensitivity s = {s:+.4f} +/- {out['enso_se']:.4f} K/K "
          f"(Amendment-6 used 0.0525 from GMST)")
    print(f"Actuator floor binds at |A| = {out['amplitude_floor_K']:.2f} K "
          f"— size the amplitude range to this")
    print(f"DONE in {(time.time() - t0) / 60:.1f} min -> {args.output}")


if __name__ == "__main__":
    main()
