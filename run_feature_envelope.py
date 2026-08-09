#!/usr/bin/env python
"""Measure the nuisance-feature envelope over an episode (Amendment 6).

Tier-2b's first distillation attempt failed because the synthetic feature
sampling was anchored by guess: the real absolute-SST features sat 3.7-8.5
sigma outside it and the imitation net froze (Amendment 4 revision 1). Rule
since then: MEASURE the operating envelope, never guess it.

This probe runs no-MCB no-ENSO baselines for a few TRAIN-split ICs over the
episode horizon and records the min/max of the two absolute-SST features
(f11 = ocean-mean SST - 288, f12 = NH - SH ocean SST) across all days and
ICs. Over 180 days (Jan-Jun) f12 traverses a large seasonal range that the
60-day Tier-2b anchors do not cover. Output feeds run_pi_imitation.py
--anchors-json.

Example (diya):
    python run_feature_envelope.py \
        --ic-dir mcb_experiments_gpu/ics_enso --split train --num-ics 3 \
        --days 180 --output mcb_experiments_gpu/enso_feature_anchors.json
"""

import argparse
import json

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_features import compute_baseline_trajectory
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.state_features import (
    compute_area_weights,
    compute_regional_mean,
    create_latitude_band_mask,
)

from run_coupled_training import coupler_workflow, setup_coupled_model
from run_stage5_training import START_DATE, load_ics


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--split", default="train",
                   choices=["train", "heldout"],
                   help="Use TRAIN ICs — the envelope must not peek at the "
                        "held-out evaluation set.")
    p.add_argument("--num-ics", type=int, default=3)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--margin", type=float, default=1.0,
                   help="Envelope widening (K) on each side.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print("FEATURE ENVELOPE PROBE (f11/f12 over the episode horizon)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")

    coupler, coords, terrain, atm_model = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True)
    template = coupler.initialize()
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)

    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template, require_realistic_terrain=True)
    ics = {"train": train_ics, "heldout": heldout_ics}[args.split]
    ics = ics[:args.num_ics]
    if not ics:
        raise SystemExit(f"No ICs in split '{args.split}' of {args.ic_dir}")

    area = compute_area_weights(coords)
    ocean_w = area * ocean_mask
    ocean_w = ocean_w / jnp.sum(ocean_w)
    nh = create_latitude_band_mask(coords, 0.0, 90.0) * ocean_mask
    sh = create_latitude_band_mask(coords, -90.0, 0.0) * ocean_mask

    f11_all, f12_all = [], []
    for entry, carry, _stored in ics:
        traj = compute_baseline_trajectory(
            carry, step_fn, num_steps=args.days, coords=coords)
        # Same definitions as extract_coupled_features' absolute block.
        f11 = jnp.einsum("txy,xy->t", traj.sst - 288.0, ocean_w)
        nh_m = jax.vmap(lambda s: compute_regional_mean(s, nh, area))(
            traj.sst)
        sh_m = jax.vmap(lambda s: compute_regional_mean(s, sh, area))(
            traj.sst)
        f11_all.append(np.asarray(f11))
        f12_all.append(np.asarray(nh_m - sh_m))
        print(f"  IC {entry['index']:02d}: f11 "
              f"[{float(f11.min()):+.2f}, {float(f11.max()):+.2f}] | "
              f"f12 [{float((nh_m - sh_m).min()):+.2f}, "
              f"{float((nh_m - sh_m).max()):+.2f}]", flush=True)

    f11_all = np.concatenate(f11_all)
    f12_all = np.concatenate(f12_all)
    anchors = {
        "f11_lo": float(f11_all.min() - args.margin),
        "f11_hi": float(f11_all.max() + args.margin),
        "f12_lo": float(f12_all.min() - args.margin),
        "f12_hi": float(f12_all.max() + args.margin),
        "days": args.days, "num_ics": len(ics), "split": args.split,
        "margin": args.margin,
    }
    with open(args.output, "w") as f:
        json.dump(anchors, f, indent=2)
    print(f"anchors -> {args.output}: {anchors}")


if __name__ == "__main__":
    main()
