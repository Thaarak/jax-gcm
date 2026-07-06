#!/usr/bin/env python
"""Stage 5 evaluation: realistic-terrain policy vs warm-start / static baselines.

Evaluates up to three policies on ALL realistic-terrain ICs (train + held-out)
from the Stage 5 IC directory:
  - stage5-realistic : the Stage 5 ensemble-trained policy (13 features,
    ocean-masked loss/features, teleconnection penalties on).
  - stage4-warmstart : the teleconnection-AGNOSTIC Stage 4 policy (13 features)
    evaluated on the SAME realistic ICs — the control for Gate 3.
  - stage1-static    : the Stage 1 optimized static pattern (11 features).

Everything is ocean-masked (loss, features, dSST) via a binary ocean_mask
(fmask > 0.95 is land, matching the slab ocean model's convention), so land's
pinned 288.15 K temperature never enters the metrics.

Per (policy, IC) cell we record: ocean-masked mean loss, day-<days> ocean-
masked paired dSST, per-region (amazon / sahel / tropics) precip-protection
loss, and the full per-interval MCB forcing field.

Gates:
  1. Terrain activation: |truncated_orography| > 0 (checked once at setup).
  2. Cooling (ocean-masked): stage5 held-out mean dSST in [-0.12, -0.08] K
     (extrapolation caveat noted; Option A is the dedicated fix).
  3. Teleconnection protection: stage5 mean regional precip-protection loss
     <= stage4-warmstart's on the same realistic ICs.
  4. Generalization + stability: stage5 held-out mean loss finite and
     <= stage1-static; all losses finite.

Usage:
    python run_stage5_eval.py --ic-dir mcb_experiments/stage5/ics \
        --checkpoint mcb_experiments/stage5/stage5_trained_policy.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.mcb import (
    CoupledControllerConfig,
    CoupledFeatureConfig,
    CoupledLossWeights,
    MCBPolicyMLP,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_train import ocean_mask_from_coupler
from jcm.mcb.coupled_controller import evaluate_coupled_policy
from jcm.mcb.coupled_loss import (
    coupled_precipitation_loss,
    tropical_precipitation_loss,
)
from jcm.mcb.state_features import compute_area_weights
from jcm.mcb.train import load_checkpoint

from run_coupled_training import setup_coupled_model, warm_start_params
from run_stage5_training import START_DATE, load_ics

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 evaluation")
    parser.add_argument("--ic-dir", type=str,
                        default="mcb_experiments/stage5/ics")
    parser.add_argument("--checkpoint", type=str,
                        default="mcb_experiments/stage5/stage5_trained_policy.pkl",
                        help="Stage 5 realistic-terrain checkpoint (13 features)")
    parser.add_argument("--stage4-checkpoint", type=str,
                        default="mcb_experiments/stage4/stage4_trained_policy.pkl",
                        help="Stage 4 aquaplanet warm-start policy (13 "
                             "features), teleconnection-agnostic control")
    parser.add_argument("--stage1", type=str,
                        default="mcb_experiments/stage1/stage1_optimized_pattern.pkl",
                        help="Stage 1 optimized pattern pickle (static policy)")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--control-interval", type=int, default=30)
    parser.add_argument("--target-cooling", type=float, default=-0.1)
    parser.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                        action="store_false", default=True)
    # Teleconnection weights used to score the regional precip-protection loss
    # (kept equal to the Stage 5 training defaults).
    parser.add_argument("--amazon", type=float, default=0.05)
    parser.add_argument("--sahel", type=float, default=0.05)
    parser.add_argument("--tropics", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle (default: <ic-dir>/../eval_results.pkl)")
    return parser.parse_args()


def make_config(args, feature_config, loss_weights):
    return CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.days,
        target_cooling=args.target_cooling,
        loss_weights=loss_weights,
        feature_config=feature_config,
        max_perturbation=0.15,
        use_checkpointing=True,
    )


def cross_ic_forcing_std(forcings, interval):
    """Mean grid-point std of forcing across ICs at a given interval."""
    stacked = jnp.stack([f[interval] for f in forcings])  # (num_ics, ix, il)
    return float(jnp.mean(jnp.std(stacked, axis=0)))


def region_precip_losses(final_carry, baseline_final, coords, area_weights):
    """Per-region precip-protection losses from a final carry (day <days>).

    Uses the same loss functions as training (amazon/sahel decrease penalty,
    tropics squared-change). Returns a dict of raw (unweighted) losses so the
    Stage 3 gate can compare policies on identical footing.
    """
    return {
        "amazon": float(coupled_precipitation_loss(
            final_carry, baseline_final.precipitation, coords,
            area_weights, region="amazon",
        )),
        "sahel": float(coupled_precipitation_loss(
            final_carry, baseline_final.precipitation, coords,
            area_weights, region="sahel",
        )),
        "tropics": float(tropical_precipitation_loss(
            final_carry, baseline_final.precipitation, coords, area_weights,
        )),
    }


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    output_path = args.output or str(Path(args.ic_dir).parent / "eval_results.pkl")

    print("=" * 70)
    print("STAGE 5 EVALUATION: realistic-terrain policy vs stage4 / stage1")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    start_datetime = jdt.to_datetime(START_DATE)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep,
        realistic_terrain=args.realistic_terrain,
    )
    template_carry = coupler.initialize()
    # Authoritative binary ocean mask read from the slab ocean model's own grid
    # bmask (the land classification that pins SST to 288.15 K), matching
    # run_stage5_training. Sourced from the coupler, not terrain.fmask, so eval
    # metrics are computed over exactly the cells whose SST evolves.
    ocean_mask = ocean_mask_from_coupler(coupler)
    area_weights = compute_area_weights(coords)

    # Gate 1: terrain activation.
    orog_max = float(jnp.max(jnp.abs(atm_model.truncated_orography)))
    print(f"  |truncated_orography|_max = {orog_max:.3e} "
          f"(land fraction {float(jnp.mean(terrain.fmask)):.3f})")
    gate1_terrain = (orog_max > 0.0) if args.realistic_terrain else True

    print(f"\nLoading ICs from {args.ic_dir}...")
    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template_carry,
        require_realistic_terrain=args.realistic_terrain,
    )
    all_ics = train_ics + heldout_ics
    print(f"  Terrain source: {manifest.get('terrain_source')}")
    print(f"  {len(train_ics)} train + {len(heldout_ics)} held-out ICs")

    output_shape = coords.horizontal.nodal_shape
    policy = MCBPolicyMLP(
        output_shape=output_shape,
        hidden_dims=(256, 256),
        max_perturbation=0.15,
    )

    fc13 = CoupledFeatureConfig(include_absolute_sst=True)
    fc11 = CoupledFeatureConfig()
    # Weights used to score losses uniformly across policies (teleconnection
    # penalties on, so the reported mean loss reflects regional protection).
    eval_weights = CoupledLossWeights(
        sst_cooling=1.0, sst_uniformity=0.1,
        amazon=args.amazon, sahel=args.sahel, tropics=args.tropics,
        regularization=0.001, smoothness=0.001,
    )

    # name -> (params, controller_config)
    policies = {}

    stage5_params, stage5_meta = load_checkpoint(args.checkpoint)
    policies["stage5-realistic"] = (stage5_params, make_config(args, fc13, eval_weights))

    if Path(args.stage4_checkpoint).exists():
        stage4_params, _ = load_checkpoint(args.stage4_checkpoint)
        policies["stage4-warmstart"] = (
            stage4_params, make_config(args, fc13, eval_weights)
        )
    else:
        print(f"  WARNING: stage4 checkpoint {args.stage4_checkpoint} "
              f"not found; skipping (Gate 3 will be SKIPPED)")

    if Path(args.stage1).exists():
        stage1_params = warm_start_params(
            policy, get_coupled_feature_dim(fc11), args.stage1
        )
        policies["stage1-static"] = (
            stage1_params, make_config(args, fc11, eval_weights)
        )
    else:
        print(f"  WARNING: stage1 pattern {args.stage1} not found; skipping")

    # --- Evaluate every (policy, IC) cell ---
    results = {"cells": {}, "manifest": manifest}
    for name, (params, config) in policies.items():
        for entry, carry, baseline in all_ics:
            ic = entry["index"]
            print(f"\nEvaluating {name} on IC {ic} "
                  f"({entry['split']}, day {entry['spinup_days']})...")
            t0 = time.time()
            result = evaluate_coupled_policy(
                coupler=coupler,
                workflow=WORKFLOW,
                policy_fn=policy.apply,
                policy_params=params,
                initial_carry=carry,
                baseline_trajectory=baseline,
                coords=coords,
                ocean_mask=ocean_mask,
                config=config,
            )
            m = result["metrics"]
            forcing = jax.device_get(result["trajectory"].mcb_forcing)
            precip = region_precip_losses(
                result["final_carry"],
                baseline.at_step(args.days),
                coords, area_weights,
            )
            print(f"  mean_loss = {m['mean_loss']:.6f}, "
                  f"final dSST = {m['final_sst_change']:+.4f} K | "
                  f"precip amazon/sahel/tropics = "
                  f"{precip['amazon']:.4e}/{precip['sahel']:.4e}/"
                  f"{precip['tropics']:.4e} "
                  f"[{time.time() - t0:.1f}s]")
            results["cells"][(name, ic)] = {
                "split": entry["split"],
                "spinup_days": entry["spinup_days"],
                "mean_loss": m["mean_loss"],
                "total_loss": m["total_loss"],
                "final_sst_change": m["final_sst_change"],
                "mean_mcb_forcing": m["mean_mcb_forcing"],
                "max_mcb_forcing": m["max_mcb_forcing"],
                "loss_trajectory": m["loss_trajectory"],
                "precip_losses": precip,
                "mcb_forcing": forcing,  # (num_intervals, ix, il)
            }

    # --- Aggregates ---
    num_intervals = args.days // args.control_interval
    agg = {}
    for name in policies:
        cells = [results["cells"][(name, e["index"])] for e, _, _ in all_ics]
        train_cells = [c for c in cells if c["split"] == "train"]
        held_cells = [c for c in cells if c["split"] == "heldout"]

        forcings = [c["mcb_forcing"] for c in cells]

        def region_total(cell):
            p = cell["precip_losses"]
            return (args.amazon * p["amazon"] + args.sahel * p["sahel"]
                    + args.tropics * p["tropics"])

        agg[name] = {
            "train_mean_loss": (sum(c["mean_loss"] for c in train_cells)
                                / len(train_cells)) if train_cells else None,
            "heldout_mean_loss": (sum(c["mean_loss"] for c in held_cells)
                                  / len(held_cells)) if held_cells else None,
            "train_mean_dsst": (sum(c["final_sst_change"] for c in train_cells)
                                / len(train_cells)) if train_cells else None,
            "heldout_mean_dsst": (sum(c["final_sst_change"] for c in held_cells)
                                  / len(held_cells)) if held_cells else None,
            "mean_region_precip_loss": sum(region_total(c) for c in cells) / len(cells),
            "cross_ic_forcing_std_interval0": cross_ic_forcing_std(forcings, 0),
            "cross_ic_forcing_std_interval1": (
                cross_ic_forcing_std(forcings, 1) if num_intervals > 1 else None
            ),
        }
    results["aggregates"] = agg

    # --- Report ---
    print("\n" + "=" * 70)
    print("AGGREGATES")
    print("=" * 70)
    header = (f"{'policy':<18} {'train loss':>11} {'held loss':>11} "
              f"{'train dSST':>11} {'held dSST':>11} {'region prec':>12} "
              f"{'xIC std i0':>11} {'xIC std i1':>11}")
    print(header)
    for name, a in agg.items():
        def fmt(v, plus=False, wide=False):
            w = 12 if wide else 11
            if v is None:
                return f"{'n/a':>{w}}"
            return f"{v:+{w}.4f}" if plus else f"{v:{w}.6f}"
        print(f"{name:<18} {fmt(a['train_mean_loss'])} "
              f"{fmt(a['heldout_mean_loss'])} "
              f"{fmt(a['train_mean_dsst'], plus=True)} "
              f"{fmt(a['heldout_mean_dsst'], plus=True)} "
              f"{fmt(a['mean_region_precip_loss'], wide=True)} "
              f"{fmt(a['cross_ic_forcing_std_interval0'])} "
              f"{fmt(a['cross_ic_forcing_std_interval1'])}")

    # --- Gates ---
    print("\n" + "=" * 70)
    print("SUCCESS GATES")
    print("=" * 70)
    s5 = agg["stage5-realistic"]
    gates = {}

    gates["1_terrain_activation"] = bool(gate1_terrain)
    print(f"  Gate 1 (terrain activation, |orography| > 0): "
          f"{orog_max:.3e} -> "
          f"{'PASS' if gates['1_terrain_activation'] else 'FAIL'}")

    held_dsst = s5["heldout_mean_dsst"]
    gates["2_heldout_cooling"] = (
        held_dsst is not None and -0.12 <= held_dsst <= -0.08
    )
    print(f"  Gate 2 (held-out ocean-masked dSST in [-0.12, -0.08] K): "
          f"{held_dsst if held_dsst is None else f'{held_dsst:+.4f}'} K "
          f"-> {'PASS' if gates['2_heldout_cooling'] else 'FAIL'}")
    print("    (note: held-out overcooling may reflect season EXTRAPOLATION; "
          "Option A bracketing is the dedicated fix — see MCB_IMPLEMENTATION_PLAN.md)")

    if "stage4-warmstart" in agg:
        s4_precip = agg["stage4-warmstart"]["mean_region_precip_loss"]
        s5_precip = s5["mean_region_precip_loss"]
        gates["3_teleconnection_protection"] = s5_precip <= s4_precip
        print(f"  Gate 3 (stage5 region precip loss <= stage4-warmstart): "
              f"{s5_precip:.6e} vs {s4_precip:.6e} "
              f"-> {'PASS' if gates['3_teleconnection_protection'] else 'FAIL'}")
    else:
        gates["3_teleconnection_protection"] = None
        print("  Gate 3 (teleconnection protection): SKIPPED "
              "(stage4 warm-start policy not found)")

    all_losses = [c["mean_loss"] for c in results["cells"].values()]
    finite = all(jnp.isfinite(x) for x in all_losses)
    if "stage1-static" in agg and s5["heldout_mean_loss"] is not None:
        s1_loss = agg["stage1-static"]["heldout_mean_loss"]
        gates["4_generalization_stability"] = (
            finite and s5["heldout_mean_loss"] <= s1_loss
        )
        print(f"  Gate 4 (stage5 held-out loss <= stage1-static AND finite): "
              f"{s5['heldout_mean_loss']:.6f} vs {s1_loss:.6f}, "
              f"finite={finite} "
              f"-> {'PASS' if gates['4_generalization_stability'] else 'FAIL'}")
    else:
        gates["4_generalization_stability"] = finite
        print(f"  Gate 4 (stability, all losses finite): "
              f"{'PASS' if finite else 'FAIL'} "
              f"(generalization SKIPPED — missing stage1 or held-out)")

    results["gates"] = gates

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
