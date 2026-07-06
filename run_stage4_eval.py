#!/usr/bin/env python
"""Stage 4 evaluation: ensemble-trained policy vs Stage 3 / Stage 1 baselines.

Evaluates up to 3 policies (stage4-ensemble @ 13 features, stage3-single
@ 11 features, stage1-static @ 11 features) on ALL ICs (train + held-out)
from the Stage 4 IC directory. Per (policy, IC) cell: mean loss, final
paired dSST, and the full per-interval MCB forcing.

Aggregates and gates:
  1. Held-out cooling: stage4 mean final paired dSST over held-out ICs in
     [-0.12, -0.08] K.
  2. Generalization: stage4 held-out mean loss <= stage1-static held-out
     mean loss (stage3 reported too).
  3. State-dependence: cross-IC forcing std at interval 1 > 0 for stage4;
     interval-0 cross-IC std > 0 (enabled by absolute-SST features);
     structurally 0 for the 11-feature policies.
  4. Stability: all losses finite.

Usage:
    python run_stage4_eval.py --ic-dir mcb_experiments/stage4/ics \
        --checkpoint mcb_experiments/stage4/stage4_trained_policy.pkl
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
    create_ocean_mask,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import evaluate_coupled_policy
from jcm.mcb.train import load_checkpoint

from run_coupled_training import setup_coupled_model, warm_start_params
from run_stage4_training import START_DATE, load_ics

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 4 evaluation")
    parser.add_argument("--ic-dir", type=str,
                        default="mcb_experiments/stage4/ics")
    parser.add_argument("--checkpoint", type=str,
                        default="mcb_experiments/stage4/stage4_trained_policy.pkl",
                        help="Stage 4 ensemble-trained checkpoint (13 features)")
    parser.add_argument("--stage3-checkpoint", type=str,
                        default="mcb_experiments/stage3_60d/coupled_trained_policy.pkl",
                        help="Stage 3 single-IC checkpoint (11 features)")
    parser.add_argument("--stage1", type=str,
                        default="mcb_experiments/stage1/stage1_optimized_pattern.pkl",
                        help="Stage 1 optimized pattern pickle (static policy)")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--control-interval", type=int, default=30)
    parser.add_argument("--target-cooling", type=float, default=-0.1)
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle (default: <ic-dir>/../eval_results.pkl)")
    return parser.parse_args()


def make_config(args, feature_config):
    return CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.days,
        target_cooling=args.target_cooling,
        loss_weights=CoupledLossWeights(
            sst_cooling=1.0, sst_uniformity=0.1,
            amazon=0.0, sahel=0.0, tropics=0.0,
            regularization=0.001, smoothness=0.001,
        ),
        feature_config=feature_config,
        max_perturbation=0.15,
        use_checkpointing=True,
    )


def cross_ic_forcing_std(forcings, interval):
    """Mean grid-point std of forcing across ICs at a given interval."""
    stacked = jnp.stack([f[interval] for f in forcings])  # (num_ics, ix, il)
    return float(jnp.mean(jnp.std(stacked, axis=0)))


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    output_path = args.output or str(Path(args.ic_dir).parent / "eval_results.pkl")

    print("=" * 70)
    print("STAGE 4 EVALUATION: ensemble policy vs stage3 / stage1 baselines")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    start_datetime = jdt.to_datetime(START_DATE)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )
    template_carry = coupler.initialize()
    ocean_mask = create_ocean_mask(coords.horizontal, terrain.fmask)

    print(f"\nLoading ICs from {args.ic_dir}...")
    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template_carry
    )
    all_ics = train_ics + heldout_ics
    print(f"  {len(train_ics)} train + {len(heldout_ics)} held-out ICs")

    output_shape = coords.horizontal.nodal_shape
    policy = MCBPolicyMLP(
        output_shape=output_shape,
        hidden_dims=(256, 256),
        max_perturbation=0.15,
    )

    fc13 = CoupledFeatureConfig(include_absolute_sst=True)
    fc11 = CoupledFeatureConfig()

    # name -> (params, controller_config)
    policies = {}

    stage4_params, stage4_meta = load_checkpoint(args.checkpoint)
    policies["stage4-ensemble"] = (stage4_params, make_config(args, fc13))

    if Path(args.stage3_checkpoint).exists():
        stage3_params, _ = load_checkpoint(args.stage3_checkpoint)
        policies["stage3-single"] = (stage3_params, make_config(args, fc11))
    else:
        print(f"  WARNING: stage3 checkpoint {args.stage3_checkpoint} "
              f"not found; skipping")

    if Path(args.stage1).exists():
        stage1_params = warm_start_params(
            policy, get_coupled_feature_dim(fc11), args.stage1
        )
        policies["stage1-static"] = (stage1_params, make_config(args, fc11))
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
            print(f"  mean_loss = {m['mean_loss']:.6f}, "
                  f"final dSST = {m['final_sst_change']:+.4f} K "
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
        interval_diffs = [
            float(jnp.max(jnp.abs(jnp.asarray(f[1]) - jnp.asarray(f[0]))))
            for f in forcings
        ] if num_intervals > 1 else [0.0]

        agg[name] = {
            "train_mean_loss": (sum(c["mean_loss"] for c in train_cells)
                                / len(train_cells)) if train_cells else None,
            "heldout_mean_loss": (sum(c["mean_loss"] for c in held_cells)
                                  / len(held_cells)) if held_cells else None,
            "train_mean_dsst": (sum(c["final_sst_change"] for c in train_cells)
                                / len(train_cells)) if train_cells else None,
            "heldout_mean_dsst": (sum(c["final_sst_change"] for c in held_cells)
                                  / len(held_cells)) if held_cells else None,
            "cross_ic_forcing_std_interval0": cross_ic_forcing_std(forcings, 0),
            "cross_ic_forcing_std_interval1": (
                cross_ic_forcing_std(forcings, 1) if num_intervals > 1 else None
            ),
            "mean_within_ic_interval_diff": sum(interval_diffs) / len(interval_diffs),
        }
    results["aggregates"] = agg

    # --- Report ---
    print("\n" + "=" * 70)
    print("AGGREGATES")
    print("=" * 70)
    header = (f"{'policy':<18} {'train loss':>11} {'held loss':>11} "
              f"{'train dSST':>11} {'held dSST':>11} {'xIC std i0':>11} "
              f"{'xIC std i1':>11} {'intvl diff':>11}")
    print(header)
    for name, a in agg.items():
        def fmt(v, plus=False):
            if v is None:
                return f"{'n/a':>11}"
            return f"{v:+11.4f}" if plus else f"{v:11.6f}"
        print(f"{name:<18} {fmt(a['train_mean_loss'])} "
              f"{fmt(a['heldout_mean_loss'])} "
              f"{fmt(a['train_mean_dsst'], plus=True)} "
              f"{fmt(a['heldout_mean_dsst'], plus=True)} "
              f"{fmt(a['cross_ic_forcing_std_interval0'])} "
              f"{fmt(a['cross_ic_forcing_std_interval1'])} "
              f"{fmt(a['mean_within_ic_interval_diff'])}")

    # --- Gates ---
    print("\n" + "=" * 70)
    print("SUCCESS GATES")
    print("=" * 70)
    s4 = agg["stage4-ensemble"]
    gates = {}

    held_dsst = s4["heldout_mean_dsst"]
    gates["1_heldout_cooling"] = (
        held_dsst is not None and -0.12 <= held_dsst <= -0.08
    )
    print(f"  Gate 1 (held-out dSST in [-0.12, -0.08] K): "
          f"{held_dsst if held_dsst is None else f'{held_dsst:+.4f}'} K "
          f"-> {'PASS' if gates['1_heldout_cooling'] else 'FAIL'}")

    if "stage1-static" in agg and s4["heldout_mean_loss"] is not None:
        s1_loss = agg["stage1-static"]["heldout_mean_loss"]
        gates["2_generalization"] = s4["heldout_mean_loss"] <= s1_loss
        print(f"  Gate 2 (stage4 held-out loss <= stage1-static): "
              f"{s4['heldout_mean_loss']:.6f} vs {s1_loss:.6f} "
              f"-> {'PASS' if gates['2_generalization'] else 'FAIL'}")
        if "stage3-single" in agg:
            print(f"    (stage3-single held-out loss: "
                  f"{agg['stage3-single']['heldout_mean_loss']:.6f})")
    else:
        gates["2_generalization"] = None
        print("  Gate 2 (generalization): SKIPPED (missing stage1 or held-out)")

    std_i1 = s4["cross_ic_forcing_std_interval1"]
    std_i0 = s4["cross_ic_forcing_std_interval0"]
    tol = 1e-6
    gates["3_state_dependence"] = (
        (std_i1 is None or std_i1 > tol) and std_i0 > tol
    )
    print(f"  Gate 3 (state-dependence, cross-IC forcing std): "
          f"interval0 = {std_i0:.2e}, "
          f"interval1 = {std_i1 if std_i1 is None else f'{std_i1:.2e}'} "
          f"-> {'PASS' if gates['3_state_dependence'] else 'FAIL'}")
    for name in ("stage3-single", "stage1-static"):
        if name in agg:
            print(f"    ({name}: interval0 std = "
                  f"{agg[name]['cross_ic_forcing_std_interval0']:.2e} — "
                  f"structurally 0 at interval 0 for 11-feature policies)")

    all_losses = [c["mean_loss"] for c in results["cells"].values()]
    gates["4_stability"] = all(jnp.isfinite(x) for x in all_losses)
    print(f"  Gate 4 (stability, all losses finite): "
          f"{'PASS' if gates['4_stability'] else 'FAIL'}")

    results["gates"] = gates

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
