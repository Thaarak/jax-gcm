#!/usr/bin/env python
"""Stage 3 evaluation: trained coupled policy vs static Stage 1 pattern.

Runs the paired 60-day rollout with (a) the trained NN policy checkpoint and
(b) the warm-start params (= static Stage 1 pattern), and reports:

  - total/mean loss for each
  - paired global SST change at day 60 (target -0.1 K)
  - MCB forcing stats (mean/max, spatial std)
  - whether the policy output is state-dependent (interval 1 vs interval 2
    forcing difference; a static pattern gives ~0)

Usage:
    python run_stage3_eval.py --checkpoint mcb_experiments/stage3_60d/coupled_trained_policy.pkl \
                              --warm-start mcb_experiments/stage1/stage1_optimized_pattern.pkl
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.mcb import (
    CoupledControllerConfig,
    CoupledFeatureConfig,
    CoupledLossWeights,
    MCBPolicyMLP,
    compute_baseline_trajectory,
    create_ocean_mask,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    evaluate_coupled_policy,
)
from jcm.mcb.train import load_checkpoint

from run_coupled_training import setup_coupled_model, warm_start_params

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default="mcb_experiments/stage3_60d/coupled_trained_policy.pkl")
    parser.add_argument("--warm-start", type=str,
                        default="mcb_experiments/stage1/stage1_optimized_pattern.pkl")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--control-interval", type=int, default=30)
    parser.add_argument("--target-cooling", type=float, default=-0.1)
    parser.add_argument("--output", type=str,
                        default="mcb_experiments/stage3_60d/eval_results.pkl")
    return parser.parse_args()


def report(name, result):
    m = result["metrics"]
    forcing = result["trajectory"].mcb_forcing  # (num_intervals, lon, lat)
    spatial_std = float(jnp.mean(jnp.std(forcing, axis=(1, 2))))
    interval_diff = float(jnp.max(jnp.abs(forcing[1] - forcing[0]))) if forcing.shape[0] > 1 else 0.0
    print(f"\n  [{name}]")
    print(f"    total_loss        = {m['total_loss']:.6f}")
    print(f"    mean_loss         = {m['mean_loss']:.6f}")
    print(f"    final paired dSST = {m['final_sst_change']:+.4f} K "
          f"(target {m['target_cooling']:+.2f} K)")
    print(f"    mcb forcing mean/max = {m['mean_mcb_forcing']:.4f} / {m['max_mcb_forcing']:.4f}")
    print(f"    spatial std       = {spatial_std:.4f}")
    print(f"    max |interval2 - interval1| forcing = {interval_diff:.6f}")
    print(f"    per-interval loss = {[f'{float(x):.6f}' for x in m['loss_trajectory']]}")
    return {
        "metrics": m,
        "spatial_std": spatial_std,
        "interval_diff": interval_diff,
        "mcb_forcing": jax.device_get(forcing),
    }


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    print("=" * 70)
    print("STAGE 3 EVALUATION: trained policy vs static Stage 1 pattern")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )
    initial_carry = coupler.initialize()
    ocean_mask = create_ocean_mask(coords.horizontal, terrain.fmask)
    step_fn = create_coupled_step_fn(coupler, WORKFLOW, jitted=True)

    print(f"\nComputing paired baseline trajectory ({args.days} days)...")
    t0 = time.time()
    trajectory = compute_baseline_trajectory(
        initial_carry, step_fn, num_steps=args.days, coords=coords
    )
    print(f"  done [{time.time() - t0:.1f}s]")

    feature_config = CoupledFeatureConfig()
    feature_dim = get_coupled_feature_dim(feature_config)
    policy = MCBPolicyMLP(
        output_shape=coords.horizontal.nodal_shape,
        hidden_dims=(256, 256),
        max_perturbation=0.15,
    )
    config = CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.days,
        target_cooling=args.target_cooling,
        loss_weights=CoupledLossWeights(
            sst_cooling=1.0, sst_uniformity=0.1,
            amazon=0.0, sahel=0.0, tropics=0.0,
            regularization=0.001, smoothness=0.001,
        ),
        feature_config=feature_config,
    )

    trained_params, meta = load_checkpoint(args.checkpoint)
    warm_params = warm_start_params(policy, feature_dim, args.warm_start)

    results = {}
    for name, params in [("trained", trained_params), ("stage1-static", warm_params)]:
        print(f"\nEvaluating {name} policy ({args.days}-day rollout)...")
        t0 = time.time()
        result = evaluate_coupled_policy(
            coupler=coupler,
            workflow=WORKFLOW,
            policy_fn=policy.apply,
            policy_params=params,
            initial_carry=initial_carry,
            baseline_trajectory=trajectory,
            coords=coords,
            ocean_mask=ocean_mask,
            config=config,
        )
        print(f"  [{time.time() - t0:.1f}s]")
        results[name] = report(name, result)

    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved evaluation results to {args.output}")

    dt = results["trained"]["metrics"]
    ds = results["stage1-static"]["metrics"]
    print("\n" + "=" * 70)
    print("SUMMARY (trained vs stage1-static)")
    print("=" * 70)
    print(f"  loss:  {dt['mean_loss']:.6f} vs {ds['mean_loss']:.6f} "
          f"({100 * (1 - dt['mean_loss'] / ds['mean_loss']):+.1f}% better)")
    print(f"  dSST:  {dt['final_sst_change']:+.4f} K vs {ds['final_sst_change']:+.4f} K "
          f"(target {args.target_cooling:+.2f} K)")
    print(f"  state-dependence (interval forcing diff): "
          f"{results['trained']['interval_diff']:.6f} vs "
          f"{results['stage1-static']['interval_diff']:.6f}")


if __name__ == "__main__":
    main()
