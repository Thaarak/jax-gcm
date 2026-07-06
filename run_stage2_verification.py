#!/usr/bin/env python
"""Stage 2: Verify paired-baseline-trajectory features and rebalanced loss.

Checks the three Stage 2 invariants BEFORE Stage 3 policy training:

  Test 1 (paired cancellation): a zero-MCB rollout must match the
      precomputed baseline trajectory exactly, so paired dSST is ~0 and all
      anomaly features are ~0 (except the time-of-rollout feature). This
      proves drift cancels and the features/loss isolate the MCB signal.

  Test 2 (loss breakdown): with the paired baseline and the rebalanced
      aquaplanet weights (amazon/sahel/tropics = 0), the zero-MCB loss must
      be dominated by sst_cooling (= target^2), i.e. the loss attractor is
      now the cooling objective, not uncontrollable constants.

  Test 3 (gradient): BPTT through unroll_coupled_with_policy with the
      trajectory baseline, time feature, and checkpointed interval runner
      must produce finite, non-zero policy gradients.

Usage:
    python run_stage2_verification.py [--days 6] [--control-interval 3]
                                      [--skip-grad]
"""

import argparse
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
    extract_coupled_features,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import (
    create_coupled_step_fn,
    run_coupled_interval,
    verify_coupled_gradients,
)
from jcm.mcb.coupled_loss import compute_coupled_loss
from jcm.mcb.state_features import compute_area_weights

# Reuse the EXACT model setup used in training
from run_coupled_training import setup_coupled_model

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 verification")
    parser.add_argument("--days", type=int, default=6,
                        help="Total rollout length (days)")
    parser.add_argument("--control-interval", type=int, default=3,
                        help="Days per control interval (must divide --days)")
    parser.add_argument("--target-cooling", type=float, default=-0.1)
    parser.add_argument("--skip-grad", action="store_true", help="Skip Test 3")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance for Test 1 (paired cancellation). On CPU the "
                             "cancellation is bitwise, so the default is strict. On GPU, "
                             "the trajectory scan and the re-run are differently compiled "
                             "XLA programs, so cross-program bitwise equality is not "
                             "guaranteed; use e.g. --tol 1e-2 (K) there. The area-mean "
                             "dSST noise must still be < tol/100.")
    return parser.parse_args()


def print_loss_components(components, weights):
    rows = [
        ("sst_cooling", components.sst_cooling, weights.sst_cooling),
        ("sst_uniformity", components.sst_uniformity, weights.sst_uniformity),
        ("amazon", components.amazon, weights.amazon),
        ("sahel", components.sahel, weights.sahel),
        ("tropics", components.tropics, weights.tropics),
        ("regularization", components.regularization, weights.regularization),
        ("smoothness", components.smoothness, weights.smoothness),
    ]
    total = float(components.total)
    print(f"  {'component':<16} {'raw':>12} {'weight':>8} {'weighted':>12} {'% of total':>10}")
    for name, raw, w in rows:
        raw = float(raw)
        weighted = raw * w
        pct = 100.0 * weighted / total if total != 0 else 0.0
        print(f"  {name:<16} {raw:>12.6f} {w:>8.3f} {weighted:>12.6f} {pct:>9.1f}%")
    print(f"  {'TOTAL':<16} {'':>12} {'':>8} {total:>12.6f}")


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    print("=" * 70)
    print("STAGE 2 VERIFICATION: paired baseline trajectory features & loss")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"days={args.days} control_interval={args.control_interval} "
          f"target={args.target_cooling} K")

    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )

    print("Initializing coupled carry...")
    initial_carry = coupler.initialize()

    ocean_mask = create_ocean_mask(coords.horizontal, terrain.fmask)
    area_weights = compute_area_weights(coords)
    step_fn = create_coupled_step_fn(coupler, WORKFLOW, jitted=True)

    weights = CoupledLossWeights()  # Stage 2 aquaplanet defaults
    feature_config = CoupledFeatureConfig()  # includes time feature
    results = {}

    # ------------------------------------------------------------------
    # Baseline trajectory (paired no-MCB run)
    # ------------------------------------------------------------------
    print(f"\nComputing paired no-MCB baseline trajectory ({args.days} days)...")
    t0 = time.time()
    trajectory = compute_baseline_trajectory(
        initial_carry, step_fn, num_steps=args.days, coords=coords
    )
    print(f"  Trajectory shape (sst): {trajectory.sst.shape} "
          f"[{time.time() - t0:.1f}s]")

    # ------------------------------------------------------------------
    # Test 1: paired cancellation (zero-MCB rollout == baseline trajectory)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 1: paired cancellation (zero MCB -> dSST ~ 0, features ~ 0)")
    print("=" * 70)

    mid = args.days // 2
    t0 = time.time()
    carry_mid, _ = run_coupled_interval(initial_carry, step_fn, mid)
    carry_end, _ = run_coupled_interval(carry_mid, step_fn, args.days - mid)
    print(f"  Zero-MCB rollout re-run [{time.time() - t0:.1f}s]")

    sst_end = carry_end["ocn"]["state"].sea_surface_temperature
    dsst_field = sst_end - trajectory.at_step(args.days).sst
    max_dsst = float(jnp.max(jnp.abs(dsst_field)))
    mean_dsst = float(jnp.sum(dsst_field * area_weights))
    print(f"  max |dSST(t={args.days})| vs trajectory: {max_dsst:.3e} K")
    print(f"  area-mean dSST:                         {mean_dsst:+.3e} K")

    features_mid = extract_coupled_features(
        carry_mid, trajectory.at_step(mid), coords, feature_config,
        time_fraction=mid / args.days,
    )
    anomaly_feats = features_mid[:-1]  # last entry is the time feature
    time_feat = float(features_mid[-1])
    max_feat = float(jnp.max(jnp.abs(anomaly_feats)))
    print(f"  features at t={mid} (paired baseline): "
          f"max |anomaly| = {max_feat:.3e}, time = {time_feat:.3f}")

    passed = (max_dsst < args.tol
              and abs(mean_dsst) < args.tol / 100
              and max_feat < args.tol
              and abs(time_feat - mid / args.days) < 1e-9)
    results["test1_paired_cancellation"] = passed
    print(f"\n  TEST 1 VERDICT: {'PASS - drift cancels exactly; features isolate MCB signal' if passed else 'FAIL - paired baseline does not cancel (nondeterminism or indexing bug)'}")

    # ------------------------------------------------------------------
    # Test 2: loss breakdown with paired baseline + rebalanced weights
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TEST 2: loss breakdown (zero MCB, paired baseline, target {args.target_cooling} K)")
    print("=" * 70)

    loss_baseline = trajectory.at_step(args.days)
    components = compute_coupled_loss(
        coupled_carry=carry_end,
        baseline_sst=loss_baseline.sst,
        baseline_precip=loss_baseline.precipitation,
        target_cooling=args.target_cooling,
        mcb_forcing=jnp.zeros(coords.horizontal.nodal_shape),
        coords=coords,
        weights=weights,
        return_components=True,
    )
    print_loss_components(components, weights)

    total = float(components.total)
    cooling_frac = weights.sst_cooling * float(components.sst_cooling) / total
    expected_cooling = args.target_cooling ** 2
    print(f"\n  sst_cooling fraction of total: {100 * cooling_frac:.1f}% "
          f"(raw {float(components.sst_cooling):.6f}, expected ~{expected_cooling:.6f} = target^2)")

    passed = (cooling_frac > 0.95
              and abs(float(components.sst_cooling) - expected_cooling) < 1e-6
              and float(components.amazon) == 0.0
              and float(components.sahel) == 0.0)
    results["test2_loss_breakdown"] = passed
    print(f"\n  TEST 2 VERDICT: {'PASS - sst_cooling dominates; land terms exactly zero' if passed else 'FAIL - loss still contains uncontrollable terms'}")

    # ------------------------------------------------------------------
    # Test 3: gradient through the full Stage 2 unroll
    # ------------------------------------------------------------------
    if not args.skip_grad:
        print("\n" + "=" * 70)
        print(f"TEST 3: policy gradient through {args.days}-day unroll "
              f"({args.days // args.control_interval} intervals)")
        print("=" * 70)

        policy = MCBPolicyMLP(
            output_shape=coords.horizontal.nodal_shape,
            hidden_dims=(16, 16),
            max_perturbation=0.15,
        )
        feature_dim = get_coupled_feature_dim(feature_config)
        params = policy.init(jax.random.PRNGKey(0), jnp.zeros(feature_dim))
        print(f"  Feature dim: {feature_dim} (includes time feature)")

        config = CoupledControllerConfig(
            control_interval_steps=args.control_interval,
            total_steps=args.days,
            target_cooling=args.target_cooling,
            loss_weights=weights,
            feature_config=feature_config,
        )

        t0 = time.time()
        grad_results = verify_coupled_gradients(
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
        print(f"  loss = {grad_results['loss']:.6f}  "
              f"|grad| = {grad_results['gradient_norm']:.3e}  "
              f"NaNs = {grad_results['has_nans']}  "
              f"all-zero = {grad_results['all_zeros']}  "
              f"[{time.time() - t0:.1f}s]")

        passed = (not grad_results["has_nans"]
                  and not grad_results["all_zeros"]
                  and grad_results["gradient_norm"] > 0.0)
        results["test3_gradient"] = passed
        print(f"\n  TEST 3 VERDICT: {'PASS - finite non-zero gradients reach the policy' if passed else 'FAIL - gradients broken'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2 SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    if all(results.values()):
        print("\n  GATE OPEN: proceed to Stage 3 (NN policy training, warm-started).")
    else:
        print("\n  GATE CLOSED: fix features/loss before policy training.")


if __name__ == "__main__":
    main()
