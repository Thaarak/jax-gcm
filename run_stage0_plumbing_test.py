#!/usr/bin/env python
"""Stage 0: Verify MCB injection plumbing and gradient flow (coupled training path).

This diagnostic answers three questions BEFORE any further training is attempted:

  Test 1 (plumbing): Does a large constant mcb_perturbation, injected through the
      TRAINING code path (carry["atm"]["derived"]["mcb_perturbation"] ->
      forcing.mcb_perturbation inside the JEM step function), actually cool the
      ocean?

      History: the original path used set_mcb_perturbation() attribute mutation,
      which was silently ignored because jcm/model.py run_from_state is @jax.jit
      with static self ("if model fields assumed to be static are changed, the
      changes will not be picked up here"). This test FAILED (exactly 0.0 K),
      after which the perturbation was rethreaded through ForcingData (a traced
      argument), and the test PASSED.

  Test 2 (gradient): Is d(global_mean_SST)/d(uniform_mcb_scalar) significantly
      non-zero and negative? If plumbing works, more MCB must mean more cooling.

  Test 3 (loss breakdown): Which components dominate the training loss (5.33 in
      the failed run)? Reproduces the training loss config to confirm the
      hypothesis that ~99.8% of the loss is constant w.r.t. the policy.

Usage:
    python run_stage0_plumbing_test.py [--days 60] [--grad-days 10] [--amplitude 0.1]
                                       [--fd-check] [--skip-grad] [--skip-forward]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.mcb import (
    CoupledBaseline,
    CoupledLossWeights,
    create_ocean_mask,
)
from jcm.mcb.coupled_loss import compute_coupled_loss
from jcm.mcb.coupled_controller import create_coupled_step_fn, run_coupled_interval
from jcm.mcb.state_features import compute_area_weights

# Reuse the EXACT model setup used in training
from run_coupled_training import setup_coupled_model

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 0 MCB plumbing test")
    parser.add_argument("--days", type=int, default=60,
                        help="Forward run length for Test 1 (days)")
    parser.add_argument("--grad-days", type=int, default=10,
                        help="Rollout length for the gradient check (days)")
    parser.add_argument("--amplitude", type=float, default=0.1,
                        help="Constant MCB albedo perturbation for Test 1")
    parser.add_argument("--fd-check", action="store_true",
                        help="Also run a finite-difference cross-check of the gradient")
    parser.add_argument("--skip-grad", action="store_true", help="Skip Test 2")
    parser.add_argument("--skip-forward", action="store_true", help="Skip Test 1")
    return parser.parse_args()


def with_perturbation(carry: dict, perturbation: jnp.ndarray) -> dict:
    """Return a copy of the coupled carry with mcb_perturbation set.

    Uses tree_map identity to rebuild containers so the original carry is not
    mutated. Matches the dtype of the existing leaf to preserve pytree types.
    """
    new_carry = jax.tree_util.tree_map(lambda x: x, carry)
    existing = new_carry["atm"]["derived"]["mcb_perturbation"]
    new_carry["atm"]["derived"]["mcb_perturbation"] = perturbation.astype(existing.dtype)
    return new_carry


def global_mean_sst(carry: dict, area_weights: jnp.ndarray) -> jnp.ndarray:
    sst = carry["ocn"]["state"].sea_surface_temperature
    return jnp.sum(sst * area_weights)


def print_loss_components(components, weights: CoupledLossWeights):
    """Print raw and weighted loss components with % of total."""
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
        print(f"  {name:<16} {raw:>12.6f} {w:>8.2f} {weighted:>12.6f} {pct:>9.1f}%")
    print(f"  {'TOTAL':<16} {'':>12} {'':>8} {total:>12.6f}")


def main():
    args = parse_args()

    print("=" * 70)
    print("STAGE 0: MCB INJECTION PLUMBING TEST (training code path)")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    # --- Setup: identical to run_coupled_training.py ---
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )

    print("Initializing coupled carry (this also compiles the 1-step init run)...")
    initial_carry = coupler.initialize()

    ocean_mask = create_ocean_mask(coords.horizontal, terrain.fmask)
    area_weights = compute_area_weights(coords)
    nodal_shape = coords.horizontal.nodal_shape

    print(f"Grid: {nodal_shape}, ocean fraction: {float(jnp.mean(ocean_mask)):.3f}")

    # Same step function factory used by the training controller
    step_fn = create_coupled_step_fn(coupler, WORKFLOW, jitted=True)

    sst_initial = float(global_mean_sst(initial_carry, area_weights))
    print(f"Initial global-mean SST: {sst_initial:.4f} K")

    results = {}

    # ------------------------------------------------------------------
    # Test 1: forward plumbing test (zero vs large constant perturbation)
    # ------------------------------------------------------------------
    final_base = final_mcb = None
    if not args.skip_forward:
        print("\n" + "=" * 70)
        print(f"TEST 1: {args.days}-day forward runs, perturbation 0.0 vs {args.amplitude}")
        print("=" * 70)

        zero_pert = jnp.zeros(nodal_shape)
        mcb_pert = args.amplitude * ocean_mask

        t0 = time.time()
        carry_base = with_perturbation(initial_carry, zero_pert)
        final_base, _ = run_coupled_interval(carry_base, step_fn, args.days)
        sst_base = float(global_mean_sst(final_base, area_weights))
        print(f"  [baseline]  final SST = {sst_base:.4f} K "
              f"(dSST = {sst_base - sst_initial:+.4f} K)  [{time.time() - t0:.1f}s]")

        t0 = time.time()
        carry_mcb = with_perturbation(initial_carry, mcb_pert)
        final_mcb, _ = run_coupled_interval(carry_mcb, step_fn, args.days)
        sst_mcb = float(global_mean_sst(final_mcb, area_weights))
        print(f"  [MCB {args.amplitude:.2f}]  final SST = {sst_mcb:.4f} K "
              f"(dSST = {sst_mcb - sst_initial:+.4f} K)  [{time.time() - t0:.1f}s]")

        effect = sst_mcb - sst_base
        print(f"\n  MCB effect (MCB - baseline): {effect:+.4f} K over {args.days} days")
        print("  Reference: static MCBConfig path gave -0.33 K / 90 days with 0.10 "
              "in stratocumulus regions only.")
        print(f"  Expectation here (0.10 over ALL ocean): substantially larger cooling.")

        passed = effect < -0.01
        results["test1_plumbing"] = passed
        print(f"\n  TEST 1 VERDICT: {'PASS - perturbation reaches physics' if passed else 'FAIL - perturbation is being IGNORED (silent zero)'}")

    # ------------------------------------------------------------------
    # Test 2: gradient check d(SST)/d(uniform scalar)
    # ------------------------------------------------------------------
    if not args.skip_grad:
        print("\n" + "=" * 70)
        print(f"TEST 2: gradient check over {args.grad_days} days")
        print("=" * 70)

        def sst_after(scalar):
            carry = with_perturbation(initial_carry, scalar * ocean_mask)
            final, _ = run_coupled_interval(carry, step_fn, args.grad_days)
            return global_mean_sst(final, area_weights)

        eval_point = args.amplitude / 2.0
        t0 = time.time()
        grad = float(jax.grad(sst_after)(jnp.asarray(eval_point)))
        print(f"  d(global_SST)/d(mcb_scalar) at {eval_point:.3f} = {grad:+.6f} K per unit albedo "
              f"[{time.time() - t0:.1f}s]")

        if args.fd_check:
            eps = 0.01
            t0 = time.time()
            hi = float(sst_after(jnp.asarray(eval_point + eps)))
            lo = float(sst_after(jnp.asarray(eval_point - eps)))
            fd = (hi - lo) / (2 * eps)
            print(f"  Finite-difference estimate: {fd:+.6f} K per unit albedo "
                  f"[{time.time() - t0:.1f}s]")
            rel_err = abs(grad - fd) / max(abs(fd), 1e-12)
            print(f"  Relative error AD vs FD: {rel_err:.3f}")

        passed = grad < -1e-3
        results["test2_gradient"] = passed
        print(f"\n  TEST 2 VERDICT: {'PASS - non-trivial negative gradient (more MCB -> cooler)' if passed else 'FAIL - gradient is zero or wrong sign'}")

    # ------------------------------------------------------------------
    # Test 3: loss component breakdown (reproduces training loss config)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 3: loss component breakdown (training config: target=-0.1 K)")
    print("=" * 70)

    baseline = CoupledBaseline.from_coupled_carry(initial_carry, coords)
    weights = CoupledLossWeights(
        sst_cooling=1.0, sst_uniformity=0.3, amazon=1.0, sahel=0.5,
        tropics=0.3, regularization=0.01, smoothness=0.01,
    )

    cases = [("initial carry (features source)", initial_carry, jnp.zeros(nodal_shape))]
    if final_base is not None:
        cases.append((f"after {args.days}d, no MCB", final_base, jnp.zeros(nodal_shape)))
    if final_mcb is not None:
        cases.append((f"after {args.days}d, MCB {args.amplitude}", final_mcb,
                      args.amplitude * ocean_mask))

    for label, carry, forcing in cases:
        print(f"\n  --- {label} ---")
        components = compute_coupled_loss(
            coupled_carry=carry,
            baseline_sst=baseline.sst,
            baseline_precip=baseline.precipitation,
            target_cooling=-0.1,
            mcb_forcing=forcing,
            coords=coords,
            weights=weights,
            return_components=True,
        )
        print_loss_components(components, weights)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 0 SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    if results and all(results.values()):
        print("\n  GATE OPEN: proceed to Stage 1 (direct pattern optimization).")
    elif results:
        print("\n  GATE CLOSED: fix injection plumbing before any training.")
        print("  Check that mcb_perturbation flows through ForcingData (traced arg)")
        print("  and is applied in jcm/physics/speedy/forcing.py set_forcing.")


if __name__ == "__main__":
    main()
