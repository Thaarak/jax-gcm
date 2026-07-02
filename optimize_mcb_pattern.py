#!/usr/bin/env python
"""Stage 1: Direct static MCB pattern optimization (no neural network).

Optimizes the 96x48 MCB albedo perturbation field directly as parameters,
by gradient descent through the coupled atmosphere-ocean model. This is a
much easier inverse problem than policy learning and serves three purposes:

  1. De-risking: if this fails, the problem is in the simulator/loss,
     not in any network architecture.
  2. Standalone scientific result: the "optimal spatial MCB pattern" for a
     given cooling target.
  3. Warm-start values and a reference loss scale for Stage 3 policy training.

Loss (per Revised Roadmap; NO land/teleconnection terms on aquaplanet):
  - cooling:    (area_mean(dSST) - target)^2, dSST vs PAIRED no-MCB baseline
  - uniformity: area-weighted variance of dSST around its mean
  - reg:        mean(pattern^2)
  - smoothness: mean squared spatial gradients of the pattern

The paired baseline is a no-MCB run of the same length from the same initial
state, so natural model drift cancels exactly in dSST.

Parameterization: pattern = max_amplitude * sigmoid(theta) * ocean_mask,
which keeps the perturbation in (0, max_amplitude) smoothly.

Usage:
    python optimize_mcb_pattern.py [--days 60] [--iters 150] [--lr 0.05]
                                   [--target-cooling -0.1] [--max-amplitude 0.15]
                                   [--output-dir mcb_experiments/stage1]
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import optax

from jcm.mcb import create_ocean_mask
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.state_features import compute_area_weights

# Reuse the EXACT model setup used in training
from run_coupled_training import setup_coupled_model

WORKFLOW = ["coupling", "atm", "ocn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 direct MCB pattern optimization")
    parser.add_argument("--days", type=int, default=60,
                        help="Rollout length in days (pattern applied constantly)")
    parser.add_argument("--iters", type=int, default=150,
                        help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Adam learning rate (on sigmoid logits)")
    parser.add_argument("--target-cooling", type=float, default=-0.1,
                        help="Target global-mean dSST vs paired baseline (K)")
    parser.add_argument("--max-amplitude", type=float, default=0.15,
                        help="Maximum albedo perturbation (sigmoid ceiling)")
    parser.add_argument("--init-amplitude", type=float, default=0.02,
                        help="Initial uniform perturbation value")
    parser.add_argument("--w-cooling", type=float, default=1.0)
    parser.add_argument("--w-uniformity", type=float, default=0.1)
    parser.add_argument("--w-reg", type=float, default=0.001)
    parser.add_argument("--w-smooth", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="mcb_experiments/stage1")
    return parser.parse_args()


def run_interval_final_carry(carry, step_fn, num_steps):
    """Run the coupler for num_steps, returning only the final carry.

    Unlike run_coupled_interval, per-step predictions are discarded, so the
    scan does not stack a full trajectory of physics outputs (major memory
    saving under reverse-mode AD). Each daily step is checkpointed.
    """
    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        return new_c, None

    body = jax.checkpoint(body)
    final_carry, _ = jax.lax.scan(body, carry, jnp.arange(num_steps))
    return final_carry


def with_perturbation(carry, perturbation):
    """Return a copy of the coupled carry with mcb_perturbation set."""
    new_carry = jax.tree_util.tree_map(lambda x: x, carry)
    existing = new_carry["atm"]["derived"]["mcb_perturbation"]
    new_carry["atm"]["derived"]["mcb_perturbation"] = perturbation.astype(existing.dtype)
    return new_carry


def smoothness_penalty(pattern):
    """Mean squared spatial gradient (periodic in longitude)."""
    d_lon = pattern - jnp.roll(pattern, 1, axis=0)
    d_lat = pattern[:, 1:] - pattern[:, :-1]
    return jnp.mean(d_lon ** 2) + jnp.mean(d_lat ** 2)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 1: DIRECT STATIC MCB PATTERN OPTIMIZATION")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"days={args.days} iters={args.iters} lr={args.lr} "
          f"target={args.target_cooling} K max_amp={args.max_amplitude}")

    # --- Setup: identical to run_coupled_training.py ---
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )

    print("Initializing coupled carry...")
    initial_carry = coupler.initialize()

    ocean_mask = create_ocean_mask(coords.horizontal, terrain.fmask)
    area_weights = compute_area_weights(coords)
    nodal_shape = coords.horizontal.nodal_shape

    step_fn = create_coupled_step_fn(coupler, WORKFLOW, jitted=True)

    # --- Paired no-MCB baseline: same initial state, same length ---
    print(f"\nRunning paired no-MCB baseline ({args.days} days)...")
    t0 = time.time()
    zero = jnp.zeros(nodal_shape)
    baseline_final = run_interval_final_carry(
        with_perturbation(initial_carry, zero), step_fn, args.days
    )
    baseline_sst = jax.device_get(
        baseline_final["ocn"]["state"].sea_surface_temperature
    )
    baseline_sst = jnp.asarray(baseline_sst)
    baseline_mean = float(jnp.sum(baseline_sst * area_weights))
    print(f"  Baseline final global-mean SST: {baseline_mean:.4f} K "
          f"[{time.time() - t0:.1f}s]")

    # --- Parameterization ---
    def pattern_from_theta(theta):
        return args.max_amplitude * jax.nn.sigmoid(theta) * ocean_mask

    init_frac = jnp.clip(args.init_amplitude / args.max_amplitude, 1e-4, 1 - 1e-4)
    theta0 = jnp.full(nodal_shape, jnp.log(init_frac / (1.0 - init_frac)))

    # --- Loss ---
    def loss_fn(theta):
        pattern = pattern_from_theta(theta)
        final = run_interval_final_carry(
            with_perturbation(initial_carry, pattern), step_fn, args.days
        )
        sst = final["ocn"]["state"].sea_surface_temperature
        dsst = sst - baseline_sst

        mean_cooling = jnp.sum(dsst * area_weights)
        cooling = (mean_cooling - args.target_cooling) ** 2
        uniformity = jnp.sum(area_weights * (dsst - mean_cooling) ** 2)
        reg = jnp.mean(pattern ** 2)
        smooth = smoothness_penalty(pattern)

        total = (args.w_cooling * cooling
                 + args.w_uniformity * uniformity
                 + args.w_reg * reg
                 + args.w_smooth * smooth)
        aux = {
            "cooling": cooling,
            "uniformity": uniformity,
            "reg": reg,
            "smooth": smooth,
            "mean_cooling": mean_cooling,
        }
        return total, aux

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(theta0)

    @jax.jit
    def update(theta, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(theta)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        grad_norm = jnp.linalg.norm(grads.ravel())
        return theta, opt_state, loss, aux, grad_norm

    # --- Optimization loop ---
    print(f"\nOptimizing ({args.iters} iterations; first includes compilation)...")
    theta = theta0
    history = {
        "loss": [], "mean_cooling": [], "grad_norm": [],
        "cooling": [], "uniformity": [], "reg": [], "smooth": [],
    }
    best_loss = float("inf")
    best_theta = theta0
    t_start = time.time()

    for it in range(args.iters):
        t0 = time.time()
        theta, opt_state, loss, aux, grad_norm = update(theta, opt_state)
        loss = float(loss)
        history["loss"].append(loss)
        history["mean_cooling"].append(float(aux["mean_cooling"]))
        history["grad_norm"].append(float(grad_norm))
        for k in ("cooling", "uniformity", "reg", "smooth"):
            history[k].append(float(aux[k]))

        if loss < best_loss:
            best_loss = loss
            best_theta = theta

        if it % args.log_interval == 0:
            pattern = pattern_from_theta(theta)
            print(f"  iter {it:4d}  loss={loss:.6f}  "
                  f"dSST={float(aux['mean_cooling']):+.4f} K  "
                  f"cooling={float(aux['cooling']):.6f}  "
                  f"unif={float(aux['uniformity']):.6f}  "
                  f"|grad|={float(grad_norm):.2e}  "
                  f"pat[min/mean/max]={float(jnp.min(pattern)):.4f}/"
                  f"{float(jnp.mean(pattern)):.4f}/{float(jnp.max(pattern)):.4f}  "
                  f"[{time.time() - t0:.1f}s]")

    total_time = time.time() - t_start

    # --- Final evaluation with the best pattern ---
    best_pattern = pattern_from_theta(best_theta)
    final = run_interval_final_carry(
        with_perturbation(initial_carry, best_pattern), step_fn, args.days
    )
    sst = final["ocn"]["state"].sea_surface_temperature
    dsst = sst - baseline_sst
    mean_cooling = float(jnp.sum(dsst * area_weights))
    pattern_np = jax.device_get(best_pattern)

    print("\n" + "=" * 70)
    print("STAGE 1 RESULT")
    print("=" * 70)
    print(f"  Best loss:            {best_loss:.6f}")
    print(f"  Achieved dSST:        {mean_cooling:+.4f} K (target {args.target_cooling} K)")
    print(f"  Pattern min/mean/max: {pattern_np.min():.4f} / "
          f"{pattern_np.mean():.4f} / {pattern_np.max():.4f}")
    print(f"  Pattern spatial std:  {pattern_np.std():.4f} "
          f"({'STRUCTURED' if pattern_np.std() > 0.005 else 'near-uniform'})")
    print(f"  Total time:           {total_time:.1f}s "
          f"({total_time / max(args.iters, 1):.1f}s/iter)")

    # --- Save ---
    result = {
        "best_theta": jax.device_get(best_theta),
        "best_pattern": pattern_np,
        "best_loss": best_loss,
        "achieved_cooling": mean_cooling,
        "baseline_sst": jax.device_get(baseline_sst),
        "history": history,
        "config": vars(args),
        "total_time": total_time,
    }
    out_path = output_dir / "stage1_optimized_pattern.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved results to {out_path}")

    target_met = abs(mean_cooling - args.target_cooling) < 0.02
    structured = pattern_np.std() > 0.005
    print(f"\n  Success criteria: target cooling {'MET' if target_met else 'NOT MET'}, "
          f"pattern {'structured' if structured else 'NOT structured'}")


if __name__ == "__main__":
    main()
