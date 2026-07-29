#!/usr/bin/env python
"""Decisive artifact-vs-real probe for the MCB feedback-controller negative.

The v2 "feedback doesn't beat static" verdict was produced by a training run
that never converged (flat loss, train-loss selection, early-stop at 22/80).
Adversarial review found two candidate explanations that the existing logs
CANNOT distinguish, because the pipeline only ever logged ||mean_i g_i|| (blind
to destructive cancellation):

  ARTIFACT  the per-IC gradients are large but conflict, so the ensemble mean
            cancels the shared signal; and/or the summed objective is <0.2%
            gate-relevant and rewards overcooling. A gate-aligned objective +
            conflict-aware training would then let feedback win.
  REAL      every per-IC gradient is itself ~0 at the static warm-start: no
            direction beats static, i.e. feedback genuinely cannot help.

This probe measures, at the exact static warm-start, the quantities that settle
it (per PREREGISTRATION.md analysis appendix):

  1. per-IC gradient norms ||g_i||, ensemble-mean norm ||mean g||, and the
     coherence ratio  C = ||mean g|| / mean_i ||g_i||  and mean pairwise cosine,
     for BOTH loss_mode="summed" (historical) and "terminal_dsst" (gate-aligned);
  2. HEADROOM: step the warm-start along the (train-averaged) gate-aligned
     gradient at several step sizes and measure held-out final-dSST gate error
     vs the static pattern's own held-out gate error.

Decision rule:
  * terminal mean||g_i|| ~ O(1) with C ~ 0.01-0.05  -> destructive cancellation
    -> ARTIFACT -> conflict-aware retrain warranted.
  * terminal mean||g_i|| ~ 0  -> flat optimum -> REAL negative, no retrain.
  * headroom: a step that drives held-out gate error below static's -> a
    gate-improving direction that GENERALIZES exists -> retrain warranted.

Runs forward/backward only; no optimizer loop. ~20-30 min on the GB10.
"""
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np

from jcm.mcb import (
    CoupledControllerConfig,
    CoupledFeatureConfig,
    MCBPolicyMLP,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_train import (
    create_coupled_grad_fn,
    ocean_mask_from_coupler,
)
from jcm.mcb.coupled_controller import evaluate_coupled_policy
from run_coupled_training import (
    coupler_workflow,
    setup_coupled_model,
    warm_start_params,
)
from run_stage5_training import START_DATE, load_ics


def parse_args():
    p = argparse.ArgumentParser(description="MCB gradient-coherence probe")
    p.add_argument("--ic-dir", type=str, required=True)
    p.add_argument("--warm-start-stage1", type=str, required=True,
                   help="stage1 pattern .pkl (the static warm-start / comparator)")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--control-interval", type=int, default=15)
    p.add_argument("--target-cooling", type=float, default=-0.1)
    p.add_argument("--max-perturbation", type=float, default=0.09)
    p.add_argument("--forcing-reg-weight", type=float, default=0.001)
    p.add_argument("--step-sizes", type=float, nargs="+",
                   default=[0.003, 0.01, 0.03, 0.1])
    p.add_argument("--output", type=str, default="mcb_experiments_gpu/gradient_probe.pkl")
    return p.parse_args()


def flat(grads):
    """Flatten a param-pytree of gradients to a single numpy vector."""
    return np.concatenate([np.asarray(g).ravel() for g in jax.tree.leaves(grads)])


def coherence_stats(grad_vecs):
    """||mean||, mean||g_i||, C, mean pairwise cosine for a list of flat grads."""
    G = np.stack(grad_vecs)                       # (N, P)
    norms = np.linalg.norm(G, axis=1)             # (N,)
    mean_vec = G.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    mean_ic_norm = float(norms.mean())
    C = mean_norm / (mean_ic_norm + 1e-12)
    # mean pairwise cosine
    unit = G / (norms[:, None] + 1e-12)
    sim = unit @ unit.T
    n = len(grad_vecs)
    iu = np.triu_indices(n, k=1)
    mean_cos = float(sim[iu].mean()) if n > 1 else float("nan")
    return {
        "mean_ensemble_norm": mean_norm,
        "mean_per_ic_norm": mean_ic_norm,
        "per_ic_norms": norms.tolist(),
        "coherence_C": C,
        "mean_pairwise_cosine": mean_cos,
    }


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0
    print("=" * 70)
    print("MCB GRADIENT-COHERENCE / HEADROOM PROBE")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    coupler, coords, terrain, atm = setup_coupled_model(
        jdt.to_datetime(START_DATE), jdt.to_timedelta(1, "day"),
        realistic_terrain=True,
    )
    template = coupler.initialize()
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)
    print(f"  workflow: {workflow}")

    _, train_ics, heldout_ics = load_ics(args.ic_dir, args.days, template)
    train_carries = [c for _, c, _ in train_ics]
    train_baselines = [b for _, _, b in train_ics]
    heldout_carries = [c for _, c, _ in heldout_ics]
    heldout_baselines = [b for _, _, b in heldout_ics]
    print(f"  {len(train_carries)} train ICs, {len(heldout_carries)} held-out ICs")

    feature_config = CoupledFeatureConfig(include_absolute_sst=True)
    feature_dim = get_coupled_feature_dim(feature_config)
    policy = MCBPolicyMLP(
        output_shape=coords.horizontal.nodal_shape,
        hidden_dims=(256, 256),
        max_perturbation=args.max_perturbation,
    )
    warm_params = warm_start_params(policy, feature_dim, args.warm_start_stage1)
    print(f"  warm-started (== static pattern) from {args.warm_start_stage1}")

    def make_config(loss_mode):
        return CoupledControllerConfig(
            control_interval_steps=args.control_interval,
            total_steps=args.days,
            target_cooling=args.target_cooling,
            feature_config=feature_config,
            max_perturbation=args.max_perturbation,
            use_checkpointing=True,
            loss_mode=loss_mode,
            forcing_reg_weight=args.forcing_reg_weight,
        )

    results = {"config": vars(args)}

    # --- 1. Coherence under both objectives, on the TRAIN ICs (what training
    #        actually averages over) --------------------------------------
    grad_trees = {}   # keep the tree-structured train-mean grad for the step test
    for mode in ("summed", "terminal_dsst"):
        cfg = make_config(mode)
        grad_fn = create_coupled_grad_fn(
            coupler=coupler, workflow=workflow, policy_fn=policy.apply,
            coords=coords, ocean_mask=ocean_mask, controller_config=cfg,
        )
        vecs, losses, grad_sum = [], [], None
        for i, (c, b) in enumerate(zip(train_carries, train_baselines)):
            loss, grads = grad_fn(warm_params, c, b)
            vecs.append(flat(grads))
            losses.append(float(loss))
            grad_sum = grads if grad_sum is None else jax.tree_util.tree_map(
                jnp.add, grad_sum, grads)
        mean_grad = jax.tree_util.tree_map(lambda g: g / len(vecs), grad_sum)
        grad_trees[mode] = mean_grad
        stats = coherence_stats(vecs)
        stats["mean_loss"] = float(np.mean(losses))
        results[f"coherence_{mode}"] = stats
        print(f"\n[{mode}] mean_loss={stats['mean_loss']:.6f}")
        print(f"  mean per-IC ||g_i|| = {stats['mean_per_ic_norm']:.4f}")
        print(f"  ||mean g||          = {stats['mean_ensemble_norm']:.4f}")
        print(f"  coherence C         = {stats['coherence_C']:.4f}")
        print(f"  mean pairwise cos   = {stats['mean_pairwise_cosine']:.4f}")

    # --- 2. Headroom: static (warm-start) held-out gate error, then step
    #        along the gate-aligned train-mean gradient -------------------
    gate_cfg = make_config("terminal_dsst")

    def heldout_gate_errors(params):
        errs = []
        for c, b in zip(heldout_carries, heldout_baselines):
            m = evaluate_coupled_policy(
                coupler=coupler, workflow=workflow, policy_fn=policy.apply,
                policy_params=params, initial_carry=c, baseline_trajectory=b,
                coords=coords, ocean_mask=ocean_mask, config=gate_cfg,
            )["metrics"]
            errs.append(abs(m["final_sst_change"] - args.target_cooling))
        return np.array(errs)

    static_err = heldout_gate_errors(warm_params)
    print(f"\nSTATIC held-out gate |dSST-target|: mean={static_err.mean():.5f} "
          f"(per-IC dSST err {np.round(static_err, 4).tolist()})")
    results["static_heldout_gate_err"] = static_err.tolist()

    mean_grad = grad_trees["terminal_dsst"]
    step_results = {}
    best = (None, static_err.mean())
    for eta in args.step_sizes:
        stepped = jax.tree_util.tree_map(
            lambda p, g: p - eta * g, warm_params, mean_grad)
        errs = heldout_gate_errors(stepped)
        step_results[eta] = errs.tolist()
        improved = errs.mean() < static_err.mean()
        print(f"  step eta={eta:<6}: held-out gate err mean={errs.mean():.5f} "
              f"{'(IMPROVES static)' if improved else ''}")
        if errs.mean() < best[1]:
            best = (eta, errs.mean())
    results["stepped_heldout_gate_err"] = step_results
    results["best_step"] = {"eta": best[0], "mean_err": best[1]}

    # --- 3. Verdict ----------------------------------------------------
    t = results["coherence_terminal_dsst"]
    print("\n" + "=" * 70)
    print("PROBE VERDICT")
    print("=" * 70)
    if t["mean_per_ic_norm"] < 0.05:
        print("  Per-IC gradients are individually ~0 at the static warm-start")
        print("  => FLAT OPTIMUM: the v2 negative looks REAL (feedback cannot help).")
    elif t["coherence_C"] < 0.2:
        print(f"  Large per-IC gradients (mean ||g_i||={t['mean_per_ic_norm']:.3f}) "
              f"but low coherence (C={t['coherence_C']:.3f})")
        print("  => DESTRUCTIVE CANCELLATION: the naive ensemble mean cannot learn.")
        print("     The v2 negative is a TRAINING ARTIFACT; conflict-aware retrain warranted.")
    else:
        print(f"  Coherent gate-aligned gradient (C={t['coherence_C']:.3f}); the")
        print("  summed objective, not conflict, was the problem. Retrain warranted.")
    if best[0] is not None:
        print(f"  HEADROOM: a gate-aligned step (eta={best[0]}) drives held-out gate")
        print(f"  error {static_err.mean():.5f} -> {best[1]:.5f} (BELOW static)")
        print("  => a gate-improving, GENERALIZING direction exists. Retrain warranted.")
    else:
        print(f"  HEADROOM: no tested step beat static ({static_err.mean():.5f}).")
        print("  Static may already be near-optimal on this metric (little headroom).")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved probe results to {args.output}")


if __name__ == "__main__":
    main()
