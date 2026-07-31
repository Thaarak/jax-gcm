"""P1 noise-floor harness — measure how much of the loss/dSST is NOISE.

The project has never measured its own noise floor, yet every stage-level
conclusion (Stage 3 "beats static", the Stage 4/5 gate verdicts, the 2026-07-09
"Option A refuted") rests on differences of a few percent, and the doc itself
admits "~15% loss variation across XLA compilations". This harness runs a FIXED
policy (the Stage-1 static pattern, which cannot learn or generalize) and
separates the two noise sources every downstream comparison is fighting:

  * sigma_compile : run-to-run spread under IDENTICAL inputs, forcing a fresh
    XLA compilation each rep (jax.clear_caches). This is chaos amplification x
    float non-associativity. On GPU it was ~15%; on CPU it is often ~0 because
    reductions are deterministic — so the DECISIVE float32-vs-x64 comparison of
    sigma_compile belongs on the GPU (run there with and without --x64).

  * sigma_IC : spread across initial conditions under the SAME fixed pattern.
    This is intrinsic per-IC climate sensitivity — hardware-independent, and
    the thing the fixed [-0.12, -0.08] K Gate-2 band conflates with policy
    quality.

Decision rules (printed against the measured floor):
  * A training run has LEARNED only if its loss improvement exceeds 2*sigma_compile.
  * A gate margin is REPORTABLE only if it exceeds 2 * s.e. of the compared means.
  * The Gate-2 cooling band is 0.04 K wide; if sigma_IC(dSST) is comparable, an
    n=2 held-out verdict is a coin flip.

Usage (CPU smoke, short rollout):
    python run_noise_floor.py --ic-dir mcb_experiments_gpu/stage5/ics \
        --days 6 --control-interval 3 --reps 3 --num-ics 2

Usage (authoritative, on the GPU box — run twice, with and without --x64):
    python run_noise_floor.py --ic-dir mcb_experiments_gpu/stage5/ics \
        --days 60 --control-interval 30 --reps 10 --output nf_f32.pkl
    python run_noise_floor.py ... --x64 --output nf_x64.pkl
"""

import os
import sys

# x64 must be enabled before JAX initializes, so honor the flag from argv here.
if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import jax
import jax_datetime as jdt

from jcm.mcb import (
    CoupledFeatureConfig,
    CoupledLossWeights,
    MCBPolicyMLP,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import (
    CoupledControllerConfig,
    create_coupled_step_fn,
    evaluate_coupled_policy,
)
from jcm.mcb.coupled_features import compute_baseline_trajectory
from jcm.mcb.coupled_train import ocean_mask_from_coupler

from run_coupled_training import (
    coupler_workflow,
    setup_coupled_model,
    warm_start_params,
)
from run_stage5_training import START_DATE, load_ics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic-dir", default="mcb_experiments_gpu/stage5/ics",
                   help="IC directory (manifest.json + carry/baseline pkls)")
    p.add_argument("--stage1", default="mcb_experiments_gpu/stage1/"
                   "stage1_optimized_pattern.pkl",
                   help="Fixed static pattern to hold constant across reps/ICs")
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--control-interval", type=int, default=30)
    p.add_argument("--reps", type=int, default=10,
                   help="Recompiled repeats per IC (isolates sigma_compile)")
    p.add_argument("--num-ics", type=int, default=None,
                   help="Cap number of ICs (default: all)")
    p.add_argument("--regen-baseline", action="store_true",
                   help="Recompute each IC's paired no-MCB baseline with the "
                        "CURRENT code instead of loading the cached pkl. "
                        "Required after any physics fix: cached baselines were "
                        "generated with the old code, so a paired dSST against "
                        "them conflates the MCB effect with the code change.")
    p.add_argument("--x64", action="store_true",
                   help="Enable float64 (env set at import time). Compare "
                        "sigma_compile with/without this ON GPU to settle P0.4.")
    p.add_argument("--cross-process", action="store_true",
                   help="Run each rep in a FRESH PYTHON PROCESS instead of "
                        "jax.clear_caches() in-process. The 2026-07-29 "
                        "meta-audit showed in-process recompiled reps are "
                        "bit-identical (sigma_compile=0 by construction) "
                        "while cross-process runs of the same computation "
                        "differ by ~0.014-0.017 K/IC (XLA autotuning x 60-day "
                        "chaos). THIS mode measures the real per-run noise.")
    p.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                   action="store_false", default=True)
    p.add_argument("--target-cooling", type=float, default=-0.1)
    p.add_argument("--output", default=None)
    p.add_argument("--max-perturbation", type=float, default=0.15,
                   help="Max albedo perturbation (forcing cap). "
                        "Lower to reduce overcooling (G2).")
    return p.parse_args()


def run_cross_process(args):
    """Spawn each rep as a fresh Python process and merge the results.

    Each worker computes ONE rep for every IC in its own process (own XLA
    compilation/autotuning), which is the noise mode the in-process
    clear_caches harness structurally cannot see.
    """
    import subprocess
    import tempfile

    print("=" * 72)
    print(f"CROSS-PROCESS NOISE FLOOR: {args.reps} worker processes")
    print("=" * 72)
    base_cmd = [sys.executable, os.path.abspath(__file__)]
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a == "--cross-process":
            continue
        if a in ("--reps", "--output"):
            skip_next = True
            continue
        base_cmd.append(a)
    merged = {}
    with tempfile.TemporaryDirectory(prefix="nf_workers_") as tmp:
        for k in range(args.reps):
            wout = str(Path(tmp) / f"worker_{k}.pkl")
            cmd = base_cmd + ["--reps", "1", "--worker-output", wout]
            print(f"\n[worker {k + 1}/{args.reps}] {' '.join(cmd)}",
                  flush=True)
            t0 = time.time()
            subprocess.run(cmd, check=True)
            with open(wout, "rb") as f:
                worker_per_ic = pickle.load(f)["per_ic"]
            for ic, blob in worker_per_ic.items():
                if ic not in merged:
                    merged[ic] = {"entry": blob["entry"], "reps": []}
                merged[ic]["reps"].extend(blob["reps"])
            print(f"[worker {k + 1}] done in {time.time() - t0:.0f}s")
    return merged


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    if args.cross_process and args.worker_output is None:
        per_ic = run_cross_process(args)
        report_and_save(per_ic, args, noise_label="sigma_run(cross-process)")
        return

    print("=" * 72)
    print("P1 NOISE-FLOOR HARNESS  (fixed Stage-1 static pattern)")
    print("=" * 72)
    print(f"JAX devices: {jax.devices()}")
    print(f"jax_enable_x64 = {jax.config.jax_enable_x64}  "
          f"(--x64 {'ON' if args.x64 else 'off'})")
    print(f"rollout: {args.days} d / {args.control_interval} d intervals | "
          f"reps={args.reps}")

    start_datetime = jdt.to_datetime(START_DATE)
    coupling_timestep = jdt.to_timedelta(1, "day")
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep,
        realistic_terrain=args.realistic_terrain,
    )
    template_carry = coupler.initialize()
    ocean_mask = ocean_mask_from_coupler(coupler)
    workflow = coupler_workflow(coupler)  # include lnd step over terrain (R7a)

    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template_carry,
        require_realistic_terrain=args.realistic_terrain,
    )
    all_ics = train_ics + heldout_ics
    if args.num_ics is not None:
        all_ics = all_ics[:args.num_ics]
    print(f"  {len(all_ics)} ICs from {args.ic_dir}")

    if args.regen_baseline:
        print("  Regenerating paired baselines with CURRENT code "
              "(cached baselines are stale after physics fixes)...")
        step_fn = create_coupled_step_fn(coupler, workflow)
        regen = []
        for entry, carry, _ in all_ics:
            b = compute_baseline_trajectory(
                carry, step_fn, num_steps=args.days, coords=coords)
            regen.append((entry, carry, b))
        all_ics = regen

    # Fixed static pattern: warm_start_params sets output bias = optimized theta
    # and zeroes the kernel, so the policy output is the Stage-1 pattern
    # regardless of features — a genuinely fixed forcing field.
    fc = CoupledFeatureConfig()  # 11-feature layout matches the Stage-1 pattern
    if not Path(args.stage1).exists():
        raise SystemExit(f"Stage-1 pattern not found: {args.stage1}")
    policy = MCBPolicyMLP(output_shape=coords.horizontal.nodal_shape,
                          hidden_dims=(256, 256), max_perturbation=args.max_perturbation)
    params = warm_start_params(policy, get_coupled_feature_dim(fc), args.stage1)

    config = CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.days,
        target_cooling=args.target_cooling,
        loss_weights=CoupledLossWeights(
            sst_cooling=1.0, sst_uniformity=0.1,
            amazon=0.05, sahel=0.05, tropics=0.05,
            regularization=0.001, smoothness=0.001,
        ),
        feature_config=fc,
        max_perturbation=args.max_perturbation,
        use_checkpointing=True,
    )

    def run_once(carry, baseline):
        # Force a fresh XLA compilation so repeated calls expose compile-level
        # non-determinism (the GPU noise source), not a cached result.
        jax.clear_caches()
        m = evaluate_coupled_policy(
            coupler=coupler, workflow=workflow, policy_fn=policy.apply,
            policy_params=params, initial_carry=carry,
            baseline_trajectory=baseline, coords=coords,
            ocean_mask=ocean_mask, config=config,
        )["metrics"]
        return {
            "dsst": float(m["final_sst_change"]),
            "loss": float(m["mean_loss"]),
            "max_forcing": float(m["max_mcb_forcing"]),
        }

    # (IC index) -> list of per-rep dicts
    per_ic = {}
    for entry, carry, baseline in all_ics:
        ic = entry["index"]
        reps = []
        for k in range(args.reps):
            t0 = time.time()
            r = run_once(carry, baseline)
            r["wall_s"] = time.time() - t0
            reps.append(r)
            print(f"  IC {ic:>2} ({entry['split']:>7}, day {entry['spinup_days']:>3}) "
                  f"rep {k+1}/{args.reps}: dSST={r['dsst']:+.5f} K  "
                  f"loss={r['loss']:.6f}  ({r['wall_s']:.1f}s)")
        per_ic[ic] = {"entry": entry, "reps": reps}

    if args.worker_output is not None:
        with open(args.worker_output, "wb") as f:
            pickle.dump({"per_ic": per_ic}, f)
        print(f"[worker] wrote {args.worker_output}")
        return

    report_and_save(per_ic, args, noise_label="sigma_compile(in-process)")


def report_and_save(per_ic, args, noise_label):
    """Compute and print the noise decomposition; save the pickle.

    noise_label distinguishes what the per-IC rep spread measures:
    in-process recompiled reps (old mode; bit-identical on GPU, so ~0 by
    construction) vs cross-process runs (the real per-run chaos noise).
    """
    # --- Statistics ---
    def col(ic, key):
        return np.array([r[key] for r in per_ic[ic]["reps"]], dtype=float)

    print("\n" + "=" * 72)
    print(f"{noise_label.upper()}  (spread across reps, per IC)")
    print("=" * 72)
    print(f"  {'IC':>3} {'split':>7} {'dSST mean':>11} {'dSST std':>10} "
          f"{'loss mean':>11} {'loss std':>11}")
    compile_dsst_stds, compile_loss_stds = [], []
    for ic in per_ic:
        d, lo = col(ic, "dsst"), col(ic, "loss")
        compile_dsst_stds.append(d.std())
        compile_loss_stds.append(lo.std())
        print(f"  {ic:>3} {per_ic[ic]['entry']['split']:>7} "
              f"{d.mean():>+11.5f} {d.std():>10.2e} "
              f"{lo.mean():>11.6f} {lo.std():>11.2e}")
    sig_comp_dsst = float(np.mean(compile_dsst_stds))
    sig_comp_loss = float(np.mean(compile_loss_stds))

    ic_means_dsst = np.array([col(ic, "dsst").mean() for ic in per_ic])
    ic_means_loss = np.array([col(ic, "loss").mean() for ic in per_ic])
    sig_ic_dsst = float(ic_means_dsst.std())
    sig_ic_loss = float(ic_means_loss.std())

    print("\n" + "=" * 72)
    print("SIGMA_IC  (spread across ICs of the per-IC mean, FIXED pattern)")
    print("=" * 72)
    print(f"  dSST : mean {ic_means_dsst.mean():+.5f} K   sigma_IC = {sig_ic_dsst:.4f} K"
          f"   range [{ic_means_dsst.min():+.4f}, {ic_means_dsst.max():+.4f}]")
    print(f"  loss : mean {ic_means_loss.mean():.6f}     sigma_IC = {sig_ic_loss:.6f}")

    n = len(per_ic)
    se_dsst = sig_ic_dsst / np.sqrt(max(n, 1))
    band = 0.04  # Gate-2 cooling band width [-0.12, -0.08] K
    print("\n" + "=" * 72)
    print("DECISION-RULE READOUT")
    print("=" * 72)
    print(f"  sigma_compile(loss) ~ {sig_comp_loss:.2e}  "
          f"({100 * sig_comp_loss / max(ic_means_loss.mean(), 1e-12):.1f}% of mean loss)")
    print(f"    -> a training run has LEARNED only if its loss improvement "
          f"exceeds 2*sigma = {2 * sig_comp_loss:.2e}")
    print(f"  sigma_IC(dSST) = {sig_ic_dsst:.4f} K  vs Gate-2 band width {band:.2f} K")
    print(f"    -> per-IC dSST scatter is {100 * sig_ic_dsst / band:.0f}% of the "
          f"whole gate band under a FIXED pattern")
    print(f"    -> s.e. of an n={n} mean dSST = {se_dsst:.4f} K; a gate margin is "
          f"only reportable if it exceeds 2*s.e. = {2 * se_dsst:.4f} K")
    if jax.config.jax_enable_x64:
        print("  NOTE: x64 ON. Compare sigma_compile(loss) to the float32 run "
              "(same command without --x64) to settle P0.4.")
    else:
        print("  NOTE: float32. On CPU sigma_compile is often ~0 (deterministic "
              "reductions); the decisive x64 comparison must run on the GPU.")

    out = args.output or str(Path(args.ic_dir).parent /
                             f"noise_floor_{'x64' if args.x64 else 'f32'}.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "config": vars(args),
            "x64": bool(jax.config.jax_enable_x64),
            "noise_label": noise_label,
            "per_ic": per_ic,
            # Key names kept for compatibility; what the rep-spread MEASURES
            # depends on the mode — see noise_label (in-process recompiles
            # are bit-identical on GPU, cross-process is the real per-run
            # chaos noise; 2026-07-29 meta-audit).
            "sigma_compile_dsst": sig_comp_dsst,
            "sigma_compile_loss": sig_comp_loss,
            "sigma_run_dsst": sig_comp_dsst,
            "sigma_run_loss": sig_comp_loss,
            "sigma_ic_dsst": sig_ic_dsst,
            "sigma_ic_loss": sig_ic_loss,
        }, f)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
