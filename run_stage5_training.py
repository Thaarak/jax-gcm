#!/usr/bin/env python
"""Stage 5 training: realistic terrain + reinstated teleconnection penalties.

Moves the varied-IC ensemble MCB controller (Stage 4 machinery) from the
aquaplanet onto REALISTIC EARTH TERRAIN (T30 climatology) and reinstates the
amazon / sahel / tropics precipitation-protection penalties that were
structurally zero on the aquaplanet.

Reuses the Stage 4 ensemble training loop unchanged. The deltas are:
  - realistic-terrain setup (orography + land physics + ocean land mask),
  - ocean-masked loss and features (land cells pinned to a fixed temperature
    otherwise corrupt the global-mean SST objective and the policy features),
  - teleconnection loss weights (amazon = sahel = tropics = 0.05),
  - warm start directly from the Stage 4 13-feature policy (SAME feature dim,
    so NO expand_policy_input is needed).

The ocean mask (binary: fmask > 0.95 is land, matching the slab ocean model's
land-sea convention) is threaded end-to-end by the ensemble trainer
(create_coupled_grad_fn / create_coupled_eval_fn already take it), and the
controller passes it into compute_coupled_loss / extract_coupled_features.

Usage:
    python run_stage5_training.py --epochs 150 \
        --init-checkpoint mcb_experiments/stage4/stage4_trained_policy.pkl \
        --ic-dir mcb_experiments/stage5/ics --output-dir mcb_experiments/stage5
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt

from jcm.mcb import (
    CoupledBaselineTrajectory,
    CoupledControllerConfig,
    CoupledFeatureConfig,
    CoupledLossWeights,
    MCBPolicyMLP,
    get_coupled_feature_dim,
    load_carry,
)
from jcm.mcb.coupled_train import (
    create_coupled_grad_fn,
    ocean_fmask_from_coupler,
    ocean_mask_from_coupler,
    train_coupled_policy_ensemble,
)
from jcm.mcb.train import TrainingConfig, load_checkpoint, save_checkpoint

from run_coupled_training import (
    coupler_workflow,
    setup_coupled_model,
    warm_start_params,
)

WORKFLOW = ["coupling", "atm", "ocn"]
START_DATE = "2000-01-01"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 ensemble training")
    parser.add_argument("--ic-dir", type=str,
                        default="mcb_experiments/stage5/ics",
                        help="Directory with manifest.json + IC pickles "
                             "(must be realistic-terrain ICs)")
    parser.add_argument("--output-dir", type=str,
                        default="mcb_experiments/stage5")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--days", type=int, default=60,
                        help="Rollout horizon per IC (days)")
    parser.add_argument("--control-interval", type=int, default=30,
                        help="Days between policy applications")
    parser.add_argument("--target-cooling", type=float, default=-0.1)
    parser.add_argument("--heldout-interval", type=int, default=10,
                        help="Epochs between held-out evaluations")
    parser.add_argument("--init-checkpoint", type=str,
                        default="mcb_experiments/stage4/stage4_trained_policy.pkl",
                        help="Stage 4 checkpoint (13 features) to warm-start "
                             "from directly — same feature dim, no expand")
    parser.add_argument("--warm-start-stage1", type=str, default=None,
                        help="Fallback: Stage 1 pattern pickle; sets the "
                             "output bias to the optimized pattern instead "
                             "of using --init-checkpoint")
    parser.add_argument("--allow-random-init", action="store_true",
                        default=False,
                        help="Permit random initialization when neither "
                             "--warm-start-stage1 nor an existing "
                             "--init-checkpoint is provided. Without this "
                             "flag a missing warm-start is a hard error, so a "
                             "mistyped/misplaced checkpoint cannot silently "
                             "train a cold-started policy and save it under a "
                             "warm-start filename.")
    parser.add_argument("--no-realistic-terrain", dest="realistic_terrain",
                        action="store_false", default=True,
                        help="Fall back to the aquaplanet (debugging only).")
    # Teleconnection weights (reinstated on realistic terrain; CLI-tunable).
    parser.add_argument("--sst-cooling", type=float, default=1.0)
    parser.add_argument("--sst-uniformity", type=float, default=0.1)
    parser.add_argument("--amazon", type=float, default=0.05)
    parser.add_argument("--sahel", type=float, default=0.05)
    parser.add_argument("--tropics", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--smoothness", type=float, default=0.001)
    parser.add_argument("--feature-mode", choices=["full", "time-only"],
                        default="full",
                        help="'full' = 13 climate-state features. 'time-only' "
                             "= only the normalized-time feature (OPEN-LOOP "
                             "ablation: a policy that cannot see the climate "
                             "state, per PREREGISTRATION.md sec 6; use with "
                             "--allow-random-init, no warm start).")
    parser.add_argument("--max-perturbation", type=float, default=0.15,
                        help="Max albedo perturbation (forcing cap); "
                             "lower to reduce overcooling (G2).")
    parser.add_argument("--loss-mode", choices=["summed", "terminal_dsst"],
                        default="summed",
                        help="'summed' = historical interval-sum loss "
                             "(<0.2%% gate-relevant, rewards overcooling). "
                             "'terminal_dsst' = gate-aligned final-step dSST "
                             "error, the quantity the pre-registered gate "
                             "scores.")
    parser.add_argument("--forcing-reg-weight", type=float, default=0.001,
                        help="Mean-square forcing penalty in terminal_dsst "
                             "mode.")
    parser.add_argument("--select-on-heldout", action="store_true",
                        help="Gate model selection + early stopping on mean "
                             "held-out loss (evaluated every epoch) instead of "
                             "train loss (finishes P0.3).")
    return parser.parse_args()


def load_baseline_trajectory(path, days):
    """Load a pickled baseline dict as a CoupledBaselineTrajectory."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    assert d["sst"].shape[0] >= days + 1, (
        f"Baseline {path} covers {d['sst'].shape[0] - 1} days, need {days}"
    )
    return CoupledBaselineTrajectory(
        sst=jnp.asarray(d["sst"][:days + 1]),
        surface_temperature=jnp.asarray(d["surface_temperature"][:days + 1]),
        precipitation=jnp.asarray(d["precipitation"][:days + 1]),
        heat_flux=jnp.asarray(d["heat_flux"][:days + 1]),
    )


def load_ics(ic_dir, days, template_carry, require_realistic_terrain=True):
    """Load manifest, carries and baselines; return (manifest, train, heldout).

    Asserts the manifest was generated under realistic terrain (Stage 5), so
    aquaplanet Stage 4 carries cannot be silently reused.
    """
    ic_dir = Path(ic_dir)
    with open(ic_dir / "manifest.json") as f:
        manifest = json.load(f)

    assert manifest["start_date"] == START_DATE, (
        f"Manifest start_date {manifest['start_date']} != {START_DATE}"
    )
    assert manifest["horizon"] >= days, (
        f"Manifest horizon {manifest['horizon']} < requested days {days}"
    )
    if require_realistic_terrain:
        assert manifest.get("realistic_terrain", False), (
            f"IC manifest at {ic_dir} was NOT generated with realistic "
            f"terrain (terrain_source={manifest.get('terrain_source')}). "
            f"Regenerate with run_stage5_generate_ics.py --realistic-terrain."
        )

    train, heldout = [], []
    for entry in manifest["ics"]:
        carry = load_carry(str(ic_dir / entry["carry_file"]), template_carry)
        baseline = load_baseline_trajectory(
            ic_dir / entry["baseline_file"], days
        )
        target = train if entry["split"] == "train" else heldout
        target.append((entry, carry, baseline))

    return manifest, train, heldout


def main():
    args = parse_args()
    assert args.days % args.control_interval == 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STAGE 5 TRAINING: realistic terrain + teleconnection penalties")
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
    # bmask (the exact land classification that pins SST to 288.15 K). Sourced
    # from the coupler, NOT terrain.fmask, whose independent interpolation
    # disagrees with the ocean grid at ~56 T30 coastal cells. train_ocean_fmask
    # feeds the trainer the matching fractional mask so its internal
    # binarization agrees cell-for-cell.
    ocean_mask = ocean_mask_from_coupler(coupler)
    train_ocean_fmask = ocean_fmask_from_coupler(coupler)
    # Derive the workflow from the coupler so the slab LAND step is included
    # over realistic terrain (["coupling","atm","ocn","lnd"]) — a hardcoded
    # 3-element workflow would leave the land component present but never
    # stepped, silently defeating prognostic land (R7a).
    workflow = coupler_workflow(coupler)
    print(f"  Coupler workflow: {workflow}")
    orog_max = float(jnp.max(jnp.abs(atm_model.truncated_orography)))
    print(f"  |truncated_orography|_max = {orog_max:.3e} "
          f"(land fraction {float(jnp.mean(terrain.fmask)):.3f})")
    if args.realistic_terrain:
        assert orog_max > 0.0, "Orography not reaching dynamics."

    print(f"\nLoading ICs from {args.ic_dir}...")
    manifest, train_ics, heldout_ics = load_ics(
        args.ic_dir, args.days, template_carry,
        require_realistic_terrain=args.realistic_terrain,
    )
    print(f"  Terrain source: {manifest.get('terrain_source')}")
    print(f"  {len(train_ics)} train ICs "
          f"(spin-up days {[e['spinup_days'] for e, _, _ in train_ics]})")
    print(f"  {len(heldout_ics)} held-out ICs "
          f"(spin-up days {[e['spinup_days'] for e, _, _ in heldout_ics]})")

    train_carries = [c for _, c, _ in train_ics]
    train_baselines = [b for _, _, b in train_ics]
    heldout_carries = [c for _, c, _ in heldout_ics]
    heldout_baselines = [b for _, _, b in heldout_ics]

    # 13-feature config (same as Stage 4): absolute-SST features distinguish
    # ICs at interval 0. Warm start loads with the SAME dim -> no expand.
    # time-only = open-loop ablation (policy sees only the normalized time).
    if args.feature_mode == "time-only":
        feature_config = CoupledFeatureConfig(
            include_sst=False, include_sst_regions=False,
            include_heat_flux=False, include_atm_temperature=False,
            include_precipitation=False, include_time=True,
            include_absolute_sst=False)
    else:
        feature_config = CoupledFeatureConfig(include_absolute_sst=True)
    feature_dim = get_coupled_feature_dim(feature_config)
    print(f"\nFeature dim: {feature_dim} (mode={args.feature_mode}, "
          f"ocean-masked on realistic terrain)")

    policy = MCBPolicyMLP(
        output_shape=coords.horizontal.nodal_shape,
        hidden_dims=(256, 256),
        max_perturbation=args.max_perturbation,
    )

    loss_weights = CoupledLossWeights(
        sst_cooling=args.sst_cooling,
        sst_uniformity=args.sst_uniformity,
        amazon=args.amazon,
        sahel=args.sahel,
        tropics=args.tropics,
        regularization=args.regularization,
        smoothness=args.smoothness,
    )
    print(f"  Loss weights: {loss_weights}")

    controller_config = CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.days,
        target_cooling=args.target_cooling,
        loss_weights=loss_weights,
        feature_config=feature_config,
        max_perturbation=args.max_perturbation,
        use_checkpointing=True,
        loss_mode=args.loss_mode,
        forcing_reg_weight=args.forcing_reg_weight,
    )
    print(f"  Loss mode: {args.loss_mode}"
          + (f" (forcing_reg={args.forcing_reg_weight})"
             if args.loss_mode == "terminal_dsst" else ""))

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_schedule='constant',
        optimizer='adam',
        grad_clip_norm=1.0,
        log_interval=1,
        early_stopping_patience=20,
    )

    # Warm start: Stage 4 13-feature checkpoint loaded directly (same feature
    # dim -> no expand_policy_input), or the Stage 1 pattern (bias trick).
    initial_params = None
    init_source = "random"
    if args.warm_start_stage1:
        initial_params = warm_start_params(
            policy, feature_dim, args.warm_start_stage1
        )
        init_source = f"stage1:{args.warm_start_stage1}"
    elif Path(args.init_checkpoint).exists():
        ckpt_params, ckpt_meta = load_checkpoint(args.init_checkpoint)
        old_dim = ckpt_params['params']['hidden_0']['kernel'].shape[0]
        assert old_dim == feature_dim, (
            f"Stage 4 checkpoint input dim {old_dim} != Stage 5 feature dim "
            f"{feature_dim}; Stage 5 reuses the 13-feature layout, so no "
            f"expand should be needed. Check the checkpoint."
        )
        initial_params = ckpt_params
        init_source = f"stage4:{args.init_checkpoint}"
        print(f"  Warm-started directly from Stage 4 policy "
              f"(dim {feature_dim}, no expand)")
    elif args.allow_random_init:
        print(f"  WARNING: init checkpoint {args.init_checkpoint} not found; "
              f"using random initialization (--allow-random-init set)")
    else:
        raise SystemExit(
            f"ERROR: no warm-start available: --warm-start-stage1 not given "
            f"and --init-checkpoint '{args.init_checkpoint}' does not exist. "
            f"Refusing to silently train a random-initialized policy and save "
            f"it as a warm-started artifact. Pass a valid checkpoint, or "
            f"--allow-random-init to run cold-start intentionally. "
            f"(Note the GPU artifacts live under mcb_experiments_gpu/, not "
            f"mcb_experiments/.)"
        )

    # Pre-flight gradient sanity check on IC 0 (ocean-masked loss/features)
    print("\nPre-flight gradient check on IC 0...")
    grad_fn = create_coupled_grad_fn(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        coords=coords,
        ocean_mask=ocean_mask,
        controller_config=controller_config,
    )
    check_params = initial_params
    if check_params is None:
        check_params = policy.init(
            jax.random.PRNGKey(training_config.random_seed),
            jnp.zeros(feature_dim),
        )
    loss, grads = grad_fn(check_params, train_carries[0], train_baselines[0])
    grad_leaves = jax.tree.leaves(grads)
    grad_norm = float(jnp.linalg.norm(
        jnp.concatenate([g.ravel() for g in grad_leaves])
    ))
    has_nans = any(bool(jnp.any(jnp.isnan(g))) for g in grad_leaves)
    all_zeros = all(bool(jnp.allclose(g, 0.0)) for g in grad_leaves)
    print(f"  loss = {float(loss):.6f}, grad norm = {grad_norm:.4f}")
    if has_nans or all_zeros or not jnp.isfinite(loss):
        print(f"  FAILED: has_nans={has_nans}, all_zeros={all_zeros}, "
              f"loss_finite={bool(jnp.isfinite(loss))}")
        return

    print("\n" + "=" * 70)
    print("Starting ensemble training...")
    print("=" * 70)

    best_params, history = train_coupled_policy_ensemble(
        coupler=coupler,
        workflow=workflow,
        policy=policy,
        coords=coords,
        terrain_fmask=train_ocean_fmask,
        train_carries=train_carries,
        train_baselines=train_baselines,
        heldout_carries=heldout_carries,
        heldout_baselines=heldout_baselines,
        training_config=training_config,
        controller_config=controller_config,
        initial_params=initial_params,
        heldout_interval=args.heldout_interval,
        select_on_heldout=args.select_on_heldout,
    )

    print("\nSaving results...")
    checkpoint_path = output_dir / "stage5_trained_policy.pkl"
    save_checkpoint(
        best_params,
        str(checkpoint_path),
        metadata={
            'epochs': history['epochs_completed'],
            'best_loss': history['best_loss'],
            'target_cooling': args.target_cooling,
            'mode': 'coupled-ensemble-realistic',
            'feature_dim': feature_dim,
            'include_absolute_sst': True,
            'realistic_terrain': bool(args.realistic_terrain),
            'terrain_source': manifest.get('terrain_source'),
            'loss_weights': loss_weights._asdict(),
            'init_source': init_source,
            'train_ics': [e['spinup_days'] for e, _, _ in train_ics],
            'heldout_ics': [e['spinup_days'] for e, _, _ in heldout_ics],
            'days': args.days,
            'control_interval': args.control_interval,
        }
    )

    history_path = output_dir / "stage5_training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Best mean loss: {history['best_loss']:.6f}")
    print(f"  Final mean loss: {history['final_loss']:.6f}")
    print(f"  Epochs: {history['epochs_completed']}")
    if history['heldout_loss_history']:
        epoch, losses = history['heldout_loss_history'][-1]
        print(f"  Last held-out eval (epoch {epoch}): "
              f"mean {sum(losses) / len(losses):.6f} {losses}")
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
