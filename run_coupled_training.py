#!/usr/bin/env python
"""Train MCB policy with JAX-ESM coupled atmosphere-ocean simulation.

This script demonstrates how to train an MCB neural network controller
using the coupled JAX-ESM framework, where the ocean responds to MCB
forcing and provides feedback to the atmosphere.

Usage:
    python run_coupled_training.py [--epochs 50] [--target-cooling -0.3]

The key advantage over atmosphere-only training is that:
- MCB reduces shortwave absorption → less heat flux to ocean
- Ocean cools → SST decreases
- Atmosphere responds to SST → global cooling

This is the physically correct mechanism for MCB-induced cooling.
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax_datetime as jdt

# JCM imports
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

# MCB imports
from jcm.mcb import (
    MCBPolicyMLP,
    CoupledControllerConfig,
    CoupledFeatureConfig,
    CoupledLossWeights,
    compute_baseline_trajectory,
    get_coupled_feature_dim,
)
from jcm.mcb.coupled_controller import create_coupled_step_fn
from jcm.mcb.coupled_train import (
    train_coupled_policy,
    validate_coupled_training_setup,
)
from jcm.mcb.train import TrainingConfig, save_checkpoint

# JAX-ESM imports
from jem.base.coupler import Coupler
from jem.components.JCM import make_jem_compatible
from jem.components.slab.slab_ocean_model.slab_ocean_model import SlabOceanModel
from jem.mapping.mapper import BasicMapper


def parse_args():
    parser = argparse.ArgumentParser(description="Coupled MCB Policy Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--target-cooling", type=float, default=-0.1, help="Target SST cooling (K)")
    parser.add_argument("--control-interval", type=int, default=30, help="Days between policy applications")
    parser.add_argument("--total-days", type=int, default=180, help="Total simulation days per epoch")
    parser.add_argument("--output-dir", type=str, default="mcb_experiments", help="Output directory")
    parser.add_argument("--validate-only", action="store_true", help="Only validate setup, don't train")
    return parser.parse_args()


def setup_coupled_model(start_datetime, coupling_timestep):
    """Set up the coupled atmosphere-ocean model."""
    print("Setting up coupled model...")

    # Get coordinates and terrain
    coords = get_speedy_coords()
    terrain = TerrainData.aquaplanet(coords)

    # Create atmosphere model (no static MCB - will be dynamic from policy)
    atm_model = jcm.model.Model(
        start_date=start_datetime,
        coords=coords,
    )

    # Make JEM-compatible
    atm_model = make_jem_compatible(atm_model, coupling_timestep)

    # Create slab ocean model (timestep needs to be in seconds as float)
    timestep_seconds = 86400.0  # 1 day in seconds
    ocn_model = SlabOceanModel(
        start_datetime=start_datetime,
        timestep=timestep_seconds,
    )

    # Create mapper for atmosphere-ocean coupling
    mapper = BasicMapper()
    mapper.add_mapping(
        source=("atm", "derived.total_heat_flux"),
        target=("ocn", "forcing.total_heat_flux"),
    )
    mapper.add_mapping(
        source=("ocn", "state.sea_surface_temperature"),
        target=("atm", "forcing.sea_surface_temperature"),
    )

    # Create coupler
    coupler = Coupler(
        components={"atm": atm_model, "ocn": ocn_model},
        mappers={"coupling": mapper},
    )

    return coupler, coords, terrain, atm_model


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup
    start_datetime = jdt.to_datetime("2000-01-01")
    coupling_timestep = jdt.to_timedelta(1, "day")

    # Create coupled model
    coupler, coords, terrain, atm_model = setup_coupled_model(
        start_datetime, coupling_timestep
    )

    # Initialize coupler
    print("Initializing coupled simulation...")
    initial_carry = coupler.initialize()

    # Paired no-MCB baseline trajectory (Stage 2): one forward run of the
    # full training window from the SAME initial state. Features and loss are
    # computed as X(t) - X_baseline(t), so natural drift cancels exactly.
    workflow = ["coupling", "atm", "ocn"]
    print(f"Computing paired no-MCB baseline trajectory ({args.total_days} days)...")
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)
    baseline_trajectory = compute_baseline_trajectory(
        initial_carry, step_fn, num_steps=args.total_days, coords=coords
    )
    baseline_path = output_dir / f"baseline_trajectory_{args.total_days}d.pkl"
    with open(baseline_path, 'wb') as f:
        pickle.dump(jax.device_get({
            'sst': baseline_trajectory.sst,
            'surface_temperature': baseline_trajectory.surface_temperature,
            'precipitation': baseline_trajectory.precipitation,
            'heat_flux': baseline_trajectory.heat_flux,
        }), f)
    print(f"  Saved baseline trajectory to {baseline_path}")

    # Configure feature extraction (include_time gives the policy
    # schedule-dependence and a non-zero input at the first interval)
    feature_config = CoupledFeatureConfig(
        include_sst=True,
        include_sst_regions=True,
        include_heat_flux=True,
        include_atm_temperature=True,
        include_precipitation=True,
        include_time=True,
    )

    # Create policy network
    print("Creating policy network...")
    feature_dim = get_coupled_feature_dim(feature_config)
    output_shape = coords.horizontal.nodal_shape
    policy = MCBPolicyMLP(
        output_shape=output_shape,
        hidden_dims=(256, 256),
        max_perturbation=0.15,
    )
    print(f"  Input features: {feature_dim}")
    print(f"  Output shape: {output_shape}")

    # Controller configuration. Loss weights are the Stage 2 aquaplanet set:
    # no land teleconnection terms (amazon/sahel/tropics = 0 — no land
    # exists), and the Stage 1-proven weights so sst_cooling dominates.
    controller_config = CoupledControllerConfig(
        control_interval_steps=args.control_interval,
        total_steps=args.total_days,
        target_cooling=args.target_cooling,
        loss_weights=CoupledLossWeights(
            sst_cooling=1.0,
            sst_uniformity=0.1,
            amazon=0.0,
            sahel=0.0,
            tropics=0.0,
            regularization=0.001,
            smoothness=0.001,
        ),
        feature_config=feature_config,
        max_perturbation=0.15,
        use_checkpointing=True,
    )

    # Training configuration
    training_config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_schedule='constant',
        optimizer='adam',
        grad_clip_norm=1.0,
        log_interval=1,
        early_stopping_patience=20,
    )

    # Validate setup
    print("\n" + "=" * 60)
    print("Validating training setup...")
    print("=" * 60)
    validation = validate_coupled_training_setup(
        coupler=coupler,
        workflow=workflow,
        policy=policy,
        coords=coords,
        terrain_fmask=terrain.fmask,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        controller_config=controller_config,
    )

    if not validation['valid']:
        print(f"\nValidation FAILED: {validation.get('error', 'Unknown error')}")
        return

    if args.validate_only:
        print("\nValidation successful. Exiting (--validate-only specified).")
        return

    # Train
    print("\n" + "=" * 60)
    print("Starting coupled MCB policy training...")
    print("=" * 60)
    print(f"  Target cooling: {args.target_cooling} K")
    print(f"  Control interval: {args.control_interval} days")
    print(f"  Total simulation: {args.total_days} days")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print()

    best_params, history = train_coupled_policy(
        coupler=coupler,
        workflow=workflow,
        policy=policy,
        coords=coords,
        terrain_fmask=terrain.fmask,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        training_config=training_config,
        controller_config=controller_config,
    )

    # Save results
    print("\nSaving results...")

    # Save checkpoint
    checkpoint_path = output_dir / "coupled_trained_policy.pkl"
    save_checkpoint(
        best_params,
        str(checkpoint_path),
        metadata={
            'epochs': history['epochs_completed'],
            'best_loss': history['best_loss'],
            'target_cooling': args.target_cooling,
            'mode': 'coupled',
        }
    )

    # Save training history
    history_path = output_dir / "coupled_training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best loss: {history['best_loss']:.6f}")
    print(f"  Final loss: {history['final_loss']:.6f}")
    print(f"  Total time: {history['total_time']:.1f}s")
    print(f"  Epochs: {history['epochs_completed']}")
    print()
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
