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

import flax
import jax
import jax.numpy as jnp
import jax_datetime as jdt

# JCM imports
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData

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
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to Stage 1 pickle (stage1_optimized_pattern.pkl); "
                             "initializes the policy output bias to best_theta and "
                             "zeros the output kernel so the initial policy output "
                             "equals the Stage 1 optimized pattern")
    return parser.parse_args()


def warm_start_params(policy, feature_dim, stage1_path):
    """Build initial policy params warm-started from the Stage 1 pattern.

    The MLP output layer produces logits that go through
    max_perturbation * sigmoid(logits) — the same parameterization Stage 1
    used (0.15 * sigmoid(theta)). Setting the output bias to theta and the
    output kernel to zero makes the initial policy output exactly the
    Stage 1 optimized pattern, regardless of input features. Gradients
    still flow to the kernel (hidden activations are non-zero), so the
    network can learn state-dependence from there.
    """
    with open(stage1_path, 'rb') as f:
        stage1 = pickle.load(f)
    theta = jnp.asarray(stage1['best_theta'])

    params = policy.init(jax.random.PRNGKey(0), jnp.zeros(feature_dim))
    params = jax.tree_util.tree_map(lambda x: x, params)  # ensure mutable copy
    if isinstance(params, flax.core.FrozenDict):
        params = flax.core.unfreeze(params)

    out = params['params']['output']
    assert out['bias'].shape == (theta.size,), (out['bias'].shape, theta.shape)
    out['bias'] = theta.reshape(-1).astype(out['bias'].dtype)
    out['kernel'] = jnp.zeros_like(out['kernel'])

    print(f"  Warm-started output bias from {stage1_path}")
    print(f"    theta range: [{float(theta.min()):.3f}, {float(theta.max()):.3f}]  "
          f"(Stage 1 loss {stage1.get('best_loss', float('nan')):.6f}, "
          f"cooling {stage1.get('achieved_cooling', float('nan')):+.4f} K)")
    return params


# T30 climatology terrain (orography + land-sea mask) shared by the
# atmosphere (Model.terrain) and the ocean land mask (SlabOceanModel.mask_file).
TERRAIN_NC = "jcm/data/bc/t30/clim/terrain.nc"
# T30 climatology surface forcing (land surface temperature, soil moisture,
# snow, albedo, SST). Required for numerical stability over realistic terrain:
# SPEEDY land-surface physics over steep orography needs real land boundary
# conditions, otherwise the atmosphere blows up (NaN within ~1 day).
FORCING_NC = "jcm/data/bc/t30/clim/forcing.nc"


def setup_coupled_model(start_datetime, coupling_timestep, realistic_terrain=False):
    """Set up the coupled atmosphere-ocean model.

    Args:
        start_datetime: Simulation start datetime (jax_datetime.Datetime).
        coupling_timestep: Coupler timestep (jax_datetime.Timedelta).
        realistic_terrain: When True (Stage 5+), load T30 climatology terrain
            (orography + land-sea mask) into BOTH the atmosphere (activating
            orography and SPEEDY land-surface physics) and the slab ocean's
            land mask (pinning land cells to a fixed temperature so only ocean
            SST evolves). When False (default), the aquaplanet is used, keeping
            all Stage 1-4 paths unchanged.

    Returns:
        Tuple of (coupler, coords, terrain, atm_model). `terrain.fmask` gives
        the land-sea mask; `1.0 - terrain.fmask` is the ocean mask used for
        MCB application and ocean-masked loss/features.

    """
    print("Setting up coupled model"
          f" ({'realistic terrain' if realistic_terrain else 'aquaplanet'})...")

    # Get coordinates and terrain
    coords = get_speedy_coords()
    if realistic_terrain:
        terrain = TerrainData.from_file(TERRAIN_NC, coords, lfluxland=True)
    else:
        terrain = TerrainData.aquaplanet(coords)

    # Create atmosphere model (no static MCB - will be dynamic from policy).
    # Passing terrain= is the one wiring point that actually activates
    # orography in the dynamics and land-surface physics in SPEEDY.
    atm_model = jcm.model.Model(
        start_date=start_datetime,
        coords=coords,
        terrain=terrain if realistic_terrain else None,
    )

    # Surface forcing: over realistic terrain, supply real land-surface
    # boundary conditions (land surface temperature, soil moisture, snow,
    # albedo) so SPEEDY land physics is stable; the coupler still overrides
    # SST every step from the ocean. On the aquaplanet, None keeps the JEM
    # default forcing (Stage 1-4 behavior unchanged).
    #
    # ForcingData.from_file returns a 365-day annual cycle: land fields are
    # 3D (ix, il, 365) and the model slices the current day internally each
    # physics step (tree_index_3d). But the coupler writes a 2D (ix, il) SST
    # into forcing.sea_surface_temperature every step; a 3D SST would break
    # the lax.scan shape invariant (input 3D vs output 2D). So we collapse
    # ONLY the SST field to 2D (day-0 climatology as a placeholder — the
    # coupler overwrites it on the first coupling step) while keeping the 3D
    # land annual cycle. tree_index_3d passes the 2D SST through unchanged.
    atm_forcing = None
    if realistic_terrain:
        atm_forcing = ForcingData.from_file(FORCING_NC, coords)
        atm_forcing = atm_forcing.copy(
            sea_surface_temperature=atm_forcing.sea_surface_temperature[:, :, 0]
        )

    # Make JEM-compatible
    atm_model = make_jem_compatible(
        atm_model, coupling_timestep, forcing=atm_forcing
    )

    # Create slab ocean model (timestep needs to be in seconds as float).
    # On realistic terrain, mask_file pins land cells to a fixed temperature
    # so only ocean SST evolves, consistent with the atmosphere land mask.
    timestep_seconds = 86400.0  # 1 day in seconds
    ocn_model = SlabOceanModel(
        start_datetime=start_datetime,
        timestep=timestep_seconds,
        mask_file=TERRAIN_NC if realistic_terrain else None,
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

    # Optional warm-start from the Stage 1 optimized pattern
    initial_params = None
    if args.warm_start:
        initial_params = warm_start_params(policy, feature_dim, args.warm_start)

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
        initial_params=initial_params,
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
            'warm_start': args.warm_start,
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
