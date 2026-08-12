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
import numpy as np
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
from jem.components.slab.slab_land_model.slab_land_model import SlabLandModel
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


def warm_start_params(policy, feature_dim, stage1_path, seed=0):
    """Build initial policy params warm-started from the Stage 1 pattern.

    The policy output head is clipped-linear (jcm/mcb/policy.py:
    clip(logits, 0, max_perturbation)). Setting the output bias to the Stage 1
    optimized PATTERN (already in [0, max_perturbation]) and the output kernel
    to zero makes the initial policy output exactly that pattern, regardless of
    input features (clip is the identity inside the bound). Gradients still flow
    to the kernel (hidden activations are non-zero) wherever the output is
    strictly inside (0, max_perturbation), so the network can learn
    state-dependence from there.
    """
    with open(stage1_path, 'rb') as f:
        stage1 = pickle.load(f)
    # Seed with the actual optimized field (parameterization-independent), not
    # the sigmoid logit `best_theta` the old head required.
    pattern = jnp.asarray(stage1['best_pattern'])

    params = policy.init(jax.random.PRNGKey(seed), jnp.zeros(feature_dim))
    params = jax.tree_util.tree_map(lambda x: x, params)  # ensure mutable copy
    if isinstance(params, flax.core.FrozenDict):
        params = flax.core.unfreeze(params)

    out = params['params']['output']
    assert out['bias'].shape == (pattern.size,), (out['bias'].shape, pattern.shape)
    out['bias'] = pattern.reshape(-1).astype(out['bias'].dtype)
    out['kernel'] = jnp.zeros_like(out['kernel'])

    print(f"  Warm-started output bias from {stage1_path}")
    print(f"    pattern range: [{float(pattern.min()):.4f}, {float(pattern.max()):.4f}]  "
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


def setup_coupled_model(start_datetime, coupling_timestep, realistic_terrain=False,
                        co2_rate=0.0, co2_year_ref=2000):
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
    # Transient CO2 (dormant until now). forcing.py computes
    #   ablco2 = ablco2_ref * exp(increase_co2 * 0.005 * (model_year + tyear
    #                             - co2_year_ref))
    # so `increase_co2`, though declared bool, acts as a continuous ramp-rate
    # multiplier. co2_year_ref MUST be set to the start year: its 1950 default
    # against a 2000 start would apply an instant x1.28 absorptivity step
    # rather than a ramp from zero.
    physics = None
    if co2_rate:
        import dataclasses
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        base = Parameters.default()
        forcing_params = dataclasses.replace(
            base.forcing,
            increase_co2=float(co2_rate),
            co2_year_ref=int(co2_year_ref),
        )
        physics = SpeedyPhysics(
            parameters=dataclasses.replace(base, forcing=forcing_params))
        print(f"  transient CO2 ENABLED: rate {co2_rate} "
              f"(x{float(np.exp(co2_rate * 0.005)):.4f} ablco2 per year), "
              f"ref year {forcing_params.co2_year_ref}")

    atm_model = jcm.model.Model(
        start_date=start_datetime,
        coords=coords,
        terrain=terrain if realistic_terrain else None,
        physics=physics,
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
        # Collapse SST and land-surface-temperature to 2D (day-0 climatology
        # placeholder). The coupler overwrites SST from the slab ocean and stl_am
        # from the slab land model every step; a 3D field would break the
        # lax.scan shape invariant (3D input vs 2D coupled output). The 3D annual
        # cycle for snow / soil moisture is kept — only these two prognostic
        # coupled fields are collapsed.
        atm_forcing = atm_forcing.copy(
            sea_surface_temperature=atm_forcing.sea_surface_temperature[:, :, 0],
            stl_am=atm_forcing.stl_am[:, :, 0],
        )

    # Make JEM-compatible
    atm_model = make_jem_compatible(
        atm_model, coupling_timestep, forcing=atm_forcing
    )

    # Create slab ocean model (timestep needs to be in seconds as float).
    # On realistic terrain, mask_file pins land cells to a fixed temperature
    # so only ocean SST evolves, consistent with the atmosphere land mask.
    timestep_seconds = 86400.0  # 1 day in seconds
    # Over realistic terrain, initialize SST from the T30 SST climatology
    # (forcing.nc `sst`) instead of the idealized 273.15 + 27·cos²(1.5·lat)
    # field, which is ~6 K too cold and was the confirmed driver of the
    # +1.4 K/60-day spin-up drift. forcing_method stays "None" (a FREE slab: no
    # relaxation), so the ocean responds freely to MCB — relaxation would damp
    # the very cooling signal we optimize. On the aquaplanet, SST_clim_file=None
    # keeps the idealized init (Stage 1-4 behavior unchanged).
    ocn_model = SlabOceanModel(
        start_datetime=start_datetime,
        timestep=timestep_seconds,
        mask_file=TERRAIN_NC if realistic_terrain else None,
        SST_clim_file=FORCING_NC if realistic_terrain else None,
    )

    # Create mapper for atmosphere-ocean coupling. The SEA slab heat flux drives
    # the ocean; the ocean SST is fed back as the atmosphere's SST boundary.
    mapper = BasicMapper()
    mapper.add_mapping(
        source=("atm", "derived.total_heat_flux"),
        target=("ocn", "forcing.total_heat_flux"),
    )
    mapper.add_mapping(
        source=("ocn", "state.sea_surface_temperature"),
        target=("atm", "forcing.sea_surface_temperature"),
    )

    components = {"atm": atm_model, "ocn": ocn_model}

    # Over realistic terrain, also couple a slab LAND model so land-surface
    # temperature is PROGNOSTIC (responds to the atmosphere) rather than pinned
    # to climatology — a prerequisite for land-driven precipitation
    # teleconnections. The LAND slab heat flux (hfluxn[..., 0]) drives the land
    # model; its land_surface_temperature is fed back as the atmosphere's stl_am.
    if realistic_terrain:
        lnd_model = SlabLandModel(
            start_datetime=start_datetime,
            timestep=timestep_seconds,
            mask_file=TERRAIN_NC,
            land_clim_file=FORCING_NC,
        )
        mapper.add_mapping(
            source=("atm", "derived.land_heat_flux"),
            target=("lnd", "forcing.total_heat_flux"),
        )
        mapper.add_mapping(
            source=("lnd", "state.land_surface_temperature"),
            target=("atm", "forcing.stl_am"),
        )
        components["lnd"] = lnd_model

    # Create coupler
    coupler = Coupler(
        components=components,
        mappers={"coupling": mapper},
    )

    return coupler, coords, terrain, atm_model


def coupler_workflow(coupler):
    """Workflow (mapper + component step order) matching a coupler's components.

    ["coupling", "atm", "ocn"] on the aquaplanet; ["coupling", "atm", "ocn",
    "lnd"] over realistic terrain (where the slab land model is present). Drivers
    should derive the workflow from the coupler with this helper rather than
    hard-coding it, so the land step is run exactly when the land component
    exists.
    """
    order = ["atm", "ocn", "lnd"]
    return ["coupling"] + [c for c in order if c in coupler.components]


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
