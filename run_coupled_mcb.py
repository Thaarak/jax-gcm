"""Run a coupled JCM + Slab Ocean simulation WITH MCB forcing.

This script tests MCB (Marine Cloud Brightening) in the coupled framework
by comparing heat flux and SST response to a baseline (no MCB) run.

Usage:
    python run_coupled_mcb.py
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
from pathlib import Path

# JCM imports
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.mcb import MCBConfig
from jcm.mcb.mcb_regions import create_stratocumulus_mask, create_ocean_mask
from jcm.terrain import get_terrain

# JEM imports
from jem import Coupler
from jem.components import JCM as JCMWrapper, SlabOceanModel
from jem.mapping import BasicMapper

print("=" * 60)
print("Coupled JCM + Slab Ocean WITH MCB Forcing")
print("=" * 60)
print()

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

# Simulation parameters
START_DATE = jdt.to_datetime("2000-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
SIMULATION_DAYS = 60

# MCB parameters
MCB_PERTURBATION = 0.10  # 10% albedo increase in stratocumulus regions

print("Configuration:")
print(f"  Start date: 2000-01-01")
print(f"  Coupling timestep: 1 day")
print(f"  Simulation length: {SIMULATION_DAYS} days")
print(f"  MCB perturbation: {MCB_PERTURBATION} (albedo increase)")
print()

# ============================================================
# Get Grid and Masks
# ============================================================

print("Setting up grid and MCB masks...")

# Get coords
coords = get_speedy_coords()
horizontal_grid = coords.horizontal
nodal_shape = horizontal_grid.nodal_shape

# Get terrain data for land-sea mask
# For aquaplanet: fmask = 0 everywhere (all ocean)
_, fmask = get_terrain(nodal_shape=nodal_shape)

# Create MCB masks
ocean_mask = create_ocean_mask(horizontal_grid, fmask)
stratocumulus_mask = create_stratocumulus_mask(horizontal_grid, fmask)

print(f"  Grid shape: {nodal_shape}")
print(f"  Ocean fraction: {float(jnp.mean(ocean_mask)):.2%}")
print(f"  Stratocumulus region fraction: {float(jnp.mean(stratocumulus_mask)):.2%}")

# ============================================================
# Create MCB Configuration
# ============================================================

print("\nCreating MCB configuration...")

mcb_config = MCBConfig.uniform(
    nodal_shape=nodal_shape,
    perturbation=MCB_PERTURBATION,
    region_mask=stratocumulus_mask,  # Already includes ocean mask
)

print(f"  MCB perturbation: {MCB_PERTURBATION}")
print(f"  MCB active cells: {float(jnp.sum(mcb_config.active_mask > 0)):.0f}")
print(f"  MCB mean forcing (in active regions): {float(jnp.mean(mcb_config.albedo_perturbation * mcb_config.active_mask)):.4f}")

# ============================================================
# Create Atmosphere Model WITH MCB
# ============================================================

print("\nInitializing atmosphere model (JCM) with MCB...")

# Create physics with MCB forcing
physics_with_mcb = SpeedyPhysics(mcb_config=mcb_config)

# Create model with MCB physics
atm_model = jcm.model.Model(
    start_date=START_DATE,
    coords=coords,
    physics=physics_with_mcb,
)

# Wrap for JEM compatibility
atm_model = JCMWrapper.make_jem_compatible(
    atm_model,
    coupling_timestep=COUPLING_TIMESTEP,
)
print("  Atmosphere model with MCB ready")

# ============================================================
# Create Ocean Model (Slab)
# ============================================================

print("Initializing ocean model (Slab)...")
ocn_model = SlabOceanModel(
    start_datetime=START_DATE,
    timestep=COUPLING_TIMESTEP / jdt.to_timedelta(1, "second"),
)
print("  Ocean model ready")

# ============================================================
# Create Coupling Mapper
# ============================================================

print("Setting up atmosphere-ocean coupling...")
mapper = BasicMapper()

# Atmosphere → Ocean: Heat flux drives SST change
mapper.add_mapping(
    source=("atm", "derived.total_heat_flux"),
    target=("ocn", "forcing.total_heat_flux"),
)

# Ocean → Atmosphere: SST as boundary condition
mapper.add_mapping(
    source=("ocn", "state.sea_surface_temperature"),
    target=("atm", "forcing.sea_surface_temperature"),
)

print("  Coupling configured")

# ============================================================
# Create Coupler
# ============================================================

print("Creating coupled model...")
coupled_model = Coupler(
    components=dict(
        atm=atm_model,
        ocn=ocn_model,
    ),
    mappers=dict(
        coupling=mapper,
    ),
)
print("  Coupled model ready")
print()

# ============================================================
# Run Coupled Simulation with MCB
# ============================================================

print(f"Running {SIMULATION_DAYS}-day coupled simulation WITH MCB...")
print("  (This may take a few minutes...)")

workflow = ["coupling", "atm", "ocn"]
iterations = SIMULATION_DAYS

initial_state, final_state, predictions = coupled_model.run(
    workflow=workflow,
    iterations=iterations,
)

print("  Simulation complete!")
print()

# ============================================================
# Analyze Results
# ============================================================

print("=" * 60)
print("RESULTS: MCB Coupled Simulation")
print("=" * 60)

# Get SST changes
final_sst = final_state["ocn"]["state"].sea_surface_temperature
initial_sst = initial_state["ocn"]["state"].sea_surface_temperature
sst_change = final_sst - initial_sst

print(f"\nSea Surface Temperature (SST):")
print(f"  Initial mean: {float(jnp.mean(initial_sst)):.2f} K")
print(f"  Final mean:   {float(jnp.mean(final_sst)):.2f} K")
print(f"  Global change: {float(jnp.mean(sst_change)):.4f} K")

# Regional SST in MCB areas
mcb_region_sst_change = jnp.where(
    stratocumulus_mask > 0,
    sst_change,
    jnp.nan
)
mcb_region_mean_change = float(jnp.nanmean(mcb_region_sst_change))
print(f"  MCB region change: {mcb_region_mean_change:.4f} K")

# Get heat flux from final state
final_heat_flux = final_state["atm"]["derived"]["total_heat_flux"]
print(f"\nHeat Flux (final step):")
print(f"  Global mean: {float(jnp.mean(final_heat_flux)):.2f} W/m²")

# Heat flux in MCB regions
mcb_heat_flux = jnp.where(stratocumulus_mask > 0, final_heat_flux, jnp.nan)
print(f"  MCB region mean: {float(jnp.nanmean(mcb_heat_flux)):.2f} W/m²")

# ============================================================
# Compare to Baseline (load from previous run)
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON: MCB vs Baseline")
print("=" * 60)

# Try to load baseline results
baseline_file = OUTPUT_DIR / "coupled_baseline_ocn.nc"
if baseline_file.exists():
    import xarray as xr
    baseline_ocn = xr.open_dataset(baseline_file)

    # Get baseline SST change (rough estimate from first/last timestep)
    if 'sea_surface_temperature' in baseline_ocn:
        baseline_sst = baseline_ocn['sea_surface_temperature']
        baseline_sst_initial = float(baseline_sst.isel(time=0).mean())
        baseline_sst_final = float(baseline_sst.isel(time=-1).mean())
        baseline_sst_change = baseline_sst_final - baseline_sst_initial

        print(f"\nSST Change Comparison:")
        print(f"  Baseline: {baseline_sst_change:+.4f} K")
        print(f"  With MCB: {float(jnp.mean(sst_change)):+.4f} K")
        print(f"  MCB effect: {float(jnp.mean(sst_change)) - baseline_sst_change:+.4f} K")

    baseline_ocn.close()
else:
    print("\n(Baseline file not found - run run_coupled_baseline.py first for comparison)")

# ============================================================
# Save Results
# ============================================================

print("\nSaving results...")

# Convert predictions to xarray
output_dict = coupled_model.predictions_to_xarray(predictions)

for component_name, ds in output_dict.items():
    output_file = OUTPUT_DIR / f"coupled_mcb_{component_name}.nc"
    print(f"  Saving: {component_name} → {output_file}")
    ds.to_netcdf(output_file)

# Save MCB config info
import pickle
mcb_results = {
    'mcb_config': {
        'perturbation': MCB_PERTURBATION,
        'active_mask': mcb_config.active_mask,
        'stratocumulus_mask': stratocumulus_mask,
    },
    'sst': {
        'initial_mean': float(jnp.mean(initial_sst)),
        'final_mean': float(jnp.mean(final_sst)),
        'global_change': float(jnp.mean(sst_change)),
        'mcb_region_change': mcb_region_mean_change,
    },
    'heat_flux': {
        'final_global_mean': float(jnp.mean(final_heat_flux)),
    },
    'simulation_days': SIMULATION_DAYS,
}

with open(OUTPUT_DIR / 'coupled_mcb_results.pkl', 'wb') as f:
    pickle.dump(mcb_results, f)
print(f"  Saved MCB results summary to coupled_mcb_results.pkl")

print()
print("=" * 60)
print("Coupled MCB simulation complete!")
print("=" * 60)
