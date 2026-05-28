"""Run a longer coupled JCM + Slab Ocean simulation WITH MCB forcing.

Extended 90-day simulation to observe sustained MCB cooling effects.

Usage:
    python run_coupled_mcb_longrun.py
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
print("Coupled MCB Long Run (90 days)")
print("=" * 60)
print()

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = jdt.to_datetime("2000-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")
SIMULATION_DAYS = 90  # Extended run

MCB_PERTURBATION = 0.10  # 10% albedo increase

print("Configuration:")
print(f"  Simulation length: {SIMULATION_DAYS} days")
print(f"  MCB perturbation: {MCB_PERTURBATION}")
print()

# ============================================================
# Setup (same as run_coupled_mcb.py)
# ============================================================

print("Setting up...")

# Get coords and masks
coords = get_speedy_coords()
horizontal_grid = coords.horizontal
nodal_shape = horizontal_grid.nodal_shape

_, fmask = get_terrain(nodal_shape=nodal_shape)
ocean_mask = create_ocean_mask(horizontal_grid, fmask)
stratocumulus_mask = create_stratocumulus_mask(horizontal_grid, fmask)

print(f"  Grid: {nodal_shape}")
print(f"  Stratocumulus coverage: {float(jnp.mean(stratocumulus_mask)):.1%}")

# MCB config
mcb_config = MCBConfig.uniform(
    nodal_shape=nodal_shape,
    perturbation=MCB_PERTURBATION,
    region_mask=stratocumulus_mask,
)

# Physics with MCB
physics_with_mcb = SpeedyPhysics(mcb_config=mcb_config)

# Atmosphere model
atm_model = jcm.model.Model(
    start_date=START_DATE,
    coords=coords,
    physics=physics_with_mcb,
)
atm_model = JCMWrapper.make_jem_compatible(atm_model, coupling_timestep=COUPLING_TIMESTEP)

# Ocean model
ocn_model = SlabOceanModel(
    start_datetime=START_DATE,
    timestep=COUPLING_TIMESTEP / jdt.to_timedelta(1, "second"),
)

# Coupling
mapper = BasicMapper()
mapper.add_mapping(("atm", "derived.total_heat_flux"), ("ocn", "forcing.total_heat_flux"))
mapper.add_mapping(("ocn", "state.sea_surface_temperature"), ("atm", "forcing.sea_surface_temperature"))

# Coupler
coupled_model = Coupler(
    components=dict(atm=atm_model, ocn=ocn_model),
    mappers=dict(coupling=mapper),
)

print("  Setup complete")
print()

# ============================================================
# Run Simulation
# ============================================================

print(f"Running {SIMULATION_DAYS}-day simulation...")
print("  (Estimated time: ~1.5 minutes)")

initial_state, final_state, predictions = coupled_model.run(
    workflow=["coupling", "atm", "ocn"],
    iterations=SIMULATION_DAYS,
)

print("  Complete!")
print()

# ============================================================
# Results
# ============================================================

print("=" * 60)
print("RESULTS: 90-Day Coupled MCB Simulation")
print("=" * 60)

# SST analysis
initial_sst = initial_state["ocn"]["state"].sea_surface_temperature
final_sst = final_state["ocn"]["state"].sea_surface_temperature
sst_change = final_sst - initial_sst

print(f"\nSea Surface Temperature:")
print(f"  Initial global mean: {float(jnp.mean(initial_sst)):.2f} K")
print(f"  Final global mean:   {float(jnp.mean(final_sst)):.2f} K")
print(f"  Global SST change:   {float(jnp.mean(sst_change)):+.4f} K")

# MCB region analysis
mcb_sst_change = jnp.where(stratocumulus_mask > 0, sst_change, jnp.nan)
print(f"  MCB region change:   {float(jnp.nanmean(mcb_sst_change)):+.4f} K")

# Heat flux
final_heat_flux = final_state["atm"]["derived"]["total_heat_flux"]
mcb_heat_flux = jnp.where(stratocumulus_mask > 0, final_heat_flux, jnp.nan)
print(f"\nHeat Flux (final):")
print(f"  Global mean:    {float(jnp.mean(final_heat_flux)):.2f} W/m²")
print(f"  MCB region:     {float(jnp.nanmean(mcb_heat_flux)):.2f} W/m²")

# Compare to 60-day baseline
baseline_60_file = OUTPUT_DIR / "coupled_baseline_ocn.nc"
if baseline_60_file.exists():
    import xarray as xr
    baseline = xr.open_dataset(baseline_60_file)
    if 'sea_surface_temperature' in baseline:
        b_sst = baseline['sea_surface_temperature']
        b_change = float(b_sst.isel(time=-1).mean()) - float(b_sst.isel(time=0).mean())

        # Scale baseline change to 90 days (rough approximation)
        b_change_90 = b_change * (90/60)

        print(f"\nComparison to Baseline (estimated for 90 days):")
        print(f"  Baseline SST change: {b_change_90:+.4f} K")
        print(f"  With MCB:            {float(jnp.mean(sst_change)):+.4f} K")
        print(f"  MCB cooling effect:  {float(jnp.mean(sst_change)) - b_change_90:+.4f} K")
    baseline.close()

# ============================================================
# Save Results
# ============================================================

print("\nSaving...")

output_dict = coupled_model.predictions_to_xarray(predictions)
for name, ds in output_dict.items():
    output_file = OUTPUT_DIR / f"coupled_mcb_90day_{name}.nc"
    ds.to_netcdf(output_file)
    print(f"  {output_file}")

# Summary
import pickle
results = {
    'simulation_days': SIMULATION_DAYS,
    'mcb_perturbation': MCB_PERTURBATION,
    'sst': {
        'initial_mean': float(jnp.mean(initial_sst)),
        'final_mean': float(jnp.mean(final_sst)),
        'global_change': float(jnp.mean(sst_change)),
        'mcb_region_change': float(jnp.nanmean(mcb_sst_change)),
    },
    'heat_flux_final_mean': float(jnp.mean(final_heat_flux)),
}
with open(OUTPUT_DIR / 'coupled_mcb_90day_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print()
print("=" * 60)
print("90-day MCB simulation complete!")
print("=" * 60)
