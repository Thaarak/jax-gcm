"""Run baseline climate simulation for MCB experiments.

This script runs the JCM model without any MCB forcing to establish
a baseline climate state for computing anomalies during policy training.

Usage:
    python run_mcb_baseline.py

Output:
    - baseline_predictions.nc: xarray Dataset with climate variables
    - baseline_climate.pkl: ClimateBaseline object for policy training
"""

import jax
import jax.numpy as jnp
import pickle
from importlib import resources
from pathlib import Path

# Configure JAX
jax.config.update("jax_debug_infs", True)

print("=" * 60)
print("MCB Baseline Simulation")
print("=" * 60)
print(f"JAX devices: {jax.devices()}")
print()

# Import JCM components
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# Import MCB components
from jcm.mcb.state_features import ClimateBaseline

# ============================================================
# Configuration
# ============================================================

# Simulation parameters
TOTAL_DAYS = 365  # 1 year baseline (can increase to 3650 for 10 years)
SAVE_INTERVAL = 30  # Save every 30 days (monthly)
OUTPUT_AVERAGES = True  # Output time-averaged quantities

# Output paths
OUTPUT_DIR = Path("mcb_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "baseline_predictions.nc"
BASELINE_PATH = OUTPUT_DIR / "baseline_climate.pkl"

# ============================================================
# Load Data
# ============================================================

print("Loading terrain and forcing data...")

data_dir = resources.files('jcm.data.bc.t30.clim')
coords = get_speedy_coords()

terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)
forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

print(f"  Grid shape: {coords.horizontal.nodal_shape}")
print(f"  Vertical levels: {coords.vertical.layers}")
print()

# ============================================================
# Create Model
# ============================================================

print("Creating model (no MCB forcing)...")

model = Model(
    coords=coords,
    terrain=terrain,
)

print("  Model created successfully")
print()

# ============================================================
# Run Baseline Simulation
# ============================================================

print(f"Running baseline simulation...")
print(f"  Total time: {TOTAL_DAYS} days ({TOTAL_DAYS/365:.1f} years)")
print(f"  Save interval: {SAVE_INTERVAL} days")
print(f"  Output averages: {OUTPUT_AVERAGES}")
print()

# Run the simulation
predictions = model.run(
    save_interval=SAVE_INTERVAL,
    total_time=TOTAL_DAYS,
    output_averages=OUTPUT_AVERAGES,
    forcing=forcing,
)

print("  Simulation complete!")
print()

# ============================================================
# Convert to xarray and Save
# ============================================================

print("Converting predictions to xarray...")
pred_ds = predictions.to_xarray()

print(f"  Dataset dimensions: {dict(pred_ds.dims)}")
print(f"  Time steps: {len(pred_ds.time)}")
print()

# Save predictions
print(f"Saving predictions to {PREDICTIONS_PATH}...")
pred_ds.to_netcdf(PREDICTIONS_PATH)
print("  Saved!")
print()

# ============================================================
# Create ClimateBaseline
# ============================================================

print("Creating ClimateBaseline object...")

# Extract mean fields over the simulation period
# Temperature (surface level is level index 0 after flipping)
surface_temp = pred_ds['temperature'].isel(level=0).mean(dim='time').values

# Total precipitation = convective + large-scale
precip = (pred_ds['convection.precnv'] + pred_ds['condensation.precls']).mean(dim='time').values

# Full temperature field (all levels)
temperature_3d = pred_ds['temperature'].mean(dim='time').values

# Transpose to match JCM convention (lon, lat, level) -> (level, lon, lat) -> need to check
# JCM uses (level, lon, lat) internally based on nodal_shape being (lon, lat)
# xarray has (level, lon, lat) so we need to transpose appropriately
# Actually looking at the predictions, it's (time, level, lon, lat)
# So after mean over time, it's (level, lon, lat)
# But ClimateBaseline expects (ix, il, sigma) = (lon, lat, level)
# So we need to transpose from (level, lon, lat) to (lon, lat, level)
temperature_3d = jnp.transpose(temperature_3d, (1, 2, 0))

# Surface temp and precip are (lon, lat) after isel and mean
# But xarray returns (lon, lat) which matches (ix, il)
surface_temp = jnp.array(surface_temp)
precip = jnp.array(precip)

# Create baseline
baseline = ClimateBaseline(
    temperature=temperature_3d,
    surface_temperature=surface_temp,
    precipitation=precip,
)

print(f"  Temperature shape: {baseline.temperature.shape}")
print(f"  Surface temperature shape: {baseline.surface_temperature.shape}")
print(f"  Precipitation shape: {baseline.precipitation.shape}")
print()

# Compute some statistics
global_mean_temp = float(jnp.mean(baseline.surface_temperature))
global_mean_precip = float(jnp.mean(baseline.precipitation))

print(f"  Global mean surface temperature: {global_mean_temp:.2f} K ({global_mean_temp - 273.15:.2f} °C)")
print(f"  Global mean precipitation: {global_mean_precip:.6f} (model units)")
print()

# Save baseline
print(f"Saving ClimateBaseline to {BASELINE_PATH}...")
with open(BASELINE_PATH, 'wb') as f:
    pickle.dump({
        'baseline': baseline,
        'coords_info': {
            'nodal_shape': coords.horizontal.nodal_shape,
            'layers': coords.vertical.layers,
        },
        'simulation_info': {
            'total_days': TOTAL_DAYS,
            'save_interval': SAVE_INTERVAL,
        },
        'statistics': {
            'global_mean_surface_temp_K': global_mean_temp,
            'global_mean_precip': global_mean_precip,
        }
    }, f)
print("  Saved!")
print()

# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Baseline Simulation Complete!")
print("=" * 60)
print()
print("Output files:")
print(f"  - {PREDICTIONS_PATH}")
print(f"  - {BASELINE_PATH}")
print()
print("Next steps:")
print("  1. Inspect baseline_predictions.nc to verify climate looks reasonable")
print("  2. Use baseline_climate.pkl for policy training")
print("  3. Run: python train_mcb_policy.py")
print()
