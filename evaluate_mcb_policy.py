"""Evaluate trained MCB policy.

Runs a 90-day simulation with the trained policy and compares
the resulting climate to the baseline (no MCB) simulation.

Usage:
    python evaluate_mcb_policy.py
"""

import jax
import jax.numpy as jnp
import pickle
import time
from importlib import resources
from pathlib import Path

print("=" * 60)
print("MCB Policy Evaluation")
print("=" * 60)
print(f"JAX devices: {jax.devices()}")
print()

# Import JCM components
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# Import MCB components
from jcm.mcb.policy import create_policy
from jcm.mcb.state_features import (
    ClimateBaseline,
    StateFeatureConfig,
    extract_state_features,
    compute_area_weights,
    compute_regional_mean,
    create_latitude_band_mask,
)
from jcm.mcb.mcb_regions import create_region_mask, TELECONNECTION_REGIONS

# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")
BASELINE_PATH = OUTPUT_DIR / "baseline_climate.pkl"
POLICY_PATH = OUTPUT_DIR / "trained_policy.pkl"
RESULTS_PATH = OUTPUT_DIR / "mcb_evaluation_results.pkl"

# ============================================================
# Load Trained Policy
# ============================================================

print("Loading trained policy...")

with open(POLICY_PATH, 'rb') as f:
    checkpoint = pickle.load(f)

params = checkpoint['params']
metadata = checkpoint['metadata']

print(f"  Policy type: {metadata['policy_type']}")
print(f"  Final training loss: {metadata['final_loss']:.6f}")
print(f"  Epochs trained: {metadata['epochs_completed']}")
print()

# ============================================================
# Load Baseline Climate
# ============================================================

print("Loading baseline climate...")

with open(BASELINE_PATH, 'rb') as f:
    baseline_data = pickle.load(f)

baseline = baseline_data['baseline']
baseline_stats = baseline_data['statistics']

print(f"  Baseline global mean temp: {baseline_stats['global_mean_surface_temp_K']:.2f} K")
print()

# ============================================================
# Load Model Components
# ============================================================

print("Loading model components...")

data_dir = resources.files('jcm.data.bc.t30.clim')
coords = get_speedy_coords()

terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)
forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

print(f"  Grid shape: {coords.horizontal.nodal_shape}")
print()

# ============================================================
# Create Model and Policy
# ============================================================

print("Creating model and policy...")

model = Model(coords=coords, terrain=terrain)

policy_kwargs = metadata['policy_kwargs']
policy = create_policy(
    policy_type=metadata['policy_type'],
    output_shape=coords.horizontal.nodal_shape,
    **policy_kwargs
)

print("  Model and policy created")
print()

# ============================================================
# Run MCB Simulation
# ============================================================

print("Running MCB simulation (90 days)...")
print("  This will take 2-3 minutes on CPU...")
print()

start_time = time.time()

# Warmup to get initial state
warmup_preds = model.run(
    save_interval=30,
    total_time=30,
    forcing=forcing,
    output_averages=True,
)
initial_state = model._final_modal_state

# Feature config
feature_config = StateFeatureConfig(
    include_temperature=True,
    include_precipitation=True,
    include_spatial=False,
)

# Run simulation with policy
TOTAL_DAYS = 90.0
CONTROL_INTERVAL = 30.0
n_steps = int(TOTAL_DAYS / CONTROL_INTERVAL)

# Track MCB forcing applied
mcb_forcings = []
all_predictions = []

state = initial_state
for step in range(n_steps):
    # Get current predictions
    predictions = model.run(
        save_interval=CONTROL_INTERVAL,
        total_time=CONTROL_INTERVAL,
        forcing=forcing,
        output_averages=True,
        initial_state=state,
    )

    # Extract features and compute MCB forcing
    features = extract_state_features(predictions, baseline, coords, feature_config)
    mcb_forcing = policy.apply(params, features)

    mcb_forcings.append(mcb_forcing)
    all_predictions.append(predictions)

    # Get state for next step
    state = model._final_modal_state

    print(f"  Step {step + 1}/{n_steps}: MCB mean = {jnp.mean(mcb_forcing):.6f}, max = {jnp.max(mcb_forcing):.6f}")

elapsed = time.time() - start_time
print()
print(f"Simulation complete in {elapsed:.1f}s")
print()

# ============================================================
# Analyze Results
# ============================================================

print("Analyzing results...")
print()

# Get final predictions
final_preds = all_predictions[-1]

# Compute area weights
area_weights = compute_area_weights(coords)

# Surface temperature
surf_temp = final_preds.physics.surface_flux.tsfc
if surf_temp.ndim == 3:
    surf_temp = surf_temp[0]

mcb_global_temp = jnp.sum(surf_temp * area_weights)
baseline_global_temp = jnp.sum(baseline.surface_temperature * area_weights)
temp_change = mcb_global_temp - baseline_global_temp

# Precipitation
precnv = final_preds.physics.convection.precnv
precls = final_preds.physics.condensation.precls
if precnv.ndim == 3:
    precnv = precnv[0]
    precls = precls[0]
mcb_precip = precnv + precls

mcb_global_precip = jnp.sum(mcb_precip * area_weights)
baseline_global_precip = jnp.sum(baseline.precipitation * area_weights)
precip_change_pct = 100 * (mcb_global_precip - baseline_global_precip) / jnp.maximum(baseline_global_precip, 1e-10)

# Regional analysis
grid = coords.horizontal

# Amazon
amazon_bounds = TELECONNECTION_REGIONS['amazon']
amazon_mask = create_region_mask(grid, amazon_bounds[:2], amazon_bounds[2:])
amazon_mcb_precip = compute_regional_mean(mcb_precip, amazon_mask, area_weights)
amazon_baseline_precip = compute_regional_mean(baseline.precipitation, amazon_mask, area_weights)
amazon_change_pct = 100 * (amazon_mcb_precip - amazon_baseline_precip) / jnp.maximum(amazon_baseline_precip, 1e-10)

# Sahel
sahel_bounds = TELECONNECTION_REGIONS['sahel']
sahel_mask = create_region_mask(grid, sahel_bounds[:2], sahel_bounds[2:])
sahel_mcb_precip = compute_regional_mean(mcb_precip, sahel_mask, area_weights)
sahel_baseline_precip = compute_regional_mean(baseline.precipitation, sahel_mask, area_weights)
sahel_change_pct = 100 * (sahel_mcb_precip - sahel_baseline_precip) / jnp.maximum(sahel_baseline_precip, 1e-10)

# Tropical temperature
tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
tropical_mcb_temp = compute_regional_mean(surf_temp, tropical_mask, area_weights)
tropical_baseline_temp = compute_regional_mean(baseline.surface_temperature, tropical_mask, area_weights)
tropical_temp_change = tropical_mcb_temp - tropical_baseline_temp

# Mean MCB forcing
mean_mcb_forcing = jnp.mean(jnp.stack(mcb_forcings))
max_mcb_forcing = jnp.max(jnp.stack(mcb_forcings))

# ============================================================
# Print Results
# ============================================================

print("=" * 60)
print("Evaluation Results")
print("=" * 60)
print()

print("Temperature Changes:")
print(f"  Global mean change:   {float(temp_change):+.4f} K (target: -0.5 K)")
print(f"  Tropical change:      {float(tropical_temp_change):+.4f} K")
print()

print("Precipitation Changes:")
print(f"  Global change:        {float(precip_change_pct):+.2f}%")
print(f"  Amazon change:        {float(amazon_change_pct):+.2f}%")
print(f"  Sahel change:         {float(sahel_change_pct):+.2f}%")
print()

print("MCB Forcing Applied:")
print(f"  Mean forcing:         {float(mean_mcb_forcing):.6f}")
print(f"  Max forcing:          {float(max_mcb_forcing):.6f}")
print()

# ============================================================
# Save Results
# ============================================================

print(f"Saving results to {RESULTS_PATH}...")

results = {
    'temperature': {
        'global_change_K': float(temp_change),
        'tropical_change_K': float(tropical_temp_change),
        'target_K': -0.5,
        'mcb_global_temp_K': float(mcb_global_temp),
        'baseline_global_temp_K': float(baseline_global_temp),
    },
    'precipitation': {
        'global_change_pct': float(precip_change_pct),
        'amazon_change_pct': float(amazon_change_pct),
        'sahel_change_pct': float(sahel_change_pct),
    },
    'mcb_forcing': {
        'mean': float(mean_mcb_forcing),
        'max': float(max_mcb_forcing),
        'final_pattern': jax.device_get(mcb_forcings[-1]),
        'all_patterns': [jax.device_get(f) for f in mcb_forcings],
    },
    'simulation': {
        'total_days': TOTAL_DAYS,
        'control_interval': CONTROL_INTERVAL,
        'n_steps': n_steps,
        'elapsed_time_s': elapsed,
    },
}

with open(RESULTS_PATH, 'wb') as f:
    pickle.dump(results, f)

print("  Saved!")
print()

# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print()

if temp_change < 0:
    cooling_pct = 100 * temp_change / -0.5
    print(f"  Achieved {float(-temp_change):.4f} K cooling ({cooling_pct:.1f}% of target)")
else:
    print(f"  WARNING: Temperature increased by {float(temp_change):.4f} K")

if amazon_change_pct < -5:
    print(f"  WARNING: Amazon precipitation decreased by {float(-amazon_change_pct):.1f}%")
else:
    print(f"  Amazon precipitation: {float(amazon_change_pct):+.1f}% (acceptable)")

if sahel_change_pct < -5:
    print(f"  WARNING: Sahel precipitation decreased by {float(-sahel_change_pct):.1f}%")
else:
    print(f"  Sahel precipitation: {float(sahel_change_pct):+.1f}% (acceptable)")

print()
print("Next steps:")
print("  1. Run: python visualize_mcb_patterns.py")
print("  2. Run: python analyze_training.py")
print()
