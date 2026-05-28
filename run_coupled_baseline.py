"""Run a simple coupled JCM + Slab Ocean baseline simulation.

This script tests the JAX-ESM coupling framework by running a short
aquaplanet simulation with atmosphere-ocean coupling.

Usage:
    python run_coupled_baseline.py
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
from pathlib import Path

# JCM imports
import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# JEM imports
from jem import Coupler
from jem.components import JCM as JCMWrapper, SlabOceanModel
from jem.mapping import BasicMapper

print("=" * 60)
print("Coupled JCM + Slab Ocean Baseline")
print("=" * 60)
print()

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("mcb_experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

# Simulation parameters
START_DATE = jdt.to_datetime("2000-01-01")
COUPLING_TIMESTEP = jdt.to_timedelta(1, "day")  # 1-day coupling
SIMULATION_DAYS = 60  # Short test run

print("Configuration:")
print(f"  Start date: {START_DATE}")
print(f"  Coupling timestep: 1 day")
print(f"  Simulation length: {SIMULATION_DAYS} days")
print()

# ============================================================
# Create Atmosphere Model (JCM)
# ============================================================

print("Initializing atmosphere model (JCM)...")
atm_model = jcm.model.Model(
    start_date=START_DATE,
    coords=get_speedy_coords(),
)

# Wrap for JEM compatibility
atm_model = JCMWrapper.make_jem_compatible(
    atm_model,
    coupling_timestep=COUPLING_TIMESTEP,
)
print("  Atmosphere model ready")

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

print("  Coupling configured:")
print("    - Atmosphere → Ocean: total_heat_flux")
print("    - Ocean → Atmosphere: sea_surface_temperature")
print()

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
# Run Coupled Simulation
# ============================================================

print(f"Running {SIMULATION_DAYS}-day coupled simulation...")
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

print("Analyzing results...")

# Extract atmosphere predictions
atm_predictions = predictions.get("atm", {})
ocn_predictions = predictions.get("ocn", {})

# Get final SST from ocean state
final_sst = final_state["ocn"]["state"].sea_surface_temperature
initial_sst = initial_state["ocn"]["state"].sea_surface_temperature

print(f"\nOcean State:")
print(f"  Initial SST mean: {float(jnp.mean(initial_sst)):.2f} K")
print(f"  Final SST mean: {float(jnp.mean(final_sst)):.2f} K")
print(f"  SST change: {float(jnp.mean(final_sst) - jnp.mean(initial_sst)):.4f} K")

# Get final atmosphere state
if "state" in final_state["atm"]:
    print(f"\nAtmosphere State:")
    print(f"  Final state type: {type(final_state['atm']['state'])}")

# ============================================================
# Save Results
# ============================================================

print("\nSaving results...")

# Convert predictions to xarray
output_dict = coupled_model.predictions_to_xarray(predictions)

for component_name, ds in output_dict.items():
    output_file = OUTPUT_DIR / f"coupled_baseline_{component_name}.nc"
    print(f"  Saving: {component_name} → {output_file}")
    ds.to_netcdf(output_file)

print()
print("=" * 60)
print("Coupled baseline simulation complete!")
print("=" * 60)
print()
print("Output files:")
for f in OUTPUT_DIR.glob("coupled_baseline_*.nc"):
    print(f"  {f}")
