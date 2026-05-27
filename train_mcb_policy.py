"""Train MCB policy network via BPTT.

This script trains a neural network policy to optimize MCB deployment
using Backpropagation Through Time through the differentiable JCM model.

Prerequisites:
    Run `python run_mcb_baseline.py` first to generate baseline climate.

Usage:
    python train_mcb_policy.py

Output:
    - mcb_experiments/trained_policy.pkl: Trained policy parameters
    - mcb_experiments/training_history.pkl: Loss curves and metrics
"""

import jax
import jax.numpy as jnp
import pickle
import time
from importlib import resources
from pathlib import Path

# Configure JAX
jax.config.update("jax_debug_infs", True)

print("=" * 60)
print("MCB Policy Training (BPTT)")
print("=" * 60)
print(f"JAX devices: {jax.devices()}")
print()

# Import JCM components
from jcm.model import Model
from jcm.terrain import TerrainData
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_coords import get_speedy_coords

# Import MCB components
from jcm.mcb.policy import MCBPolicyMLP, create_policy
from jcm.mcb.state_features import ClimateBaseline, StateFeatureConfig, get_feature_dim
from jcm.mcb.loss import LossWeights
from jcm.mcb.controller import ControllerConfig
from jcm.mcb.train import (
    TrainingConfig,
    train_policy,
    validate_training_setup,
    save_checkpoint,
)

# ============================================================
# Configuration
# ============================================================

# Paths
OUTPUT_DIR = Path("mcb_experiments")
BASELINE_PATH = OUTPUT_DIR / "baseline_climate.pkl"
POLICY_PATH = OUTPUT_DIR / "trained_policy.pkl"
HISTORY_PATH = OUTPUT_DIR / "training_history.pkl"

# Training parameters
TRAINING_CONFIG = TrainingConfig(
    num_epochs=50,          # Number of training epochs
    learning_rate=1e-3,     # Initial learning rate
    lr_schedule='constant', # Learning rate schedule
    optimizer='adam',       # Optimizer type
    grad_clip_norm=1.0,     # Gradient clipping
    log_interval=5,         # Log every N epochs
    early_stopping_patience=None,  # Disable early stopping for now
    random_seed=42,
)

# Controller parameters
CONTROLLER_CONFIG = ControllerConfig(
    control_interval_days=30.0,  # Apply policy every 30 days
    total_days=90.0,             # 3-month rollout per training step (short for speed)
    target_cooling=-0.5,         # Target 0.5K global cooling
    loss_weights=LossWeights(
        temperature=1.0,      # Weight for temperature objective
        amazon=1.0,           # Weight for Amazon protection
        sahel=0.5,            # Weight for Sahel protection
        tropics=0.3,          # Weight for tropical stability
        regularization=0.01,  # Weight for forcing magnitude
        smoothness=0.01,      # Weight for spatial smoothness
    ),
    feature_config=StateFeatureConfig(
        include_temperature=True,
        include_precipitation=True,
        include_spatial=False,  # Use scalar features for MLP
    ),
    max_perturbation=0.15,
    use_checkpointing=True,
)

# Policy architecture
POLICY_TYPE = 'mlp'
POLICY_KWARGS = {
    'hidden_dims': (128, 128),  # Smaller network for faster training
    'max_perturbation': 0.15,
}

# ============================================================
# Load Baseline
# ============================================================

print("Loading baseline climate...")

if not BASELINE_PATH.exists():
    print(f"ERROR: Baseline file not found at {BASELINE_PATH}")
    print("Please run `python run_mcb_baseline.py` first.")
    exit(1)

with open(BASELINE_PATH, 'rb') as f:
    baseline_data = pickle.load(f)

baseline = baseline_data['baseline']
coords_info = baseline_data['coords_info']
statistics = baseline_data['statistics']

print(f"  Loaded baseline with shape: {baseline.surface_temperature.shape}")
print(f"  Baseline global mean temp: {statistics['global_mean_surface_temp_K']:.2f} K")
print()

# ============================================================
# Load Model Components
# ============================================================

print("Loading terrain and forcing data...")

data_dir = resources.files('jcm.data.bc.t30.clim')
coords = get_speedy_coords()

terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)
forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

print(f"  Grid shape: {coords.horizontal.nodal_shape}")
print()

# ============================================================
# Create Model (no MCB - policy will generate forcing)
# ============================================================

print("Creating model...")

model = Model(
    coords=coords,
    terrain=terrain,
)

print("  Model created")
print()

# ============================================================
# Create Policy Network
# ============================================================

print("Creating policy network...")

output_shape = coords.horizontal.nodal_shape
policy = create_policy(
    policy_type=POLICY_TYPE,
    output_shape=output_shape,
    **POLICY_KWARGS
)

feature_dim = get_feature_dim(CONTROLLER_CONFIG.feature_config)
print(f"  Policy type: {POLICY_TYPE}")
print(f"  Input features: {feature_dim}")
print(f"  Output shape: {output_shape}")
print()

# ============================================================
# Get Initial State
# ============================================================

print("Preparing initial state...")

# Run a short warmup to get a reasonable initial state
warmup_predictions = model.run(
    save_interval=30,
    total_time=30,
    forcing=forcing,
    output_averages=True,
)

initial_state = model._final_modal_state

print("  Initial state prepared")
print()

# ============================================================
# Validate Training Setup
# ============================================================

print("Validating training setup...")
print()

validation = validate_training_setup(
    model=model,
    policy=policy,
    coords=coords,
    terrain=terrain,
    forcing=forcing,
    baseline=baseline,
    initial_state=initial_state,
    controller_config=CONTROLLER_CONFIG,
)

if not validation['valid']:
    print(f"ERROR: Training setup validation failed!")
    print(f"  Reason: {validation.get('error', 'Unknown')}")
    exit(1)

print()
print("  Validation passed!")
print()

# ============================================================
# Train Policy
# ============================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)
print()

start_time = time.time()

best_params, history = train_policy(
    model=model,
    policy=policy,
    coords=coords,
    terrain=terrain,
    forcing=forcing,
    baseline=baseline,
    initial_state=initial_state,
    training_config=TRAINING_CONFIG,
    controller_config=CONTROLLER_CONFIG,
)

total_time = time.time() - start_time

print()
print("=" * 60)
print("Training Complete!")
print("=" * 60)
print()

# ============================================================
# Save Results
# ============================================================

print("Saving trained policy...")

save_checkpoint(
    params=best_params,
    path=str(POLICY_PATH),
    metadata={
        'policy_type': POLICY_TYPE,
        'policy_kwargs': POLICY_KWARGS,
        'training_config': TRAINING_CONFIG._asdict(),
        'controller_config': CONTROLLER_CONFIG._asdict(),
        'final_loss': history['final_loss'],
        'best_loss': history['best_loss'],
        'epochs_completed': history['epochs_completed'],
    }
)

print(f"Saving training history to {HISTORY_PATH}...")
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f)
print("  Saved!")
print()

# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("Training Summary")
print("=" * 60)
print()
print(f"  Epochs completed: {history['epochs_completed']}")
print(f"  Final loss: {history['final_loss']:.6f}")
print(f"  Best loss: {history['best_loss']:.6f}")
print(f"  Total time: {total_time:.1f}s")
print()
print("Output files:")
print(f"  - {POLICY_PATH}")
print(f"  - {HISTORY_PATH}")
print()
print("Next steps:")
print("  1. Evaluate trained policy: python evaluate_mcb_policy.py")
print("  2. Visualize MCB forcing patterns")
print("  3. Analyze climate impacts")
print()
