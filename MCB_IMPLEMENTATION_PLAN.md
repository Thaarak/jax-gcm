# MCB Neural Network Controller Implementation Plan

## Overview

This plan outlines the implementation of a **neural network-based feedback controller** for Marine Cloud Brightening (MCB) optimization using JAX-GCM's differentiable simulation capabilities.

**Approach**: Train a policy network via **Backpropagation Through Time (BPTT)** by unrolling the climate model and computing gradients of a climate loss function with respect to network weights.

---

## Implementation Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Foundation | ✅ Complete | MCB forcing module and physics integration |
| Phase 2: Policy Network | ✅ Complete | Flax-based neural network architectures |
| Phase 3: Loss Function | ✅ Complete | Differentiable climate loss with teleconnection penalties |
| Phase 4: Differentiable Unrolling | ✅ Complete | `jax.lax.scan`-based simulation unrolling |
| Phase 5: BPTT Training Loop | ✅ Complete | Full training infrastructure with checkpointing |
| Phase 6: Experimental Design | ⏳ Pending | Run experiments and evaluate results |

**Test Coverage**: 54 unit tests passing | **Linting**: All checks pass

---

## Phase 1: Foundation (Completed)

### 1.1 MCB Forcing Module ✅

| File | Status | Description |
|------|--------|-------------|
| `jcm/mcb/__init__.py` | ✅ Complete | Public API exports (updated for all new modules) |
| `jcm/mcb/mcb_config.py` | ✅ Complete | `MCBConfig` struct with `albedo_perturbation`, `active_mask`, `temporal_weights` |
| `jcm/mcb/mcb_regions.py` | ✅ Complete | Region masks for stratocumulus zones and teleconnection monitoring |
| `jcm/mcb/mcb_forcing.py` | ✅ Complete | `compute_mcb_sea_albedo()` core forcing function |
| `jcm/mcb/mcb_test.py` | ✅ Complete | 54 unit tests passing (27 original + 27 new) |

### 1.2 Physics Integration ✅

| File | Status | Description |
|------|--------|-------------|
| `jcm/physics/speedy/forcing.py` | ✅ Modified | Added `mcb_config` parameter to `set_forcing()` |
| `jcm/physics/speedy/speedy_physics.py` | ✅ Modified | `SpeedyPhysics` accepts and propagates `mcb_config` |

---

## Phase 2: Policy Network (Completed)

### 2.1 Architecture Design ✅

**Input**: Climate state features extracted from `PhysicsState` / `Predictions`
- Global mean surface temperature anomaly
- Regional temperature anomalies (tropics, mid-latitudes, poles)
- Precipitation rate anomalies (global and tropical)
- Teleconnection region features (Amazon, Sahel, South Asia)

**Output**: Spatial MCB albedo perturbation field `(ix, il)`
- Constrained to valid range `[0, max_perturbation]` via sigmoid activation
- Masked to ocean-only regions

**Networks Implemented**:
- `MCBPolicyMLP`: Flattened state → hidden layers → output grid
- `MCBPolicyCNN`: Spatial state → conv layers → output grid (preserves spatial structure)
- `MCBPolicyResNet`: ResNet-style with residual connections for better gradient flow
- `MCBPolicyHybrid`: Combines global scalar features with spatial processing

### 2.2 Implementation ✅

**File**: `jcm/mcb/policy.py`

Key features implemented:
- All networks use `nn.sigmoid` to constrain output to `[0, max_perturbation]`
- Support for both batched and unbatched inputs
- LayerNorm and GELU activations for stable training
- Optional dropout for regularization
- `create_policy()` factory function for easy instantiation
- `init_policy_params()` helper for parameter initialization

### 2.3 State Feature Extraction ✅

**File**: `jcm/mcb/state_features.py`

Key features implemented:
- `StateFeatureConfig`: Configurable feature extraction
- `ClimateBaseline`: Baseline climate state for anomaly computation
- `extract_scalar_features()`: 12 scalar features for MLP policies
  - Global/tropical/midlat/polar temperature anomalies
  - Global/tropical precipitation anomalies
  - Teleconnection region features (Amazon, Sahel, South Asia Monsoon)
- `extract_spatial_features()`: Gridded features for CNN policies
- Area-weighted regional averaging with `compute_area_weights()`
- Latitude band masking with `create_latitude_band_mask()`

---

## Phase 3: Loss Function (Completed)

### 3.1 Components ✅

| Term | Weight | Description | Implementation |
|------|--------|-------------|----------------|
| `L_temperature` | `λ_T = 1.0` | Penalize deviation from target global temperature | `temperature_loss()` |
| `L_amazon` | `λ_A = 1.0` | Penalize precipitation reduction in Amazon basin | `amazon_precipitation_loss()` - asymmetric (softplus) |
| `L_sahel` | `λ_S = 0.5` | Penalize precipitation reduction in Sahel | `sahel_precipitation_loss()` - asymmetric |
| `L_tropics` | `λ_tr = 0.3` | Penalize precipitation changes in global tropics | `tropical_precipitation_loss()` - symmetric |
| `L_regularization` | `λ_reg = 0.01` | Penalize excessive MCB forcing (L2 norm) | `regularization_loss()` |
| `L_smoothness` | `λ_sm = 0.01` | Penalize non-smooth MCB patterns (total variation) | `smoothness_loss()` |

### 3.2 Implementation ✅

**File**: `jcm/mcb/loss.py`

Key features implemented:
- `LossWeights` NamedTuple for configurable loss term weights
- `LossComponents` struct for detailed loss breakdown/logging
- `compute_climate_loss()`: Main loss function combining all terms
- `compute_climate_loss_from_baseline()`: Convenience wrapper for ClimateBaseline
- `create_loss_fn()`: Curried loss function for training loops
- All loss functions fully differentiable (no `jnp.maximum`, using `softplus` instead)

---

## Phase 4: Differentiable Unrolling (Completed)

### 4.1 Control Loop Architecture ✅

```
For each control interval (e.g., 30 days):
    1. Run model forward to get current climate state
    2. Extract state features from predictions
    3. Pass features through policy network → get MCB forcing
    4. Apply ocean mask to MCB forcing
    5. Compute interval loss
    6. Accumulate loss and continue to next interval

After full rollout:
    7. Return total loss for gradient computation
    8. Backpropagate through entire trajectory (BPTT)
```

### 4.2 Implementation ✅

**File**: `jcm/mcb/controller.py`

Key features implemented:
- `ControllerConfig` NamedTuple with all control parameters
- `ControlStep` NamedTuple for step outputs (state, predictions, forcing, loss)
- `create_controlled_step()`: Factory for single control step function
- `unroll_with_policy()`: Full unrolling with `jax.lax.scan`
- `unroll_with_policy_simple()`: Returns only scalar loss (memory efficient)
- `create_loss_fn()`: Curried loss function depending only on policy params
- `evaluate_policy()`: Run policy and compute detailed metrics
- `compute_policy_gradient()`: Compute gradients via BPTT
- `verify_gradients()`: Check for NaN/zero gradients
- Gradient checkpointing via `jax.checkpoint` for memory efficiency

---

## Phase 5: BPTT Training Loop (Completed)

### 5.1 Training Function ✅

**File**: `jcm/mcb/train.py`

Key features implemented:
- `TrainingConfig` NamedTuple with all training hyperparameters
- `TrainingState` NamedTuple tracking training progress
- `create_optimizer()`: Optimizer factory with learning rate schedules
  - Constant, cosine decay, warmup + cosine decay
  - Adam, AdamW, SGD with momentum
  - Optional gradient clipping
- `create_train_step()`: JIT-compiled training step
- `train_policy()`: Main training loop with:
  - Progress logging
  - Best parameter tracking
  - Early stopping support
  - Callback hooks for custom logging
- `validate_training_setup()`: Pre-training validation
- `save_checkpoint()` / `load_checkpoint()`: Model persistence

### 5.2 Memory Optimization ✅

Gradient checkpointing implemented via `jax.checkpoint` decorator on step functions.
Configurable via `ControllerConfig.use_checkpointing = True` (default).

---

## Phase 6: Experimental Design (Pending)

### 6.1 Experiments

| Experiment | Description | Duration | Status |
|------------|-------------|----------|--------|
| Baseline | No MCB, establish climate reference | 10 years | ⏳ Pending |
| Static MCB | Uniform 5% albedo increase in SE Pacific | 10 years | ⏳ Pending |
| Policy Training | Train neural controller | 100 epochs × 1 year | ⏳ Pending |
| Policy Evaluation | Run trained policy | 10 years | ⏳ Pending |
| Ablation | Vary loss weights, network size | Variable | ⏳ Pending |

### 6.2 Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Global ΔT | -0.5 K | Temperature reduction |
| Amazon ΔP | ≥ 0% | No precipitation decrease |
| Tropical ΔP | ≤ 5% | Minimal tropical impact |
| MCB Efficiency | max | Cooling per unit forcing |
| Policy Smoothness | high | Avoid erratic forcing changes |

---

## File Structure

```
jcm/mcb/
├── __init__.py           # ✅ Exports all public API
├── mcb_config.py         # ✅ MCBConfig struct
├── mcb_regions.py        # ✅ Region masks (stratocumulus + teleconnection)
├── mcb_forcing.py        # ✅ Core forcing computation
├── mcb_test.py           # ✅ 54 unit tests
├── policy.py             # ✅ Policy networks (MLP, CNN, ResNet, Hybrid)
├── state_features.py     # ✅ Feature extraction (scalar + spatial)
├── loss.py               # ✅ Climate loss function (6 components)
├── controller.py         # ✅ Differentiable unrolling with lax.scan
└── train.py              # ✅ BPTT training loop
```

---

## Implementation Checklist

### Step 1: Policy Network (`policy.py`) ✅
- [x] Implement `MCBPolicyMLP` class
- [x] Implement `MCBPolicyCNN` class
- [x] Implement `MCBPolicyResNet` class (added for better gradient flow)
- [x] Implement `MCBPolicyHybrid` class (added for combined global/spatial)
- [x] Add `create_policy()` factory function
- [x] Add `init_policy_params()` helper

### Step 2: State Features (`state_features.py`) ✅
- [x] Implement `StateFeatureConfig` configuration
- [x] Implement `ClimateBaseline` struct
- [x] Implement `extract_scalar_features()` (12 features)
- [x] Implement `extract_spatial_features()` (for CNN policies)
- [x] Implement `extract_state_features()` main entry point
- [x] Add `compute_area_weights()` for proper spatial averaging
- [x] Add `create_latitude_band_mask()` utility
- [x] Add `get_feature_dim()` for dimension lookup

### Step 3: Loss Function (`loss.py`) ✅
- [x] Implement `LossWeights` configuration
- [x] Implement `LossComponents` for detailed breakdown
- [x] Implement `temperature_loss()`
- [x] Implement `amazon_precipitation_loss()` (asymmetric)
- [x] Implement `sahel_precipitation_loss()` (asymmetric)
- [x] Implement `tropical_precipitation_loss()` (symmetric)
- [x] Implement `regularization_loss()` (L2 norm)
- [x] Implement `smoothness_loss()` (total variation)
- [x] Implement `compute_climate_loss()` main function
- [x] Ensure full differentiability (no hard thresholds)

### Step 4: Controller (`controller.py`) ✅
- [x] Implement `ControllerConfig` configuration
- [x] Implement `ControlStep` output struct
- [x] Implement `create_controlled_step()` factory
- [x] Implement `unroll_with_policy()` with `jax.lax.scan`
- [x] Implement `unroll_with_policy_simple()` for memory efficiency
- [x] Implement `create_loss_fn()` curried function
- [x] Implement `evaluate_policy()` for metrics
- [x] Implement `compute_policy_gradient()` for BPTT
- [x] Implement `verify_gradients()` for debugging
- [x] Add gradient checkpointing option

### Step 5: Training (`train.py`) ✅
- [x] Implement `TrainingConfig` configuration
- [x] Implement `TrainingState` state tracking
- [x] Implement `create_optimizer()` with LR schedules
- [x] Implement `create_train_step()` JIT-compiled step
- [x] Implement `initialize_training()` helper
- [x] Implement `train_policy()` main training loop
- [x] Implement `validate_training_setup()` pre-check
- [x] Implement `save_checkpoint()` / `load_checkpoint()`
- [x] Add early stopping support
- [x] Add callback hooks for logging

### Step 6: Tests (`mcb_test.py`) ✅
- [x] Test `MCBPolicyMLP` output shape and bounds
- [x] Test `MCBPolicyMLP` batched input handling
- [x] Test `MCBPolicyMLP` JIT compatibility
- [x] Test `MCBPolicyMLP` gradient flow
- [x] Test `MCBPolicyCNN` output shape and bounds
- [x] Test `MCBPolicyResNet` gradient flow
- [x] Test `create_policy()` factory function
- [x] Test `StateFeatureConfig` defaults
- [x] Test `ClimateBaseline.global_mean()`
- [x] Test `get_feature_dim()`
- [x] Test `LossWeights` configuration
- [x] Test `regularization_loss()` (zero and positive cases)
- [x] Test `smoothness_loss()` (uniform and non-uniform)
- [x] Test `ControllerConfig` defaults
- [x] Test `TrainingConfig` defaults
- [x] Test `create_optimizer()` optimizer creation

### Step 7: Exports (`__init__.py`) ✅
- [x] Export all policy network classes
- [x] Export state feature extraction functions
- [x] Export loss function components
- [x] Export controller functions
- [x] Export training functions

---

## Verification

### Unit Tests ✅
```bash
pytest jcm/mcb/mcb_test.py -v
# Result: 54 passed
```

### Linting ✅
```bash
ruff check jcm/mcb/
# Result: All checks passed!
```

### Integration Test (Pending)
```python
# Verify end-to-end gradient flow
def test_bptt_gradient_flow():
    policy = MCBPolicyMLP(output_shape=coords.horizontal.nodal_shape)
    params = policy.init(key, dummy_input)

    def loss_fn(params):
        return unroll_with_policy(model, policy.apply, params, ...)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Check no NaNs
    assert not any(jnp.any(jnp.isnan(g)) for g in jax.tree.leaves(grads))
    # Check non-zero gradients
    assert any(jnp.any(g != 0) for g in jax.tree.leaves(grads))
```

### Training Verification (Pending)
```python
# Verify loss decreases over training
losses = train_policy(model, policy, ...)
assert losses[-1] < losses[0]  # Loss should decrease
```

---

## Next Steps

1. **Run baseline simulation** to establish reference climate state
2. **Create ClimateBaseline** from baseline run
3. **Train policy** with `train_policy()` function
4. **Evaluate trained policy** with `evaluate_policy()`
5. **Run ablation studies** varying loss weights and network architectures
6. **Visualize results** using existing `visualize_mcb_globe.py`

---

## Change Log

### 2024-XX-XX: Neural Network Controller Implementation
- Created `jcm/mcb/policy.py` with 4 policy network architectures (MLP, CNN, ResNet, Hybrid)
- Created `jcm/mcb/state_features.py` with scalar and spatial feature extraction
- Created `jcm/mcb/loss.py` with 6-component differentiable loss function
- Created `jcm/mcb/controller.py` with `jax.lax.scan`-based unrolling
- Created `jcm/mcb/train.py` with full BPTT training infrastructure
- Updated `jcm/mcb/__init__.py` to export all new components
- Added 27 new unit tests (54 total) to `jcm/mcb/mcb_test.py`
- All tests passing, all linting checks passing
