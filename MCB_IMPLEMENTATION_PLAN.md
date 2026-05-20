# MCB Neural Network Controller Implementation Plan

## Overview

This plan outlines the implementation of a **neural network-based feedback controller** for Marine Cloud Brightening (MCB) optimization using JAX-GCM's differentiable simulation capabilities.

**Approach**: Train a policy network via **Backpropagation Through Time (BPTT)** by unrolling the climate model and computing gradients of a climate loss function with respect to network weights.

---

## Phase 1: Foundation (Completed)

### 1.1 MCB Forcing Module ✅

| File | Status | Description |
|------|--------|-------------|
| `jcm/mcb/__init__.py` | ✅ Complete | Public API exports |
| `jcm/mcb/mcb_config.py` | ✅ Complete | `MCBConfig` struct with `albedo_perturbation`, `active_mask`, `temporal_weights` |
| `jcm/mcb/mcb_regions.py` | ✅ Complete | Region masks for stratocumulus zones and teleconnection monitoring |
| `jcm/mcb/mcb_forcing.py` | ✅ Complete | `compute_mcb_sea_albedo()` core forcing function |
| `jcm/mcb/mcb_test.py` | ✅ Complete | 27 unit tests passing |

### 1.2 Physics Integration ✅

| File | Status | Description |
|------|--------|-------------|
| `jcm/physics/speedy/forcing.py` | ✅ Modified | Added `mcb_config` parameter to `set_forcing()` |
| `jcm/physics/speedy/speedy_physics.py` | ✅ Modified | `SpeedyPhysics` accepts and propagates `mcb_config` |

---

## Phase 2: Policy Network

### 2.1 Architecture Design

**Input**: Climate state features extracted from `PhysicsState` / `Predictions`
- Global mean surface temperature anomaly
- Regional temperature anomalies (tropics, mid-latitudes, poles)
- Precipitation rate fields (or anomalies from baseline)
- Optional: TOA radiation imbalance, cloud cover

**Output**: Spatial MCB albedo perturbation field `(ix, il)`
- Constrained to valid range `[0, max_perturbation]`
- Masked to ocean-only regions

**Network**: Lightweight MLP or CNN
- MLP: Flattened state → hidden layers → output grid
- CNN: Spatial state → conv layers → output grid (preserves spatial structure)

### 2.2 Implementation

**File**: `jcm/mcb/policy.py`

```python
import flax.linen as nn
import jax.numpy as jnp

class MCBPolicyMLP(nn.Module):
    """MLP policy network for MCB control."""
    hidden_dims: tuple = (256, 256)
    output_shape: tuple = (64, 32)  # nodal_shape
    max_perturbation: float = 0.15

    @nn.compact
    def __call__(self, state_features):
        x = state_features
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_shape[0] * self.output_shape[1])(x)
        x = x.reshape(self.output_shape)
        # Constrain to valid range
        x = self.max_perturbation * nn.sigmoid(x)
        return x

class MCBPolicyCNN(nn.Module):
    """CNN policy network preserving spatial structure."""
    features: tuple = (32, 64, 32)
    max_perturbation: float = 0.15

    @nn.compact
    def __call__(self, state_grid):
        x = state_grid
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
        x = nn.Conv(1, kernel_size=(1, 1))(x)  # Output single channel
        x = x.squeeze(-1)
        x = self.max_perturbation * nn.sigmoid(x)
        return x
```

### 2.3 State Feature Extraction

**File**: `jcm/mcb/state_features.py`

```python
def extract_state_features(predictions, baseline, coords):
    """Extract climate state features for policy input."""
    # Temperature anomaly
    temp = predictions.dynamics.temperature
    temp_anomaly = temp - baseline.temperature

    # Global mean
    global_temp_anomaly = jnp.mean(temp_anomaly)

    # Regional means (tropics, etc.)
    tropical_mask = create_tropical_mask(coords)
    tropical_temp = jnp.mean(temp_anomaly * tropical_mask)

    # Precipitation
    precip = predictions.physics.convection.precnv + predictions.physics.condensation.precls
    precip_anomaly = precip - baseline.precip

    # Flatten for MLP or stack for CNN
    return jnp.concatenate([...])
```

---

## Phase 3: Loss Function

### 3.1 Components

| Term | Weight | Description |
|------|--------|-------------|
| `L_temperature` | `λ_T` | Penalize deviation from target global temperature |
| `L_amazon` | `λ_A` | Penalize precipitation reduction in Amazon basin |
| `L_tropics` | `λ_tr` | Penalize precipitation changes in global tropics |
| `L_regularization` | `λ_reg` | Penalize excessive/non-smooth MCB forcing |

### 3.2 Implementation

**File**: `jcm/mcb/loss.py`

```python
def compute_climate_loss(
    predictions,
    baseline,
    target_cooling,
    mcb_forcing,
    coords,
    weights,
):
    """Differentiable climate loss function.

    Args:
        predictions: Model output (Predictions object)
        baseline: Baseline climate state (no MCB)
        target_cooling: Target temperature reduction (K)
        mcb_forcing: Applied MCB perturbation field
        coords: Model coordinates
        weights: Dict of loss term weights

    Returns:
        Scalar loss value
    """
    # Temperature loss
    temp = predictions.dynamics.temperature
    global_temp = jnp.mean(temp)
    baseline_temp = jnp.mean(baseline.temperature)
    temp_anomaly = global_temp - baseline_temp
    L_temp = (temp_anomaly - target_cooling) ** 2

    # Amazon precipitation loss
    precip = predictions.physics.convection.precnv + predictions.physics.condensation.precls
    amazon_mask = create_teleconnection_mask(coords.horizontal, 'amazon')
    amazon_precip = jnp.sum(precip * amazon_mask) / jnp.sum(amazon_mask)
    baseline_amazon = jnp.sum(baseline.precip * amazon_mask) / jnp.sum(amazon_mask)
    L_amazon = jnp.maximum(0, baseline_amazon - amazon_precip) ** 2  # Penalize decreases

    # Tropical precipitation loss
    tropical_mask = create_tropical_mask(coords)
    tropical_precip = jnp.sum(precip * tropical_mask) / jnp.sum(tropical_mask)
    baseline_tropical = jnp.sum(baseline.precip * tropical_mask) / jnp.sum(tropical_mask)
    L_tropics = (tropical_precip - baseline_tropical) ** 2

    # Regularization
    L_reg = jnp.mean(mcb_forcing ** 2) + jnp.mean(jnp.abs(jnp.diff(mcb_forcing)))

    # Weighted sum
    loss = (
        weights['temperature'] * L_temp +
        weights['amazon'] * L_amazon +
        weights['tropics'] * L_tropics +
        weights['regularization'] * L_reg
    )

    return loss
```

---

## Phase 4: Differentiable Unrolling

### 4.1 Control Loop Architecture

```
For each control interval (e.g., 30 days):
    1. Extract current climate state from simulation
    2. Pass state through policy network → get MCB forcing
    3. Apply MCB forcing to model physics
    4. Step model forward by control interval
    5. Accumulate loss

After full rollout:
    6. Compute total loss
    7. Backpropagate through entire trajectory
    8. Update policy network weights
```

### 4.2 Implementation

**File**: `jcm/mcb/controller.py`

```python
def unroll_with_policy(
    model,
    policy_fn,
    policy_params,
    initial_state,
    forcing,
    terrain,
    baseline,
    coords,
    control_interval_days=30,
    total_days=365,
    loss_weights=None,
):
    """Unroll climate simulation with neural network control.

    Args:
        model: JCM Model instance
        policy_fn: Policy network apply function
        policy_params: Policy network parameters
        initial_state: Starting modal state
        forcing: Base ForcingData
        terrain: TerrainData
        baseline: Baseline climate (for anomalies)
        coords: Model coordinates
        control_interval_days: Days between policy applications
        total_days: Total simulation length
        loss_weights: Dict of loss term weights

    Returns:
        (total_loss, final_state, trajectory)
    """
    state = initial_state
    total_loss = 0.0
    trajectory = []

    num_intervals = total_days // control_interval_days

    def step_interval(carry, _):
        state, cumulative_loss = carry

        # Run model for one interval to get current climate
        predictions = model.run_from_state(
            state, forcing,
            save_interval=control_interval_days,
            total_time=control_interval_days
        )

        # Extract state features
        state_features = extract_state_features(predictions, baseline, coords)

        # Get MCB forcing from policy
        mcb_perturbation = policy_fn(policy_params, state_features)

        # Apply ocean mask
        ocean_mask = 1.0 - terrain.fmask
        mcb_perturbation = mcb_perturbation * ocean_mask

        # Create MCB config
        mcb_config = MCBConfig.from_spatial_field(
            mcb_perturbation,
            active_mask=ocean_mask,
        )

        # Update physics with MCB
        physics_with_mcb = SpeedyPhysics(mcb_config=mcb_config)
        model_with_mcb = model.copy(physics=physics_with_mcb)

        # Step forward with MCB applied
        final_state, predictions = model_with_mcb.run_from_state(
            state, forcing,
            save_interval=control_interval_days,
            total_time=control_interval_days
        )

        # Compute loss for this interval
        interval_loss = compute_climate_loss(
            predictions, baseline, target_cooling=-0.5,
            mcb_forcing=mcb_perturbation, coords=coords,
            weights=loss_weights
        )

        return (final_state, cumulative_loss + interval_loss), predictions

    # Use lax.scan for efficient unrolling
    (final_state, total_loss), trajectory = jax.lax.scan(
        step_interval,
        (initial_state, 0.0),
        None,
        length=num_intervals
    )

    return total_loss, final_state, trajectory
```

---

## Phase 5: BPTT Training Loop

### 5.1 Training Function

**File**: `jcm/mcb/train.py`

```python
import jax
import optax

def create_train_step(model, policy, coords, terrain, baseline, loss_weights):
    """Create JIT-compiled training step."""

    @jax.jit
    def train_step(policy_params, opt_state, initial_state, forcing):
        def loss_fn(params):
            total_loss, _, _ = unroll_with_policy(
                model=model,
                policy_fn=policy.apply,
                policy_params=params,
                initial_state=initial_state,
                forcing=forcing,
                terrain=terrain,
                baseline=baseline,
                coords=coords,
                control_interval_days=30,
                total_days=365,
                loss_weights=loss_weights,
            )
            return total_loss

        loss, grads = jax.value_and_grad(loss_fn)(policy_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, policy_params)
        new_params = optax.apply_updates(policy_params, updates)

        return new_params, new_opt_state, loss

    return train_step


def train_policy(
    model,
    policy,
    coords,
    terrain,
    forcing,
    baseline,
    num_epochs=100,
    learning_rate=1e-3,
    loss_weights=None,
):
    """Train MCB policy network via BPTT."""

    # Initialize policy
    dummy_input = jnp.zeros(state_feature_dim)
    policy_params = policy.init(jax.random.PRNGKey(0), dummy_input)

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(policy_params)

    # Training step
    train_step = create_train_step(
        model, policy, coords, terrain, baseline, loss_weights
    )

    # Initial state
    initial_state = model._prepare_initial_modal_state()

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        policy_params, opt_state, loss = train_step(
            policy_params, opt_state, initial_state, forcing
        )
        losses.append(float(loss))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return policy_params, losses
```

### 5.2 Memory Optimization

For long rollouts, use gradient checkpointing:

```python
from jax import checkpoint

@checkpoint
def step_interval_checkpointed(carry, _):
    # Same as step_interval but memory-efficient
    ...
```

---

## Phase 6: Experimental Design

### 6.1 Experiments

| Experiment | Description | Duration | Notes |
|------------|-------------|----------|-------|
| Baseline | No MCB, establish climate reference | 10 years | Extract baseline temperature/precip |
| Static MCB | Uniform 5% albedo increase in SE Pacific | 10 years | Compare to trained policy |
| Policy Training | Train neural controller | 100 epochs × 1 year | BPTT through simulation |
| Policy Evaluation | Run trained policy | 10 years | Assess cooling + teleconnections |
| Ablation | Vary loss weights, network size | Variable | Sensitivity analysis |

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
├── __init__.py           # Exports (update)
├── mcb_config.py         # ✅ MCBConfig struct
├── mcb_regions.py        # ✅ Region masks
├── mcb_forcing.py        # ✅ Core forcing
├── mcb_test.py           # ✅ Tests (update)
├── policy.py             # NEW: Policy networks (Flax)
├── state_features.py     # NEW: Feature extraction
├── loss.py               # NEW: Climate loss function
├── controller.py         # NEW: Differentiable unrolling
└── train.py              # NEW: BPTT training loop
```

---

## Implementation Order

### Step 1: Policy Network (`policy.py`)
- [ ] Implement `MCBPolicyMLP` class
- [ ] Implement `MCBPolicyCNN` class (optional)
- [ ] Add parameter initialization helpers

### Step 2: State Features (`state_features.py`)
- [ ] Implement `extract_state_features()`
- [ ] Add tropical/regional mask utilities
- [ ] Handle baseline anomaly computation

### Step 3: Loss Function (`loss.py`)
- [ ] Implement `compute_climate_loss()`
- [ ] Add individual loss term functions
- [ ] Ensure full differentiability

### Step 4: Controller (`controller.py`)
- [ ] Implement `unroll_with_policy()`
- [ ] Add `jax.lax.scan` for efficient unrolling
- [ ] Add gradient checkpointing option

### Step 5: Training (`train.py`)
- [ ] Implement `create_train_step()`
- [ ] Implement `train_policy()`
- [ ] Add logging and checkpointing

### Step 6: Tests (`mcb_test.py`)
- [ ] Test policy forward pass
- [ ] Test loss function gradients
- [ ] Test single unroll step
- [ ] Test BPTT gradient flow

---

## Verification

### Unit Tests
```bash
pytest jcm/mcb/mcb_test.py -v
```

### Integration Test
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

### Training Verification
```python
# Verify loss decreases over training
losses = train_policy(model, policy, ...)
assert losses[-1] < losses[0]  # Loss should decrease
```
