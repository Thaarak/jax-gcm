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
| Phase 6: Experimental Design | 🔄 In Progress | Training + evaluation done, needs GPU retraining |
| Phase 7: Coupled Earth System | ✅ Framework Complete | JAX-ESM + coupled training framework, MCB cooling achieved (-0.33 K) |
| Phase 8: GPU Training | ⚠️ Complete (Trivial Solution) | 196 epochs on diya GPU, loss minimally decreased, policy outputs constant |

**Test Coverage**: 54 unit tests passing | **Linting**: All checks pass

### Current Status (as of latest update)
- ✅ Framework complete and working (gradients flow correctly)
- ✅ Baseline simulation complete (1 year, all metrics)
- ✅ First training attempt complete (50 epochs, CPU)
- ✅ Evaluation complete (policy not working - causes warming)
- ✅ Diagnostics complete (identified 4 root causes)
- ✅ JAX-ESM coupling verified (MCB achieves -0.33 K cooling)
- ✅ **Coupled training framework complete** (4 new modules + training script)
- ✅ **GPU training complete (diya)** - 196 epochs, 50.7 minutes
- ⚠️ **Policy learned trivial solution** - constant output, no cooling
- ✅ **Post-mortem complete** - 5 compounding issues identified (see Revised Roadmap below)
- ✅ **Stage 0 complete** - plumbing was BROKEN (confirmed silent zero); fixed by threading `mcb_perturbation` through `ForcingData`; gradient verified vs finite differences (1.6% error). **GATE OPEN**
- ✅ **Stage 1 complete** - direct pattern optimization on diya GPU: achieved **-0.097 K** (target -0.1 K) with a **spatially structured** pattern (peak ~50°N). Both success criteria MET
- ✅ **Stage 2 complete** - paired no-MCB baseline **trajectory** features/loss implemented and verified (`run_stage2_verification.py`): bitwise drift cancellation, loss 100% cooling-dominated at zero MCB, finite non-zero BPTT gradients. **GATE OPEN**
- ✅ **Stage 3 complete** - NN policy trained on diya (60-day horizon, warm-started from Stage 1): **-0.1015 K** paired cooling (target -0.1 K, beats static pattern's -0.0972 K), state-dependent output (interval forcing differs by up to 0.141), loss ≈ Stage 1 reference. 180-day horizon FAILED (exploding BPTT gradients 1e7–1e13) — 60 days is the proven trainable horizon
- ⚠️ **Stage 4 complete (3/4 gates)** - varied-IC ensemble training on diya (4 train ICs @ spin-up days 0/45/90/135 + 2 held-out @ 180/225; 60-day/2×30-day BPTT; warm-started from Stage 3, input dim 11→13 via zero-padded `expand_policy_input` + 2 absolute-SST features). Early-stopped epoch 26 (best epoch 6, mean train loss 0.012925; ~122 s/epoch; no NaNs). Eval over 6 ICs × 3 policies: **Gate 2 PASS** (stage4 held-out loss 0.008416 ≤ stage1-static 0.009495, < stage3 0.009607); **Gate 3 PASS** (genuine state-dependence: cross-IC forcing std interval-0 = 6.6e-5, interval-1 = 4.2e-3, vs ~1e-9 structural zero for 11-feature policies — the absolute-SST features enable interval-0 state-dependence); **Gate 4 PASS** (stable, early-stopped ≪150 epochs). **Gate 1 FAIL**: held-out mean day-60 dSST −0.1285 K overshoots the [−0.12, −0.08] window (train dSST −0.1044 K is on-target; stage4's held-out cooling is nonetheless the closest-to-target of all three policies vs stage3 −0.1503, stage1 −0.1325). Artifacts in `mcb_experiments_gpu/stage4/`
- ⏳ **NEXT**: Stage 4 follow-up (tune held-out cooling magnitude / target weighting to bring held-out dSST into the gate window) or Stage 5 (realistic terrain + teleconnection penalties)

---

## ⭐ Revised Roadmap (Current Strategy - Supersedes Previous "Retrain" Plan)

Post-mortem of the GPU training run (Phase 8) identified **five compounding issues**, only one of
which (zero features) was previously known. Fixing features alone would likely NOT have produced a
working policy. This staged roadmap de-risks each issue in order of cheapness and severity.

### Full Post-Mortem: Diagnosed Issues

| # | Issue | Evidence | Consequence |
|---|-------|----------|-------------|
| 1 | **Unverified MCB injection path** | Training uses `carry["derived"]["mcb_perturbation"]` → `model.physics.set_mcb_perturbation()` — an attribute mutation *inside a traced function* (`jax-esm/jem/components/JCM.py:90`). Never tested with a large perturbation. The verified -0.33 K result used a *different* path (`MCBConfig → SpeedyPhysics(mcb_config=...)`). | The observed 0.0000 K MCB effect may mean the perturbation **never reaches the physics** (baked in as zeros at JIT trace time) — not merely that the output was small |
| 2 | **Zero input features** | Baseline computed from initial state; training starts from same state → all anomalies = 0 | Weight gradients are zero (inputs are zero); only bias can learn → constant output |
| 3 | **Loss dominated by uncontrollable terms** | Final loss = 5.33, but the `sst_cooling` term with ΔSST≈0 vs target -0.1 K contributes only ~0.01 | ~99.8% of the loss is (near-)constant w.r.t. the policy → vanishing gradients; meanwhile reg + smoothness actively push output to zero. **The trivial solution is the attractor of this loss** |
| 4 | **Aquaplanet vs land loss terms** | `TerrainData.aquaplanet(coords)` has no land, but the loss penalizes Amazon/Sahel precipitation | Physically meaningless terms contribute noise/constants to the loss |
| 5 | **Baseline is a state, not a trajectory** | Coupled aquaplanet drifts +0.29 K / 90 days with no MCB | Anomalies vs a static baseline conflate MCB effect with natural model drift |

### Stage 0: Verify Plumbing & Diagnose Loss ✅ COMPLETE (2026-07-01) — GATE OPEN

Diagnostic script: `run_stage0_plumbing_test.py` (reuses the exact training path via
`setup_coupled_model` + `create_coupled_step_fn` + `run_coupled_interval`).

- [x] Inject constant `mcb_perturbation = 0.1` through the **training code path** → **initially FAILED**: MCB effect was *bitwise* +0.0000 K over the forward run. Issue #1 confirmed: `set_mcb_perturbation()` attribute mutation is invisible to `run_from_state`'s `@jax.jit` with static `self` — the perturbation was baked in as `None` at first trace
- [x] **Plumbing fixed** by threading the perturbation through a **traced argument**:
  - `jcm/forcing.py`: added `mcb_perturbation` field to `ForcingData` (zeros = inactive; defaults to zeros in `zeros()`, `ones()`, `copy()`, `from_file`, `default_forcing`)
  - `jcm/physics/speedy/forcing.py`: `set_forcing` now applies `alb_s += (1 - fmask) * forcing.mcb_perturbation` unconditionally (removed the dead `mcb_perturbation` kwarg); static `mcb_config` path unchanged
  - `jcm/physics/speedy/speedy_physics.py`: removed `set_mcb_perturbation()` / `clear_mcb_perturbation()` / `self.mcb_perturbation` (broken-by-design under jit)
  - `jax-esm/jem/components/JCM.py`: step function now does `forcing = forcing.copy(mcb_perturbation=...)` from `carry["derived"]["mcb_perturbation"]`
  - Regression: `forcing_test.py`, `speedy_physics_test.py`, `shortwave_radiation_test.py` all pass
- [x] **Test 1 after fix (60-day forward, 0.1 over all ocean)**: baseline dSST = +0.189 K, MCB dSST = -0.218 K → **MCB effect = -0.407 K / 60 days** (consistent with the -0.33 K / 90 days static-path result which covered only stratocumulus regions)
- [x] **Test 2 gradient check (10-day rollout)**: `d(global_SST)/d(uniform_mcb_scalar)` = **-0.752 K per unit albedo** (AD) vs **-0.740** (central finite difference), relative error **1.6%** — gradient is real, negative, and correct
- [x] **Test 3 loss breakdown (60-day states, training config)**: `sst_uniformity` dominates (**92-98% of total**) because it measures spatial deviation vs the *initial-state* baseline, which conflates model drift with MCB effect (Issue #5). `sst_cooling` is only 2-7%. On aquaplanet, amazon/sahel are ~0. → Confirms Stage 2 must use a **paired baseline trajectory** and rebalanced weights before policy training

**Gate: OPEN.** Both plumbing and gradient checks pass; proceed to Stage 1.

### Stage 1: Direct Static Pattern Optimization ✅ COMPLETE (2026-07-01) — BOTH CRITERIA MET

Script: `optimize_mcb_pattern.py`. Run on diya GPU: 150 Adam iterations, 60-day coupled rollouts,
paired no-MCB baseline, sigmoid parameterization (`0.15 * sigmoid(theta) * ocean_mask`),
loss = cooling (w=1.0) + uniformity (w=0.1) + reg (w=0.001) + smoothness (w=0.001), **no land terms**.

- [x] Optimized the 96×48 MCB field directly (no NN) via gradient descent through the coupled model
- [x] **Achieved dSST = -0.0972 K vs paired baseline (target -0.1 K)** — cooling-loss component driven to ~0 by iteration ~87
- [x] **Pattern is spatially structured** (std 0.029; range 0.0045–0.132): brightening concentrated in NH mid-latitudes (zonal-mean peak 0.116 at ~50°N, band ~40–55°N) over a ~0.02 background, with zonal structure (within-band std up to 0.03)
- [x] Reference loss scale for Stage 3: best total loss **0.004497** (uniformity now dominates the residual, fluctuating ~0.05–0.07 raw — weather noise in the 60-day paired difference)
- [x] Warm-start artifact saved: `mcb_experiments(_gpu)/stage1/stage1_optimized_pattern.pkl` (best_theta, best_pattern, baseline SST, full history, config)
- Performance: 26.0 s/iter on GB10 (60-day rollout fwd+bwd), 65 min total. Memory fine with per-day `jax.checkpoint` + trajectory-discarding scan

**Rationale (validated)**: the simulator, loss, and gradients are now all proven end-to-end. Any Stage 3 failure is attributable to the policy/feature setup, not the substrate.

### Stage 2: Fix Features & Loss for Policy Training ✅ COMPLETE (2026-07-02) — GATE OPEN

Verification script: `run_stage2_verification.py` (3 tests, all PASS — see Change Log for numbers).

- [x] Precompute **paired no-MCB baseline trajectory** over the full training window: `CoupledBaselineTrajectory` struct + `compute_baseline_trajectory()` (single `lax.scan` from the same initial carry; per-step SST, surface T, precipitation, heat flux; `at_step(t)` supports traced indices)
- [x] Features: `X(t) − X_baseline(t)` — `extract_coupled_features` takes `baseline_trajectory.at_step(t_start)`; verified anomalies are **bitwise zero** under zero MCB (drift cancels exactly)
- [x] Loss: paired difference vs baseline trajectory at matching timestep (`at_step(t_end)` inside each control interval)
- [x] Removed Amazon/Sahel/tropics terms for aquaplanet (`CoupledLossWeights` defaults now 0.0; zero-weight components skipped at trace time to kill constant offsets like softplus(0))
- [x] Rebalanced weights so `sst_cooling` dominates: verified breakdown = **100.0%** sst_cooling at zero MCB (raw = target² exactly)
- [x] Added time-of-rollout feature (`include_time=True`, feature dim 11) — non-zero policy input at the first interval where paired anomalies are exactly 0

### Stage 3: Neural Network Policy Training (the end goal) ✅ COMPLETE (2026-07-06)

Training script: `run_coupled_training.py --warm-start`; evaluation: `run_stage3_eval.py`.
See Change Log for the full story (180-day failure → 60-day success) and numbers.

- [x] Warm-start implemented (`warm_start_params`): output bias = Stage 1 `best_theta`, output kernel = 0 → initial policy output **exactly** the Stage 1 pattern (verified to 2.2e-08); MLP (256, 256), feature dim 11
- [x] **180-day horizon FAILED**: grad norms 1e7–1e13 (chaos-amplified BPTT noise), loss climbed 0.225 → 0.61, early-stopped with zero learning. **60 days (2×30-day intervals) is the proven trainable horizon** — matches Stage 1
- [x] 60-day retrain on diya GPU: warm-start loss 0.009911 → best **0.007421** (epoch 12, −25% vs static under the same objective); sane grads (0.003–0.02); early-stopped epoch 32; 31.7 s/epoch, 18.6 min total
- [x] Success criteria (eval, `run_stage3_eval.py` — trained vs stage1-static under identical fresh compilation):
  - **Cooling: -0.1015 K** vs target -0.1 K (static pattern: -0.0972 K) → **MET, beats static**
  - **Non-constant, state-dependent output**: spatial std 0.0441; interval-2 forcing differs from interval-1 by up to **0.141** (static: 0.000) → **MET**
  - **Loss ≈ Stage 1 reference**: eval mean loss 0.004461 vs gate 0.004497; trained-vs-static loss comparison is within GPU cross-compilation chaos noise (~15% level) → **MET (matched)**. Note: on a single deterministic trajectory a static pattern is near-optimal by construction; beating it decisively requires Stage 4's varied initial conditions
- Artifacts: `mcb_experiments(_gpu)/stage3_60d/` (checkpoint, training history, baseline trajectory, eval results, logs)

### Stage 4: Make Feedback Meaningful (controller justification) ⏳ NEXT

In a deterministic single-trajectory setup, a "policy" is effectively an open-loop schedule — the
feedback aspect adds nothing. To demonstrate a genuine adaptive *controller* (the novel contribution
per MCB_CONTEXT.md):

- [ ] Train across varied initial conditions / perturbed initial states
- [ ] Verify policy output actually depends on state (responds differently to different climates)
- [ ] Later: switch to realistic terrain + seasonal boundary conditions → reinstate Amazon/Sahel teleconnection penalties (this is when the original research goal — cooling with minimized teleconnections — becomes fully testable)
- [ ] Long-horizon stability evaluation (multi-year rollout with trained policy)

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

**Bug Fix Applied**: Updated field access to use correct JCM prediction structure:
- Surface temperature: `predictions.physics.surface_flux.tsfc` (not `temperature_tendency.surface_temp`)
- Precipitation: `predictions.physics.convection.precnv` + `predictions.physics.condensation.precls`
- Added time dimension squeezing for fields with shape `(1, ix, il)`

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

**Bug Fix Applied**: Updated field access to match JCM prediction structure (same as state_features.py)

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

## Phase 6: Experimental Design (In Progress)

### 6.1 Experiments

| Experiment | Description | Duration | Status |
|------------|-------------|----------|--------|
| Baseline | No MCB, establish climate reference | 1 year | ✅ Complete |
| Baseline Visualization | Globe plots of climate variables | - | ✅ Complete |
| Policy Training (CPU) | Train neural controller on CPU | 50 epochs × 90 days | ✅ Complete (insufficient) |
| Policy Evaluation | Run trained policy, compare to baseline | 90 days | ✅ Complete |
| Training Analysis | Diagnose why training was insufficient | - | ✅ Complete |
| Static MCB | Uniform 5% albedo increase in SE Pacific | 10 years | ⏳ Pending |
| Policy Training (GPU) | Retrain with tuned hyperparameters | 500 epochs × 180 days | ⏳ Pending |
| Ablation | Vary loss weights, network size | Variable | ⏳ Pending |

### 6.2 Baseline Simulation Results ✅

**Completed**: 1-year baseline simulation without MCB forcing.

| Variable | Value |
|----------|-------|
| Grid | 96 × 48 (lon × lat), 8 vertical levels |
| Duration | 365 days (12 monthly outputs) |
| Global Mean Surface Temp | 279.67 K (6.52°C) |
| Global Mean Precipitation | 0.030 model units |
| Global Mean Cloud Cover | 59.4% |

**Output Files**:
- `mcb_experiments/baseline_predictions.nc` - Full xarray dataset
- `mcb_experiments/baseline_climate.pkl` - ClimateBaseline object for training
- `mcb_experiments/baseline_globe_visualization.png` - Globe visualizations

### 6.3 Training Results ✅ (Insufficient - Needs Retraining)

**Completed**: 50 epochs of BPTT training on CPU.

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Learning Rate | 0.001 (too low) |
| Rollout Length | 90 days (too short) |
| Control Interval | 30 days |
| Target Cooling | -0.5 K (too aggressive) |
| Policy Type | MLP (128, 128) |
| Total Training Time | ~2.15 hours (7,745 seconds) |
| Time per Epoch | ~100-150 seconds |

**Loss Curve**:
| Metric | Value |
|--------|-------|
| Initial Loss | 8.372972 |
| Final Loss | 8.371943 |
| Best Loss | 8.371943 |
| Loss Reduction | **0.012%** (insufficient!) |
| Initial Gradient Norm | 0.0008 |
| Final Gradient Norm | 0.0001 (vanishing gradients) |

**Output Files**:
- `mcb_experiments/trained_policy.pkl` - Trained policy parameters + metadata
- `mcb_experiments/training_history.pkl` - Loss and gradient history

### 6.4 Evaluation Results ✅ (Policy Not Working)

**Completed**: 90-day simulation with trained policy.

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Global Temperature Change | **+2.55 K** (warming!) | -0.5 K (cooling) | ❌ FAILED |
| Tropical Temperature Change | +2.55 K | - | ❌ |
| Amazon Precipitation | +14.4% | ≥ 0% | ✅ Protected |
| Sahel Precipitation | **-97.4%** | ≥ 0% | ❌ SEVERE |
| Global Precipitation | +3.5% | - | - |
| Mean MCB Forcing | 0.068 | - | Applied |
| Max MCB Forcing | 0.133 | ≤ 0.15 | ✅ In range |

**Key Finding**: The trained policy causes **warming** instead of cooling. Training was insufficient - the policy learned to apply forcing but not in the correct pattern.

**Output Files**:
- `mcb_experiments/mcb_evaluation_results.pkl` - Evaluation metrics and MCB patterns
- `mcb_experiments/mcb_pattern_visualization.png` - Globe visualization of MCB forcing
- `mcb_experiments/training_analysis.png` - Training diagnostics plots

### 6.5 Training Failure Diagnosis ✅

**Root Causes Identified**:

| Issue | Problem | Impact | Fix |
|-------|---------|--------|-----|
| Learning Rate | 0.001 too conservative | Minimal parameter updates | Increase to 0.01-0.1 |
| Rollout Length | 90 days too short | Climate doesn't respond in time | Use 180-365 days |
| Target Cooling | -0.5 K too aggressive | Loss dominated by gap | Start with -0.1 K |
| Training Duration | 50 epochs insufficient | Policy undertrained | Train 200-500 epochs |
| Hardware | CPU too slow | Limited experimentation | Use GPU/TPU |
| Gradient Flow | Norms decreased 0.0008→0.0001 | Vanishing gradients | Check architecture |

**Recommendations for Next Training Run**:
1. **Learning Rate**: Start with 0.01, consider 0.1
2. **Target Cooling**: Start easy with -0.1 K, then increase
3. **Rollout Length**: Use 180 days minimum on GPU
4. **Epochs**: Train for 200+ epochs with early stopping
5. **Architecture**: Try CNN policy for spatial patterns
6. **Curriculum Learning**: Start with easy targets, gradually increase difficulty

### 6.6 Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Global ΔT | -0.5 K | Temperature reduction |
| Amazon ΔP | ≥ 0% | No precipitation decrease |
| Tropical ΔP | ≤ 5% | Minimal tropical impact |
| MCB Efficiency | max | Cooling per unit forcing |
| Policy Smoothness | high | Avoid erratic forcing changes |

---

## Phase 7: Coupled Earth System MCB (Pending)

### 7.1 Why Coupled Modeling Matters

**Problem with Atmosphere-Only MCB**: The standalone JCM atmospheric model cannot properly simulate global mean temperature changes from MCB because:
- Ocean stores ~93% of Earth's excess heat
- Without ocean feedback, atmospheric temperature changes don't persist
- SST (Sea Surface Temperature) is prescribed, not responsive to forcing changes
- Energy budget is not closed → unrealistic temperature response

**Solution: JAX-ESM Coupling**
- **JAX-ESM** (`jax-esm/`) is a JAX-based Earth System Model coupler
- Couples JCM atmosphere with slab ocean and land models
- Captures large-scale energy flows that influence global mean temperature
- MCB → reduced heat flux to ocean → ocean cools → SST decreases → atmosphere responds
- This is the **physically correct** mechanism for MCB-induced cooling

### 7.2 JAX-ESM Architecture

**Location**: `/Users/thaaraksriram/workspace/jax-gcm/jax-esm/`

| Component | File | Description |
|-----------|------|-------------|
| Coupler | `jem/base/coupler.py` | Orchestrates coupled simulation with `jax.lax.scan` |
| JCM Wrapper | `jem/components/JCM.py` | Adapts JCM for coupled mode via `make_jem_compatible()` |
| Slab Ocean | `jem/components/slab/slab_ocean_model/` | Mixed-layer ocean with heat flux response |
| Slab Land | `jem/components/slab/slab_land_model/` | Land surface temperature model |
| Mapper | `jem/mapping/mapper.py` | Variable exchange between components |

**Coupling Workflow**:
```
For each coupling timestep (e.g., 1 day):
    1. Mapper: Ocean SST → Atmosphere boundary condition
    2. Atmosphere (JCM): Run physics → compute heat flux
    3. Mapper: Heat flux → Ocean forcing
    4. Ocean: Respond to heat flux → update SST
    5. (Optional) Land: Respond to heat flux → update land temperature
```

### 7.3 Key Variable Exchanges

| Direction | Variable | Units | Description |
|-----------|----------|-------|-------------|
| Atm → Ocean | `derived.total_heat_flux` | W/m² | Net heat flux (positive upward) |
| Ocean → Atm | `state.sea_surface_temperature` | K | SST boundary condition |
| Atm → Land | `derived.total_heat_flux` | W/m² | Land surface heat flux |
| Land → Atm | `state.land_surface_temperature` | K | Land temperature |

**Ocean Physics (Slab Model)**:
```
dT/dt = -F_net / (ρ × cp × h)

Where:
  T     = SST (K)
  F_net = net heat flux from atmosphere (W/m², positive upward)
  ρ     = 1025 kg/m³ (ocean density)
  cp    = 3992 J/(kg·K) (specific heat)
  h     = mixed layer depth (40-60 m typical)
```

### 7.4 MCB Integration Strategies

**Option A: Modify Atmospheric Albedo (Current Approach)**
- MCB increases sea surface albedo in JCM
- Reduces shortwave absorbed by surface
- Reduces heat flux to ocean
- Ocean cools → SST decreases → global cooling
- **Status**: Already implemented in `jcm/mcb/`

**Option B: Direct Heat Flux Reduction (Alternative)**
- Apply MCB as a heat flux modifier in the mapper
- `H_net_modified = H_net - MCB_cooling_effect`
- More direct control over energy balance
- **Status**: Not yet implemented

**Recommended**: Start with **Option A** since MCB forcing is already integrated into JCM.

### 7.5 Implementation Checklist

#### Step 1: Setup JAX-ESM ✅
- [x] Install jax-esm: `pip install -e jax-esm/`
- [x] Verify JCM compatibility: Imports work
- [x] Test coupled JCM + slab ocean simulation (no MCB)

#### Step 2: Coupled Baseline ✅
- [x] Create `run_coupled_baseline.py` script
- [x] Run 60-day coupled simulation without MCB
- [x] Verify SST evolution (+0.19 K over 60 days)
- [x] Save coupled baseline for comparison

#### Step 3: MCB in Coupled Mode ✅
- [x] Create `run_coupled_mcb.py` script
- [x] Pass MCB config through `SpeedyPhysics(mcb_config=...)`
- [x] Verify heat flux reduction in MCB regions
- [x] Confirm SST cooling response (-0.14 K in MCB regions)
- [x] Create `run_coupled_mcb_longrun.py` for 90-day run
- [x] Achieved global cooling: -0.035 K (vs baseline +0.29 K)

#### Step 4: Coupled Policy Training ✅
- [x] Adapt training loop for coupled model
- [x] Update loss function for coupled state
- [x] Create coupled feature extraction
- [x] Create coupled controller for BPTT through JAX-ESM
- [x] Create training script `run_coupled_training.py`
- [ ] Train MCB policy with ocean feedback (pending GPU)
- [ ] Compare to atmosphere-only training

### 7.6 Expected Differences from Atmosphere-Only

| Aspect | Atmosphere-Only | Coupled (JAX-ESM) |
|--------|-----------------|-------------------|
| Temperature Response | Unrealistic (no persistence) | Realistic (ocean thermal inertia) |
| Timescale | Fast (days) | Slow (months-years) |
| Energy Budget | Not closed | Closed (heat flux → ocean) |
| SST Feedback | None (prescribed) | Full (interactive) |
| Training Rollout | 90-180 days sufficient | May need 1-2 years |
| Physical Accuracy | Low | High |

### 7.7 Coupled Experiments

| Experiment | Description | Duration | Status |
|------------|-------------|----------|--------|
| Coupled Baseline | JCM + Slab Ocean, no MCB | 60 days | ✅ Complete |
| Static MCB Coupled (short) | Fixed 10% MCB in Sc regions | 60 days | ✅ Complete |
| Static MCB Coupled (long) | Fixed 10% MCB in Sc regions | 90 days | ✅ Complete |
| Policy Training Coupled | BPTT with ocean feedback | TBD | ✅ Framework Ready |
| Long-term Stability | Trained policy, multi-year | TBD | ⏳ Pending |

### 7.8 Coupled MCB Results ✅

**60-Day Coupled Baseline (no MCB)**:
| Metric | Value |
|--------|-------|
| Initial SST | 282.25 K |
| Final SST | 282.44 K |
| SST Change | +0.195 K |

**60-Day Coupled MCB (10% albedo increase in stratocumulus)**:
| Metric | Baseline | With MCB | MCB Effect |
|--------|----------|----------|------------|
| Global SST change | +0.195 K | +0.164 K | -0.031 K |
| MCB region SST change | - | -0.139 K | Local cooling |
| MCB region heat flux | - | 1.48 W/m² | Reduced |

**90-Day Coupled MCB**:
| Metric | Baseline (est.) | With MCB | MCB Effect |
|--------|-----------------|----------|------------|
| Global SST change | +0.293 K | **-0.035 K** | **-0.328 K** |
| MCB region SST change | - | **-0.378 K** | Strong cooling |

**Key Finding**: With coupled atmosphere-ocean modeling, MCB achieves **actual global cooling** (negative SST change), not just reduced warming. The 90-day simulation shows the ocean thermal response accumulating over time.

**Physical Mechanism Verified**:
1. MCB increases surface albedo (+10% in stratocumulus regions)
2. Less solar radiation absorbed → reduced heat flux to ocean
3. Ocean receives less energy → SST decreases
4. SST feedback to atmosphere → global cooling

**Output Files**:
- `mcb_experiments/coupled_baseline_atm.nc` - 60-day baseline atmosphere
- `mcb_experiments/coupled_baseline_ocn.nc` - 60-day baseline ocean
- `mcb_experiments/coupled_mcb_atm.nc` - 60-day MCB atmosphere
- `mcb_experiments/coupled_mcb_ocn.nc` - 60-day MCB ocean
- `mcb_experiments/coupled_mcb_90day_atm.nc` - 90-day MCB atmosphere
- `mcb_experiments/coupled_mcb_90day_ocn.nc` - 90-day MCB ocean
- `mcb_experiments/coupled_mcb_results.pkl` - 60-day summary
- `mcb_experiments/coupled_mcb_90day_results.pkl` - 90-day summary

### 7.8 Code Example: Coupled MCB Simulation

```python
from jem import Coupler
from jem.components import JCM, SlabOceanModel
from jem.mapping import BasicMapper
import jcm
from jcm.mcb import MCBConfig
import jax_datetime as jdt

# Setup
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(1, "day")

# Create MCB config
mcb_config = MCBConfig(
    albedo_perturbation=albedo_field,  # From policy network
    active_mask=ocean_mask,
)

# Atmosphere with MCB
atm_model = jcm.model.Model(
    start_date=start_datetime,
    coords=get_speedy_coords(),
    mcb_config=mcb_config,  # Pass MCB forcing
)
atm_model = JCM.make_jem_compatible(atm_model, coupling_timestep)

# Ocean
ocn_model = SlabOceanModel(
    start_datetime=start_datetime,
    timestep=coupling_timestep,
)

# Coupling
mapper = BasicMapper()
mapper.add_mapping(("atm", "derived.total_heat_flux"), ("ocn", "forcing.total_heat_flux"))
mapper.add_mapping(("ocn", "state.sea_surface_temperature"), ("atm", "forcing.sea_surface_temperature"))

# Coupler
model = Coupler(
    components={"atm": atm_model, "ocn": ocn_model},
    mappers={"coupling": mapper},
)

# Run coupled simulation
workflow = ["coupling", "atm", "ocn"]
initial_state, final_state, predictions = model.run(
    workflow=workflow,
    iterations=365,  # 1 year
)
```

---

## File Structure

```
jcm/mcb/                          # MCB forcing and policy (existing)
├── __init__.py           # ✅ Exports all public API (updated for coupled)
├── mcb_config.py         # ✅ MCBConfig struct
├── mcb_regions.py        # ✅ Region masks (stratocumulus + teleconnection)
├── mcb_forcing.py        # ✅ Core forcing computation
├── mcb_test.py           # ✅ 54 unit tests
├── policy.py             # ✅ Policy networks (MLP, CNN, ResNet, Hybrid)
├── state_features.py     # ✅ Feature extraction (scalar + spatial) - FIXED
├── loss.py               # ✅ Climate loss function (6 components) - FIXED
├── controller.py         # ✅ Differentiable unrolling with lax.scan
├── train.py              # ✅ BPTT training loop
├── coupled_features.py   # ✅ NEW: Coupled feature extraction (SST, heat flux)
├── coupled_loss.py       # ✅ NEW: Loss functions using ocean SST
├── coupled_controller.py # ✅ NEW: BPTT through JAX-ESM coupler
└── coupled_train.py      # ✅ NEW: Training loop for coupled simulation

jax-esm/                          # Earth System Model Coupler (NEW)
├── jem/                          # Main package
│   ├── base/
│   │   ├── coupler.py            # Core coupling engine (jax.lax.scan)
│   │   └── typing.py             # Component interface definitions
│   ├── components/
│   │   ├── JCM.py                # JCM wrapper (make_jem_compatible)
│   │   └── slab/
│   │       ├── slab_ocean_model/ # Mixed-layer ocean
│   │       └── slab_land_model/  # Land surface model
│   └── mapping/
│       └── mapper.py             # Variable exchange (heat flux ↔ SST)
├── notebooks/                    # Example coupled simulations
│   ├── 01_basic/                 # Aquaplanet setup
│   └── 02_experimental/          # Advanced features
├── tests/                        # Unit tests
└── pyproject.toml               # Package config

Experiment Scripts (root directory):
├── run_mcb_baseline.py           # ✅ Baseline simulation script
├── train_mcb_policy.py           # ✅ Policy training script
├── evaluate_mcb_policy.py        # ✅ Policy evaluation script
├── visualize_baseline_globe.py   # ✅ Baseline visualization script
├── visualize_mcb_patterns.py     # ✅ MCB pattern visualization script
├── analyze_training.py           # ✅ Training analysis/diagnostics script
├── MCB_PRESENTATION.html         # ✅ Interactive presentation (16 slides)
├── run_coupled_baseline.py       # ✅ Coupled baseline (Phase 7)
├── run_coupled_mcb.py            # ✅ Coupled MCB 60-day (Phase 7)
├── run_coupled_mcb_longrun.py    # ✅ Coupled MCB 90-day (Phase 7)
├── run_coupled_training.py       # ✅ NEW: Coupled policy training script
└── mcb_experiments/              # ✅ Output directory
    ├── baseline_predictions.nc         # 1-year baseline simulation
    ├── baseline_climate.pkl            # ClimateBaseline object
    ├── baseline_globe_visualization.png # Baseline climate plots
    ├── trained_policy.pkl              # Trained policy parameters
    ├── training_history.pkl            # Loss/gradient history
    ├── mcb_evaluation_results.pkl      # Evaluation metrics
    ├── mcb_pattern_visualization.png   # MCB forcing pattern plots
    └── training_analysis.png           # Training diagnostics
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
- [x] **Fix field access for JCM predictions** (surface_flux.tsfc, etc.)

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
- [x] **Fix field access for JCM predictions**

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

### Step 8: Experiments (Partially Complete)
- [x] Create `run_mcb_baseline.py` script
- [x] Run 1-year baseline simulation
- [x] Save baseline predictions to NetCDF
- [x] Create `ClimateBaseline` object and save
- [x] Create `visualize_baseline_globe.py` script
- [x] Generate globe visualizations of baseline climate
- [x] Create `train_mcb_policy.py` script
- [x] Run policy training on CPU (50 epochs, ~2 hours)
- [x] Create `evaluate_mcb_policy.py` script
- [x] Run trained policy evaluation (90 days)
- [x] Create `visualize_mcb_patterns.py` script
- [x] Generate MCB pattern visualizations
- [x] Create `analyze_training.py` script
- [x] Diagnose training issues (learning rate, rollout, target)
- [x] Create `MCB_PRESENTATION.html` (16-slide presentation)
- [ ] **Retrain with tuned hyperparameters on GPU**
- [ ] Achieve actual cooling (current: warming)
- [ ] Run ablation studies (loss weights, architectures)

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

### Baseline Simulation ✅
```bash
python run_mcb_baseline.py
# Result: Successfully generated 1-year baseline
# Output: mcb_experiments/baseline_predictions.nc
#         mcb_experiments/baseline_climate.pkl
```

### Baseline Visualization ✅
```bash
python visualize_baseline_globe.py
# Result: Generated globe plots
# Output: mcb_experiments/baseline_globe_visualization.png
```

### Training (CPU) ✅
```bash
python train_mcb_policy.py
# Result: 50 epochs completed in ~2 hours
# Loss: 8.373 → 8.372 (0.012% reduction - INSUFFICIENT)
# Output: mcb_experiments/trained_policy.pkl
#         mcb_experiments/training_history.pkl
```

### Evaluation ✅
```bash
python evaluate_mcb_policy.py
# Result: Policy causes WARMING (+2.55 K) instead of cooling (-0.5 K)
# Amazon: +14.4% (protected)
# Sahel: -97.4% (SEVERE impact)
# Output: mcb_experiments/mcb_evaluation_results.pkl
```

### Visualization ✅
```bash
python visualize_mcb_patterns.py
# Result: Generated MCB forcing pattern visualization
# Output: mcb_experiments/mcb_pattern_visualization.png

python analyze_training.py
# Result: Generated training diagnostics
# Output: mcb_experiments/training_analysis.png
```

---

## Next Steps (TODO)

**⭐ The authoritative task list is the [Revised Roadmap](#-revised-roadmap-current-strategy---supersedes-previous-retrain-plan) at the top of this document.** Summary:

### ✅ Stage 0 — verify plumbing (COMPLETE 2026-07-01, GATE OPEN)
- [x] Test large constant perturbation (0.1) through the dynamic injection path → initially FAILED (bitwise 0.0 K); after fix: **-0.407 K / 60 days**
- [x] Fixed plumbing: `mcb_perturbation` is now a traced `ForcingData` field (attribute mutation removed) — see Stage 0 section in Revised Roadmap for file-level details
- [x] Gradient check: `d(SST)/d(uniform_mcb_scalar)` = -0.752 K/unit (AD), matches finite differences within 1.6%
- [x] Per-component loss breakdown logged (`run_stage0_plumbing_test.py`): uniformity-vs-initial-state dominates 92-98% → confirms Stage 2 paired-baseline requirement

### ✅ Stage 1 — static pattern optimization (COMPLETE 2026-07-01)
- [x] Created `optimize_mcb_pattern.py`: optimizes 96×48 field directly (no NN) through coupled model
- [x] Loss: paired SST cooling + uniformity + reg + smoothness (no Amazon/Sahel — aquaplanet)
- [x] Result: **-0.097 K achieved (target -0.1 K)**, structured pattern (peak ~50°N); best loss 0.004497 = Stage 3 reference; artifact: `stage1/stage1_optimized_pattern.pkl`

### ✅ Stage 2 — fix features & loss (COMPLETE 2026-07-02, GATE OPEN)
- [x] Paired no-MCB baseline **trajectory**: `CoupledBaselineTrajectory` + `compute_baseline_trajectory()`; `run_coupled_training.py` precomputes and pickles it
- [x] Features and loss computed as `X(t) − X_baseline(t)` at matching timesteps — verified **bitwise** drift cancellation
- [x] Land teleconnection terms removed for aquaplanet (weights 0.0, skipped at trace time)
- [x] Weights rebalanced: zero-MCB loss is **100% sst_cooling** (= target²)
- [x] Time-of-rollout feature added (dim 11); BPTT gradient through interval-indexed unroll verified (|grad| = 7.29, no NaNs)
- [x] Gate script: `run_stage2_verification.py` — all 3 tests PASS

### ✅ Stage 3 — NN policy training (COMPLETE 2026-07-06)
- [x] Codebase re-synced to diya; Stage 2 gate re-verified on GPU (needs `--tol 1e-2`: cross-program XLA noise, not a logic bug — CPU stays bitwise)
- [x] Warm-start from Stage 1 pattern (`--warm-start`, output bias = best_theta, kernel = 0)
- [x] Trained on diya GPU at **60-day** horizon (180 days fails: exploding BPTT gradients)
- [x] Gates: **-0.1015 K** cooling (beats static -0.0972 K); state-dependent non-constant output (interval diff 0.141); loss ≈ Stage 1 reference (0.004461 vs 0.004497)

### ⏳ NEXT (Stage 4 — genuine feedback controller)
- [ ] Varied initial conditions; verify state-dependence of policy output
- [ ] Realistic terrain + seasonal BCs → reinstate Amazon/Sahel penalties (original research goal)
- [ ] Multi-year stability run; architecture/loss-weight ablations

### ✅ COMPLETED
- [x] **Coupled Earth System (Phase 7)**: jax-esm install, coupling verified, static MCB -0.328 K
- [x] **Coupled training framework**: 4 modules + `run_coupled_training.py`
- [x] **GPU environment (diya)**: JAX + CUDA working, ~15.5 s/epoch (6-10x CPU speedup)
- [x] **First GPU training run**: 196 epochs — trivial solution, post-mortem complete

### ❌ SUPERSEDED (do not pursue)
- ~~Fix atmosphere-only training~~ — atmosphere-only cannot capture MCB cooling (no ocean feedback); all training is now coupled
- ~~Hyperparameter sweeps / curriculum learning before Stage 0-2~~ — pointless while gradients don't reach the policy; revisit after Stage 3 works
- ~~Climatological-mean baseline~~ — paired baseline trajectory chosen instead (isolates MCB effect from drift exactly)

---

## Known Issues

### ✅ RESOLVED: Baseline Feature Issue (Was Blocking Training)

**Problem**: `CoupledBaseline.from_coupled_carry(initial_carry, coords)` created the baseline from the **initial simulation state**. When training starts from the same initial state, all anomaly features were zero → policy could only learn the bias term (constant output, no cooling). Anomalies vs a static baseline also conflated MCB effect with natural drift (+0.29 K / 90 days).

**Fix (implemented, Stage 2 2026-07-02)**: Paired no-MCB baseline **trajectory**
(`CoupledBaselineTrajectory` + `compute_baseline_trajectory()` in `jcm/mcb/coupled_features.py`).
Features and loss are computed as `X(t) − X_baseline(t)` at matching timesteps, isolating the
MCB-caused signal from drift exactly. A time-of-rollout feature (`include_time=True`) gives the
policy a non-zero input at the first interval, where paired anomalies are exactly 0 by construction.

**Verified** (`run_stage2_verification.py` Test 1): zero-MCB rollout matches the precomputed
trajectory **bitwise** (max |dSST| = 0.0); all anomaly features = 0.0; time feature exact.

### ✅ RESOLVED: Dynamic MCB Injection Path Was Broken (Silent Zero)

**Problem**: The training loop injected MCB via `carry["derived"]["mcb_perturbation"]` →
`model.physics.set_mcb_perturbation()` inside the JEM step function — an **attribute mutation
inside a traced/JIT'd function**. `run_from_state` is `@jax.jit` with static `self`, so the
perturbation was baked in as `None` at first trace and all runtime values silently ignored.

**Confirmation (Stage 0, 2026-07-01)**: A 0.1 albedo perturbation over ALL ocean produced a
*bitwise* +0.0000 K SST difference, and `d(SST)/d(mcb_scalar)` was exactly 0.0. All GPU training
gradients w.r.t. the ocean pathway were structurally zero — the policy could never have learned.

**Fix (implemented)**: `mcb_perturbation` is now a field of `ForcingData` (a traced argument of
`run_from_state`), applied in `set_forcing` as `alb_s += (1 - fmask) * forcing.mcb_perturbation`.
The JEM step injects via `forcing.copy(mcb_perturbation=...)`. The mutation-based API was removed.
Verified: -0.407 K / 60 days effect; AD gradient matches finite differences within 1.6%.

### ✅ RESOLVED: Loss Dominated by Terms the Policy Cannot Influence

**Problem**: Final training loss was 5.33, but with ΔSST≈0 and target -0.1 K, the `sst_cooling` term
contributed only ~0.01. The remaining ~5.3 came from precipitation/uniformity terms computed against
broken zero baselines — large near-constants with near-zero gradients w.r.t. the policy.
Regularization + smoothness then actively pushed output toward zero: the trivial solution was the
attractor of that loss configuration.

**Fix (implemented, Stage 2 2026-07-02)**: `CoupledLossWeights` defaults rebalanced to the
Stage 1-proven set (sst_cooling=1.0, sst_uniformity=0.1, reg=0.001, smoothness=0.001); zero-weight
components are skipped at trace time (kills constant offsets like softplus(0)/100). Verified
(`run_stage2_verification.py` Test 2): at zero MCB with the paired baseline, the loss is
**100.0% sst_cooling** (raw = target² exactly) — the cooling objective is now the attractor.

### ✅ RESOLVED: Aquaplanet Configuration vs Land Teleconnection Terms

**Problem**: Experiments use `TerrainData.aquaplanet(coords)` — there is **no land** — yet the coupled
loss penalized Amazon and Sahel precipitation. These terms are physically meaningless in this
configuration and only added noise/constants to the loss.

**Fix (implemented, Stage 2 2026-07-02)**: Amazon/Sahel/tropics weights default to 0.0 and
zero-weight terms are skipped entirely (verified exactly 0.0 in the Test 2 breakdown). Reinstate
them when switching to realistic terrain + boundary conditions (Stage 4), which is when the original
research goal (cooling with minimized teleconnections) becomes fully testable.

### NumPy 2.x Compatibility Warning
Some dependencies (numexpr, sklearn) show warnings about NumPy 2.x compatibility but continue to function. The warnings can be ignored or resolved by reinstalling affected packages.

### BPTT Computational Cost
Backpropagation through a climate simulation is computationally expensive:
- CPU: ~100-150 seconds per epoch (50 epochs = 2 hours)
- GPU/TPU: Required for practical training times (estimate 10-20x speedup)

**Workaround options**:
1. Use shorter rollouts (30-90 days instead of 365)
2. Use smaller policy networks
3. Run on GPU/TPU hardware
4. Use gradient accumulation with smaller batches

### Pickle Loading Issues
When loading `.pkl` files that contain `jcm.mcb` objects, Python triggers the full import chain which can cause NumPy compatibility warnings. For scripts that only need numeric data (like visualizations), avoid importing jcm modules.

### Training Convergence Issues (IMPORTANT)
First training attempt showed minimal convergence:
- Loss reduced by only 0.012% over 50 epochs
- Gradient norms decreased (potential vanishing gradients)
- Policy learned to apply forcing but in wrong patterns

**Root causes identified**:
1. Learning rate too low (0.001)
2. Rollout too short (90 days) for climate response
3. Target too aggressive (-0.5 K)
4. Insufficient epochs (50)

**DO NOT repeat these mistakes** - see Section 6.5 for fixes.

---

## Change Log

### 2026-07-06: Stage 3 Complete — NN Policy Trained (60-Day Horizon), Beats Static Pattern on Cooling
- **Warm-start implemented** (`run_coupled_training.py`): new `--warm-start` flag + `warm_start_params()` — loads Stage 1 `best_theta`, sets the MLP output bias to `theta.reshape(-1)` and zeros the output kernel. Since the output layer is `0.15 * sigmoid(logits)` (identical to Stage 1's parameterization), the initial policy output equals the Stage 1 pattern **exactly** (verified to 2.2e-08) regardless of input features. Hidden-layer grads are 0 at step 0 (zero kernel) but unblock after the first update (observed: epoch-0 |grad| 1e-4 → epoch-1 |grad| 52). `jcm/mcb/coupled_train.py`: `train_coupled_policy` gained `initial_params` parameter
- **GPU gate tolerance** (`run_stage2_verification.py`): added `--tol` (default 1e-6). On GPU, the trajectory scan and interval re-runs are differently compiled XLA programs → cross-program equality is not bitwise (max local |dSST| 7.6e-04 K over 6 days; area-mean 2.6e-06 K — 5 orders below the -0.1 K signal). CPU remains bitwise. GPU gate re-run with `--tol 1e-2`: all 3 tests PASS
- **diya incident (resolved)**: launching JAX wedged the machine hard (ping alive, SSH dead; required physical reboot). Root cause: the auto-starting vLLM Docker container (`aeon-ultimate-xs`, `--gpu-memory-utilization 0.75`) holds ~96 GB of the GB10's 121 GB **unified** memory; JAX's default 75% preallocation on top wedged the OS. Mitigations (now standard procedure): `docker stop aeon-ultimate-xs` before training (restart after), and `XLA_PYTHON_CLIENT_PREALLOCATE=false` on every run
- **180-day training FAILED (diagnosed)**: warm-start loss 0.225; grad norms 1e7–1e13 (random-init validation: 2.6e13) — chaos-amplified BPTT gradient noise beyond the proven 60-day regime; loss climbed monotonically to 0.61, early-stopped at epoch 21 with best = epoch 0 (zero learning). Artifacts kept in `stage3/` as a negative result
- **60-day retrain SUCCEEDED** (200 epochs max, lr 0.01, Adam, clip 1.0, 2×30-day intervals, target -0.1 K): warm-start epoch-0 loss 0.009911 → best **0.007421** at epoch 12 (**-25%** vs the static Stage 1 pattern under the same compiled objective); grad norms sane throughout (0.003–0.02); early-stopped epoch 32; 31.7 s/epoch, 1115 s total
- **Created `run_stage3_eval.py`**: paired 60-day rollout comparing trained checkpoint vs stage1-static (warm-start params) under one fresh compilation:
  - trained: mean loss 0.004461, **dSST -0.1015 K**, forcing mean/max 0.037/0.150, spatial std 0.0441, max |interval2−interval1| forcing **0.141** (state-dependent)
  - stage1-static: mean loss 0.004227, dSST -0.0972 K, spatial std 0.0291, interval diff 0.000
  - Caveats: (a) the 60d policy loss averages the day-30 and day-60 interval losses, so it is not numerically identical to Stage 1's single end-of-rollout loss 0.004497; (b) absolute losses shift ~15% across XLA compilations (chaotic 60-day divergence), so trained-vs-static loss ordering is within noise — the physical metric (day-60 paired dSST) favors the trained policy
- **Gate verdict**: cooling MET (-0.1015 K, beats static), structure/state-dependence MET, loss MET (matched Stage 1 reference). On a single deterministic trajectory a static pattern is near-optimal by construction — decisively beating it is exactly Stage 4's job (varied initial conditions)
- **Artifacts**: diya + local `mcb_experiments(_gpu)/stage3_60d/`: `coupled_trained_policy.pkl` (best params, metadata), `coupled_training_history.pkl`, `baseline_trajectory_60d.pkl`, `eval_results.pkl`, `train.log`, `eval.log`
- **Next**: Stage 4 — train across varied initial conditions, verify genuine state-dependence, then realistic terrain + teleconnection penalties

### 2026-07-02: Stage 2 Complete — Paired Baseline Trajectory Features & Loss (GATE OPEN)
- **`jcm/mcb/coupled_features.py`**: added `CoupledBaselineTrajectory` (tree_math.struct; per-step `sst`/`surface_temperature`/`precipitation`/`heat_flux`, shape (T+1, ix, il); `at_step(t)` works with traced indices) and `compute_baseline_trajectory()` (single zero-MCB `lax.scan` from the same initial carry). `extract_coupled_features` gained `time_fraction`; `CoupledFeatureConfig.include_time=True` appends a normalized time-of-rollout feature (feature dim now 11)
- **`jcm/mcb/coupled_loss.py`**: `CoupledLossWeights` defaults rebalanced to the Stage 1-proven aquaplanet set (sst_cooling=1.0, sst_uniformity=0.1, amazon/sahel/tropics=0.0, reg/smoothness=0.001); zero-weight components skipped at trace time (removes constant offsets like softplus(0)/100 ≈ 0.00693 and wasted compute)
- **`jcm/mcb/coupled_controller.py`**: control step now scans over interval indices, extracting features vs `trajectory.at_step(t_start)` (with `time_fraction`) and computing loss vs `trajectory.at_step(t_end)`; added `run_interval_final_carry` (per-day `jax.checkpoint`, trajectory-discarding scan — the proven Stage 1 memory pattern, now inside BPTT); defaults `total_steps=180`, `target_cooling=-0.1`; `baseline` → `baseline_trajectory` throughout
- **`jcm/mcb/coupled_train.py`**, **`jcm/mcb/__init__.py`**: threaded `baseline_trajectory`; exported `CoupledBaselineTrajectory`, `compute_baseline_trajectory`
- **`run_coupled_training.py`**: precomputes and pickles the paired 180-day baseline trajectory before training; `include_time=True`; Stage 2 loss weights; `--target-cooling` default -0.1; dead imports removed (ruff clean)
- **Created `run_stage2_verification.py`** (gate script) — all 3 tests PASS (local CPU, 6-day/2-interval config):
  - Test 1 (paired cancellation): zero-MCB rollout vs trajectory max |dSST| = **0.000e+00 K** (bitwise); mid-rollout anomaly features all 0.0; time feature = 0.500 exact
  - Test 2 (loss breakdown): zero-MCB loss = **100.0% sst_cooling**, raw 0.010000 = target² exactly; amazon/sahel/tropics/uniformity/reg/smoothness all exactly 0
  - Test 3 (gradient): BPTT through the full Stage 2 unroll (trajectory indexing + time feature + checkpointed interval runner): loss 0.012060, **|grad| = 7.292**, no NaNs, not all-zero
- **Resolves post-mortem Issues #2, #3, #4, #5** (zero features, uncontrollable loss terms, land terms on aquaplanet, static baseline)
- **Next**: Stage 3 (NN policy training on diya, warm-started from Stage 1 pattern; re-sync codebase to diya first)

### 2026-07-01: Stage 1 Complete — Direct Pattern Optimization Succeeded (BOTH CRITERIA MET)
- **Created `optimize_mcb_pattern.py`**: optimizes the 96x48 albedo pattern directly (no NN) through the coupled model; sigmoid-bounded parameterization (`max_amplitude * sigmoid(theta) * ocean_mask`); paired no-MCB baseline from the same initial state; loss = cooling + uniformity + reg + smoothness (no land terms on aquaplanet); memory-efficient trajectory-discarding scan with per-day `jax.checkpoint`
- **Run on diya GPU**: 150 iters, 60-day rollouts, lr 0.05, target -0.1 K, max amplitude 0.15; 26.0 s/iter, ~65 min total
- **Results**:
  - Achieved dSST **-0.0972 K** vs target -0.1 K (criterion |diff| < 0.02 → **MET**)
  - Pattern spatially **STRUCTURED**: std 0.0291 (criterion > 0.005 → MET); range 0.0045–0.132; zonal-mean peak 0.116 at ~50°N; ~0.02 background elsewhere; within-band longitudinal std up to 0.03
  - Best loss **0.004497** — reference loss scale for Stage 3 policy training
  - Residual loss dominated by uniformity (~0.05–0.07 raw), i.e. weather noise in the paired 60-day difference
- **Artifacts**: `mcb_experiments_gpu/stage1/stage1_optimized_pattern.pkl` (best_theta for warm-start, best_pattern, history, baseline_sst) + `stage1.log`, retrieved from diya
- **Conclusion**: the inverse problem is solvable through the coupled model; gradients are informative end-to-end over 60 days. Any Stage 3 failure is now attributable to features/loss/architecture, not the simulator
- **Next**: Stage 2 (paired baseline trajectory features + loss rebalancing)

### 2026-07-01: Stage 0 Complete — Injection Plumbing Was Broken, Now Fixed (GATE OPEN)
- **Created `run_stage0_plumbing_test.py`**: reuses the exact training path (`setup_coupled_model`, `create_coupled_step_fn`, `run_coupled_interval`, carry-based injection); 3 tests: forward plumbing, AD gradient (+ FD cross-check), loss breakdown
- **Confirmed Issue #1**: dynamic path gave *bitwise* 0.0000 K effect and exactly-zero gradient — `set_mcb_perturbation()` attribute mutation invisible to `run_from_state`'s jit (static `self`). All prior coupled training was structurally incapable of learning
- **Fix implemented (4 files)**:
  - `jcm/forcing.py`: new `ForcingData.mcb_perturbation` field (traced; zeros = inactive)
  - `jcm/physics/speedy/forcing.py`: `set_forcing` applies `alb_s += (1 - fmask) * forcing.mcb_perturbation`; dead kwarg removed
  - `jcm/physics/speedy/speedy_physics.py`: removed mutation API (`set_mcb_perturbation`/`clear_mcb_perturbation`/`self.mcb_perturbation`)
  - `jax-esm/jem/components/JCM.py`: injects via `forcing.copy(mcb_perturbation=...)`
- **Verification (local CPU)**:
  - Test 1: MCB 0.1 over all ocean → **-0.407 K / 60 days** vs baseline (baseline drift +0.189 K)
  - Test 2: `d(SST)/d(scalar)` = -0.752 K/unit (AD) vs -0.740 (FD), rel. error 1.6%
  - Test 3: with initial-state baseline, `sst_uniformity` dominates loss at 92-98% after 60 days → drift conflation confirmed; Stage 2 paired-baseline is mandatory
  - Regression tests pass (`forcing_test`, `speedy_physics_test`, `shortwave_radiation_test`)
- **Next**: Stage 1 (direct static pattern optimization)

### 2026-07-01: Strategy Revision — Staged Roadmap (Post-Mortem)
- **Full post-mortem of Phase 8** identified 5 compounding issues (only zero-features was previously known):
  1. Dynamic MCB injection path unverified (0.0000 K effect may mean perturbation never reaches physics)
  2. Zero input features (baseline = initial state)
  3. Loss dominated by uncontrollable terms (5.33 total vs ~0.01 from sst_cooling) — trivial solution is the loss attractor
  4. Amazon/Sahel loss terms meaningless on aquaplanet (no land exists)
  5. Baseline is a state, not a trajectory (drift conflated with MCB effect)
- **Added Revised Roadmap (Stages 0-4)**:
  - Stage 0: Verify injection plumbing + gradient check + loss breakdown (gate for all training)
  - Stage 1: Direct static pattern optimization (de-risking + standalone scientific result + warm-start)
  - Stage 2: Paired baseline trajectory for features/loss; drop land terms; rebalance weights
  - Stage 3: NN policy training, warm-started from Stage 1
  - Stage 4: Varied initial conditions (genuine feedback), realistic terrain, teleconnection penalties
- **Design decisions confirmed**:
  - Aquaplanet retained for now; land terms dropped until Stage 4
  - Paired no-MCB baseline trajectory chosen over climatological mean
  - Pattern optimization inserted before policy retraining
- **Superseded**: atmosphere-only training fixes, pre-Stage-3 hyperparameter sweeps, climatological baseline
- **Rewrote Next Steps (TODO)** around the staged roadmap; added 3 new Known Issues entries

### 2024-XX-XX: GPU Training on diya (Phase 8 - Trivial Solution)
- **Completed GPU training**: 196 epochs in 50.7 minutes on NVIDIA GB10
- **Training metrics**:
  - Loss: 5.3354 → 5.3320 (0.064% reduction)
  - Gradient norm: 0.0117 → 0.000028 (vanished)
  - Time per epoch: ~15.5 seconds (6-10x speedup vs CPU)
- **Evaluation showed trivial solution**:
  - MCB output: constant ~0.002 everywhere (1.3% of max)
  - MCB effect: 0.0000 K (no additional cooling)
  - Policy learned to output bias term (all-zero features)
- **Root cause identified**: Input features all zeros
  - `CoupledBaseline` created from initial state
  - Training starts from same initial state
  - `current - baseline = 0` at t=0, stays near zero
- **Technical fixes applied** (all JIT tracing issues resolved):
  - Memory: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - Terrain: `TerrainData.aquaplanet(coords)` instead of `get_terrain()`
  - Timestep: `float` seconds instead of `Timedelta`
  - Pre-compute datetime values outside JIT closures
  - Replace boolean indexing with `jnp.where()`
  - Ensure pytree structure consistency (mcb_perturbation key)
  - Initialize `mcb_perturbation = jnp.zeros(...)` not `None`
- **Recommendations added** to fix baseline issue before next training

### 2024-XX-XX: Coupled Policy Training Framework (Phase 7 complete)
- **Created 4 new coupled modules** for training with ocean feedback:
  - `jcm/mcb/coupled_features.py` - Feature extraction from coupled state
    - `CoupledFeatureConfig` - configurable feature selection
    - `CoupledBaseline` - baseline SST, heat flux, precipitation for anomalies
    - `extract_coupled_features()` - extracts ~10 scalar features from coupled carry
    - `get_coupled_feature_dim()` - returns feature dimension
  - `jcm/mcb/coupled_loss.py` - Loss functions using ocean SST
    - `CoupledLossWeights` - weights for sst_cooling, sst_uniformity, amazon, sahel, tropics, regularization, smoothness
    - `sst_cooling_loss()` - penalizes deviation from target SST change
    - `sst_uniformity_loss()` - penalizes non-uniform cooling (teleconnections)
    - `compute_coupled_loss()` - combines all loss components
  - `jcm/mcb/coupled_controller.py` - BPTT through JAX-ESM coupler
    - `CoupledControllerConfig` - control interval, total steps, target cooling
    - `unroll_coupled_with_policy()` - main entry point using `jax.lax.scan`
    - `unroll_coupled_simple()` - returns only scalar loss (memory efficient)
    - `verify_coupled_gradients()` - gradient flow verification
  - `jcm/mcb/coupled_train.py` - Training loop for coupled simulation
    - `train_coupled_policy()` - main training loop with early stopping
    - `validate_coupled_training_setup()` - pre-training checks
    - `resume_coupled_training()` - resume from checkpoint
- **Modified 4 existing files** for dynamic MCB injection:
  - `jcm/physics/speedy/forcing.py` - Added `mcb_perturbation` parameter to `set_forcing()`
  - `jcm/physics/speedy/speedy_physics.py` - Added `set_mcb_perturbation()` method, closure-based access
  - `jax-esm/jem/components/JCM.py` - Reads `mcb_perturbation` from `carry["derived"]`, sets on physics
  - `jcm/mcb/__init__.py` - Exports all new coupled modules
- **Created training script**: `run_coupled_training.py`
  - Demonstrates coupled MCB training usage
  - Configurable epochs, learning rate, target cooling
  - Validation mode for setup verification
- **Key design decisions**:
  - MCB injection via `carry["atm"]["derived"]["mcb_perturbation"]` (no physics rebuild)
  - Primary loss uses ocean SST (`coupled_carry["ocn"]["state"].sea_surface_temperature`)
  - Gradient checkpointing per control interval for memory efficiency
  - Closure pattern in SpeedyPhysics for dynamic mcb_perturbation access
- **All 54 existing tests still passing**

### 2024-XX-XX: Coupled MCB Experiments (Phase 7 continued)
- **Created coupled MCB scripts**:
  - `run_coupled_mcb.py` - 60-day MCB simulation
  - `run_coupled_mcb_longrun.py` - 90-day MCB simulation
- **Key results achieved**:
  - 60-day: MCB reduced warming by 0.031 K (baseline +0.195 K → MCB +0.164 K)
  - 90-day: MCB achieved **global cooling** of -0.035 K (vs baseline +0.293 K)
  - MCB region SST change: -0.378 K (strong local cooling)
  - MCB cooling effect: **-0.328 K** (90 days)
- **Physical mechanism verified**:
  - MCB increases albedo → less solar absorbed
  - Reduced heat flux to ocean → ocean cools
  - SST feedback → global temperature decrease
- **Integration path confirmed**:
  - `MCBConfig` → `SpeedyPhysics(mcb_config=...)` → `Model(physics=...)` → JEM wrapper

### 2024-XX-XX: JAX-ESM Integration (Phase 7)
- **Added jax-esm** to repository at `jax-esm/`
  - JAX-based Earth System Model coupler
  - Couples JCM atmosphere with slab ocean and land models
  - Uses `jax.lax.scan` for efficient time integration
- **Created Phase 7** in implementation plan
  - Documented JAX-ESM architecture and component interface
  - Defined MCB integration strategy (use existing albedo forcing)
  - Added implementation checklist for coupled experiments
  - Documented expected differences from atmosphere-only
- **Key insight**: Atmosphere-only model cannot capture global temperature response
  - Ocean stores ~93% of heat → must be modeled for realistic MCB effects
  - Coupled mode: MCB → reduced heat flux → ocean cools → SST drops → global cooling
- **Updated Next Steps**: Coupled MCB is now top priority

### 2024-XX-XX: Training Evaluation and Analysis
- **Ran full training**: 50 epochs on CPU (~2.15 hours)
  - Learning rate: 0.001, Rollout: 90 days, Target: -0.5 K
  - Result: Loss reduced by only 0.012% (8.373 → 8.372)
  - Gradient norms decreased from 0.0008 to 0.0001 (vanishing)
- **Created evaluation script**: `evaluate_mcb_policy.py`
  - Runs 90-day simulation with trained policy
  - Compares temperature/precipitation to baseline
  - Computes regional metrics (Amazon, Sahel, tropics)
- **Ran evaluation**: Policy causes +2.55 K WARMING (not cooling!)
  - Amazon precipitation: +14.4% (protected)
  - Sahel precipitation: -97.4% (severe impact)
  - MCB forcing applied (mean: 0.068, max: 0.133)
- **Created visualization script**: `visualize_mcb_patterns.py`
  - Globe projections of MCB forcing pattern
  - Zonal mean distribution
  - Results summary panel
- **Created analysis script**: `analyze_training.py`
  - Loss curve plotting
  - Gradient norm tracking
  - Automated diagnostics with recommendations
- **Diagnosed training failure** (Section 6.5):
  - Learning rate too low → increase to 0.01-0.1
  - Rollout too short → use 180+ days on GPU
  - Target too aggressive → start with -0.1 K
  - Insufficient epochs → train 200-500 epochs
- **Created presentation**: `MCB_PRESENTATION.html`
  - 16 interactive slides
  - Includes all visualizations and results
  - Professional speaking notes provided

### 2024-XX-XX: Experimental Phase Started
- Created `run_mcb_baseline.py` for baseline simulation
- Ran 1-year baseline simulation (365 days, monthly outputs)
- Generated baseline climate data:
  - `mcb_experiments/baseline_predictions.nc`
  - `mcb_experiments/baseline_climate.pkl`
- Created `visualize_baseline_globe.py` for globe visualizations
- Generated baseline visualization with 6 climate variables
- Created `train_mcb_policy.py` for policy training
- **Bug fix**: Updated `state_features.py` field access:
  - `predictions.physics.surface_flux.tsfc` for surface temperature
  - Added time dimension squeezing for `(1, ix, il)` shaped fields
- **Bug fix**: Updated `loss.py` with same field access corrections
- Training script validates successfully but times out on CPU (needs GPU)

### 2024-XX-XX: Neural Network Controller Implementation
- Created `jcm/mcb/policy.py` with 4 policy network architectures (MLP, CNN, ResNet, Hybrid)
- Created `jcm/mcb/state_features.py` with scalar and spatial feature extraction
- Created `jcm/mcb/loss.py` with 6-component differentiable loss function
- Created `jcm/mcb/controller.py` with `jax.lax.scan`-based unrolling
- Created `jcm/mcb/train.py` with full BPTT training infrastructure
- Updated `jcm/mcb/__init__.py` to export all new components
- Added 27 new unit tests (54 total) to `jcm/mcb/mcb_test.py`
- All tests passing, all linting checks passing

---

## Lessons Learned

### What Worked
- JAX differentiability through complex physics simulations
- Modular architecture - easy to swap components
- Gradient checkpointing for memory management
- Comprehensive test suite caught bugs early
- Framework is sound - gradients flow correctly
- **JAX-ESM coupled modeling** - MCB achieves realistic cooling with ocean feedback
- **MCBConfig → SpeedyPhysics → Model** integration path works seamlessly

### What Didn't Work (First Attempt - CPU, Atmosphere-Only)
- Conservative learning rate (0.001) → minimal updates
- Short rollout (90 days) → climate doesn't respond
- Aggressive target (-0.5 K) → loss dominated by gap
- CPU training → too slow for adequate epochs
- MLP architecture → may lack spatial awareness

### What Didn't Work (GPU Training - Phase 8)
- **Baseline from initial state** → all anomaly features = 0
- Policy saw all-zero inputs → learned only bias term (constant output)
- No spatial structure, no meaningful forcing
- 0.0000 K MCB effect (trivial solution)
- **Key lesson**: Data pipeline (features) is as important as model architecture

### Key Insight
**The framework works. Gradients flow. Training converges (slowly).**
The issue is hyperparameter tuning, not fundamental problems.
With GPU + tuned hyperparameters, achieving cooling is feasible.

### Critical Finding: Coupled Modeling Required
**Atmosphere-only simulations cannot capture global mean temperature changes from MCB.**

Why the first attempt showed warming (+2.55 K) instead of cooling:
- Ocean stores ~93% of Earth's excess heat
- Without ocean model, energy changes don't persist
- SST is prescribed (boundary condition), not responsive to forcing
- MCB reduces heat flux, but ocean doesn't cool → no SST feedback → no cooling

**Solution: JAX-ESM coupling (Phase 7)** ✅ VERIFIED
- Atmosphere ↔ Ocean coupling captures energy flow
- MCB → reduced heat flux → ocean absorbs less heat → SST decreases → atmosphere cools
- This is the physically correct mechanism for MCB-induced cooling
- **Result**: 90-day coupled MCB achieved -0.328 K cooling effect (vs baseline)

---

## Phase 8: GPU Training on Diya (Complete - Trivial Solution)

### 8.1 Training Environment

| Parameter | Value |
|-----------|-------|
| Machine | diya (Tailscale: 100.76.85.47) |
| OS | Ubuntu 24.04 LTS (aarch64/ARM64) |
| GPU | NVIDIA GB10, CUDA 13.0 |
| RAM | 121GB total |
| JAX Memory | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` |

### 8.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 (completed 196 before early stopping) |
| Learning Rate | 0.01 |
| Target Cooling | -0.1 K |
| Total Simulation | 180 days per epoch |
| Control Interval | 30 days |
| Policy Architecture | MLP (256, 256) with 1.25M parameters |
| Mode | Coupled (atmosphere + slab ocean) |
| Total Training Time | **50.7 minutes** |

### 8.3 Training Results

| Metric | Value |
|--------|-------|
| Initial Loss | 5.3354 |
| Final Loss | 5.3320 |
| Best Loss | 5.3320 |
| Loss Reduction | **0.064%** (minimal!) |
| Initial Gradient Norm | 0.0117 |
| Final Gradient Norm | 0.000028 (vanished to near-zero) |
| Time per Epoch | ~15.5 seconds (GPU speedup: ~6-10x vs CPU) |

### 8.4 Evaluation Results

**MCB Pattern Analysis**:
| Metric | Value |
|--------|-------|
| Mean MCB Output | 0.00220 |
| Min MCB Output | 0.00217 |
| Max MCB Output | 0.00224 |
| Output Range | 0.00007 (essentially constant!) |
| As % of Max (0.15) | **1.3%** (near-zero forcing) |

**Cooling Performance**:
| Metric | Baseline | With Policy | MCB Effect |
|--------|----------|-------------|------------|
| Global SST Change | -0.0004 K | -0.0004 K | **0.0000 K** |

**Key Finding**: The policy learned a **trivial solution** - outputting a near-constant, near-zero value everywhere. There is no spatial structure and no additional cooling beyond baseline.

### 8.5 Root Cause Analysis

**Critical Bug Identified**: The input features were all zeros throughout training!

```
Feature statistics:
  sst_global_anomaly:     mean=0.0000, std=0.0000
  sst_tropics_anomaly:    mean=0.0000, std=0.0000
  sst_atlantic_anomaly:   mean=0.0000, std=0.0000
  sst_pacific_anomaly:    mean=0.0000, std=0.0000
  heat_flux_global:       mean=0.0000, std=0.0000
  precip_tropics_anomaly: mean=0.0000, std=0.0000
```

**Why Features Were Zero**:
1. `CoupledBaseline.from_coupled_carry(initial_carry, coords)` creates baseline from **initial state**
2. Training starts from the **same initial state**
3. At step 0: `current_state - baseline = initial_state - initial_state = 0`
4. Policy sees all-zero features → learns constant output (the bias term)
5. Near-zero forcing → no SST change → features stay near zero
6. **Feedback loop**: zero features → constant output → no change → zero features

**This is a data/baseline issue, not a model architecture issue.**

**Update (post-mortem)**: Deeper analysis found this was only 1 of 5 compounding issues. The others:
unverified dynamic MCB injection plumbing (the 0.0000 K effect may be a broken path, not just tiny
output), loss dominated by uncontrollable terms (5.33 total vs ~0.01 from sst_cooling), meaningless
Amazon/Sahel terms on an aquaplanet, and state-based (rather than trajectory-based) baseline. See the
**Revised Roadmap** section at the top of this document for the full diagnosis and staged fix plan.

### 8.6 Technical Issues Encountered & Resolved

During GPU training setup, several JAX-related issues were fixed:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Memory crash | 91GB GPU pre-allocation on 121GB system | `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| Terrain API | `get_terrain()` not found | Use `TerrainData.aquaplanet(coords)` |
| Timestep type | `float + Timedelta` error | Pass `timestep=86400.0` (float seconds) |
| JIT tracing | `.item()` inside closure | Pre-compute `_timestep_days` outside closure |
| Datetime tracing | `_compute_start_day_offset()` traced | Pre-compute in `__init__`, cache as `_start_day_offset` |
| Boolean indexing | `NonConcreteBooleanIndexError` | Replace `.at[idx].set()` with `jnp.where()` |
| Pytree mismatch | `mcb_perturbation` key missing | Add key to both input and output dicts |
| Pytree type | `NoneType vs ShapedArray` | Initialize `mcb_perturbation = jnp.zeros(...)` |

### 8.7 Output Files

- `mcb_experiments_gpu/coupled_trained_policy.pkl` - Trained policy (1.25M params)
- `mcb_experiments_gpu/coupled_training_history.pkl` - Loss/gradient history
- `mcb_experiments_gpu/coupled_training_analysis.png` - Training visualization
- `mcb_experiments_gpu/policy_evaluation.png` - MCB pattern visualization
