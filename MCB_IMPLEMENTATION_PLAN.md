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

**Test Coverage**: 86 `jcm/mcb` unit tests passing | **Linting**: `ruff` clean

### Current Position

Foundational phases 1–8 are complete (table above). The project now follows the staged
[Revised Roadmap](#-revised-roadmap-current-strategy---supersedes-previous-retrain-plan) below,
where per-stage detail, numbers, and gate verdicts live. One-line status:

| Stage | Status |
|-------|--------|
| 0 — verify plumbing / diagnose loss | ✅ complete, gate open |
| 1 — static pattern optimization | ✅ complete (−0.097 K, spatially structured) |
| 2 — paired-baseline features & loss | ✅ complete, gate open |
| 3 — NN policy training (60-day) | ✅ complete (−0.1015 K, beats static) |
| 4 — varied-IC ensemble controller | ⚠️ complete, 3/4 gates (Gate 1: held-out overcooling) |
| 5 — realistic terrain + teleconnections | ✅ code-complete + local smoke; GPU run pending |

**NEXT:** Stage 5 diya GPU run (150-epoch warm-start from Stage 4, 6-IC eval, 4 gates), then the
Stage 4 Follow-up (Option A) seasonal-bracketing fix, which also applies to the Stage 5 held-out gate.

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

### Stage 4 Follow-up (Option A) — TODO: Fix Gate-1 held-out overcooling via seasonal bracketing

**Gate 1 FAILED in Stage 4**: held-out mean day-60 dSST −0.1285 K overshot the target band
[−0.12, −0.08] K (train dSST −0.1044 K is on-target).

**Root cause (from Stage 4 eval):** the held-out ICs (spin-up days 180, 225) lie *beyond* the
training range {0, 45, 90, 135}, so the policy must **extrapolate**. The system's cooling
sensitivity rises monotonically with spin-up day — the teleconnection-free stage1-static baseline
dSST steepens −0.095 → −0.148 K across the spin-up axis — so extrapolating past day 135 makes the
policy **overcool** the later (more sensitive) held-out states.

**Fix — bracket held-out seasons with training seasons (interpolation, not extrapolation):**
regenerate ICs so every held-out spin-up day is *inside* the convex hull of the training days.
For example:

- Train spin-up days: {0, 45, 90, 135, 180, 225}
- Held out (interpolated): {70, 200}

Both held-out points now fall *between* training points, converting extrapolation into
interpolation. This is expected to bring held-out day-60 dSST into the [−0.12, −0.08] K gate window
without changing any loss weights or the policy architecture — only the IC-generation schedule in
`run_stage4_generate_ics.py` (`--spinup-interval`, `--num-train`, `--num-heldout`) changes.

**Status:** deferred. Stage 5 (realistic terrain) proceeds first per the current roadmap; this
bracketing fix is orthogonal and applies to both the aquaplanet Stage 4 and the realistic Stage 5
held-out gates.

### Stage 5: Realistic Terrain + Teleconnection Penalties ✅ CODE-COMPLETE + LOCAL SMOKE VERIFIED (GPU run pending)

Moves the controller from the aquaplanet to **realistic Earth terrain** (T30 climatology: orography
+ land-sea mask) and **reinstates the teleconnection precipitation penalties** (amazon / sahel /
tropics @ 0.05) that were structurally zero on the aquaplanet. Reuses the Stage 4 ensemble
machinery (varied-IC BPTT, 13-feature state-dependent policy, paired baselines, host-side gradient
averaging); the deltas are terrain wiring and ocean-masking of the objective/features.

**Implemented (all `ocean_mask=None` defaults keep Stages 1–4 aquaplanet paths bit-identical):**
- `jcm/mcb/coupled_loss.py`: `compute_coupled_loss` takes optional `ocean_mask`; renormalizes area
  weights over ocean for `sst_cooling_loss` / `sst_uniformity_loss`.
- `jcm/mcb/coupled_features.py`: `extract_coupled_features` takes optional `ocean_mask`;
  ocean-renormalizes the area-weighted global means (SST, heat flux, precip, absolute-SST, NH-SH).
- `jcm/mcb/coupled_controller.py`: threads `ocean_mask` into the per-interval loss/feature calls.
- `run_coupled_training.py` `setup_coupled_model(realistic_terrain=)`: passes
  `terrain=TerrainData.from_file(TERRAIN_NC)` into `Model(...)` and `mask_file=TERRAIN_NC` into the
  slab ocean model.
- Drivers `run_stage5_generate_ics.py`, `run_stage5_training.py`, `run_stage5_eval.py` (warm-start
  from Stage 4 dim-13 policy, no expand).

**Two correctness findings during verification (not in the original plan):**

1. **Realistic surface forcing is REQUIRED for numerical stability over terrain.** The atmosphere
   NaNs within ~1 day over steep orography without land-surface boundary conditions. The JEM wrapper
   (`jax-esm/jem/components/JCM.py:make_jem_compatible`) previously hardcoded aquaplanet
   `default_forcing()` (zero land fields). Fix: thread `ForcingData.from_file(FORCING_NC)` (stl_am /
   soilw_am / snowc_am / alb0) into the wrapper. Verified decisive: 40 min + no forcing → NaN; 30 min
   + forcing → stable. Only the SST field is collapsed to 2D (day-0; the coupler overwrites it each
   step) to satisfy the `lax.scan` shape invariant while keeping the 3D land annual cycle.
2. **The ocean mask must come from the slab ocean model's own grid `bmask`, not `terrain.fmask`.**
   The ocean model pins land (SST ≡ 288.15 K) where its grid `bmask` = `fmask > 0.95`.
   `TerrainData.from_file` interpolates `fmask` on an independent path that disagrees with the ocean
   grid by up to ~0.1 at coastlines (**56 mismatched T30 cells**). Using `1 - terrain.fmask` would
   drop genuine evolving-SST ocean cells and could weight pinned-land cells. Fix: new
   `ocean_mask_from_coupler(coupler)` (= `1 - bmask`) and `ocean_fmask_from_coupler(coupler)` in
   `jcm/mcb/coupled_train.py`; drivers source the mask from the coupler. Verified: mask matches the
   ocean bmask cell-for-cell (1197 land cells), **zero ocean-weighted cells carry the pinned 288.15 K
   temperature**, and the trainer/eval/pre-flight masks are identical.

**Local CPU smoke (6-day / 2-epoch):** `pytest jcm/mcb/ -m "not slow"` 86 passed; `ruff` clean;
full IC-gen → training → eval pipeline runs with finite decreasing losses, clean baselines, and
populated per-region precip diagnostics. Gate 1 (orography 812.7, terrain reaches dynamics) PASS;
Gate 4 (no NaNs, finite losses) PASS. Gate 2 (cooling band) and Gate 3 (teleconnection protection)
are only meaningful at the 60-day / warm-start GPU scale.

**Pending:** full diya GPU run (150-epoch training warm-started from `stage4_trained_policy.pkl`;
6-IC eval); check the 4 gates; record results here. The Stage 4 Follow-up (Option A) seasonal
bracketing applies to the Stage 5 held-out Gate 2 as well.

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
jcm/mcb/                          # MCB forcing, policy, and coupled training
├── __init__.py           # Public API exports
├── mcb_config.py         # MCBConfig struct
├── mcb_regions.py        # Region masks (stratocumulus + teleconnection); create_ocean_mask
├── mcb_forcing.py        # Core forcing computation
├── carry_io.py           # Coupled-carry (de)serialization for IC checkpoints
├── policy.py             # Policy networks (MLP, CNN, ResNet, Hybrid); expand_policy_input
├── state_features.py     # Atmosphere-only feature extraction (scalar + spatial)
├── loss.py               # Atmosphere-only climate loss (6 components)
├── controller.py         # Differentiable unrolling with lax.scan
├── train.py              # BPTT training loop
├── coupled_features.py   # Coupled feature extraction (SST, heat flux, ...); ocean-masked
├── coupled_loss.py       # Loss using ocean SST; ocean-masked cooling/uniformity
├── coupled_controller.py # BPTT through the JAX-ESM coupler
├── coupled_train.py      # Coupled/ensemble training loop; ocean_mask_from_coupler
└── *_test.py             # Co-located unit tests (86 passing)

jax-esm/                          # Earth System Model coupler
└── jem/
    ├── base/coupler.py           # Core coupling engine (jax.lax.scan)
    ├── components/JCM.py         # JCM wrapper (make_jem_compatible); MCB + forcing injection
    ├── components/slab/          # Mixed-layer slab ocean + land models
    └── mapping/                  # Variable exchange (heat flux ↔ SST); bmask/fmask grids

Root scripts (chronological by stage):
├── run_mcb_baseline.py           # Atmosphere-only baseline (Phase 6, legacy)
├── run_coupled_baseline.py       # Coupled baseline (Phase 7)
├── run_coupled_mcb.py / _longrun.py  # Static coupled MCB, 60d / 90d (Phase 7)
├── run_stage0_plumbing_test.py   # Stage 0 injection + gradient diagnostic
├── optimize_mcb_pattern.py       # Stage 1 direct static pattern optimization
├── run_stage2_verification.py    # Stage 2 paired-baseline gate (3 tests)
├── run_coupled_training.py       # Stage 2/3 coupled policy training (--warm-start)
├── run_stage3_eval.py            # Stage 3 trained-vs-static eval
├── run_stage4_generate_ics.py / _training.py / _eval.py   # Stage 4 varied-IC ensemble
└── run_stage5_generate_ics.py / _training.py / _eval.py   # Stage 5 realistic terrain

Artifacts: mcb_experiments/ (local CPU), mcb_experiments_gpu/ (diya GPU),
mcb_experiments_gpu/stage{1,3_60d,4,5}/ (per-stage policies, histories, eval results, logs).
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

## Known Issues

> **Resolved blockers** (zero features, silent MCB injection, uncontrollable loss terms, aquaplanet
> land terms, state-vs-trajectory baseline) are documented in the [Full Post-Mortem](#full-post-mortem-diagnosed-issues)
> table and the Stage 0–2 sections of the Revised Roadmap. The operational notes below remain live.

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

### 8.2 Result: Trivial Solution

200 epochs (196 before early stop, 50.7 min; lr 0.01, target -0.1 K, 180 days/6×30-day intervals,
MLP (256,256), 1.25M params, coupled): loss barely moved (5.3354 → 5.3320, **0.064%**), gradient
norm vanished (0.0117 → 0.000028), and the policy output a **near-constant ~0.0022** (1.3% of max)
everywhere → **0.0000 K** additional cooling. The immediate cause was all-zero input features
(baseline built from the initial state, training started from the same state → `current − baseline
= 0` at every step, so only the bias could learn). Post-mortem found this was 1 of **5 compounding
issues** — see the [Full Post-Mortem](#full-post-mortem-diagnosed-issues) table and the staged fix
in the Revised Roadmap.

### 8.3 Technical Issues Encountered & Resolved

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

### 8.4 Output Files

- `mcb_experiments_gpu/coupled_trained_policy.pkl` - Trained policy (1.25M params)
- `mcb_experiments_gpu/coupled_training_history.pkl` - Loss/gradient history
- `mcb_experiments_gpu/coupled_training_analysis.png` - Training visualization
- `mcb_experiments_gpu/policy_evaluation.png` - MCB pattern visualization
