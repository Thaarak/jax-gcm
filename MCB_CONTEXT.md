# Marine Cloud Brightening (MCB) Research Context

## What is Marine Cloud Brightening?

Marine Cloud Brightening (MCB) is a proposed **Solar Radiation Management (SRM)** geoengineering technique designed to temporarily mitigate the impacts of global warming.

### Mechanism

1. **Aerosol Injection**: Sub-micron sea salt aerosols are continuously injected into the lower marine boundary layer over the ocean
2. **Cloud Condensation Nuclei (CCN)**: These aerosols act as CCN, causing marine stratocumulus clouds to form with a higher concentration of smaller droplets
3. **Twomey Effect**: This process increases the reflectivity (albedo) of low-lying clouds, bouncing more incoming shortwave solar radiation back into space
4. **Net Cooling**: The increased reflection reduces the amount of solar energy absorbed by Earth's surface

### Target Deployment Regions

MCB research typically focuses on subtropical stratocumulus regions:
- **SE Pacific** (off Peru/Chile coast)
- **SE Atlantic** (off Namibia/Angola coast)
- **NE Pacific** (off California coast)
- **Canary Current** (off NW Africa)
- **Benguela Current** (off SW Africa)

These regions are characterized by persistent low cloud decks, cool upwelling waters, and stable atmospheric conditions.

---

## The Core Problem: Climate Teleconnections

### What Are Teleconnections?

MCB introduces **highly localized cooling** over specific ocean regions. This cooling inherently disrupts established global atmospheric and oceanic circulation patterns, triggering severe **remote side effects** known as **climate teleconnections**.

### Documented Risks

| MCB Location | Remote Effect | Mechanism |
|--------------|---------------|-----------|
| Subtropical South Atlantic | Drastic reduction in Amazon rainfall | Alters Walker circulation |
| North Pacific (mid-century) | Spin-up of AMOC, European summer heatwaves | Perturbs Atlantic overturning |
| Unilateral regional deployment | Precipitation shifts in distant continents | Disrupts Hadley/Ferrel cells |

### The Fundamental Challenge

A uniform, continuous MCB spray creates unpredictable and potentially catastrophic remote impacts. The challenge is to find a **constrained, optimized deployment strategy** that achieves global cooling benefits while explicitly minimizing adverse teleconnections.

---

## Research Goal: Solving the MCB Inverse Problem

### Problem Statement

Given:
- A desired global cooling target (e.g., offset 2°C of warming)
- Constraints on regional climate impacts (e.g., Amazon precipitation must not decrease)

Find:
- The optimal **spatial distribution** of MCB radiative forcing
- The optimal **temporal pattern** of MCB deployment
- A **dynamic control policy** that adapts to evolving climate state

### Novel Contribution

Rather than static optimization, we aim to develop a **neural network-based feedback controller** that:

1. **Observes** the current climate state (temperature, precipitation patterns)
2. **Decides** the optimal MCB forcing to apply at each timestep
3. **Adapts** dynamically as the climate evolves over multi-year timescales
4. **Minimizes** both the deviation from cooling targets AND teleconnection side effects

This transforms the problem from static optimization to **reinforcement learning / optimal control** through differentiable simulation.

---

## Why JAX-GCM (JCM)?

### The Differentiability Barrier

Historically, solving this inverse problem was **computationally impossible** because traditional GCMs are:
- Written in legacy Fortran code
- Non-differentiable (no automatic gradients)
- Treat the atmosphere as a black box

### JCM's Capabilities

JAX Circulation Model (JCM) v1.0 is built entirely in Python/JAX, providing:

| Feature | Benefit for MCB Research |
|---------|--------------------------|
| **Automatic Differentiation** | Compute exact analytical gradients via VJP/JVP |
| **End-to-End Differentiability** | Backpropagate through multi-year simulations |
| **Spectral Dynamical Core** | Accurate atmospheric dynamics (Dinosaur) |
| **Modular Physics** | SPEEDY parameterizations for clouds, radiation, convection |
| **JAX Transformations** | `jit`, `grad`, `vmap` for efficient optimization |

### Key Gradient Computations

With JCM, we can directly compute sensitivities like:

```
∂(Amazon_precipitation) / ∂(SE_Pacific_albedo)
∂(Global_mean_temperature) / ∂(Spatial_MCB_pattern)
∂(AMOC_strength) / ∂(Temporal_MCB_schedule)
```

These gradients enable **gradient descent optimization** of MCB strategies and **backpropagation through time** for training neural network controllers.

---

## Implementation Status

### Completed: MCB Forcing Module (`jcm/mcb/`)

| Component | File | Description |
|-----------|------|-------------|
| `MCBConfig` | `mcb_config.py` | Data structure for spatial/temporal MCB forcing |
| Region Masks | `mcb_regions.py` | Predefined stratocumulus and teleconnection regions |
| Forcing Logic | `mcb_forcing.py` | `compute_mcb_sea_albedo()` - applies MCB to surface albedo |
| Integration | `speedy_physics.py` | MCB flows through physics pipeline via `mcb_config` parameter |

### Next Phase: Neural Network Controller

The next implementation phase involves:
1. **Policy Network**: Neural network mapping climate state → MCB forcing
2. **Loss Function**: Differentiable objective combining cooling + teleconnection penalties
3. **Differentiable Unrolling**: Step JCM forward with periodic policy application
4. **BPTT Training**: Backpropagate through unrolled simulation to train controller

---

## Key References

- **Twomey Effect**: Twomey, S. (1977). The influence of pollution on the shortwave albedo of clouds.
- **MCB Proposal**: Latham, J. (1990). Control of global warming?
- **Teleconnection Risks**: Jones et al. (2009), Robock et al. (2008)
- **JCM/Dinosaur**: Google Research spectral dynamical core
- **Differentiable Climate**: Various works on differentiable physics for climate
