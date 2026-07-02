"""Marine Cloud Brightening (MCB) forcing module.

This module provides tools for simulating and optimizing MCB geoengineering
strategies using JAX-GCM's differentiable framework.

Main components:
- MCBConfig: Configuration struct for MCB forcing parameters
- Region utilities: Create masks for stratocumulus deployment zones
- Forcing functions: Compute MCB albedo perturbations
- Policy networks: Neural networks for MCB control (Flax)
- State features: Climate state extraction for policy input
- Loss functions: Differentiable climate objectives
- Controller: Differentiable simulation unrolling
- Training: BPTT training loop for policy optimization

Example usage:
    from jcm.mcb import MCBConfig, create_stratocumulus_mask
    from jcm.physics.speedy.speedy_coords import get_speedy_coords

    coords = get_speedy_coords()
    mcb_config = MCBConfig.uniform(
        coords.horizontal.nodal_shape,
        perturbation=0.05,
        region_mask=create_stratocumulus_mask(
            coords.horizontal, terrain.fmask, ['se_pacific']
        )
    )

For neural network control:
    from jcm.mcb import MCBPolicyMLP, train_policy, ControllerConfig

    policy = MCBPolicyMLP(output_shape=coords.horizontal.nodal_shape)
    params, history = train_policy(model, policy, coords, ...)
"""

from jcm.mcb.mcb_config import MCBConfig
from jcm.mcb.mcb_regions import (
    STRATOCUMULUS_REGIONS,
    TELECONNECTION_REGIONS,
    create_region_mask,
    create_ocean_mask,
    create_stratocumulus_mask,
    create_teleconnection_mask,
)
from jcm.mcb.mcb_forcing import (
    compute_mcb_sea_albedo,
    compute_mcb_radiative_forcing,
)

# Policy networks
from jcm.mcb.policy import (
    MCBPolicyMLP,
    MCBPolicyCNN,
    MCBPolicyResNet,
    MCBPolicyHybrid,
    create_policy,
    init_policy_params,
)

# State feature extraction
from jcm.mcb.state_features import (
    StateFeatureConfig,
    ClimateBaseline,
    extract_state_features,
    extract_scalar_features,
    extract_spatial_features,
    get_feature_dim,
)

# Loss functions
from jcm.mcb.loss import (
    LossWeights,
    LossComponents,
    compute_climate_loss,
    compute_climate_loss_from_baseline,
    create_loss_fn,
)

# Controller
from jcm.mcb.controller import (
    ControllerConfig,
    ControlStep,
    unroll_with_policy,
    unroll_with_policy_simple,
    create_controlled_step,
    evaluate_policy,
    compute_policy_gradient,
    verify_gradients,
)

# Training
from jcm.mcb.train import (
    TrainingConfig,
    TrainingState,
    train_policy,
    validate_training_setup,
    save_checkpoint,
    load_checkpoint,
)

# Coupled (JAX-ESM) modules
from jcm.mcb.coupled_features import (
    CoupledFeatureConfig,
    CoupledBaseline,
    CoupledBaselineTrajectory,
    compute_baseline_trajectory,
    extract_coupled_features,
    get_coupled_feature_dim,
    create_coupled_feature_extractor,
)

from jcm.mcb.coupled_loss import (
    CoupledLossWeights,
    CoupledLossComponents,
    compute_coupled_loss,
    compute_coupled_loss_from_baseline,
    create_coupled_loss_fn,
    sst_cooling_loss,
)

from jcm.mcb.coupled_controller import (
    CoupledControllerConfig,
    CoupledControlStep,
    unroll_coupled_with_policy,
    unroll_coupled_simple,
    create_coupled_loss_fn as create_coupled_controller_loss_fn,
    evaluate_coupled_policy,
    verify_coupled_gradients,
)

from jcm.mcb.coupled_train import (
    train_coupled_policy,
    validate_coupled_training_setup,
    resume_coupled_training,
)

__all__ = [
    # Config
    'MCBConfig',
    # Regions
    'STRATOCUMULUS_REGIONS',
    'TELECONNECTION_REGIONS',
    'create_region_mask',
    'create_ocean_mask',
    'create_stratocumulus_mask',
    'create_teleconnection_mask',
    # Forcing
    'compute_mcb_sea_albedo',
    'compute_mcb_radiative_forcing',
    # Policy networks
    'MCBPolicyMLP',
    'MCBPolicyCNN',
    'MCBPolicyResNet',
    'MCBPolicyHybrid',
    'create_policy',
    'init_policy_params',
    # State features
    'StateFeatureConfig',
    'ClimateBaseline',
    'extract_state_features',
    'extract_scalar_features',
    'extract_spatial_features',
    'get_feature_dim',
    # Loss
    'LossWeights',
    'LossComponents',
    'compute_climate_loss',
    'compute_climate_loss_from_baseline',
    'create_loss_fn',
    # Controller
    'ControllerConfig',
    'ControlStep',
    'unroll_with_policy',
    'unroll_with_policy_simple',
    'create_controlled_step',
    'evaluate_policy',
    'compute_policy_gradient',
    'verify_gradients',
    # Training
    'TrainingConfig',
    'TrainingState',
    'train_policy',
    'validate_training_setup',
    'save_checkpoint',
    'load_checkpoint',
    # Coupled features
    'CoupledFeatureConfig',
    'CoupledBaseline',
    'CoupledBaselineTrajectory',
    'compute_baseline_trajectory',
    'extract_coupled_features',
    'get_coupled_feature_dim',
    'create_coupled_feature_extractor',
    # Coupled loss
    'CoupledLossWeights',
    'CoupledLossComponents',
    'compute_coupled_loss',
    'compute_coupled_loss_from_baseline',
    'create_coupled_loss_fn',
    'sst_cooling_loss',
    # Coupled controller
    'CoupledControllerConfig',
    'CoupledControlStep',
    'unroll_coupled_with_policy',
    'unroll_coupled_simple',
    'create_coupled_controller_loss_fn',
    'evaluate_coupled_policy',
    'verify_coupled_gradients',
    # Coupled training
    'train_coupled_policy',
    'validate_coupled_training_setup',
    'resume_coupled_training',
]
