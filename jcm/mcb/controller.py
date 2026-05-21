"""Differentiable controller for MCB policy optimization.

This module implements differentiable climate simulation unrolling with
neural network policy application. It enables Backpropagation Through Time
(BPTT) for training MCB controllers.

Key functions:
- unroll_with_policy: Unroll simulation with periodic policy application
- create_controlled_step: Create a single control step function
- gradient_checkpoint_unroll: Memory-efficient unrolling with checkpointing

Example usage:
    from jcm.mcb.controller import unroll_with_policy

    loss, final_state, trajectory = unroll_with_policy(
        model=model,
        policy_fn=policy.apply,
        policy_params=params,
        initial_state=state0,
        forcing=forcing,
        baseline=baseline,
        coords=coords,
        control_interval_days=30,
        total_days=365,
    )
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, Any, NamedTuple
from functools import partial

from jcm.mcb.state_features import (
    extract_state_features,
    StateFeatureConfig,
    ClimateBaseline,
)
from jcm.mcb.loss import compute_climate_loss, LossWeights


class ControllerConfig(NamedTuple):
    """Configuration for MCB controller.

    Attributes:
        control_interval_days: Days between policy applications.
        total_days: Total simulation duration in days.
        target_cooling: Target temperature change (K, negative for cooling).
        loss_weights: Weights for loss components.
        feature_config: Configuration for state feature extraction.
        max_perturbation: Maximum MCB albedo perturbation.
        use_checkpointing: Enable gradient checkpointing for memory efficiency.

    """

    control_interval_days: float = 30.0
    total_days: float = 365.0
    target_cooling: float = -0.5
    loss_weights: LossWeights = LossWeights()
    feature_config: StateFeatureConfig = StateFeatureConfig()
    max_perturbation: float = 0.15
    use_checkpointing: bool = True


class ControlStep(NamedTuple):
    """Output from a single control step.

    Attributes:
        state: Final modal state after the control interval.
        predictions: Model predictions from the interval.
        mcb_forcing: Applied MCB forcing field.
        loss: Loss value for this interval.

    """

    state: Any
    predictions: Any
    mcb_forcing: jnp.ndarray
    loss: jnp.ndarray


def create_controlled_step(
    model,
    policy_fn: Callable,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig,
):
    """Create a function for a single controlled simulation step.

    This factory creates a differentiable function that:
    1. Runs the model for one control interval
    2. Extracts state features
    3. Applies the policy to get MCB forcing
    4. Computes the loss

    Args:
        model: JCM Model instance.
        policy_fn: Policy network apply function.
        forcing: Base ForcingData.
        terrain: TerrainData with land-sea mask.
        baseline: Climate baseline for anomalies.
        coords: Model coordinates.
        config: Controller configuration.

    Returns:
        Function (state, policy_params) -> ControlStep.

    """
    ocean_mask = 1.0 - terrain.fmask

    def step_fn(state, policy_params):
        """Execute one control interval."""
        # Run model for control interval to get current climate state
        final_state, predictions = model.run_from_state(
            initial_state=state,
            forcing=forcing,
            save_interval=config.control_interval_days,
            total_time=config.control_interval_days,
            output_averages=False,
        )

        # Extract state features for policy
        state_features = extract_state_features(
            predictions,
            baseline,
            coords,
            config.feature_config,
        )

        # Get MCB forcing from policy
        mcb_perturbation = policy_fn(policy_params, state_features)

        # Apply ocean mask - MCB only over ocean
        mcb_perturbation = mcb_perturbation * ocean_mask

        # Compute loss for this interval
        loss = compute_climate_loss(
            predictions=predictions,
            baseline_temp=baseline.surface_temperature,
            baseline_precip=baseline.precipitation,
            target_cooling=config.target_cooling,
            mcb_forcing=mcb_perturbation,
            coords=coords,
            weights=config.loss_weights,
        )

        return ControlStep(
            state=final_state,
            predictions=predictions,
            mcb_forcing=mcb_perturbation,
            loss=loss,
        )

    return step_fn


def unroll_with_policy(
    model,
    policy_fn: Callable,
    policy_params: dict,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> Tuple[jnp.ndarray, Any, list]:
    """Unroll climate simulation with neural network control.

    Main entry point for differentiable MCB control. Runs the climate model
    forward with periodic policy application, computing a total loss that
    can be differentiated with respect to policy parameters.

    Args:
        model: JCM Model instance.
        policy_fn: Policy network apply function (params, features) -> forcing.
        policy_params: Policy network parameters.
        initial_state: Initial modal state (primitive_equations.State).
        forcing: Base ForcingData.
        terrain: TerrainData with land-sea mask.
        baseline: Climate baseline for computing anomalies.
        coords: Model coordinates.
        config: Controller configuration.

    Returns:
        Tuple of:
        - total_loss: Sum of losses over all control intervals
        - final_state: Modal state after full simulation
        - trajectory: List of ControlStep outputs for each interval

    """
    num_intervals = int(config.total_days / config.control_interval_days)

    # Create step function
    step_fn = create_controlled_step(
        model=model,
        policy_fn=policy_fn,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
        config=config,
    )

    # Optionally wrap with checkpointing
    if config.use_checkpointing:
        step_fn = jax.checkpoint(step_fn)

    # Use scan for efficient unrolling
    def scan_body(carry, _):
        state, cumulative_loss = carry
        control_step = step_fn(state, policy_params)
        new_carry = (control_step.state, cumulative_loss + control_step.loss)
        return new_carry, control_step

    (final_state, total_loss), trajectory = lax.scan(
        scan_body,
        (initial_state, jnp.array(0.0)),
        None,
        length=num_intervals,
    )

    return total_loss, final_state, trajectory


def unroll_with_policy_simple(
    model,
    policy_fn: Callable,
    policy_params: dict,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> jnp.ndarray:
    """Simplified unroll returning only the total loss.

    Useful for training where we only need the scalar loss for gradients.
    Does not return trajectory, reducing memory usage.

    Args:
        Same as unroll_with_policy.

    Returns:
        Scalar total loss over all control intervals.

    """
    total_loss, _, _ = unroll_with_policy(
        model=model,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_state=initial_state,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
        config=config,
    )
    return total_loss


def create_loss_fn(
    model,
    policy_fn: Callable,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> Callable[[dict], jnp.ndarray]:
    """Create a loss function that only depends on policy parameters.

    Useful for creating training loops with jax.value_and_grad.

    Args:
        model: JCM Model instance.
        policy_fn: Policy network apply function.
        initial_state: Initial modal state.
        forcing: ForcingData.
        terrain: TerrainData.
        baseline: ClimateBaseline.
        coords: Model coordinates.
        config: Controller configuration.

    Returns:
        Function (policy_params) -> scalar loss.

    """
    @partial(jax.jit, static_argnums=())
    def loss_fn(policy_params):
        return unroll_with_policy_simple(
            model=model,
            policy_fn=policy_fn,
            policy_params=policy_params,
            initial_state=initial_state,
            forcing=forcing,
            terrain=terrain,
            baseline=baseline,
            coords=coords,
            config=config,
        )

    return loss_fn


def evaluate_policy(
    model,
    policy_fn: Callable,
    policy_params: dict,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> dict:
    """Evaluate a trained policy and return detailed metrics.

    Runs the full simulation with the policy and computes various
    metrics for analysis.

    Args:
        Same as unroll_with_policy.

    Returns:
        Dictionary with:
        - total_loss: Sum of interval losses
        - mean_loss: Average loss per interval
        - final_state: Final modal state
        - trajectory: Full control step trajectory
        - metrics: Dictionary of aggregated climate metrics

    """
    total_loss, final_state, trajectory = unroll_with_policy(
        model=model,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_state=initial_state,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
        config=config,
    )

    num_intervals = len(trajectory.loss)

    # Aggregate metrics
    metrics = {
        'total_loss': float(total_loss),
        'mean_loss': float(total_loss / num_intervals),
        'mean_mcb_forcing': float(jnp.mean(trajectory.mcb_forcing)),
        'max_mcb_forcing': float(jnp.max(trajectory.mcb_forcing)),
        'loss_trajectory': jax.device_get(trajectory.loss),
    }

    return {
        'total_loss': total_loss,
        'mean_loss': total_loss / num_intervals,
        'final_state': final_state,
        'trajectory': trajectory,
        'metrics': metrics,
    }


def compute_policy_gradient(
    model,
    policy_fn: Callable,
    policy_params: dict,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> Tuple[jnp.ndarray, dict]:
    """Compute gradient of total loss with respect to policy parameters.

    Main function for BPTT training. Backpropagates through the entire
    simulation trajectory to compute policy gradients.

    Args:
        Same as unroll_with_policy.

    Returns:
        Tuple of:
        - loss: Total loss value
        - grads: Gradient dictionary matching policy_params structure

    """
    loss_fn = create_loss_fn(
        model=model,
        policy_fn=policy_fn,
        initial_state=initial_state,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
        config=config,
    )

    loss, grads = jax.value_and_grad(loss_fn)(policy_params)

    return loss, grads


def verify_gradients(
    model,
    policy_fn: Callable,
    policy_params: dict,
    initial_state,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    config: ControllerConfig = ControllerConfig(),
) -> dict:
    """Verify gradient computation is working correctly.

    Checks for NaN gradients and zero gradients, which would indicate
    problems with differentiability or gradient flow.

    Args:
        Same as unroll_with_policy.

    Returns:
        Dictionary with verification results:
        - has_nans: Whether any gradient is NaN
        - all_zeros: Whether all gradients are zero
        - gradient_norm: L2 norm of flattened gradients
        - loss: Loss value

    """
    loss, grads = compute_policy_gradient(
        model=model,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_state=initial_state,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
        config=config,
    )

    # Check for NaNs
    grad_leaves = jax.tree.leaves(grads)
    has_nans = any(jnp.any(jnp.isnan(g)) for g in grad_leaves)

    # Check for all zeros
    all_zeros = all(jnp.allclose(g, 0.0) for g in grad_leaves)

    # Compute gradient norm
    grad_flat = jnp.concatenate([g.ravel() for g in grad_leaves])
    gradient_norm = jnp.linalg.norm(grad_flat)

    return {
        'has_nans': bool(has_nans),
        'all_zeros': bool(all_zeros),
        'gradient_norm': float(gradient_norm),
        'loss': float(loss),
    }
