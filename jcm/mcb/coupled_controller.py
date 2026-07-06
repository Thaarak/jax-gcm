"""Coupled MCB controller for training with ocean feedback.

Adapts the atmosphere-only controller to work with JAX-ESM
coupled simulations, enabling gradient flow through ocean feedback.

The key difference is that we:
1. Extract features from coupled state (both atm and ocn)
2. Inject MCB perturbation into the coupled carry
3. Run the coupler's step function
4. Compute loss using ocean SST

Features and loss are computed against a PAIRED no-MCB baseline trajectory
(CoupledBaselineTrajectory): anomalies at coupling step t are taken against
the no-MCB run at the same step t, so natural model drift cancels exactly and
only the MCB-caused signal remains (Stage 2 fix).

Example usage:
    from jcm.mcb.coupled_features import compute_baseline_trajectory
    from jcm.mcb.coupled_controller import (
        create_coupled_step_fn,
        unroll_coupled_with_policy,
        CoupledControllerConfig,
    )

    step_fn = create_coupled_step_fn(coupler, ["coupling", "atm", "ocn"])
    baseline_trajectory = compute_baseline_trajectory(
        initial_carry, step_fn, num_steps=config.total_steps, coords=coords
    )
    total_loss, final_carry, trajectory = unroll_coupled_with_policy(
        coupler=coupler,
        workflow=["coupling", "atm", "ocn"],
        policy_fn=policy.apply,
        policy_params=params,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        ocean_mask=ocean_mask,
    )
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Tuple, Any, NamedTuple
from functools import partial

from jcm.mcb.coupled_features import (
    extract_coupled_features,
    CoupledBaselineTrajectory,
    CoupledFeatureConfig,
)
from jcm.mcb.coupled_loss import (
    compute_coupled_loss,
    CoupledLossWeights,
)


class CoupledControllerConfig(NamedTuple):
    """Configuration for coupled MCB controller.

    Attributes:
        control_interval_steps: Coupler steps between policy applications.
        total_steps: Total coupler steps to run.
        target_cooling: Target SST change (K, negative for cooling).
        loss_weights: Weights for loss components.
        feature_config: Configuration for feature extraction.
        max_perturbation: Maximum MCB albedo perturbation.
        use_checkpointing: Enable gradient checkpointing for memory.

    """

    control_interval_steps: int = 30  # e.g., 30 days if daily coupling
    total_steps: int = 180  # tuned training window (days at daily coupling)
    target_cooling: float = -0.1
    loss_weights: CoupledLossWeights = CoupledLossWeights()
    feature_config: CoupledFeatureConfig = CoupledFeatureConfig()
    max_perturbation: float = 0.15
    use_checkpointing: bool = True


class CoupledControlStep(NamedTuple):
    """Output from a single coupled control step.

    Attributes:
        carry: Final coupled carry after the control interval.
        mcb_forcing: Applied MCB forcing field.
        loss: Loss value for this interval.

    """

    carry: dict  # CoupledCarry
    mcb_forcing: jnp.ndarray
    loss: jnp.ndarray


def create_coupled_step_fn(
    coupler,
    workflow: list,
    jitted: bool = True,
):
    """Create a single-step function for the coupler.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow (e.g., ["coupling", "atm", "ocn"]).
        jitted: Whether to JIT compile the step function.

    Returns:
        Function (carry, step) -> (new_carry, predictions).

    """
    step_fn = coupler.generate_step_function(
        workflow=workflow,
        jitted=jitted,
        show_progress=False,
        verbose=False,
    )
    return step_fn


def run_coupled_interval(
    carry: dict,
    step_fn: Callable,
    num_steps: int,
) -> Tuple[dict, Any]:
    """Run coupled simulation for a control interval.

    Args:
        carry: Initial coupled carry.
        step_fn: Coupler step function.
        num_steps: Number of coupling steps to run.

    Returns:
        Tuple of (final_carry, stacked_predictions).

    """
    def scan_body(c, step_idx):
        new_c, preds = step_fn(c, step_idx)
        return new_c, preds

    final_carry, trajectory = lax.scan(
        scan_body,
        carry,
        jnp.arange(num_steps),
    )
    return final_carry, trajectory


def run_interval_final_carry(
    carry: dict,
    step_fn: Callable,
    num_steps: int,
) -> dict:
    """Run a control interval, returning only the final carry.

    Unlike run_coupled_interval, per-step predictions are discarded, so the
    scan does not stack a full trajectory of physics outputs (major memory
    saving under reverse-mode AD). Each coupling step is checkpointed.
    Proven in Stage 1 pattern optimization (60-day rollouts fwd+bwd on GPU).

    Args:
        carry: Initial coupled carry.
        step_fn: Coupler step function.
        num_steps: Number of coupling steps to run.

    Returns:
        Final coupled carry.

    """
    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        return new_c, None

    body = jax.checkpoint(body)
    final_carry, _ = lax.scan(body, carry, jnp.arange(num_steps))
    return final_carry


def create_coupled_control_step(
    coupler,
    workflow: list,
    policy_fn: Callable,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    config: CoupledControllerConfig,
    ocean_mask: jnp.ndarray,
) -> Callable:
    """Create a single coupled control step function.

    This creates a differentiable function that:
    1. Extracts features from coupled state (anomalies vs the PAIRED no-MCB
       baseline at the same coupling step, plus time-of-rollout)
    2. Applies policy to get MCB forcing
    3. Injects MCB into coupled carry
    4. Runs coupler for control interval
    5. Computes loss from SST (paired difference at the interval's end step)

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow.
        policy_fn: Policy network apply function.
        baseline_trajectory: Paired no-MCB baseline trajectory covering at
            least config.total_steps coupling steps.
        coords: Model coordinates.
        config: Controller configuration.
        ocean_mask: Ocean cell mask for MCB application.

    Returns:
        Function (carry, policy_params, interval_idx) -> CoupledControlStep.

    """
    # Get coupler step function
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)

    def control_step(
        carry: dict,
        policy_params: dict,
        interval_idx: jnp.ndarray,
    ) -> CoupledControlStep:
        """Execute one control interval with policy application."""
        t_start = interval_idx * config.control_interval_steps
        t_end = t_start + config.control_interval_steps

        # Extract features vs the paired baseline at the interval start.
        # ocean_mask restricts the area-weighted means to ocean cells on
        # realistic terrain (aquaplanet mask is all ones -> no-op).
        features = extract_coupled_features(
            coupled_carry=carry,
            baseline=baseline_trajectory.at_step(t_start),
            coords=coords,
            config=config.feature_config,
            time_fraction=t_start / config.total_steps,
            ocean_mask=ocean_mask,
        )

        # Get MCB perturbation from policy
        mcb_perturbation = policy_fn(policy_params, features)

        # Clip and apply ocean mask
        mcb_perturbation = jnp.clip(mcb_perturbation, 0.0, config.max_perturbation)
        mcb_perturbation = mcb_perturbation * ocean_mask

        # Inject MCB into atmosphere forcing
        # The JCM wrapper will read this and pass to physics
        carry["atm"]["derived"]["mcb_perturbation"] = mcb_perturbation

        # Run coupled simulation for control interval (predictions discarded
        # for memory efficiency under BPTT)
        final_carry = run_interval_final_carry(
            carry=carry,
            step_fn=step_fn,
            num_steps=config.control_interval_steps,
        )

        # Compute loss vs the paired baseline at the interval end. The SST
        # cooling / uniformity terms are ocean-masked on realistic terrain
        # (aquaplanet mask is all ones -> identical to prior behavior).
        loss_baseline = baseline_trajectory.at_step(t_end)
        loss = compute_coupled_loss(
            coupled_carry=final_carry,
            baseline_sst=loss_baseline.sst,
            baseline_precip=loss_baseline.precipitation,
            target_cooling=config.target_cooling,
            mcb_forcing=mcb_perturbation,
            coords=coords,
            weights=config.loss_weights,
            ocean_mask=ocean_mask,
        )

        return CoupledControlStep(
            carry=final_carry,
            mcb_forcing=mcb_perturbation,
            loss=loss,
        )

    return control_step


def unroll_coupled_with_policy(
    coupler,
    workflow: list,
    policy_fn: Callable,
    policy_params: dict,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> Tuple[jnp.ndarray, dict, Any]:
    """Unroll coupled simulation with neural network control.

    Main entry point for differentiable coupled MCB control.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow (e.g., ["coupling", "atm", "ocn"]).
        policy_fn: Policy network apply function (params, features) -> forcing.
        policy_params: Policy network parameters.
        initial_carry: Initial coupled carry (must be the same state the
            baseline trajectory was computed from, for exact pairing).
        baseline_trajectory: Paired no-MCB baseline trajectory.
        coords: Model coordinates.
        ocean_mask: Ocean mask for MCB application.
        config: Controller configuration.

    Returns:
        Tuple of:
        - total_loss: Sum of losses over all control intervals
        - final_carry: Coupled carry after full simulation
        - trajectory: Stacked CoupledControlStep outputs

    """
    num_intervals = config.total_steps // config.control_interval_steps

    # Create control step function
    control_step = create_coupled_control_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        config=config,
        ocean_mask=ocean_mask,
    )

    # Optionally wrap with checkpointing for memory efficiency
    if config.use_checkpointing:
        control_step = jax.checkpoint(control_step)

    # Unroll with scan over interval indices (needed to index the paired
    # baseline trajectory at the right timesteps)
    def scan_body(state, interval_idx):
        carry, cumulative_loss = state
        step_output = control_step(carry, policy_params, interval_idx)
        new_state = (step_output.carry, cumulative_loss + step_output.loss)
        return new_state, step_output

    (final_carry, total_loss), trajectory = lax.scan(
        scan_body,
        (initial_carry, jnp.array(0.0)),
        jnp.arange(num_intervals),
    )

    return total_loss, final_carry, trajectory


def unroll_coupled_simple(
    coupler,
    workflow: list,
    policy_fn: Callable,
    policy_params: dict,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> jnp.ndarray:
    """Simplified unroll returning only total loss (for training).

    More memory efficient as it doesn't store trajectory.

    Args:
        Same as unroll_coupled_with_policy.

    Returns:
        Scalar total loss over all control intervals.

    """
    total_loss, _, _ = unroll_coupled_with_policy(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        ocean_mask=ocean_mask,
        config=config,
    )
    return total_loss


def create_coupled_loss_fn(
    coupler,
    workflow: list,
    policy_fn: Callable,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> Callable[[dict], jnp.ndarray]:
    """Create loss function that only depends on policy parameters.

    Useful for training with jax.value_and_grad.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow.
        policy_fn: Policy network apply function.
        initial_carry: Initial coupled carry.
        baseline_trajectory: Paired no-MCB baseline trajectory.
        coords: Model coordinates.
        ocean_mask: Ocean mask.
        config: Controller configuration.

    Returns:
        Function (policy_params) -> scalar loss.

    """
    @partial(jax.jit)
    def loss_fn(policy_params: dict) -> jnp.ndarray:
        return unroll_coupled_simple(
            coupler=coupler,
            workflow=workflow,
            policy_fn=policy_fn,
            policy_params=policy_params,
            initial_carry=initial_carry,
            baseline_trajectory=baseline_trajectory,
            coords=coords,
            ocean_mask=ocean_mask,
            config=config,
        )

    return loss_fn


def evaluate_coupled_policy(
    coupler,
    workflow: list,
    policy_fn: Callable,
    policy_params: dict,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> dict:
    """Evaluate a trained policy and return detailed metrics.

    Args:
        Same as unroll_coupled_with_policy.

    Returns:
        Dictionary with metrics:
        - total_loss: Sum of interval losses
        - mean_loss: Average loss per interval
        - final_sst_change: Final global SST change
        - mean_mcb_forcing: Average MCB forcing magnitude
        - max_mcb_forcing: Maximum MCB forcing
        - trajectory: Full control step trajectory

    """
    from jcm.mcb.state_features import compute_area_weights

    total_loss, final_carry, trajectory = unroll_coupled_with_policy(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        ocean_mask=ocean_mask,
        config=config,
    )

    num_intervals = config.total_steps // config.control_interval_steps
    area_weights = compute_area_weights(coords)

    # Weight the reported dSST over ocean cells (identical to the plain
    # area-weighted mean on the aquaplanet, where ocean_mask is all ones).
    sst_weights = area_weights * ocean_mask
    sst_weights = sst_weights / jnp.sum(sst_weights)

    # Compute SST change vs the paired baseline at the same final step
    final_sst = final_carry["ocn"]["state"].sea_surface_temperature
    baseline_sst = baseline_trajectory.at_step(config.total_steps).sst
    global_final_sst = jnp.sum(final_sst * sst_weights)
    global_baseline_sst = jnp.sum(baseline_sst * sst_weights)
    sst_change = global_final_sst - global_baseline_sst

    # Aggregate metrics
    metrics = {
        'total_loss': float(total_loss),
        'mean_loss': float(total_loss / num_intervals),
        'final_sst_change': float(sst_change),
        'target_cooling': config.target_cooling,
        'mean_mcb_forcing': float(jnp.mean(trajectory.mcb_forcing)),
        'max_mcb_forcing': float(jnp.max(trajectory.mcb_forcing)),
        'loss_trajectory': jax.device_get(trajectory.loss),
    }

    return {
        'total_loss': total_loss,
        'final_carry': final_carry,
        'trajectory': trajectory,
        'metrics': metrics,
    }


def compute_coupled_policy_gradient(
    coupler,
    workflow: list,
    policy_fn: Callable,
    policy_params: dict,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> Tuple[jnp.ndarray, dict]:
    """Compute gradient of total loss with respect to policy parameters.

    Main function for BPTT training. Backpropagates through the entire
    coupled simulation trajectory.

    Args:
        Same as unroll_coupled_with_policy.

    Returns:
        Tuple of:
        - loss: Total loss value
        - grads: Gradient dictionary matching policy_params structure

    """
    loss_fn = create_coupled_loss_fn(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        ocean_mask=ocean_mask,
        config=config,
    )

    loss, grads = jax.value_and_grad(loss_fn)(policy_params)
    return loss, grads


def verify_coupled_gradients(
    coupler,
    workflow: list,
    policy_fn: Callable,
    policy_params: dict,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    ocean_mask: jnp.ndarray,
    config: CoupledControllerConfig = CoupledControllerConfig(),
) -> dict:
    """Verify gradient computation is working correctly.

    Checks for NaN gradients and zero gradients.

    Args:
        Same as unroll_coupled_with_policy.

    Returns:
        Dictionary with verification results:
        - has_nans: Whether any gradient is NaN
        - all_zeros: Whether all gradients are zero
        - gradient_norm: L2 norm of flattened gradients
        - loss: Loss value

    """
    loss, grads = compute_coupled_policy_gradient(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        policy_params=policy_params,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        ocean_mask=ocean_mask,
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
