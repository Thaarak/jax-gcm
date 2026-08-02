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
        loss_mode: Training objective. "summed" (default) sums
            compute_coupled_loss over every control interval — the historical
            behavior, dominated by early-interval transients and the spatial
            uniformity penalty (<0.2% of its magnitude is the gated quantity).
            "terminal_dsst" trains on the squared error of the FINAL-STEP
            global-mean ocean dSST vs the paired baseline, target_cooling,
            area x ocean weighted (plus a small forcing-magnitude
            regularizer). "tail_dsst" trains on the squared error of the
            TIME-MEAN dSST over the final ``tail_days`` coupling steps — the
            registered gate metric exactly (PREREGISTRATION.md section 3 /
            Amendment 3); in the slab's ramp-like 60-day response regime a
            terminal-targeting controller systematically under-corrects the
            tail metric, so Tier-2 training uses tail_dsst.
        forcing_reg_weight: Weight of the mean-square MCB forcing penalty
            added in "terminal_dsst"/"tail_dsst" modes (keeps the pattern
            from wandering without materially shifting the gated dSST).
        tail_days: Terminal-mean window for "tail_dsst" (coupling steps).

    """

    control_interval_steps: int = 30  # e.g., 30 days if daily coupling
    total_steps: int = 180  # tuned training window (days at daily coupling)
    target_cooling: float = -0.1
    loss_weights: CoupledLossWeights = CoupledLossWeights()
    feature_config: CoupledFeatureConfig = CoupledFeatureConfig()
    max_perturbation: float = 0.15
    use_checkpointing: bool = True
    loss_mode: str = "summed"
    forcing_reg_weight: float = 0.001
    tail_days: int = 10


class CoupledControlStep(NamedTuple):
    """Output from a single coupled control step.

    Attributes:
        carry: Final coupled carry after the control interval.
        mcb_forcing: Applied MCB forcing field.
        loss: Loss value for this interval.
        sst_ocean_mean: Optional (interval_steps,) per-coupling-step
            area-weighted ocean-mean SST scalars, collected only when the
            unroll is asked for them (needed for the pre-registered
            final-10-day time-mean dSST metric). None in training mode.

    """

    carry: dict  # CoupledCarry
    mcb_forcing: jnp.ndarray
    loss: jnp.ndarray
    sst_ocean_mean: Any = None


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


def run_interval_final_carry_with_sst(
    carry: dict,
    step_fn: Callable,
    num_steps: int,
    sst_weights: jnp.ndarray,
) -> Tuple[dict, jnp.ndarray]:
    """Like run_interval_final_carry, also collecting per-step ocean-mean SST.

    Emits one scalar per coupling step (the area x ocean weighted mean SST
    after that step) — negligible memory — so evaluation can compute the
    pre-registered final-10-day time-mean dSST (PREREGISTRATION.md section 3)
    instead of the noisier day-60 snapshot.

    Args:
        carry: Initial coupled carry.
        step_fn: Coupler step function.
        num_steps: Number of coupling steps to run.
        sst_weights: Normalized area x ocean weights (ix, il).

    Returns:
        (final_carry, sst_means) with sst_means of shape (num_steps,);
        sst_means[s] is the mean SST after step s+1 of the interval.

    """
    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        sst = new_c["ocn"]["state"].sea_surface_temperature
        return new_c, jnp.sum(sst * sst_weights)

    body = jax.checkpoint(body)
    final_carry, sst_means = lax.scan(body, carry, jnp.arange(num_steps))
    return final_carry, sst_means


def create_coupled_control_step(
    coupler,
    workflow: list,
    policy_fn: Callable,
    baseline_trajectory: CoupledBaselineTrajectory,
    coords,
    config: CoupledControllerConfig,
    ocean_mask: jnp.ndarray,
    collect_sst_weights: jnp.ndarray = None,
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
        collect_sst_weights: Optional normalized area x ocean weights. When
            given, each control step also records the per-coupling-step
            ocean-mean SST (CoupledControlStep.sst_ocean_mean) so evaluation
            can form the pre-registered final-10-day time-mean dSST. Leave
            None in training (no extra state saved under BPTT).

    Returns:
        Function (carry, policy_params, interval_idx) -> CoupledControlStep.

    """
    # Get coupler step function
    step_fn = create_coupled_step_fn(coupler, workflow, jitted=True)

    def control_step(
        carry: dict,
        policy_params: dict,
        interval_idx: jnp.ndarray,
        efficacy: jnp.ndarray = 1.0,
    ) -> CoupledControlStep:
        """Execute one control interval with policy application.

        ``efficacy`` (Tier-2 meta-audit experiment) scales the APPLIED
        perturbation: the policy commands a clipped albedo field, and the
        cloud responds with efficacy x command — modeling uncertain seeding
        efficacy, the dominant real-world MCB uncertainty. The policy never
        observes efficacy directly; a feedback controller can only infer it
        from the realized cooling in the paired-anomaly features. Traced
        scalar, so one compiled function serves every draw.
        """
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

        # Clip the COMMAND and apply ocean mask, then scale by the episode's
        # (unobserved) efficacy to get the APPLIED perturbation. The cap
        # bounds the command; applied albedo remains physically clipped to
        # [0, 1] inside shortwave_radiation.
        mcb_perturbation = jnp.clip(mcb_perturbation, 0.0, config.max_perturbation)
        mcb_perturbation = mcb_perturbation * ocean_mask * efficacy

        # Inject MCB into atmosphere forcing
        # The JCM wrapper will read this and pass to physics
        carry["atm"]["derived"]["mcb_perturbation"] = mcb_perturbation

        # Run coupled simulation for control interval (predictions discarded
        # for memory efficiency under BPTT)
        sst_means = None
        if collect_sst_weights is not None:
            final_carry, sst_means = run_interval_final_carry_with_sst(
                carry=carry,
                step_fn=step_fn,
                num_steps=config.control_interval_steps,
                sst_weights=collect_sst_weights,
            )
        else:
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
            sst_ocean_mean=sst_means,
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
    collect_sst: bool = False,
    efficacy=1.0,
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
        collect_sst: When True, trajectory.sst_ocean_mean holds the
            per-coupling-step ocean-mean SST, shape (num_intervals,
            interval_steps) — used by evaluation for the pre-registered
            final-10-day time-mean dSST. Keep False for training.
        efficacy: Per-episode MCB efficacy factor (Tier-2 experiment):
            applied perturbation = efficacy x clipped command. Scalar
            (float or traced jnp scalar); unobserved by the policy.

    Returns:
        Tuple of:
        - total_loss: Sum of losses over all control intervals
        - final_carry: Coupled carry after full simulation
        - trajectory: Stacked CoupledControlStep outputs

    """
    num_intervals = config.total_steps // config.control_interval_steps

    collect_sst_weights = None
    if collect_sst or config.loss_mode == "tail_dsst":
        from jcm.mcb.state_features import compute_area_weights
        w = compute_area_weights(coords) * ocean_mask
        collect_sst_weights = w / jnp.sum(w)

    # Create control step function
    control_step = create_coupled_control_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy_fn,
        baseline_trajectory=baseline_trajectory,
        coords=coords,
        config=config,
        ocean_mask=ocean_mask,
        collect_sst_weights=collect_sst_weights,
    )

    # Optionally wrap with checkpointing for memory efficiency
    if config.use_checkpointing:
        control_step = jax.checkpoint(control_step)

    # Unroll with scan over interval indices (needed to index the paired
    # baseline trajectory at the right timesteps)
    efficacy_arr = jnp.asarray(efficacy, dtype=jnp.float32)

    def scan_body(state, interval_idx):
        carry, cumulative_loss = state
        step_output = control_step(carry, policy_params, interval_idx,
                                   efficacy_arr)
        new_state = (step_output.carry, cumulative_loss + step_output.loss)
        return new_state, step_output

    (final_carry, total_loss), trajectory = lax.scan(
        scan_body,
        (initial_carry, jnp.array(0.0)),
        jnp.arange(num_intervals),
    )

    if config.loss_mode == "terminal_dsst":
        # Gate-aligned objective: score exactly what the pre-registered gate
        # scores (evaluate_coupled_policy below) — the final-step global-mean
        # ocean dSST vs the paired baseline — instead of the interval sum,
        # whose gradient is dominated by early-interval "not cooled yet" terms
        # and the uniformity penalty and which rewards overcooling.
        from jcm.mcb.state_features import compute_area_weights
        area_weights = compute_area_weights(coords)
        sst_weights = area_weights * ocean_mask
        sst_weights = sst_weights / jnp.sum(sst_weights)
        final_sst = final_carry["ocn"]["state"].sea_surface_temperature
        baseline_sst = baseline_trajectory.at_step(config.total_steps).sst
        dsst = jnp.sum((final_sst - baseline_sst) * sst_weights)
        terminal = (dsst - config.target_cooling) ** 2
        forcing_reg = config.forcing_reg_weight * jnp.mean(
            trajectory.mcb_forcing ** 2
        )
        total_loss = terminal + forcing_reg
    elif config.loss_mode == "tail_dsst":
        # Registered-metric objective (Amendment 3): squared error of the
        # TIME-MEAN dSST over the final tail_days steps — identical to
        # evaluate_coupled_policy's final_sst_change_10d. In the ramp-like
        # slab response a terminal-only objective leaves the tail window
        # systematically under-corrected (2026-07-31 adversarial review).
        tail = int(min(config.tail_days, config.total_steps))
        policy_daily = jnp.reshape(trajectory.sst_ocean_mean, (-1,))
        baseline_tail = jnp.stack([
            jnp.sum(baseline_trajectory.sst[t] * collect_sst_weights)
            for t in range(config.total_steps - tail + 1,
                           config.total_steps + 1)
        ])
        dsst_tail = jnp.mean(policy_daily[-tail:] - baseline_tail)
        forcing_reg = config.forcing_reg_weight * jnp.mean(
            trajectory.mcb_forcing ** 2
        )
        total_loss = (dsst_tail - config.target_cooling) ** 2 + forcing_reg

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
    efficacy=1.0,
) -> jnp.ndarray:
    """Simplified unroll returning only total loss (for training).

    More memory efficient as it doesn't store trajectory.

    Args:
        Same as unroll_coupled_with_policy (incl. per-episode efficacy).

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
        efficacy=efficacy,
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
    tail_mean_days: int = 10,
    efficacy=1.0,
) -> dict:
    """Evaluate a trained policy and return detailed metrics.

    Args:
        Same as unroll_coupled_with_policy, plus:
        tail_mean_days: Window for the pre-registered terminal metric
            (PREREGISTRATION.md section 3): dSST time-mean over the final
            ``tail_mean_days`` coupling steps. 0 disables collection and
            reproduces the old snapshot-only behavior.
        efficacy: Per-episode MCB efficacy (applied = efficacy x command);
            recorded in metrics. The policy does not observe it.

    Returns:
        Dictionary with metrics:
        - total_loss: Sum of interval losses
        - mean_loss: Average loss per interval
        - final_sst_change: Final global SST change (day-60 snapshot; kept
          for continuity with pre-2026-07-30 artifacts)
        - final_sst_change_10d: dSST time-mean over the final
          ``tail_mean_days`` days — the REGISTERED cooling metric
        - mean_mcb_forcing: Average MCB forcing magnitude
        - max_mcb_forcing: Maximum MCB forcing
        - trajectory: Full control step trajectory

    """
    from jcm.mcb.state_features import compute_area_weights

    collect_sst = tail_mean_days > 0
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
        collect_sst=collect_sst,
        efficacy=efficacy,
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
        'efficacy': float(efficacy),
        'loss_trajectory': jax.device_get(trajectory.loss),
    }

    if collect_sst:
        # Pre-registered terminal metric: dSST time-mean over the final
        # ``tail`` days. trajectory.sst_ocean_mean[i, s] is the ocean-mean
        # SST after coupling step i*L + s + 1, so flat index t-1 = day t;
        # baseline_trajectory.sst[t] is the state after t steps.
        tail = int(min(tail_mean_days, config.total_steps))
        policy_daily = jnp.reshape(trajectory.sst_ocean_mean, (-1,))
        baseline_tail = jnp.stack([
            jnp.sum(baseline_trajectory.sst[t] * sst_weights)
            for t in range(config.total_steps - tail + 1,
                           config.total_steps + 1)
        ])
        dsst_tail = policy_daily[-tail:] - baseline_tail
        metrics['final_sst_change_10d'] = float(jnp.mean(dsst_tail))
        metrics['tail_mean_days'] = tail
        metrics['dsst_daily_tail'] = jax.device_get(dsst_tail)

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
