"""Training loop for coupled MCB policy optimization.

Extends train.py to work with JAX-ESM coupled simulations,
enabling gradient flow through ocean feedback.

Example usage:
    from jcm.mcb.coupled_train import train_coupled_policy
    from jcm.mcb.policy import MCBPolicyMLP

    policy = MCBPolicyMLP(output_shape=(96, 48))

    trained_params, history = train_coupled_policy(
        coupler=coupler,
        workflow=["coupling", "atm", "ocn"],
        policy=policy,
        coords=coords,
        terrain_fmask=fmask,
        initial_carry=initial_carry,
        baseline=baseline,
    )
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Callable, Any, Tuple, Dict
import time

from jcm.mcb.coupled_controller import (
    CoupledControllerConfig,
    unroll_coupled_simple,
    verify_coupled_gradients,
)
from jcm.mcb.coupled_features import (
    CoupledBaseline,
    get_coupled_feature_dim,
)
from jcm.mcb.train import (
    TrainingConfig,
    TrainingState,
    create_optimizer,
    load_checkpoint,
)


def create_coupled_train_step(
    coupler,
    workflow: list,
    policy_fn: Callable,
    baseline: CoupledBaseline,
    coords,
    ocean_mask: jnp.ndarray,
    controller_config: CoupledControllerConfig,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """Create JIT-compiled training step for coupled simulation.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow.
        policy_fn: Policy apply function.
        baseline: Climate baseline.
        coords: Model coordinates.
        ocean_mask: Ocean mask for MCB.
        controller_config: Controller config.
        optimizer: Optax optimizer.

    Returns:
        JIT-compiled train step function.

    """
    @jax.jit
    def train_step(
        params: dict,
        opt_state: Any,
        initial_carry: dict,
    ) -> Tuple[dict, Any, jnp.ndarray, jnp.ndarray]:
        """Execute one training step.

        Args:
            params: Current policy parameters.
            opt_state: Current optimizer state.
            initial_carry: Initial coupled carry.

        Returns:
            Tuple of (new_params, new_opt_state, loss, grad_norm).

        """
        def loss_fn(p):
            return unroll_coupled_simple(
                coupler=coupler,
                workflow=workflow,
                policy_fn=policy_fn,
                policy_params=p,
                initial_carry=initial_carry,
                baseline=baseline,
                coords=coords,
                ocean_mask=ocean_mask,
                config=controller_config,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Compute gradient norm for monitoring
        grad_leaves = jax.tree.leaves(grads)
        grad_flat = jnp.concatenate([g.ravel() for g in grad_leaves])
        grad_norm = jnp.linalg.norm(grad_flat)

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss, grad_norm

    return train_step


def initialize_coupled_training(
    policy,
    coords,
    config: TrainingConfig,
    controller_config: CoupledControllerConfig,
) -> Tuple[dict, Any, optax.GradientTransformation]:
    """Initialize policy parameters and optimizer for coupled training.

    Args:
        policy: Flax policy module.
        coords: Model coordinates.
        config: Training configuration.
        controller_config: Controller configuration.

    Returns:
        Tuple of (initial_params, initial_opt_state, optimizer).

    """
    # Get feature dimension for coupled features
    feature_dim = get_coupled_feature_dim(controller_config.feature_config)
    input_shape = (feature_dim,)

    # Initialize parameters
    rng_key = jax.random.PRNGKey(config.random_seed)
    dummy_input = jnp.zeros(input_shape)
    params = policy.init(rng_key, dummy_input)

    # Create optimizer
    optimizer = create_optimizer(config, config.num_epochs)
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def train_coupled_policy(
    coupler,
    workflow: list,
    policy,  # Flax module
    coords,
    terrain_fmask: jnp.ndarray,
    initial_carry: dict,
    baseline: CoupledBaseline,
    training_config: TrainingConfig = TrainingConfig(),
    controller_config: CoupledControllerConfig = CoupledControllerConfig(),
    callback: Optional[Callable[[TrainingState], None]] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """Train MCB policy with coupled ocean feedback.

    Main training function for coupled MCB optimization. Uses BPTT
    through the coupled atmosphere-ocean simulation.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow (e.g., ["coupling", "atm", "ocn"]).
        policy: Flax policy module.
        coords: Model coordinates.
        terrain_fmask: Land mask (1.0 = land, 0.0 = ocean).
        initial_carry: Initial coupled state.
        baseline: Climate baseline for anomalies.
        training_config: Training hyperparameters.
        controller_config: Controller configuration.
        callback: Optional epoch callback.

    Returns:
        Tuple of:
        - best_params: Best policy parameters (lowest loss)
        - history: Dictionary with training history and metrics

    """
    ocean_mask = 1.0 - terrain_fmask

    # Initialize
    params, opt_state, optimizer = initialize_coupled_training(
        policy=policy,
        coords=coords,
        config=training_config,
        controller_config=controller_config,
    )

    # Create training step
    train_step = create_coupled_train_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        baseline=baseline,
        coords=coords,
        ocean_mask=ocean_mask,
        controller_config=controller_config,
        optimizer=optimizer,
    )

    # Training state
    state = TrainingState(
        epoch=0,
        params=params,
        opt_state=opt_state,
        best_params=params,
        best_loss=float('inf'),
        loss_history=[],
        grad_norm_history=[],
    )

    # Early stopping tracking
    patience_counter = 0

    print("Starting coupled MCB policy training")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Control interval: {controller_config.control_interval_steps} steps")
    print(f"  Total steps: {controller_config.total_steps}")
    print(f"  Target cooling: {controller_config.target_cooling} K")
    print("  Using ocean SST for loss (coupled mode)")
    print()

    start_time = time.time()

    for epoch in range(training_config.num_epochs):
        epoch_start = time.time()

        # Training step
        new_params, new_opt_state, loss, grad_norm = train_step(
            state.params, state.opt_state, initial_carry
        )

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        # Update best params
        if loss_val < state.best_loss:
            best_params = new_params
            best_loss = loss_val
            patience_counter = 0
        else:
            best_params = state.best_params
            best_loss = state.best_loss
            patience_counter += 1

        # Update state
        state = TrainingState(
            epoch=epoch + 1,
            params=new_params,
            opt_state=new_opt_state,
            best_params=best_params,
            best_loss=best_loss,
            loss_history=state.loss_history + [loss_val],
            grad_norm_history=state.grad_norm_history + [grad_norm_val],
        )

        # Logging
        if epoch % training_config.log_interval == 0:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.6f} | "
                  f"Grad norm: {grad_norm_val:.4f} | "
                  f"Best: {best_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")

        # Callback
        if callback is not None:
            callback(state)

        # Early stopping
        if training_config.early_stopping_patience is not None:
            if patience_counter >= training_config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_counter} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Final loss: {state.loss_history[-1]:.6f}")
    print(f"Best loss: {state.best_loss:.6f}")

    # Return best params and history
    history = {
        'loss_history': state.loss_history,
        'grad_norm_history': state.grad_norm_history,
        'best_loss': state.best_loss,
        'final_loss': state.loss_history[-1],
        'total_time': total_time,
        'epochs_completed': state.epoch,
        'mode': 'coupled',
    }

    return state.best_params, history


def validate_coupled_training_setup(
    coupler,
    workflow: list,
    policy,
    coords,
    terrain_fmask: jnp.ndarray,
    initial_carry: dict,
    baseline: CoupledBaseline,
    controller_config: CoupledControllerConfig = CoupledControllerConfig(),
) -> Dict[str, Any]:
    """Validate coupled training setup before full training.

    Runs a quick check to verify gradients flow correctly.

    Args:
        Same as train_coupled_policy.

    Returns:
        Dictionary with validation results and any warnings.

    """
    print("Validating coupled training setup...")

    ocean_mask = 1.0 - terrain_fmask

    # Initialize policy
    rng_key = jax.random.PRNGKey(0)
    feature_dim = get_coupled_feature_dim(controller_config.feature_config)
    dummy_input = jnp.zeros(feature_dim)
    params = policy.init(rng_key, dummy_input)

    # Test forward pass
    print("  Testing forward pass...", end=" ")
    try:
        output = policy.apply(params, dummy_input)
        print(f"OK (output shape: {output.shape})")
    except Exception as e:
        print(f"FAILED: {e}")
        return {'valid': False, 'error': f"Forward pass failed: {e}"}

    # Test gradient computation
    print("  Testing gradient computation...", end=" ")
    try:
        results = verify_coupled_gradients(
            coupler=coupler,
            workflow=workflow,
            policy_fn=policy.apply,
            policy_params=params,
            initial_carry=initial_carry,
            baseline=baseline,
            coords=coords,
            ocean_mask=ocean_mask,
            config=controller_config,
        )
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return {'valid': False, 'error': f"Gradient computation failed: {e}"}

    # Check results
    warnings = []
    if results['has_nans']:
        warnings.append("WARNING: Gradients contain NaN values")
    if results['all_zeros']:
        warnings.append("WARNING: All gradients are zero (no gradient flow)")
    if results['gradient_norm'] > 1e6:
        warnings.append(f"WARNING: Very large gradient norm ({results['gradient_norm']:.2e})")

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    {w}")

    print(f"\n  Initial loss: {results['loss']:.6f}")
    print(f"  Gradient norm: {results['gradient_norm']:.4f}")

    return {
        'valid': not results['has_nans'] and not results['all_zeros'],
        'loss': results['loss'],
        'gradient_norm': results['gradient_norm'],
        'has_nans': results['has_nans'],
        'all_zeros': results['all_zeros'],
        'warnings': warnings,
    }


def resume_coupled_training(
    coupler,
    workflow: list,
    policy,
    coords,
    terrain_fmask: jnp.ndarray,
    initial_carry: dict,
    baseline: CoupledBaseline,
    checkpoint_path: str,
    training_config: TrainingConfig = TrainingConfig(),
    controller_config: CoupledControllerConfig = CoupledControllerConfig(),
    callback: Optional[Callable[[TrainingState], None]] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """Resume training from a checkpoint.

    Args:
        Same as train_coupled_policy, plus:
        checkpoint_path: Path to saved checkpoint.

    Returns:
        Same as train_coupled_policy.

    """
    # Load checkpoint
    params, metadata = load_checkpoint(checkpoint_path)
    start_epoch = metadata.get('epoch', 0)

    print(f"Resuming from epoch {start_epoch}")

    ocean_mask = 1.0 - terrain_fmask

    # Create optimizer (will need to reinitialize state)
    optimizer = create_optimizer(training_config, training_config.num_epochs)
    opt_state = optimizer.init(params)

    # Create training step
    train_step = create_coupled_train_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        baseline=baseline,
        coords=coords,
        ocean_mask=ocean_mask,
        controller_config=controller_config,
        optimizer=optimizer,
    )

    # Training state
    state = TrainingState(
        epoch=start_epoch,
        params=params,
        opt_state=opt_state,
        best_params=params,
        best_loss=metadata.get('best_loss', float('inf')),
        loss_history=metadata.get('loss_history', []),
        grad_norm_history=metadata.get('grad_norm_history', []),
    )

    patience_counter = 0
    start_time = time.time()

    for epoch in range(start_epoch, training_config.num_epochs):
        epoch_start = time.time()

        new_params, new_opt_state, loss, grad_norm = train_step(
            state.params, state.opt_state, initial_carry
        )

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        if loss_val < state.best_loss:
            best_params = new_params
            best_loss = loss_val
            patience_counter = 0
        else:
            best_params = state.best_params
            best_loss = state.best_loss
            patience_counter += 1

        state = TrainingState(
            epoch=epoch + 1,
            params=new_params,
            opt_state=new_opt_state,
            best_params=best_params,
            best_loss=best_loss,
            loss_history=state.loss_history + [loss_val],
            grad_norm_history=state.grad_norm_history + [grad_norm_val],
        )

        if epoch % training_config.log_interval == 0:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.6f} | "
                  f"Grad norm: {grad_norm_val:.4f} | "
                  f"Best: {best_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")

        if callback is not None:
            callback(state)

        if training_config.early_stopping_patience is not None:
            if patience_counter >= training_config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best loss: {state.best_loss:.6f}")

    history = {
        'loss_history': state.loss_history,
        'grad_norm_history': state.grad_norm_history,
        'best_loss': state.best_loss,
        'final_loss': state.loss_history[-1] if state.loss_history else None,
        'total_time': total_time,
        'epochs_completed': state.epoch,
        'mode': 'coupled',
        'resumed_from': checkpoint_path,
    }

    return state.best_params, history
