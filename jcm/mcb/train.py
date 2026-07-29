"""BPTT training loop for MCB policy optimization.

This module provides the training infrastructure for MCB neural network
controllers using Backpropagation Through Time (BPTT).

Key components:
- TrainingConfig: Configuration for the training process
- TrainingState: Current state of training (params, optimizer, metrics)
- train_policy: Main training function
- create_train_step: JIT-compiled training step

Example usage:
    from jcm.mcb.train import train_policy, TrainingConfig
    from jcm.mcb.policy import MCBPolicyMLP

    policy = MCBPolicyMLP(output_shape=(96, 48))
    config = TrainingConfig(num_epochs=100, learning_rate=1e-3)

    trained_params, history = train_policy(
        model=model,
        policy=policy,
        coords=coords,
        terrain=terrain,
        forcing=forcing,
        baseline=baseline,
        config=config,
    )
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Optional, Callable, Any, Tuple, Dict
import time

from jcm.mcb.controller import (
    ControllerConfig,
    unroll_with_policy_simple,
    verify_gradients,
)
from jcm.mcb.state_features import ClimateBaseline, get_feature_dim, StateFeatureConfig


class TrainingConfig(NamedTuple):
    """Configuration for MCB policy training.

    Attributes:
        num_epochs: Number of training epochs.
        learning_rate: Initial learning rate for optimizer.
        lr_schedule: Learning rate schedule ('constant', 'cosine', 'warmup_cosine').
        warmup_epochs: Epochs for learning rate warmup (if using warmup schedule).
        optimizer: Optimizer type ('adam', 'adamw', 'sgd').
        weight_decay: Weight decay coefficient (for adamw).
        grad_clip_norm: Maximum gradient norm (None to disable).
        log_interval: Epochs between logging.
        checkpoint_interval: Epochs between saving checkpoints.
        early_stopping_patience: Stop if no improvement for this many epochs.
        random_seed: Random seed for initialization.

    """

    num_epochs: int = 100
    learning_rate: float = 1e-3
    lr_schedule: str = 'constant'
    warmup_epochs: int = 5
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    log_interval: int = 10
    checkpoint_interval: int = 25
    early_stopping_patience: Optional[int] = None
    random_seed: int = 42


class TrainingState(NamedTuple):
    """Current state of training.

    Attributes:
        epoch: Current epoch number.
        params: Current policy parameters.
        opt_state: Optimizer state.
        best_params: Best parameters seen so far (lowest loss).
        best_loss: Best loss seen so far.
        loss_history: List of losses per epoch.
        grad_norm_history: List of gradient norms per epoch.

    """

    epoch: int
    params: dict
    opt_state: Any
    best_params: dict
    best_loss: float
    loss_history: list
    grad_norm_history: list


def create_optimizer(config: TrainingConfig, num_epochs: int = None):
    """Create optimizer with optional learning rate schedule.

    Args:
        config: Training configuration.
        num_epochs: Total epochs (needed for cosine schedule).

    Returns:
        Optax optimizer.

    """
    # Learning rate schedule
    if config.lr_schedule == 'constant':
        lr = config.learning_rate
    elif config.lr_schedule == 'cosine':
        lr = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=num_epochs or config.num_epochs,
        )
    elif config.lr_schedule == 'warmup_cosine':
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=config.warmup_epochs,
        )
        cosine = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=(num_epochs or config.num_epochs) - config.warmup_epochs,
        )
        lr = optax.join_schedules([warmup, cosine], [config.warmup_epochs])
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    # Base optimizer
    if config.optimizer == 'adam':
        base_opt = optax.adam(lr)
    elif config.optimizer == 'adamw':
        base_opt = optax.adamw(lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        base_opt = optax.sgd(lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Optional gradient clipping
    if config.grad_clip_norm is not None:
        opt = optax.chain(
            optax.clip_by_global_norm(config.grad_clip_norm),
            base_opt,
        )
    else:
        opt = base_opt

    return opt


def create_train_step(
    model,
    policy_fn: Callable,
    forcing,
    terrain,
    baseline: ClimateBaseline,
    coords,
    controller_config: ControllerConfig,
    optimizer: optax.GradientTransformation,
):
    """Create a JIT-compiled training step function.

    Args:
        model: JCM Model instance.
        policy_fn: Policy network apply function.
        forcing: ForcingData.
        terrain: TerrainData.
        baseline: ClimateBaseline.
        coords: Model coordinates.
        controller_config: Controller configuration.
        optimizer: Optax optimizer.

    Returns:
        JIT-compiled function (params, opt_state, initial_state) -> (new_params, new_opt_state, loss, grad_norm).

    """
    @jax.jit
    def train_step(params, opt_state, initial_state):
        def loss_fn(p):
            return unroll_with_policy_simple(
                model=model,
                policy_fn=policy_fn,
                policy_params=p,
                initial_state=initial_state,
                forcing=forcing,
                terrain=terrain,
                baseline=baseline,
                coords=coords,
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


def initialize_training(
    policy,
    coords,
    config: TrainingConfig,
    feature_config: StateFeatureConfig = StateFeatureConfig(),
) -> Tuple[dict, Any, optax.GradientTransformation]:
    """Initialize policy parameters and optimizer.

    Args:
        policy: Flax policy module.
        coords: Model coordinates.
        config: Training configuration.
        feature_config: Feature extraction config (to determine input dim).

    Returns:
        Tuple of (initial_params, initial_opt_state, optimizer).

    """
    # Determine input shape based on policy type and feature config
    if feature_config.include_spatial:
        # CNN policy - input is spatial grid
        # Estimate number of channels based on config
        num_channels = 1  # surface temp
        if feature_config.include_temperature:
            num_channels += len(feature_config.temperature_levels)
        if feature_config.include_precipitation:
            num_channels += 1
        input_shape = coords.horizontal.nodal_shape + (num_channels,)
    else:
        # MLP policy - input is flattened scalar features
        input_shape = (get_feature_dim(feature_config),)

    # Initialize parameters
    rng_key = jax.random.PRNGKey(config.random_seed)
    dummy_input = jnp.zeros(input_shape)
    params = policy.init(rng_key, dummy_input)

    # Create optimizer
    optimizer = create_optimizer(config, config.num_epochs)
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def train_policy(
    model,
    policy,
    coords,
    terrain,
    forcing,
    baseline: ClimateBaseline,
    initial_state,
    training_config: TrainingConfig = TrainingConfig(),
    controller_config: ControllerConfig = ControllerConfig(),
    callback: Optional[Callable[[TrainingState], None]] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """Train MCB policy network via BPTT.

    Main training function that runs the full training loop.

    Args:
        model: JCM Model instance.
        policy: Flax policy module.
        coords: Model coordinates.
        terrain: TerrainData.
        forcing: ForcingData.
        baseline: ClimateBaseline for anomaly computation.
        initial_state: Initial modal state for simulation.
        training_config: Training hyperparameters.
        controller_config: Controller configuration.
        callback: Optional callback called after each epoch with TrainingState.

    Returns:
        Tuple of:
        - best_params: Best policy parameters (lowest loss)
        - history: Dictionary with training history and metrics

    """
    # Initialize
    params, opt_state, optimizer = initialize_training(
        policy=policy,
        coords=coords,
        config=training_config,
        feature_config=controller_config.feature_config,
    )

    # Create training step
    train_step = create_train_step(
        model=model,
        policy_fn=policy.apply,
        forcing=forcing,
        terrain=terrain,
        baseline=baseline,
        coords=coords,
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

    print("Starting MCB policy training")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Control interval: {controller_config.control_interval_days} days")
    print(f"  Total simulation: {controller_config.total_days} days")
    print(f"  Target cooling: {controller_config.target_cooling} K")
    print()

    start_time = time.time()

    for epoch in range(training_config.num_epochs):
        epoch_start = time.time()

        # Training step. `loss` is measured at state.params (pre-update);
        # `new_params` are AFTER the optimizer step.
        new_params, new_opt_state, loss, grad_norm = train_step(
            state.params, state.opt_state, initial_state
        )

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        # Update best params. Save state.params — the params that produced
        # loss_val — not the post-update new_params (unmeasured this epoch).
        if loss_val < state.best_loss:
            best_params = state.params
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
    }

    return state.best_params, history


def validate_training_setup(
    model,
    policy,
    coords,
    terrain,
    forcing,
    baseline: ClimateBaseline,
    initial_state,
    controller_config: ControllerConfig = ControllerConfig(),
) -> Dict[str, Any]:
    """Validate training setup before full training.

    Runs a quick check to verify:
    - Forward pass works
    - Gradients are computable
    - No NaN gradients
    - Gradients are non-zero

    Args:
        Same as train_policy.

    Returns:
        Dictionary with validation results and any warnings.

    """
    print("Validating training setup...")

    # Initialize policy
    rng_key = jax.random.PRNGKey(0)
    feature_dim = get_feature_dim(controller_config.feature_config)
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
        results = verify_gradients(
            model=model,
            policy_fn=policy.apply,
            policy_params=params,
            initial_state=initial_state,
            forcing=forcing,
            terrain=terrain,
            baseline=baseline,
            coords=coords,
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


def save_checkpoint(params: dict, path: str, metadata: Optional[dict] = None):
    """Save policy parameters to disk.

    Args:
        params: Policy parameters to save.
        path: File path for checkpoint.
        metadata: Optional metadata to include (epoch, loss, etc.).

    """
    import pickle

    checkpoint = {
        'params': jax.device_get(params),
        'metadata': metadata or {},
    }

    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str) -> Tuple[dict, dict]:
    """Load policy parameters from disk.

    Args:
        path: File path to checkpoint.

    Returns:
        Tuple of (params, metadata).

    """
    import pickle

    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    params = jax.device_put(checkpoint['params'])
    metadata = checkpoint.get('metadata', {})

    print(f"Loaded checkpoint from {path}")
    if metadata:
        print(f"  Metadata: {metadata}")

    return params, metadata
