"""Training loop for coupled MCB policy optimization.

Extends train.py to work with JAX-ESM coupled simulations,
enabling gradient flow through ocean feedback.

Features and loss use a PAIRED no-MCB baseline trajectory (Stage 2): compute
it once with compute_baseline_trajectory from the same initial carry used for
training, then pass it as baseline_trajectory.

Example usage:
    from jcm.mcb.coupled_features import compute_baseline_trajectory
    from jcm.mcb.coupled_controller import create_coupled_step_fn
    from jcm.mcb.coupled_train import train_coupled_policy
    from jcm.mcb.policy import MCBPolicyMLP

    policy = MCBPolicyMLP(output_shape=(96, 48))
    step_fn = create_coupled_step_fn(coupler, ["coupling", "atm", "ocn"])
    baseline_trajectory = compute_baseline_trajectory(
        initial_carry, step_fn, num_steps=180, coords=coords
    )

    trained_params, history = train_coupled_policy(
        coupler=coupler,
        workflow=["coupling", "atm", "ocn"],
        policy=policy,
        coords=coords,
        terrain_fmask=fmask,
        initial_carry=initial_carry,
        baseline_trajectory=baseline_trajectory,
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
    CoupledBaselineTrajectory,
    get_coupled_feature_dim,
)
from jcm.mcb.train import (
    TrainingConfig,
    TrainingState,
    create_optimizer,
    load_checkpoint,
)

# Land-fraction threshold above which a cell is treated as land. This matches
# the slab ocean model's binary land-sea convention (builtin_grid_generator.
# load_jcm_mask: `bmask = fmask > 0.95`). The ocean model pins any cell with
# fmask > 0.95 to default_land_surface_temperature (288.15 K) and evolves SST
# everywhere else. The MCB loss/features must weight exactly those evolving-SST
# cells, so the ocean mask is binary (1 - bmask), NOT the fractional 1 - fmask:
# a fractional mask would down-weight coastal ocean cells whose SST is live.
_LAND_FMASK_THRESHOLD = 0.95


def ocean_mask_from_fmask(terrain_fmask: jnp.ndarray) -> jnp.ndarray:
    """Binary ocean mask matching the slab ocean model's land-sea convention.

    Args:
        terrain_fmask: Fractional land mask in [0, 1] (1.0 = all land).

    Returns:
        1.0 for ocean cells (fmask <= 0.95, SST evolves), 0.0 for land cells
        (fmask > 0.95, SST pinned to 288.15 K by the slab ocean model).

    """
    return jnp.where(terrain_fmask > _LAND_FMASK_THRESHOLD, 0.0, 1.0)


def ocean_mask_from_coupler(coupler) -> jnp.ndarray:
    """Authoritative ocean mask read from the slab ocean model's own bmask.

    The slab ocean model pins land cells to 288.15 K using its grid's binary
    `bmask` (1 = land, 0 = ocean), built via load_jcm_mask from the `lsm`
    variable and thresholded at fmask > 0.95. `TerrainData.from_file` uses a
    DIFFERENT interpolation path, so `terrain.fmask` and the ocean grid's fmask
    can disagree by up to ~0.1 at coastlines (observed 56 T30 cells). Sourcing
    the mask from the ocean grid guarantees the masked loss/features weight
    EXACTLY the cells whose SST actually evolves — no pinned-land contamination
    and no spuriously dropped ocean cells.

    Args:
        coupler: JEM Coupler with an "ocn" component (SlabOceanModel).

    Returns:
        1.0 for ocean cells (SST evolves), 0.0 for land cells (pinned to
        288.15 K). On the aquaplanet the ocean grid has no land, so this is
        all ones.

    """
    ocn = coupler.components["ocn"]
    raw = getattr(ocn, "raw_component", ocn)
    bmask = raw.horizontal_grids["T"].bmask
    return 1.0 - bmask


def ocean_fmask_from_coupler(coupler) -> jnp.ndarray:
    """Fractional land mask from the ocean model's own grid.

    Returns the ocean grid's `fmask` (fraction of cell that is land). Threshold
    at > 0.95 reproduces the grid's binary `bmask` exactly, so passing this as
    `terrain_fmask` into the trainer makes the internal ocean_mask_from_fmask
    binarization agree cell-for-cell with the ocean model's SST pinning —
    unlike TerrainData.from_file's independently-interpolated fmask.

    Args:
        coupler: JEM Coupler with an "ocn" component (SlabOceanModel).

    Returns:
        Fractional land mask (ix, il) from the ocean model's grid.

    """
    ocn = coupler.components["ocn"]
    raw = getattr(ocn, "raw_component", ocn)
    return raw.horizontal_grids["T"].fmask


def create_coupled_train_step(
    coupler,
    workflow: list,
    policy_fn: Callable,
    baseline_trajectory: CoupledBaselineTrajectory,
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
        baseline_trajectory: Paired no-MCB baseline trajectory.
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
                baseline_trajectory=baseline_trajectory,
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


def create_coupled_grad_fn(
    coupler,
    workflow: list,
    policy_fn: Callable,
    coords,
    ocean_mask: jnp.ndarray,
    controller_config: CoupledControllerConfig,
) -> Callable:
    """Create a jitted (params, carry, baseline) -> (loss, grads) function.

    Unlike create_coupled_train_step, the baseline trajectory is a traced
    argument rather than a jit-closure constant, so a SINGLE compiled
    function serves every initial condition in an ensemble (same shapes ->
    one compile).

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow.
        policy_fn: Policy apply function.
        coords: Model coordinates.
        ocean_mask: Ocean mask for MCB.
        controller_config: Controller config.

    Returns:
        Jitted function (params, initial_carry, baseline_trajectory) ->
        (loss, grads).

    """
    @jax.jit
    def _grad_fn(
        params: dict,
        initial_carry: dict,
        baseline_trajectory: CoupledBaselineTrajectory,
        efficacy: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, dict]:
        def loss_fn(p):
            return unroll_coupled_simple(
                coupler=coupler,
                workflow=workflow,
                policy_fn=policy_fn,
                policy_params=p,
                initial_carry=initial_carry,
                baseline_trajectory=baseline_trajectory,
                coords=coords,
                ocean_mask=ocean_mask,
                config=controller_config,
                efficacy=efficacy,
            )

        return jax.value_and_grad(loss_fn)(params)

    def grad_fn(params, initial_carry, baseline_trajectory, efficacy=1.0):
        # efficacy is a TRACED scalar (Tier-2 domain randomization): one
        # compiled function serves every per-episode draw.
        return _grad_fn(params, initial_carry, baseline_trajectory,
                        jnp.asarray(efficacy, dtype=jnp.float32))

    return grad_fn


def create_coupled_eval_fn(
    coupler,
    workflow: list,
    policy_fn: Callable,
    coords,
    ocean_mask: jnp.ndarray,
    controller_config: CoupledControllerConfig,
) -> Callable:
    """Create a jitted forward-only (params, carry, baseline) -> loss function.

    Used for held-out IC evaluation during ensemble training (no gradients,
    so much cheaper than the grad function).

    Args:
        Same as create_coupled_grad_fn.

    Returns:
        Jitted function (params, initial_carry, baseline_trajectory) -> loss.

    """
    @jax.jit
    def _eval_fn(
        params: dict,
        initial_carry: dict,
        baseline_trajectory: CoupledBaselineTrajectory,
        efficacy: jnp.ndarray,
    ) -> jnp.ndarray:
        return unroll_coupled_simple(
            coupler=coupler,
            workflow=workflow,
            policy_fn=policy_fn,
            policy_params=params,
            initial_carry=initial_carry,
            baseline_trajectory=baseline_trajectory,
            coords=coords,
            ocean_mask=ocean_mask,
            config=controller_config,
            efficacy=efficacy,
        )

    def eval_fn(params, initial_carry, baseline_trajectory, efficacy=1.0):
        return _eval_fn(params, initial_carry, baseline_trajectory,
                        jnp.asarray(efficacy, dtype=jnp.float32))

    return eval_fn


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
    baseline_trajectory: CoupledBaselineTrajectory,
    training_config: TrainingConfig = TrainingConfig(),
    controller_config: CoupledControllerConfig = CoupledControllerConfig(),
    callback: Optional[Callable[[TrainingState], None]] = None,
    initial_params: Optional[dict] = None,
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
        baseline_trajectory: Paired no-MCB baseline trajectory for anomalies.
        training_config: Training hyperparameters.
        controller_config: Controller configuration.
        callback: Optional epoch callback.
        initial_params: Optional policy parameters to start from (e.g. a
            warm-start from the Stage 1 optimized pattern). If None, the
            policy is initialized from training_config.random_seed.

    Returns:
        Tuple of:
        - best_params: Best policy parameters (lowest loss)
        - history: Dictionary with training history and metrics

    """
    ocean_mask = ocean_mask_from_fmask(terrain_fmask)

    # Initialize
    params, opt_state, optimizer = initialize_coupled_training(
        policy=policy,
        coords=coords,
        config=training_config,
        controller_config=controller_config,
    )
    if initial_params is not None:
        params = initial_params
        opt_state = optimizer.init(params)

    # Create training step
    train_step = create_coupled_train_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        baseline_trajectory=baseline_trajectory,
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

        # Training step. `loss` is measured at state.params (pre-update);
        # `new_params` are AFTER the optimizer step.
        new_params, new_opt_state, loss, grad_norm = train_step(
            state.params, state.opt_state, initial_carry
        )

        loss_val = float(loss)
        grad_norm_val = float(grad_norm)

        # Update best params. Save state.params — the params that produced
        # loss_val — not new_params (whose loss is not measured until next
        # epoch). See the off-by-one note in train_coupled_policy_ensemble.
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
        'mode': 'coupled',
    }

    return state.best_params, history


def train_coupled_policy_ensemble(
    coupler,
    workflow: list,
    policy,  # Flax module
    coords,
    terrain_fmask: jnp.ndarray,
    train_carries: list,
    train_baselines: list,
    heldout_carries: tuple = (),
    heldout_baselines: tuple = (),
    training_config: TrainingConfig = TrainingConfig(),
    controller_config: CoupledControllerConfig = CoupledControllerConfig(),
    callback: Optional[Callable[[TrainingState], None]] = None,
    initial_params: Optional[dict] = None,
    heldout_interval: int = 10,
    select_on_heldout: bool = False,
    train_efficacies: tuple = (),
    heldout_efficacies: tuple = (),
    efficacy_resample: Optional[dict] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """Train MCB policy across an ensemble of varied initial conditions.

    Each epoch computes per-IC losses/gradients with a SINGLE jitted
    (params, carry, baseline) -> (loss, grads) function (baseline is a traced
    argument, so all ICs share one compile), averages the gradients on the
    host, and applies one optimizer update. Gradient clipping (if configured)
    acts on the AVERAGED gradient.

    Model selection and early stopping use the mean HELD-OUT loss when
    ``select_on_heldout`` is True (held-out is then evaluated every epoch on
    the pre-update params, i.e. exactly the params that would be shipped),
    and fall back to the mean TRAIN loss otherwise (the historical behavior).
    Per-IC gradient norms and the gradient-coherence ratio
    C = ||mean_i g_i|| / mean_i ||g_i|| are recorded every epoch: C near 1
    means the ICs agree on a descent direction; C near 0 means large per-IC
    gradients are destructively cancelling in the ensemble mean.

    Args:
        coupler: JEM Coupler instance.
        workflow: Coupling workflow (e.g., ["coupling", "atm", "ocn"]).
        policy: Flax policy module.
        coords: Model coordinates.
        terrain_fmask: Land mask (1.0 = land, 0.0 = ocean).
        train_carries: Initial coupled carries for training ICs.
        train_baselines: Paired no-MCB baseline trajectories, one per
            training IC (same order as train_carries).
        heldout_carries: Initial carries for held-out ICs.
        heldout_baselines: Paired baselines for held-out ICs.
        training_config: Training hyperparameters.
        controller_config: Controller configuration.
        callback: Optional epoch callback.
        initial_params: Optional policy parameters to start from (e.g. an
            expanded Stage 3 checkpoint). If None, initialized from
            training_config.random_seed.
        heldout_interval: Epochs between held-out evaluations when
            ``select_on_heldout`` is False (logging only).
        select_on_heldout: Gate model selection and early stopping on the
            mean held-out loss (evaluated every epoch) instead of train loss.
        train_efficacies: Optional per-episode MCB efficacy factors, one per
            training IC (Tier-2 domain randomization: applied perturbation =
            efficacy x command; the policy never observes efficacy directly).
            Fixed across epochs so each (IC, efficacy) pair is one episode.
            Empty means 1.0 everywhere (Tier-1 behavior).
        heldout_efficacies: Same for held-out (validation) episodes, drawn
            independently of the training draws. Held-out efficacies are
            ALWAYS fixed across epochs so the selection metric stays
            comparable epoch to epoch.
        efficacy_resample: Optional {"range": (lo, hi), "seed": int}. When
            set, TRAINING efficacies are REDRAWN every epoch (epoch-seeded)
            instead of fixed — the 2026-07-31 adversarial review showed that
            with one fixed eta per IC and absolute-SST features that
            fingerprint the IC, memorizing the IC->eta map strictly dominates
            learning the feedback law on the training loss. Per-epoch
            redraws give num_epochs x num_ics draws. Zero compile cost:
            efficacy is a traced scalar.

    Returns:
        Tuple of:
        - best_params: Best policy parameters (lowest mean training loss)
        - history: Dictionary with per-IC and held-out training history

    """
    assert len(train_carries) == len(train_baselines)
    assert len(heldout_carries) == len(heldout_baselines)
    if select_on_heldout and not heldout_carries:
        raise ValueError(
            "select_on_heldout=True requires held-out ICs to gate on."
        )
    num_train = len(train_carries)

    ocean_mask = ocean_mask_from_fmask(terrain_fmask)

    # Initialize
    params, opt_state, optimizer = initialize_coupled_training(
        policy=policy,
        coords=coords,
        config=training_config,
        controller_config=controller_config,
    )
    if initial_params is not None:
        params = initial_params
        opt_state = optimizer.init(params)

    grad_fn = create_coupled_grad_fn(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        coords=coords,
        ocean_mask=ocean_mask,
        controller_config=controller_config,
    )
    eval_fn = create_coupled_eval_fn(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        coords=coords,
        ocean_mask=ocean_mask,
        controller_config=controller_config,
    )

    @jax.jit
    def apply_update(params, opt_state, grads):
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    best_params = params
    best_loss = float('inf')
    patience_counter = 0

    loss_history = []            # mean train loss per epoch
    per_ic_loss_history = []     # list of per-IC loss lists per epoch
    grad_norm_history = []       # norm of the AVERAGED gradient per epoch
    per_ic_grad_norm_history = []  # list of per-IC ||g_i|| per epoch
    coherence_history = []       # ||mean g|| / mean ||g_i|| per epoch
    heldout_loss_history = []    # (epoch, [per-IC held-out losses])

    if not train_efficacies:
        train_efficacies = tuple(1.0 for _ in train_carries)
    if not heldout_efficacies:
        heldout_efficacies = tuple(1.0 for _ in heldout_carries)
    assert len(train_efficacies) == len(train_carries), (
        f"{len(train_efficacies)} efficacies for {len(train_carries)} train ICs")
    assert len(heldout_efficacies) == len(heldout_carries), (
        f"{len(heldout_efficacies)} efficacies for "
        f"{len(heldout_carries)} held-out ICs")

    train_efficacy_history = []

    def epoch_train_efficacies(epoch):
        if efficacy_resample is None:
            return train_efficacies
        import numpy as _np
        lo, hi = efficacy_resample["range"]
        rng = _np.random.default_rng(efficacy_resample["seed"] + epoch)
        return tuple(float(x) for x in rng.uniform(lo, hi, num_train))

    print("Starting coupled MCB policy ensemble training")
    print(f"  Training ICs: {num_train}, held-out ICs: {len(heldout_carries)}")
    if efficacy_resample is not None:
        print(f"  Efficacy randomization ON (redrawn PER EPOCH from "
              f"U{tuple(efficacy_resample['range'])}, seed "
              f"{efficacy_resample['seed']}); held-out fixed: "
              f"{list(heldout_efficacies)}")
    elif any(e != 1.0 for e in train_efficacies + heldout_efficacies):
        print(f"  Efficacy randomization ON: train {list(train_efficacies)}, "
              f"held-out {list(heldout_efficacies)}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Control interval: {controller_config.control_interval_steps} steps")
    print(f"  Total steps: {controller_config.total_steps}")
    print(f"  Target cooling: {controller_config.target_cooling} K")
    print()

    start_time = time.time()

    for epoch in range(training_config.num_epochs):
        epoch_start = time.time()

        # Per-IC gradients, averaged on the host (memory stays at the
        # single-IC level; dispatch overhead is negligible vs the unrolls)
        ic_losses = []
        grad_sum = None
        per_ic_grad_norms = []
        epoch_efficacies = epoch_train_efficacies(epoch)
        train_efficacy_history.append(list(epoch_efficacies))
        for carry, baseline, eff in zip(train_carries, train_baselines,
                                        epoch_efficacies):
            loss, grads = grad_fn(params, carry, baseline, eff)
            ic_losses.append(float(loss))
            g_flat = jnp.concatenate(
                [g.ravel() for g in jax.tree.leaves(grads)]
            )
            per_ic_grad_norms.append(float(jnp.linalg.norm(g_flat)))
            if grad_sum is None:
                grad_sum = grads
            else:
                grad_sum = jax.tree_util.tree_map(jnp.add, grad_sum, grads)

        avg_grads = jax.tree_util.tree_map(lambda g: g / num_train, grad_sum)

        grad_leaves = jax.tree.leaves(avg_grads)
        grad_flat = jnp.concatenate([g.ravel() for g in grad_leaves])
        grad_norm_val = float(jnp.linalg.norm(grad_flat))

        # Gradient coherence: ||mean g|| / mean ||g_i||. Near 1 => the ICs
        # agree; near 0 => large per-IC gradients cancel in the mean (the
        # naive ensemble average cannot learn a per-IC-conflicting signal).
        mean_ic_norm = sum(per_ic_grad_norms) / num_train
        coherence_val = grad_norm_val / (mean_ic_norm + 1e-12)

        # Capture the params that PRODUCED this epoch's measured mean_loss
        # before the optimizer step overwrites them. Selecting best_params on
        # the post-update params (as the original code did) ships a checkpoint
        # whose loss was never measured — a silent off-by-one that, in this
        # noisy loss landscape, saves a one-Adam-step perturbation of the
        # actual best-measured params.
        measured_params = params
        params, opt_state = apply_update(params, opt_state, avg_grads)

        mean_loss = sum(ic_losses) / num_train

        # Held-out evaluation. When gating on held-out we evaluate EVERY epoch
        # on the pre-update params (measured_params) — the exact params that
        # would be shipped as best — so selection and the shipped checkpoint
        # agree. Otherwise it is a periodic forward-only log on the post-update
        # params (historical behavior).
        heldout_mean = None
        heldout_msg = ""
        do_heldout = bool(heldout_carries) and (
            select_on_heldout or epoch % heldout_interval == 0
        )
        if do_heldout:
            eval_params = measured_params if select_on_heldout else params
            heldout_losses = [
                float(eval_fn(eval_params, carry, baseline, eff))
                for carry, baseline, eff in zip(
                    heldout_carries, heldout_baselines, heldout_efficacies)
            ]
            heldout_loss_history.append((epoch, heldout_losses))
            heldout_mean = sum(heldout_losses) / len(heldout_losses)
            heldout_msg = f" | Held-out: {heldout_mean:.6f}"

        # Best-params / early-stopping on the selection metric (held-out when
        # requested, else train mean loss).
        selection_metric = heldout_mean if select_on_heldout else mean_loss
        if selection_metric < best_loss:
            best_params = measured_params
            best_loss = selection_metric
            patience_counter = 0
        else:
            patience_counter += 1

        loss_history.append(mean_loss)
        per_ic_loss_history.append(ic_losses)
        grad_norm_history.append(grad_norm_val)
        per_ic_grad_norm_history.append(per_ic_grad_norms)
        coherence_history.append(coherence_val)

        # Logging
        if epoch % training_config.log_interval == 0:
            epoch_time = time.time() - epoch_start
            per_ic_str = "/".join(f"{v:.4f}" for v in ic_losses)
            print(f"Epoch {epoch:4d} | Mean loss: {mean_loss:.6f} "
                  f"[{per_ic_str}] | Grad norm: {grad_norm_val:.4f} "
                  f"| Coh: {coherence_val:.3f} | "
                  f"Best: {best_loss:.6f}{heldout_msg} | "
                  f"Time: {epoch_time:.1f}s")

        # Callback
        if callback is not None:
            callback(TrainingState(
                epoch=epoch + 1,
                params=params,
                opt_state=opt_state,
                best_params=best_params,
                best_loss=best_loss,
                loss_history=loss_history,
                grad_norm_history=grad_norm_history,
            ))

        # Early stopping on mean train loss
        if training_config.early_stopping_patience is not None:
            if patience_counter >= training_config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience_counter} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Final mean loss: {loss_history[-1]:.6f}")
    print(f"Best mean loss: {best_loss:.6f}")

    history = {
        'loss_history': loss_history,
        'per_ic_loss_history': per_ic_loss_history,
        'grad_norm_history': grad_norm_history,
        'per_ic_grad_norm_history': per_ic_grad_norm_history,
        'coherence_history': coherence_history,
        'heldout_loss_history': heldout_loss_history,
        'best_loss': best_loss,
        'selection_metric': 'heldout' if select_on_heldout else 'train',
        'final_loss': loss_history[-1],
        'total_time': total_time,
        'epochs_completed': len(loss_history),
        'num_train_ics': num_train,
        'num_heldout_ics': len(heldout_carries),
        'train_efficacies': list(train_efficacies),
        'heldout_efficacies': list(heldout_efficacies),
        'efficacy_resample': efficacy_resample,
        'train_efficacy_history': train_efficacy_history,
        'mode': 'coupled-ensemble',
    }

    return best_params, history


def validate_coupled_training_setup(
    coupler,
    workflow: list,
    policy,
    coords,
    terrain_fmask: jnp.ndarray,
    initial_carry: dict,
    baseline_trajectory: CoupledBaselineTrajectory,
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

    ocean_mask = ocean_mask_from_fmask(terrain_fmask)

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
            baseline_trajectory=baseline_trajectory,
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
    baseline_trajectory: CoupledBaselineTrajectory,
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

    ocean_mask = ocean_mask_from_fmask(terrain_fmask)

    # Create optimizer (will need to reinitialize state)
    optimizer = create_optimizer(training_config, training_config.num_epochs)
    opt_state = optimizer.init(params)

    # Create training step
    train_step = create_coupled_train_step(
        coupler=coupler,
        workflow=workflow,
        policy_fn=policy.apply,
        baseline_trajectory=baseline_trajectory,
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
