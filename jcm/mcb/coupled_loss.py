"""Coupled loss functions for MCB optimization.

Loss functions that use ocean SST directly instead of atmospheric
surface temperature, enabling gradient flow through ocean feedback.

The key difference from atmosphere-only loss is that we penalize
deviation from target SST change rather than atmospheric temperature,
which captures the physically correct MCB mechanism:
MCB -> reduced heat flux -> ocean cools -> SST decreases -> global cooling

Example usage:
    from jcm.mcb.coupled_loss import compute_coupled_loss, CoupledLossWeights

    weights = CoupledLossWeights(sst_cooling=1.0, amazon=0.5)
    loss = compute_coupled_loss(coupled_carry, baseline_sst, baseline_precip,
                                target_cooling=-0.5, mcb_forcing=forcing,
                                coords=coords, weights=weights)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
import tree_math

from jcm.mcb.mcb_regions import create_region_mask, TELECONNECTION_REGIONS
from jcm.mcb.state_features import (
    compute_area_weights,
    compute_regional_mean,
    create_latitude_band_mask,
)


class CoupledLossWeights(NamedTuple):
    """Weights for coupled loss components.

    Defaults are the Stage 2 aquaplanet configuration: land teleconnection
    terms (amazon/sahel) and tropics are OFF (weight 0.0 — there is no land
    on an aquaplanet), and the remaining weights match the values that
    succeeded in Stage 1 direct pattern optimization, so `sst_cooling`
    dominates the loss. Components with weight 0.0 are skipped entirely in
    compute_coupled_loss (exactly zero contribution, no constant offsets).

    Reinstate amazon/sahel/tropics when switching to realistic terrain
    (Stage 4).

    Attributes:
        sst_cooling: Weight for SST-based cooling objective.
        sst_uniformity: Weight for uniform SST change (avoid hotspots).
        amazon: Weight for Amazon precipitation protection.
        sahel: Weight for Sahel precipitation protection.
        tropics: Weight for tropical precipitation stability.
        regularization: Weight for MCB forcing magnitude penalty.
        smoothness: Weight for MCB spatial smoothness penalty.

    """

    sst_cooling: float = 1.0
    sst_uniformity: float = 0.1
    amazon: float = 0.0
    sahel: float = 0.0
    tropics: float = 0.0
    regularization: float = 0.001
    smoothness: float = 0.001


@tree_math.struct
class CoupledLossComponents:
    """Individual coupled loss components for logging/analysis.

    Attributes:
        total: Total weighted loss.
        sst_cooling: SST deviation loss.
        sst_uniformity: SST uniformity loss.
        amazon: Amazon precipitation loss.
        sahel: Sahel precipitation loss.
        tropics: Tropical precipitation loss.
        regularization: MCB forcing magnitude penalty.
        smoothness: MCB forcing smoothness penalty.

    """

    total: jnp.ndarray
    sst_cooling: jnp.ndarray
    sst_uniformity: jnp.ndarray
    amazon: jnp.ndarray
    sahel: jnp.ndarray
    tropics: jnp.ndarray
    regularization: jnp.ndarray
    smoothness: jnp.ndarray


def sst_cooling_loss(
    coupled_carry: dict,
    baseline_sst: jnp.ndarray,
    target_cooling: float,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute SST-based cooling loss.

    Uses actual ocean SST instead of atmospheric surface temperature.
    This is the primary objective for coupled MCB optimization.

    Args:
        coupled_carry: Coupled simulation state with "ocn" component.
        baseline_sst: Baseline SST from control run.
        target_cooling: Target SST change (negative for cooling).
        area_weights: Area weights for global averaging.

    Returns:
        Scalar loss value.

    """
    # Current ocean SST
    current_sst = coupled_carry["ocn"]["state"].sea_surface_temperature

    # Global mean SST
    global_sst = jnp.sum(current_sst * area_weights)
    baseline_global = jnp.sum(baseline_sst * area_weights)

    # Actual cooling achieved
    actual_cooling = global_sst - baseline_global

    # Squared deviation from target
    loss = (actual_cooling - target_cooling) ** 2
    return loss


def sst_uniformity_loss(
    coupled_carry: dict,
    baseline_sst: jnp.ndarray,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Penalize non-uniform SST changes.

    Encourages MCB to produce globally uniform cooling rather than
    regional cooling patterns that might cause teleconnections.

    Args:
        coupled_carry: Coupled simulation state.
        baseline_sst: Baseline SST.
        area_weights: Area weights for averaging.

    Returns:
        Scalar loss value (variance of SST change).

    """
    current_sst = coupled_carry["ocn"]["state"].sea_surface_temperature
    sst_change = current_sst - baseline_sst

    # Mean SST change
    mean_change = jnp.sum(sst_change * area_weights)

    # Variance of SST change (penalize non-uniformity)
    variance = jnp.sum((sst_change - mean_change) ** 2 * area_weights)
    return variance


def coupled_precipitation_loss(
    coupled_carry: dict,
    baseline_precip: jnp.ndarray,
    coords,
    area_weights: jnp.ndarray,
    region: str = 'amazon',
) -> jnp.ndarray:
    """Compute precipitation protection loss for a region.

    Penalizes precipitation DECREASES in the specified region.
    Uses softplus for smooth differentiability.

    Args:
        coupled_carry: Coupled simulation state.
        baseline_precip: Baseline precipitation.
        coords: Model coordinates.
        area_weights: Area weights.
        region: Region name from TELECONNECTION_REGIONS.

    Returns:
        Scalar loss (penalizes decreases only).

    """
    # Get precipitation from atmosphere physics
    atm_physics = coupled_carry["atm"]["derived"]["physics"]
    precip = atm_physics.convection.precnv + atm_physics.condensation.precls

    # Get region mask
    bounds = TELECONNECTION_REGIONS[region]
    region_mask = create_region_mask(coords.horizontal, bounds[:2], bounds[2:])

    # Regional means
    region_precip = compute_regional_mean(precip, region_mask, area_weights)
    region_baseline = compute_regional_mean(baseline_precip, region_mask, area_weights)

    # Asymmetric loss: penalize decreases only
    precip_change = region_precip - region_baseline
    loss = jax.nn.softplus(-precip_change * 100) / 100
    return loss


def tropical_precipitation_loss(
    coupled_carry: dict,
    baseline_precip: jnp.ndarray,
    coords,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute tropical precipitation stability loss.

    Penalizes large changes (either direction) in tropical precipitation.

    Args:
        coupled_carry: Coupled simulation state.
        baseline_precip: Baseline precipitation.
        coords: Model coordinates.
        area_weights: Area weights.

    Returns:
        Scalar loss value.

    """
    # Tropical mask (30S - 30N)
    tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)

    # Current precipitation
    atm_physics = coupled_carry["atm"]["derived"]["physics"]
    precip = atm_physics.convection.precnv + atm_physics.condensation.precls

    # Regional means
    tropical_precip = compute_regional_mean(precip, tropical_mask, area_weights)
    baseline_tropical = compute_regional_mean(baseline_precip, tropical_mask, area_weights)

    # Squared change (penalize both increases and decreases)
    precip_change = tropical_precip - baseline_tropical
    loss = precip_change ** 2
    return loss


def regularization_loss(mcb_forcing: jnp.ndarray) -> jnp.ndarray:
    """Compute MCB forcing magnitude penalty.

    Encourages minimal MCB intervention.

    Args:
        mcb_forcing: MCB albedo perturbation field (ix, il).

    Returns:
        Scalar L2 regularization loss.

    """
    return jnp.mean(mcb_forcing ** 2)


def smoothness_loss(mcb_forcing: jnp.ndarray) -> jnp.ndarray:
    """Compute MCB forcing smoothness penalty.

    Encourages spatially smooth MCB patterns using total variation.

    Args:
        mcb_forcing: MCB albedo perturbation field (ix, il).

    Returns:
        Scalar total variation loss.

    """
    # Finite differences with periodic boundary in longitude
    dx = mcb_forcing - jnp.roll(mcb_forcing, 1, axis=0)
    dy = mcb_forcing - jnp.roll(mcb_forcing, 1, axis=1)
    return jnp.mean(jnp.abs(dx)) + jnp.mean(jnp.abs(dy))


def compute_coupled_loss(
    coupled_carry: dict,
    baseline_sst: jnp.ndarray,
    baseline_precip: jnp.ndarray,
    target_cooling: float,
    mcb_forcing: jnp.ndarray,
    coords,
    weights: CoupledLossWeights = CoupledLossWeights(),
    return_components: bool = False,
    ocean_mask: jnp.ndarray | None = None,
):
    """Compute total coupled loss for MCB optimization.

    Main loss function for coupled training. Uses ocean SST for the
    primary cooling objective while protecting regional precipitation.

    For Stage 2+ training, `baseline_sst`/`baseline_precip` should be the
    paired no-MCB snapshots at the SAME timestep (from
    CoupledBaselineTrajectory.at_step), so the differences isolate the
    MCB-caused signal from natural drift.

    Components whose weight is exactly 0.0 (a static Python float) are
    skipped entirely: they contribute exactly zero to the total (avoiding
    constant offsets such as softplus(0) from the precipitation terms) and
    cost nothing to compute.

    Args:
        coupled_carry: Current coupled simulation state.
        baseline_sst: Baseline ocean SST (paired, same timestep).
        baseline_precip: Baseline precipitation (paired, same timestep).
        target_cooling: Target SST change (K, negative for cooling).
        mcb_forcing: Applied MCB forcing field.
        coords: Model coordinates.
        weights: Loss component weights.
        return_components: If True, return CoupledLossComponents.
        ocean_mask: Optional ocean mask (1.0 ocean, 0.0 land) with shape
            (ix, il). When provided (realistic terrain, Stage 5+), the
            SST cooling and uniformity losses are area-weighted over OCEAN
            only, so land cells pinned at a fixed temperature do not corrupt
            the global-mean SST or its variance. Default None reproduces the
            aquaplanet behavior (all cells weighted) bit-for-bit.

    Returns:
        Total loss (scalar) or CoupledLossComponents if return_components=True.

    """
    area_weights = compute_area_weights(coords)
    zero = jnp.array(0.0)

    # For the SST objective/uniformity, weight over ocean only when a mask is
    # given (realistic terrain): land cells are pinned to a fixed temperature
    # and would otherwise corrupt the global-mean SST and its variance.
    if ocean_mask is not None:
        sst_weights = area_weights * ocean_mask
        sst_weights = sst_weights / jnp.sum(sst_weights)
    else:
        sst_weights = area_weights

    # SST-based losses
    L_sst = (
        sst_cooling_loss(coupled_carry, baseline_sst, target_cooling, sst_weights)
        if weights.sst_cooling != 0.0 else zero
    )
    L_uniform = (
        sst_uniformity_loss(coupled_carry, baseline_sst, sst_weights)
        if weights.sst_uniformity != 0.0 else zero
    )

    # Precipitation protection (off by default on aquaplanet)
    L_amazon = (
        coupled_precipitation_loss(
            coupled_carry, baseline_precip, coords, area_weights, 'amazon'
        ) if weights.amazon != 0.0 else zero
    )
    L_sahel = (
        coupled_precipitation_loss(
            coupled_carry, baseline_precip, coords, area_weights, 'sahel'
        ) if weights.sahel != 0.0 else zero
    )
    L_tropics = (
        tropical_precipitation_loss(
            coupled_carry, baseline_precip, coords, area_weights
        ) if weights.tropics != 0.0 else zero
    )

    # Regularization
    L_reg = regularization_loss(mcb_forcing) if weights.regularization != 0.0 else zero
    L_smooth = smoothness_loss(mcb_forcing) if weights.smoothness != 0.0 else zero

    # Total weighted loss
    total = (
        weights.sst_cooling * L_sst +
        weights.sst_uniformity * L_uniform +
        weights.amazon * L_amazon +
        weights.sahel * L_sahel +
        weights.tropics * L_tropics +
        weights.regularization * L_reg +
        weights.smoothness * L_smooth
    )

    if return_components:
        return CoupledLossComponents(
            total=total,
            sst_cooling=L_sst,
            sst_uniformity=L_uniform,
            amazon=L_amazon,
            sahel=L_sahel,
            tropics=L_tropics,
            regularization=L_reg,
            smoothness=L_smooth,
        )
    else:
        return total


def compute_coupled_loss_from_baseline(
    coupled_carry: dict,
    baseline,  # CoupledBaseline
    target_cooling: float,
    mcb_forcing: jnp.ndarray,
    coords,
    weights: CoupledLossWeights = CoupledLossWeights(),
    return_components: bool = False,
):
    """Compute coupled loss using CoupledBaseline object.

    Convenience wrapper that extracts fields from baseline object.

    Args:
        coupled_carry: Current coupled simulation state.
        baseline: CoupledBaseline object.
        target_cooling: Target SST change (K).
        mcb_forcing: MCB forcing field.
        coords: Model coordinates.
        weights: Loss weights.
        return_components: If True, return components.

    Returns:
        Total loss or CoupledLossComponents.

    """
    return compute_coupled_loss(
        coupled_carry=coupled_carry,
        baseline_sst=baseline.sst,
        baseline_precip=baseline.precipitation,
        target_cooling=target_cooling,
        mcb_forcing=mcb_forcing,
        coords=coords,
        weights=weights,
        return_components=return_components,
    )


def create_coupled_loss_fn(
    baseline_sst: jnp.ndarray,
    baseline_precip: jnp.ndarray,
    target_cooling: float,
    coords,
    weights: CoupledLossWeights = CoupledLossWeights(),
):
    """Create a curried coupled loss function.

    Returns a function that only takes (coupled_carry, mcb_forcing)
    as arguments, with other parameters pre-bound.

    Args:
        baseline_sst: Baseline ocean SST.
        baseline_precip: Baseline precipitation.
        target_cooling: Target SST change.
        coords: Model coordinates.
        weights: Loss weights.

    Returns:
        Function (coupled_carry, mcb_forcing) -> loss.

    """
    def loss_fn(coupled_carry: dict, mcb_forcing: jnp.ndarray) -> jnp.ndarray:
        return compute_coupled_loss(
            coupled_carry=coupled_carry,
            baseline_sst=baseline_sst,
            baseline_precip=baseline_precip,
            target_cooling=target_cooling,
            mcb_forcing=mcb_forcing,
            coords=coords,
            weights=weights,
        )
    return loss_fn
