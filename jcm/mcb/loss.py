"""Differentiable climate loss functions for MCB optimization.

This module provides loss functions for training MCB policy networks.
The loss combines multiple objectives:
- Temperature control: Achieve target global cooling
- Teleconnection constraints: Minimize adverse regional impacts
- Regularization: Encourage smooth, efficient MCB forcing

All functions are fully differentiable for use with BPTT.

Example usage:
    from jcm.mcb.loss import compute_climate_loss, LossWeights

    weights = LossWeights(temperature=1.0, amazon=0.5, regularization=0.01)
    loss = compute_climate_loss(predictions, baseline, target_cooling=-0.5,
                                mcb_forcing=forcing, coords=coords, weights=weights)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
import tree_math

from jcm.mcb.mcb_regions import create_region_mask, TELECONNECTION_REGIONS
from jcm.mcb.state_features import compute_area_weights, compute_regional_mean, create_latitude_band_mask


class LossWeights(NamedTuple):
    """Weights for different loss components.

    Attributes:
        temperature: Weight for global temperature loss.
        amazon: Weight for Amazon precipitation protection.
        sahel: Weight for Sahel precipitation protection.
        tropics: Weight for tropical precipitation stability.
        regularization: Weight for MCB forcing regularization.
        smoothness: Weight for spatial smoothness penalty.

    """

    temperature: float = 1.0
    amazon: float = 1.0
    sahel: float = 0.5
    tropics: float = 0.3
    regularization: float = 0.01
    smoothness: float = 0.01


@tree_math.struct
class LossComponents:
    """Individual loss components for logging/analysis.

    Attributes:
        total: Total weighted loss.
        temperature: Temperature deviation loss.
        amazon: Amazon precipitation loss.
        sahel: Sahel precipitation loss.
        tropics: Tropical precipitation loss.
        regularization: MCB forcing magnitude penalty.
        smoothness: MCB forcing smoothness penalty.

    """

    total: jnp.ndarray
    temperature: jnp.ndarray
    amazon: jnp.ndarray
    sahel: jnp.ndarray
    tropics: jnp.ndarray
    regularization: jnp.ndarray
    smoothness: jnp.ndarray


def temperature_loss(
    predictions,
    baseline_temp: jnp.ndarray,
    target_cooling: float,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute temperature deviation loss.

    Penalizes deviation from target global cooling.

    Args:
        predictions: Model Predictions object.
        baseline_temp: Baseline surface temperature field.
        target_cooling: Target temperature change (negative for cooling).
        area_weights: Area weights for global averaging.

    Returns:
        Scalar temperature loss.

    """
    # Current global mean surface temperature
    surf_temp = predictions.physics.temperature_tendency.surface_temp
    global_temp = jnp.sum(surf_temp * area_weights)

    # Baseline global mean
    baseline_global = jnp.sum(baseline_temp * area_weights)

    # Actual cooling achieved
    actual_cooling = global_temp - baseline_global

    # Loss: squared deviation from target
    loss = (actual_cooling - target_cooling) ** 2

    return loss


def amazon_precipitation_loss(
    predictions,
    baseline_precip: jnp.ndarray,
    coords,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Amazon precipitation protection loss.

    Penalizes precipitation DECREASES in the Amazon basin.
    No penalty for increases (rain is good for the Amazon).

    Args:
        predictions: Model Predictions object.
        baseline_precip: Baseline precipitation field.
        coords: Model coordinates.
        area_weights: Area weights.

    Returns:
        Scalar Amazon precipitation loss.

    """
    # Get Amazon mask
    amazon_bounds = TELECONNECTION_REGIONS['amazon']
    amazon_mask = create_region_mask(
        coords.horizontal,
        amazon_bounds[:2],
        amazon_bounds[2:]
    )

    # Current precipitation
    precip = predictions.physics.convection.precnv + predictions.physics.condensation.precls

    # Regional means
    amazon_precip = compute_regional_mean(precip, amazon_mask, area_weights)
    baseline_amazon = compute_regional_mean(baseline_precip, amazon_mask, area_weights)

    # Precipitation change (negative = decrease)
    precip_change = amazon_precip - baseline_amazon

    # Only penalize decreases (asymmetric loss)
    # Using softplus for smooth differentiability
    loss = jax.nn.softplus(-precip_change * 100) / 100  # Scale for numerical stability

    return loss


def sahel_precipitation_loss(
    predictions,
    baseline_precip: jnp.ndarray,
    coords,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Sahel precipitation protection loss.

    Similar to Amazon loss - penalizes precipitation decreases in the Sahel.

    Args:
        predictions: Model Predictions object.
        baseline_precip: Baseline precipitation field.
        coords: Model coordinates.
        area_weights: Area weights.

    Returns:
        Scalar Sahel precipitation loss.

    """
    # Get Sahel mask
    sahel_bounds = TELECONNECTION_REGIONS['sahel']
    sahel_mask = create_region_mask(
        coords.horizontal,
        sahel_bounds[:2],
        sahel_bounds[2:]
    )

    # Current precipitation
    precip = predictions.physics.convection.precnv + predictions.physics.condensation.precls

    # Regional means
    sahel_precip = compute_regional_mean(precip, sahel_mask, area_weights)
    baseline_sahel = compute_regional_mean(baseline_precip, sahel_mask, area_weights)

    # Precipitation change
    precip_change = sahel_precip - baseline_sahel

    # Penalize decreases
    loss = jax.nn.softplus(-precip_change * 100) / 100

    return loss


def tropical_precipitation_loss(
    predictions,
    baseline_precip: jnp.ndarray,
    coords,
    area_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute tropical precipitation stability loss.

    Penalizes large changes (either direction) in tropical precipitation.
    Aims to maintain tropical rainfall patterns.

    Args:
        predictions: Model Predictions object.
        baseline_precip: Baseline precipitation field.
        coords: Model coordinates.
        area_weights: Area weights.

    Returns:
        Scalar tropical precipitation loss.

    """
    # Tropical mask (30S - 30N)
    tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)

    # Current precipitation
    precip = predictions.physics.convection.precnv + predictions.physics.condensation.precls

    # Regional means
    tropical_precip = compute_regional_mean(precip, tropical_mask, area_weights)
    baseline_tropical = compute_regional_mean(baseline_precip, tropical_mask, area_weights)

    # Squared change (penalize both increases and decreases)
    precip_change = tropical_precip - baseline_tropical
    loss = precip_change ** 2

    return loss


def regularization_loss(mcb_forcing: jnp.ndarray) -> jnp.ndarray:
    """Compute MCB forcing magnitude penalty.

    Encourages minimal MCB intervention - use only as much forcing as needed.

    Args:
        mcb_forcing: MCB albedo perturbation field (ix, il).

    Returns:
        Scalar regularization loss.

    """
    # L2 norm of forcing
    loss = jnp.mean(mcb_forcing ** 2)
    return loss


def smoothness_loss(mcb_forcing: jnp.ndarray) -> jnp.ndarray:
    """Compute MCB forcing smoothness penalty.

    Encourages spatially smooth MCB patterns, avoiding sharp gradients
    that may be physically unrealistic.

    Args:
        mcb_forcing: MCB albedo perturbation field (ix, il).

    Returns:
        Scalar smoothness loss.

    """
    # Finite differences in both directions
    # Note: Using roll for periodic boundary in longitude
    dx = mcb_forcing - jnp.roll(mcb_forcing, 1, axis=0)
    dy = mcb_forcing - jnp.roll(mcb_forcing, 1, axis=1)

    # Total variation
    loss = jnp.mean(jnp.abs(dx)) + jnp.mean(jnp.abs(dy))

    return loss


def compute_climate_loss(
    predictions,
    baseline_temp: jnp.ndarray,
    baseline_precip: jnp.ndarray,
    target_cooling: float,
    mcb_forcing: jnp.ndarray,
    coords,
    weights: LossWeights = LossWeights(),
    return_components: bool = False,
) -> jnp.ndarray:
    """Compute total climate loss for MCB optimization.

    Main loss function combining all objectives for training MCB policies.

    Args:
        predictions: Model Predictions object from simulation.
        baseline_temp: Baseline surface temperature (ix, il).
        baseline_precip: Baseline precipitation (ix, il).
        target_cooling: Target global temperature change (K, negative for cooling).
        mcb_forcing: Applied MCB albedo perturbation (ix, il).
        coords: Model coordinates.
        weights: Loss component weights.
        return_components: If True, return LossComponents instead of scalar.

    Returns:
        Total loss (scalar) or LossComponents if return_components=True.

    """
    # Compute area weights
    area_weights = compute_area_weights(coords)

    # Individual loss components
    L_temp = temperature_loss(predictions, baseline_temp, target_cooling, area_weights)
    L_amazon = amazon_precipitation_loss(predictions, baseline_precip, coords, area_weights)
    L_sahel = sahel_precipitation_loss(predictions, baseline_precip, coords, area_weights)
    L_tropics = tropical_precipitation_loss(predictions, baseline_precip, coords, area_weights)
    L_reg = regularization_loss(mcb_forcing)
    L_smooth = smoothness_loss(mcb_forcing)

    # Weighted sum
    total_loss = (
        weights.temperature * L_temp +
        weights.amazon * L_amazon +
        weights.sahel * L_sahel +
        weights.tropics * L_tropics +
        weights.regularization * L_reg +
        weights.smoothness * L_smooth
    )

    if return_components:
        return LossComponents(
            total=total_loss,
            temperature=L_temp,
            amazon=L_amazon,
            sahel=L_sahel,
            tropics=L_tropics,
            regularization=L_reg,
            smoothness=L_smooth,
        )
    else:
        return total_loss


def compute_climate_loss_from_baseline(
    predictions,
    baseline,
    target_cooling: float,
    mcb_forcing: jnp.ndarray,
    coords,
    weights: LossWeights = LossWeights(),
    return_components: bool = False,
):
    """Compute climate loss using ClimateBaseline object.

    Args:
        predictions: Model Predictions object.
        baseline: ClimateBaseline object.
        target_cooling: Target temperature change (K).
        mcb_forcing: MCB forcing field.
        coords: Model coordinates.
        weights: Loss weights.
        return_components: If True, return components.

    Returns:
        Total loss or LossComponents.

    """
    return compute_climate_loss(
        predictions=predictions,
        baseline_temp=baseline.surface_temperature,
        baseline_precip=baseline.precipitation,
        target_cooling=target_cooling,
        mcb_forcing=mcb_forcing,
        coords=coords,
        weights=weights,
        return_components=return_components,
    )


def create_loss_fn(
    baseline_temp: jnp.ndarray,
    baseline_precip: jnp.ndarray,
    target_cooling: float,
    coords,
    weights: LossWeights = LossWeights(),
):
    """Create a curried loss function for use in training loops.

    Returns a function that only takes (predictions, mcb_forcing) as arguments,
    with other parameters pre-bound.

    Args:
        baseline_temp: Baseline surface temperature.
        baseline_precip: Baseline precipitation.
        target_cooling: Target temperature change.
        coords: Model coordinates.
        weights: Loss weights.

    Returns:
        Function (predictions, mcb_forcing) -> loss.

    """
    def loss_fn(predictions, mcb_forcing):
        return compute_climate_loss(
            predictions=predictions,
            baseline_temp=baseline_temp,
            baseline_precip=baseline_precip,
            target_cooling=target_cooling,
            mcb_forcing=mcb_forcing,
            coords=coords,
            weights=weights,
        )
    return loss_fn
