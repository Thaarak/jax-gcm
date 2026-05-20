"""Core MCB forcing application functions.

These functions modify the surface albedo to simulate MCB effects.
All functions are pure and JAX-transformable (jit, grad, vmap).
"""

import jax.numpy as jnp
from jax import jit

from jcm.mcb.mcb_config import MCBConfig


@jit
def compute_mcb_sea_albedo(
    mcb_config: MCBConfig,
    base_sea_albedo: jnp.ndarray,
    day_of_year: int,
) -> jnp.ndarray:
    """Compute the effective sea albedo with MCB forcing applied.

    This function computes the albedo enhancement due to MCB, applying:
    1. The spatial perturbation field
    2. The active region mask
    3. Temporal modulation based on day of year

    The result is clipped to physically reasonable values [base_albedo, 1.0].

    Args:
        mcb_config: MCB configuration with perturbation field and masks.
        base_sea_albedo: Background sea albedo (scalar or spatial field).
            Typically from ModRadConParameters.albsea (~0.07).
        day_of_year: Day of year for temporal modulation (0-364).

    Returns:
        Effective sea albedo field with shape matching mcb_config.albedo_perturbation.
        Values are clipped to [base_sea_albedo, 1.0].

    """
    # Get temporal weight for this day
    # Use modulo to handle day_of_year >= 365
    day_idx = jnp.mod(day_of_year, 365)
    temporal_weight = mcb_config.temporal_weights[day_idx]

    # Compute effective perturbation: spatial field * mask * temporal weight
    effective_perturbation = (
        mcb_config.albedo_perturbation *
        mcb_config.active_mask *
        temporal_weight
    )

    # Add perturbation to base albedo
    mcb_albedo = base_sea_albedo + effective_perturbation

    # Clamp to physically reasonable range [base_albedo, 1.0]
    mcb_albedo = jnp.clip(mcb_albedo, base_sea_albedo, 1.0)

    return mcb_albedo


@jit
def compute_mcb_radiative_forcing(
    mcb_config: MCBConfig,
    base_sea_albedo: jnp.ndarray,
    incoming_shortwave: jnp.ndarray,
    day_of_year: int,
) -> jnp.ndarray:
    """Compute the instantaneous radiative forcing from MCB.

    This is a diagnostic function to estimate the radiative impact of MCB
    without running the full model. The forcing is computed as the change
    in reflected shortwave radiation due to the albedo increase.

    Args:
        mcb_config: MCB configuration.
        base_sea_albedo: Background sea albedo.
        incoming_shortwave: Incoming shortwave radiation at surface (W/m^2).
        day_of_year: Day of year for temporal modulation.

    Returns:
        Radiative forcing in W/m^2 (negative = cooling). Shape matches
        mcb_config.albedo_perturbation.

    """
    mcb_albedo = compute_mcb_sea_albedo(mcb_config, base_sea_albedo, day_of_year)
    albedo_change = mcb_albedo - base_sea_albedo

    # Additional reflected radiation = albedo_change * incoming
    # Forcing is negative (cooling) when albedo increases
    forcing = -albedo_change * incoming_shortwave

    return forcing
