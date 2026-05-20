"""Marine Cloud Brightening (MCB) forcing module.

This module provides tools for simulating and optimizing MCB geoengineering
strategies using JAX-GCM's differentiable framework.

Main components:
- MCBConfig: Configuration struct for MCB forcing parameters
- Region utilities: Create masks for stratocumulus deployment zones
- Forcing functions: Compute MCB albedo perturbations

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

__all__ = [
    'MCBConfig',
    'STRATOCUMULUS_REGIONS',
    'TELECONNECTION_REGIONS',
    'create_region_mask',
    'create_ocean_mask',
    'create_stratocumulus_mask',
    'create_teleconnection_mask',
    'compute_mcb_sea_albedo',
    'compute_mcb_radiative_forcing',
]
