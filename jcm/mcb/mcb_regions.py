"""Predefined region masks for common MCB deployment zones.

MCB research typically targets subtropical stratocumulus regions where
persistent low clouds can be brightened effectively:
- SE Pacific (off Peru/Chile coast)
- SE Atlantic (off Namibia/Angola coast)
- NE Pacific (off California coast)
- Canary current (off NW Africa)
- Benguela current (off SW Africa)

These regions are characterized by:
- Persistent stratocumulus cloud decks
- Cool upwelling waters
- Stable atmospheric conditions
"""

import jax.numpy as jnp
from dinosaur.coordinate_systems import HorizontalGridTypes


# Region definitions: (lat_min, lat_max, lon_min, lon_max) in degrees
# Latitude: -90 to 90, Longitude: -180 to 180
STRATOCUMULUS_REGIONS = {
    'se_pacific': (-30.0, -5.0, -120.0, -70.0),
    'se_atlantic': (-25.0, 0.0, -15.0, 15.0),
    'ne_pacific': (15.0, 35.0, -150.0, -110.0),
    'canary': (15.0, 35.0, -35.0, -15.0),
    'benguela': (-35.0, -15.0, 0.0, 20.0),
}

# Teleconnection regions for monitoring remote impacts
TELECONNECTION_REGIONS = {
    'amazon': (-15.0, 5.0, -75.0, -50.0),
    'sahel': (10.0, 20.0, -15.0, 35.0),
    'south_asia_monsoon': (5.0, 30.0, 70.0, 100.0),
    'north_atlantic': (40.0, 65.0, -60.0, -10.0),
}


def create_region_mask(
    grid: HorizontalGridTypes,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> jnp.ndarray:
    """Create a binary mask for a rectangular lat/lon region.

    Args:
        grid: Horizontal grid from coords.horizontal. Must have `latitudes`,
            `longitudes` (in radians), and `nodal_shape` attributes.
        lat_bounds: (lat_min, lat_max) in degrees.
        lon_bounds: (lon_min, lon_max) in degrees. Handles wraparound
            (e.g., lon_min=170, lon_max=-170 selects Pacific crossing dateline).

    Returns:
        Binary mask array of shape nodal_shape = (ix, il), where ix is
        longitude dimension and il is latitude dimension. Values are
        1.0 inside the region, 0.0 outside.

    """
    # Convert grid coordinates from radians to degrees
    lats_deg = jnp.rad2deg(grid.latitudes)
    lons_deg = jnp.rad2deg(grid.longitudes)

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # Normalize lon_bounds to 0-360 range (grid uses 0-360)
    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    # Create 2D coordinate grids
    # nodal_shape is (ix, il) where ix is longitude, il is latitude
    # meshgrid with indexing='ij' gives (ix, il) output
    lon_grid, lat_grid = jnp.meshgrid(lons_deg, lats_deg, indexing='ij')

    # Latitude mask
    lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)

    # Longitude mask (handle wraparound)
    if lon_min_360 <= lon_max_360:
        lon_mask = (lon_grid >= lon_min_360) & (lon_grid <= lon_max_360)
    else:
        # Wraparound case (e.g., region crosses 0 degrees)
        lon_mask = (lon_grid >= lon_min_360) | (lon_grid <= lon_max_360)

    return (lat_mask & lon_mask).astype(jnp.float32)


def create_ocean_mask(
    grid: HorizontalGridTypes,
    fmask: jnp.ndarray,
) -> jnp.ndarray:
    """Create ocean-only mask from land-sea mask.

    Args:
        grid: Horizontal grid (used for shape validation).
        fmask: Land-sea mask from TerrainData with shape (ix, il).
            Values are 0.0 for ocean, 1.0 for land.

    Returns:
        Ocean mask with shape (ix, il). Values are 1.0 for ocean, 0.0 for land.

    """
    return 1.0 - fmask


def create_stratocumulus_mask(
    grid: HorizontalGridTypes,
    fmask: jnp.ndarray,
    regions: list[str] = None,
) -> jnp.ndarray:
    """Create combined mask for stratocumulus MCB deployment regions.

    This creates a mask that is 1.0 over ocean areas within the specified
    stratocumulus regions, and 0.0 elsewhere.

    Args:
        grid: Horizontal grid from coords.horizontal.
        fmask: Land-sea mask from TerrainData with shape (ix, il).
        regions: List of region names from STRATOCUMULUS_REGIONS.
            Defaults to all defined regions if None.

    Returns:
        Combined ocean + stratocumulus region mask with shape (ix, il).

    Raises:
        ValueError: If an unknown region name is specified.

    """
    if regions is None:
        regions = list(STRATOCUMULUS_REGIONS.keys())

    ocean_mask = create_ocean_mask(grid, fmask)

    combined_region = jnp.zeros(grid.nodal_shape)
    for region_name in regions:
        if region_name not in STRATOCUMULUS_REGIONS:
            raise ValueError(
                f"Unknown region: {region_name}. "
                f"Available: {list(STRATOCUMULUS_REGIONS.keys())}"
            )
        bounds = STRATOCUMULUS_REGIONS[region_name]
        lat_bounds = (bounds[0], bounds[1])
        lon_bounds = (bounds[2], bounds[3])
        region_mask = create_region_mask(grid, lat_bounds, lon_bounds)
        combined_region = jnp.maximum(combined_region, region_mask)

    # Intersect with ocean mask to exclude any land areas
    return ocean_mask * combined_region


def create_teleconnection_mask(
    grid: HorizontalGridTypes,
    region_name: str,
) -> jnp.ndarray:
    """Create mask for a teleconnection monitoring region.

    Args:
        grid: Horizontal grid from coords.horizontal.
        region_name: Name of region from TELECONNECTION_REGIONS.

    Returns:
        Mask with shape (ix, il). Values are 1.0 inside the region, 0.0 outside.

    Raises:
        ValueError: If an unknown region name is specified.

    """
    if region_name not in TELECONNECTION_REGIONS:
        raise ValueError(
            f"Unknown teleconnection region: {region_name}. "
            f"Available: {list(TELECONNECTION_REGIONS.keys())}"
        )

    bounds = TELECONNECTION_REGIONS[region_name]
    lat_bounds = (bounds[0], bounds[1])
    lon_bounds = (bounds[2], bounds[3])
    return create_region_mask(grid, lat_bounds, lon_bounds)
