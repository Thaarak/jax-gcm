"""Climate state feature extraction for policy networks.

This module provides functions to extract relevant climate features from
JCM model predictions for use as input to MCB policy networks.

Features extracted include:
- Global and regional temperature anomalies
- Precipitation patterns and anomalies
- TOA radiation imbalance
- Surface conditions

Example usage:
    from jcm.mcb.state_features import extract_state_features, StateFeatureConfig

    config = StateFeatureConfig(include_precipitation=True)
    features = extract_state_features(predictions, baseline, coords, config)
"""

import jax.numpy as jnp
from typing import NamedTuple, Optional, Tuple
import tree_math

from jcm.mcb.mcb_regions import (
    create_region_mask,
    TELECONNECTION_REGIONS,
)


class StateFeatureConfig(NamedTuple):
    """Configuration for state feature extraction.

    Attributes:
        include_temperature: Include temperature-based features.
        include_precipitation: Include precipitation-based features.
        include_radiation: Include TOA radiation features.
        include_surface: Include surface flux features.
        include_spatial: Include full spatial fields (for CNN policies).
        temperature_levels: Vertical levels to include for temperature.

    """

    include_temperature: bool = True
    include_precipitation: bool = True
    include_radiation: bool = True
    include_surface: bool = True
    include_spatial: bool = False
    temperature_levels: Tuple[int, ...] = (0, 4, 7)  # Surface, mid, top


@tree_math.struct
class ClimateBaseline:
    """Baseline climate state for computing anomalies.

    Attributes:
        temperature: Mean temperature field (ix, il, sigma) or scalar.
        surface_temperature: Mean surface temperature (ix, il) or scalar.
        precipitation: Mean total precipitation (ix, il) or scalar.
        toa_sw_down: Mean TOA downward shortwave (ix, il) or scalar.
        toa_sw_up: Mean TOA upward shortwave (ix, il) or scalar.
        toa_lw_up: Mean TOA upward longwave (ix, il) or scalar.

    """

    temperature: jnp.ndarray
    surface_temperature: jnp.ndarray
    precipitation: jnp.ndarray
    toa_sw_down: Optional[jnp.ndarray] = None
    toa_sw_up: Optional[jnp.ndarray] = None
    toa_lw_up: Optional[jnp.ndarray] = None

    @classmethod
    def from_predictions(cls, predictions, num_samples: int = 1):
        """Create baseline from model predictions.

        Args:
            predictions: Model Predictions object or sequence thereof.
            num_samples: Number of prediction samples to average.

        Returns:
            ClimateBaseline with mean fields.

        """
        # Handle single prediction or sequence
        if hasattr(predictions, '__len__') and num_samples > 1:
            # Average over multiple predictions
            temp = jnp.mean(jnp.stack([p.dynamics.temperature for p in predictions]), axis=0)
            surf_temp = jnp.mean(jnp.stack([p.physics.temperature_tendency.surface_temp for p in predictions]), axis=0)

            # Precipitation = convective + large-scale
            precip = jnp.mean(jnp.stack([
                p.physics.convection.precnv + p.physics.condensation.precls
                for p in predictions
            ]), axis=0)
        else:
            # Single prediction
            p = predictions[0] if hasattr(predictions, '__len__') else predictions
            temp = p.dynamics.temperature
            surf_temp = p.physics.temperature_tendency.surface_temp

            precip = p.physics.convection.precnv + p.physics.condensation.precls

        return cls(
            temperature=temp,
            surface_temperature=surf_temp,
            precipitation=precip,
        )

    @classmethod
    def global_mean(cls, nodal_shape: Tuple[int, int], target_temp: float = 288.0):
        """Create baseline with global mean values.

        Args:
            nodal_shape: Grid shape (ix, il).
            target_temp: Target global mean temperature in K.

        Returns:
            ClimateBaseline with uniform mean values.

        """
        return cls(
            temperature=jnp.full(nodal_shape + (8,), target_temp),
            surface_temperature=jnp.full(nodal_shape, target_temp),
            precipitation=jnp.zeros(nodal_shape),
        )


def compute_area_weights(coords) -> jnp.ndarray:
    """Compute area weights for spatial averaging.

    Args:
        coords: Model coordinates with horizontal grid info.

    Returns:
        Area weights array (ix, il) normalized to sum to 1.

    """
    # Get latitude values
    lats = coords.horizontal.latitudes  # (il,)

    # Area weight proportional to cos(latitude)
    weights = jnp.cos(jnp.radians(lats))

    # Broadcast to full grid
    weights_2d = jnp.broadcast_to(weights[None, :], coords.horizontal.nodal_shape)

    # Normalize
    weights_2d = weights_2d / jnp.sum(weights_2d)

    return weights_2d


def compute_regional_mean(field: jnp.ndarray, mask: jnp.ndarray,
                          weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute area-weighted mean over a masked region.

    Args:
        field: 2D field (ix, il) to average.
        mask: Binary mask (ix, il) defining the region.
        weights: Optional area weights (ix, il).

    Returns:
        Scalar regional mean.

    """
    if weights is not None:
        weighted_field = field * mask * weights
        weighted_mask = mask * weights
        return jnp.sum(weighted_field) / jnp.maximum(jnp.sum(weighted_mask), 1e-10)
    else:
        return jnp.sum(field * mask) / jnp.maximum(jnp.sum(mask), 1e-10)


def create_latitude_band_mask(coords, lat_min: float, lat_max: float) -> jnp.ndarray:
    """Create mask for a latitude band.

    Args:
        coords: Model coordinates.
        lat_min: Minimum latitude (degrees).
        lat_max: Maximum latitude (degrees).

    Returns:
        Binary mask (ix, il) for the latitude band.

    """
    lats = coords.horizontal.latitudes  # (il,)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    return jnp.broadcast_to(lat_mask[None, :], coords.horizontal.nodal_shape)


def extract_scalar_features(
    predictions,
    baseline: ClimateBaseline,
    coords,
    area_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Extract scalar (global/regional mean) features.

    Args:
        predictions: Model Predictions object.
        baseline: Climate baseline for anomalies.
        coords: Model coordinates.
        area_weights: Optional pre-computed area weights.

    Returns:
        1D array of scalar features.

    """
    if area_weights is None:
        area_weights = compute_area_weights(coords)

    features = []

    # --- Temperature features ---

    # Surface temperature - squeeze out time dimension if present
    surf_temp = predictions.physics.surface_flux.tsfc
    if surf_temp.ndim == 3:
        surf_temp = surf_temp[0]  # Remove time dimension (1, ix, il) -> (ix, il)

    # Global mean surface temperature anomaly
    global_temp = jnp.sum(surf_temp * area_weights)
    baseline_temp = jnp.sum(baseline.surface_temperature * area_weights)
    global_temp_anomaly = global_temp - baseline_temp
    features.append(global_temp_anomaly)

    # Tropical temperature (30S - 30N)
    tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
    tropical_temp = compute_regional_mean(surf_temp, tropical_mask, area_weights)
    tropical_baseline = compute_regional_mean(baseline.surface_temperature, tropical_mask, area_weights)
    tropical_temp_anomaly = tropical_temp - tropical_baseline
    features.append(tropical_temp_anomaly)

    # Mid-latitude temperature (30-60 both hemispheres)
    midlat_nh_mask = create_latitude_band_mask(coords, 30.0, 60.0)
    midlat_sh_mask = create_latitude_band_mask(coords, -60.0, -30.0)
    midlat_mask = midlat_nh_mask | midlat_sh_mask

    midlat_temp = compute_regional_mean(surf_temp, midlat_mask.astype(jnp.float32), area_weights)
    midlat_baseline = compute_regional_mean(baseline.surface_temperature, midlat_mask.astype(jnp.float32), area_weights)
    midlat_temp_anomaly = midlat_temp - midlat_baseline
    features.append(midlat_temp_anomaly)

    # Polar temperature (60-90 both hemispheres)
    polar_nh_mask = create_latitude_band_mask(coords, 60.0, 90.0)
    polar_sh_mask = create_latitude_band_mask(coords, -90.0, -60.0)
    polar_mask = polar_nh_mask | polar_sh_mask

    polar_temp = compute_regional_mean(surf_temp, polar_mask.astype(jnp.float32), area_weights)
    polar_baseline = compute_regional_mean(baseline.surface_temperature, polar_mask.astype(jnp.float32), area_weights)
    polar_temp_anomaly = polar_temp - polar_baseline
    features.append(polar_temp_anomaly)

    # --- Precipitation features ---

    # Total precipitation - squeeze out time dimension if present
    precnv = predictions.physics.convection.precnv
    precls = predictions.physics.condensation.precls
    if precnv.ndim == 3:
        precnv = precnv[0]
        precls = precls[0]
    precip = precnv + precls

    # Global mean precipitation
    global_precip = jnp.sum(precip * area_weights)
    baseline_precip = jnp.sum(baseline.precipitation * area_weights)
    global_precip_anomaly = global_precip - baseline_precip
    features.append(global_precip_anomaly)

    # Tropical precipitation
    tropical_precip = compute_regional_mean(precip, tropical_mask, area_weights)
    tropical_precip_baseline = compute_regional_mean(baseline.precipitation, tropical_mask, area_weights)
    tropical_precip_anomaly = tropical_precip - tropical_precip_baseline
    features.append(tropical_precip_anomaly)

    # --- Teleconnection region features ---
    grid = coords.horizontal

    for region_name in ['amazon', 'sahel', 'south_asia_monsoon']:
        region_bounds = TELECONNECTION_REGIONS[region_name]
        region_mask = create_region_mask(grid, region_bounds[:2], region_bounds[2:])

        # Temperature
        region_temp = compute_regional_mean(surf_temp, region_mask, area_weights)
        region_temp_baseline = compute_regional_mean(baseline.surface_temperature, region_mask, area_weights)
        features.append(region_temp - region_temp_baseline)

        # Precipitation
        region_precip = compute_regional_mean(precip, region_mask, area_weights)
        region_precip_baseline = compute_regional_mean(baseline.precipitation, region_mask, area_weights)
        features.append(region_precip - region_precip_baseline)

    return jnp.array(features)


def extract_spatial_features(
    predictions,
    baseline: ClimateBaseline,
    coords,
    config: StateFeatureConfig = StateFeatureConfig(),
) -> jnp.ndarray:
    """Extract spatial (gridded) features for CNN policies.

    Args:
        predictions: Model Predictions object.
        baseline: Climate baseline for anomalies.
        coords: Model coordinates.
        config: Feature extraction configuration.

    Returns:
        3D array of spatial features (ix, il, channels).

    """
    channels = []

    if config.include_temperature:
        # Surface temperature anomaly - squeeze out time dimension if present
        surf_temp = predictions.physics.surface_flux.tsfc
        if surf_temp.ndim == 3:
            surf_temp = surf_temp[0]
        surf_temp_anomaly = surf_temp - baseline.surface_temperature
        channels.append(surf_temp_anomaly)

        # Multi-level temperature anomalies
        # predictions.dynamics.temperature has shape (time, level, lon, lat)
        temp_3d = predictions.dynamics.temperature
        if temp_3d.ndim == 4:
            temp_3d = temp_3d[0]  # Remove time dim: (level, lon, lat)
        for level in config.temperature_levels:
            temp_level = temp_3d[level]  # (lon, lat)
            baseline_level = baseline.temperature[:, :, level]
            channels.append(temp_level - baseline_level)

    if config.include_precipitation:
        # Precipitation anomaly - squeeze out time dimension if present
        precnv = predictions.physics.convection.precnv
        precls = predictions.physics.condensation.precls
        if precnv.ndim == 3:
            precnv = precnv[0]
            precls = precls[0]
        precip = precnv + precls
        precip_anomaly = precip - baseline.precipitation
        channels.append(precip_anomaly)

    # Stack channels
    spatial_features = jnp.stack(channels, axis=-1)

    return spatial_features


def extract_state_features(
    predictions,
    baseline: ClimateBaseline,
    coords,
    config: StateFeatureConfig = StateFeatureConfig(),
) -> jnp.ndarray:
    """Extract climate state features for policy input.

    Main entry point for feature extraction. Returns either scalar features
    (for MLP policies) or spatial features (for CNN policies) based on config.

    Args:
        predictions: Model Predictions object.
        baseline: Climate baseline for computing anomalies.
        coords: Model coordinates.
        config: Feature extraction configuration.

    Returns:
        Feature array - 1D for scalar features, 3D for spatial features.

    """
    if config.include_spatial:
        return extract_spatial_features(predictions, baseline, coords, config)
    else:
        return extract_scalar_features(predictions, baseline, coords)


def get_feature_dim(config: StateFeatureConfig = StateFeatureConfig()) -> int:
    """Get the dimension of scalar features for a given config.

    Args:
        config: Feature extraction configuration.

    Returns:
        Number of scalar features.

    """
    # Base features:
    # - Global temp anomaly (1)
    # - Tropical temp anomaly (1)
    # - Midlat temp anomaly (1)
    # - Polar temp anomaly (1)
    # - Global precip anomaly (1)
    # - Tropical precip anomaly (1)
    # - 3 teleconnection regions × 2 (temp + precip) = 6
    # Total: 12
    return 12
