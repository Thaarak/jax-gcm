"""Coupled state feature extraction for policy networks.

Extracts features from both atmosphere and ocean components in
the coupled carry structure for use as policy network inputs.

This module is designed for JAX-ESM coupled simulations where
the coupled_carry dict contains both "atm" and "ocn" components.

Example usage:
    from jcm.mcb.coupled_features import extract_coupled_features, CoupledBaseline

    baseline = CoupledBaseline.from_coupled_carry(initial_carry, coords)
    features = extract_coupled_features(coupled_carry, baseline, coords)
"""

import jax.numpy as jnp
from typing import NamedTuple, Optional
import tree_math

from jcm.mcb.state_features import (
    compute_area_weights,
    compute_regional_mean,
    create_latitude_band_mask,
)
from jcm.mcb.mcb_regions import create_region_mask, TELECONNECTION_REGIONS


# Stratocumulus region definitions for SST monitoring
STRATOCUMULUS_SST_REGIONS = {
    'peru_sc': (-30.0, -5.0, -120.0, -70.0),
    'namibia_sc': (-25.0, 0.0, -15.0, 15.0),
    'california_sc': (15.0, 35.0, -150.0, -110.0),
}


class CoupledFeatureConfig(NamedTuple):
    """Configuration for coupled feature extraction.

    Attributes:
        include_sst: Include ocean SST features.
        include_sst_regions: Include stratocumulus region SST anomalies.
        include_heat_flux: Include air-sea heat flux.
        include_atm_temperature: Include atmospheric temperature.
        include_precipitation: Include precipitation features.

    """

    include_sst: bool = True
    include_sst_regions: bool = True
    include_heat_flux: bool = True
    include_atm_temperature: bool = True
    include_precipitation: bool = True


@tree_math.struct
class CoupledBaseline:
    """Baseline for coupled climate state.

    Stores reference values for computing anomalies from coupled simulations.

    Attributes:
        sst: Sea surface temperature baseline from ocean component.
        surface_temperature: Atmospheric surface temperature baseline.
        precipitation: Precipitation baseline.
        heat_flux: Air-sea heat flux baseline (optional).

    """

    sst: jnp.ndarray
    surface_temperature: jnp.ndarray
    precipitation: jnp.ndarray
    heat_flux: Optional[jnp.ndarray] = None

    @classmethod
    def from_coupled_carry(cls, coupled_carry: dict, coords):
        """Create baseline from coupled simulation state.

        Args:
            coupled_carry: JEM coupled carry dict with "atm" and "ocn" components.
            coords: Model coordinates.

        Returns:
            CoupledBaseline with fields from the coupled state.

        """
        # Ocean SST
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        # Atmospheric surface temp (from physics predictions)
        atm_physics = coupled_carry["atm"]["derived"]["physics"]
        surf_temp = atm_physics.surface_flux.tsfc

        # Precipitation
        precip = (
            atm_physics.convection.precnv +
            atm_physics.condensation.precls
        )

        # Heat flux
        heat_flux = coupled_carry["atm"]["derived"]["total_heat_flux"]

        return cls(
            sst=sst,
            surface_temperature=surf_temp,
            precipitation=precip,
            heat_flux=heat_flux,
        )

    @classmethod
    def global_mean(cls, nodal_shape, target_sst: float = 288.0):
        """Create baseline with global mean values.

        Args:
            nodal_shape: Grid shape (ix, il).
            target_sst: Target global mean SST in K.

        Returns:
            CoupledBaseline with uniform mean values.

        """
        return cls(
            sst=jnp.full(nodal_shape, target_sst),
            surface_temperature=jnp.full(nodal_shape, target_sst),
            precipitation=jnp.zeros(nodal_shape),
            heat_flux=jnp.zeros(nodal_shape),
        )


def extract_coupled_features(
    coupled_carry: dict,
    baseline: CoupledBaseline,
    coords,
    config: CoupledFeatureConfig = CoupledFeatureConfig(),
) -> jnp.ndarray:
    """Extract features from coupled simulation state.

    Main entry point for coupled feature extraction. Returns scalar features
    for use with MLP policy networks.

    Args:
        coupled_carry: JEM coupled carry dict with "atm" and "ocn" components.
        baseline: CoupledBaseline for anomaly computation.
        coords: Model coordinates.
        config: Feature extraction configuration.

    Returns:
        1D array of scalar features for policy input.

    """
    area_weights = compute_area_weights(coords)
    features = []
    grid = coords.horizontal

    # --- Ocean SST features ---
    if config.include_sst:
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        # Global SST anomaly
        global_sst = jnp.sum(sst * area_weights)
        baseline_sst = jnp.sum(baseline.sst * area_weights)
        features.append(global_sst - baseline_sst)

        # Tropical SST (important for MCB targeting)
        tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
        tropical_sst = compute_regional_mean(sst, tropical_mask, area_weights)
        tropical_baseline = compute_regional_mean(baseline.sst, tropical_mask, area_weights)
        features.append(tropical_sst - tropical_baseline)

    # --- Stratocumulus region SST features ---
    if config.include_sst_regions:
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        for region_name, bounds in STRATOCUMULUS_SST_REGIONS.items():
            region_mask = create_region_mask(grid, bounds[:2], bounds[2:])
            region_sst = compute_regional_mean(sst, region_mask, area_weights)
            region_baseline = compute_regional_mean(baseline.sst, region_mask, area_weights)
            features.append(region_sst - region_baseline)

    # --- Atmospheric temperature features ---
    if config.include_atm_temperature:
        atm_physics = coupled_carry["atm"]["derived"]["physics"]
        surf_temp = atm_physics.surface_flux.tsfc

        global_temp = jnp.sum(surf_temp * area_weights)
        baseline_temp = jnp.sum(baseline.surface_temperature * area_weights)
        features.append(global_temp - baseline_temp)

    # --- Heat flux features ---
    if config.include_heat_flux:
        heat_flux = coupled_carry["atm"]["derived"]["total_heat_flux"]
        global_flux = jnp.sum(heat_flux * area_weights)
        if baseline.heat_flux is not None:
            baseline_flux = jnp.sum(baseline.heat_flux * area_weights)
            features.append(global_flux - baseline_flux)
        else:
            features.append(global_flux)

    # --- Precipitation features ---
    if config.include_precipitation:
        atm_physics = coupled_carry["atm"]["derived"]["physics"]
        precip = atm_physics.convection.precnv + atm_physics.condensation.precls

        # Tropical precipitation
        tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
        tropical_precip = compute_regional_mean(precip, tropical_mask, area_weights)
        tropical_precip_baseline = compute_regional_mean(
            baseline.precipitation, tropical_mask, area_weights
        )
        features.append(tropical_precip - tropical_precip_baseline)

        # Amazon precipitation
        amazon_bounds = TELECONNECTION_REGIONS['amazon']
        amazon_mask = create_region_mask(grid, amazon_bounds[:2], amazon_bounds[2:])
        amazon_precip = compute_regional_mean(precip, amazon_mask, area_weights)
        amazon_baseline = compute_regional_mean(baseline.precipitation, amazon_mask, area_weights)
        features.append(amazon_precip - amazon_baseline)

        # Sahel precipitation
        sahel_bounds = TELECONNECTION_REGIONS['sahel']
        sahel_mask = create_region_mask(grid, sahel_bounds[:2], sahel_bounds[2:])
        sahel_precip = compute_regional_mean(precip, sahel_mask, area_weights)
        sahel_baseline = compute_regional_mean(baseline.precipitation, sahel_mask, area_weights)
        features.append(sahel_precip - sahel_baseline)

    return jnp.array(features)


def get_coupled_feature_dim(config: CoupledFeatureConfig = CoupledFeatureConfig()) -> int:
    """Get dimension of coupled feature vector.

    Args:
        config: Feature extraction configuration.

    Returns:
        Number of scalar features for the given config.

    """
    dim = 0
    if config.include_sst:
        dim += 2  # global + tropical SST
    if config.include_sst_regions:
        dim += 3  # peru, namibia, california stratocumulus regions
    if config.include_atm_temperature:
        dim += 1  # global atmospheric temp
    if config.include_heat_flux:
        dim += 1  # global heat flux
    if config.include_precipitation:
        dim += 3  # tropical + amazon + sahel
    return dim


def create_coupled_feature_extractor(
    baseline: CoupledBaseline,
    coords,
    config: CoupledFeatureConfig = CoupledFeatureConfig(),
):
    """Create a curried feature extraction function.

    Useful for creating feature extractors with pre-bound parameters.

    Args:
        baseline: CoupledBaseline for anomaly computation.
        coords: Model coordinates.
        config: Feature extraction configuration.

    Returns:
        Function (coupled_carry) -> features array.

    """
    def extractor(coupled_carry: dict) -> jnp.ndarray:
        return extract_coupled_features(coupled_carry, baseline, coords, config)
    return extractor
