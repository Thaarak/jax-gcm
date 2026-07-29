"""Coupled state feature extraction for policy networks.

Extracts features from both atmosphere and ocean components in
the coupled carry structure for use as policy network inputs.

This module is designed for JAX-ESM coupled simulations where
the coupled_carry dict contains both "atm" and "ocn" components.

Example usage (Stage 2, paired baseline trajectory):
    from jcm.mcb.coupled_features import (
        compute_baseline_trajectory,
        extract_coupled_features,
    )

    trajectory = compute_baseline_trajectory(initial_carry, step_fn, num_steps=180)
    baseline_t = trajectory.at_step(t)  # CoupledBaseline at day t
    features = extract_coupled_features(coupled_carry, baseline_t, coords,
                                        time_fraction=t / 180)
"""

import jax
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
        include_time: Include normalized time-of-rollout (0 at start, 1 at
            end). Gives the policy schedule-dependence, and guarantees a
            non-constant input even when paired anomalies are zero (e.g. at
            the very first control interval).
        include_absolute_sst: Include absolute (non-paired) SST features:
            global-mean SST minus 288 K, and NH-minus-SH hemispheric mean
            SST difference. Unlike the paired anomalies (which are exactly
            zero at interval 0), these distinguish different initial
            conditions and seasons, enabling state-dependent behavior at
            the very first control interval. Default False preserves the
            11-feature layout of existing checkpoints.

    """

    include_sst: bool = True
    include_sst_regions: bool = True
    include_heat_flux: bool = True
    include_atm_temperature: bool = True
    include_precipitation: bool = True
    include_time: bool = True
    include_absolute_sst: bool = False


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


@tree_math.struct
class CoupledBaselineTrajectory:
    """Paired no-MCB baseline trajectory for features and loss.

    Stores per-coupling-step snapshots of the baseline fields from a no-MCB
    run started from the SAME initial state as training. Anomalies computed
    against `at_step(t)` isolate the MCB-caused signal exactly: natural model
    drift is identical in both runs and cancels in the difference.

    All fields have a leading time dimension of length num_steps + 1
    (index 0 = initial state, index t = state after t coupling steps).

    Attributes:
        sst: (T+1, ix, il) sea surface temperature.
        surface_temperature: (T+1, ix, il) atmospheric surface temperature.
        precipitation: (T+1, ix, il) total precipitation.
        heat_flux: (T+1, ix, il) air-sea heat flux.

    """

    sst: jnp.ndarray
    surface_temperature: jnp.ndarray
    precipitation: jnp.ndarray
    heat_flux: jnp.ndarray

    @property
    def num_steps(self) -> int:
        return self.sst.shape[0] - 1

    def at_step(self, step: jnp.ndarray) -> CoupledBaseline:
        """Return the CoupledBaseline snapshot at a given coupling step.

        Works with traced integer indices (usable inside jit/scan).
        """
        return CoupledBaseline(
            sst=self.sst[step],
            surface_temperature=self.surface_temperature[step],
            precipitation=self.precipitation[step],
            heat_flux=self.heat_flux[step],
        )


def compute_baseline_trajectory(
    initial_carry: dict,
    step_fn,
    num_steps: int,
    coords,
) -> CoupledBaselineTrajectory:
    """Run the paired no-MCB baseline and record per-step snapshots.

    Runs the coupler forward from `initial_carry` with the MCB perturbation
    forced to zero, recording the baseline fields after every coupling step
    (plus the initial state at index 0). Forward-only: not differentiated.

    Args:
        initial_carry: Initial coupled carry (same one used for training).
        step_fn: Coupler step function (carry, step_idx) -> (carry, preds).
        num_steps: Number of coupling steps (e.g. total training days).
        coords: Model coordinates.

    Returns:
        CoupledBaselineTrajectory with num_steps + 1 snapshots.

    """
    # Copy the carry and zero the MCB perturbation without mutating the input.
    carry = jax.tree_util.tree_map(lambda x: x, initial_carry)
    existing = carry["atm"]["derived"]["mcb_perturbation"]
    carry["atm"]["derived"]["mcb_perturbation"] = jnp.zeros_like(existing)

    def body(c, step_idx):
        new_c, _ = step_fn(c, step_idx)
        return new_c, CoupledBaseline.from_coupled_carry(new_c, coords)

    _, snapshots = jax.lax.scan(body, carry, jnp.arange(num_steps))

    initial = CoupledBaseline.from_coupled_carry(carry, coords)
    stacked = jax.tree_util.tree_map(
        lambda first, rest: jnp.concatenate([first[None], rest], axis=0),
        initial, snapshots,
    )
    return CoupledBaselineTrajectory(
        sst=stacked.sst,
        surface_temperature=stacked.surface_temperature,
        precipitation=stacked.precipitation,
        heat_flux=stacked.heat_flux,
    )


def extract_coupled_features(
    coupled_carry: dict,
    baseline: CoupledBaseline,
    coords,
    config: CoupledFeatureConfig = CoupledFeatureConfig(),
    time_fraction: float = 0.0,
    ocean_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Extract features from coupled simulation state.

    Main entry point for coupled feature extraction. Returns scalar features
    for use with MLP policy networks.

    For Stage 2+ training, `baseline` should be the paired no-MCB snapshot at
    the SAME timestep (from CoupledBaselineTrajectory.at_step), so that the
    anomalies isolate the MCB-caused signal from natural drift.

    Args:
        coupled_carry: JEM coupled carry dict with "atm" and "ocn" components.
        baseline: CoupledBaseline for anomaly computation.
        coords: Model coordinates.
        config: Feature extraction configuration.
        time_fraction: Normalized time-of-rollout in [0, 1]; used when
            config.include_time is True.
        ocean_mask: Optional ocean mask (1.0 ocean, 0.0 land) with shape
            (ix, il). When provided (realistic terrain, Stage 5+), every
            area-weighted global/regional mean is restricted to ocean cells,
            so land points pinned to a fixed temperature do not corrupt the
            state-dependent features that drive the policy. Default None
            reproduces the aquaplanet behavior (all cells) bit-for-bit.

    Returns:
        1D array of scalar features for policy input.

    """
    area_weights = compute_area_weights(coords)
    features = []
    grid = coords.horizontal

    # Ocean-weighted global average helper: over ocean cells when a mask is
    # given, else identical to jnp.sum(field * area_weights). Combining the
    # ocean mask with any region mask and renormalizing keeps every mean on
    # the ocean-only support (via compute_regional_mean's renormalization).
    if ocean_mask is not None:
        ocean_weights = area_weights * ocean_mask
        ocean_weights = ocean_weights / jnp.sum(ocean_weights)
    else:
        ocean_weights = area_weights

    def _global_mean(field):
        return jnp.sum(field * ocean_weights)

    def _regional_mean(field, region_mask):
        if ocean_mask is not None:
            region_mask = region_mask * ocean_mask
        return compute_regional_mean(field, region_mask, area_weights)

    # --- Ocean SST features ---
    if config.include_sst:
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        # Global SST anomaly
        features.append(_global_mean(sst) - _global_mean(baseline.sst))

        # Tropical SST (important for MCB targeting)
        tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
        tropical_sst = _regional_mean(sst, tropical_mask)
        tropical_baseline = _regional_mean(baseline.sst, tropical_mask)
        features.append(tropical_sst - tropical_baseline)

    # --- Stratocumulus region SST features ---
    if config.include_sst_regions:
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        for region_name, bounds in STRATOCUMULUS_SST_REGIONS.items():
            region_mask = create_region_mask(grid, bounds[:2], bounds[2:])
            region_sst = _regional_mean(sst, region_mask)
            region_baseline = _regional_mean(baseline.sst, region_mask)
            features.append(region_sst - region_baseline)

    # --- Atmospheric temperature features ---
    if config.include_atm_temperature:
        atm_physics = coupled_carry["atm"]["derived"]["physics"]
        surf_temp = atm_physics.surface_flux.tsfc

        features.append(
            _global_mean(surf_temp) - _global_mean(baseline.surface_temperature)
        )

    # --- Heat flux features ---
    if config.include_heat_flux:
        heat_flux = coupled_carry["atm"]["derived"]["total_heat_flux"]
        global_flux = _global_mean(heat_flux)
        if baseline.heat_flux is not None:
            features.append(global_flux - _global_mean(baseline.heat_flux))
        else:
            features.append(global_flux)

    # --- Precipitation features ---
    if config.include_precipitation:
        atm_physics = coupled_carry["atm"]["derived"]["physics"]
        precip = atm_physics.convection.precnv + atm_physics.condensation.precls

        # Tropical precipitation
        tropical_mask = create_latitude_band_mask(coords, -30.0, 30.0)
        tropical_precip = _regional_mean(precip, tropical_mask)
        tropical_precip_baseline = _regional_mean(
            baseline.precipitation, tropical_mask
        )
        features.append(tropical_precip - tropical_precip_baseline)

        # Amazon precipitation
        amazon_bounds = TELECONNECTION_REGIONS['amazon']
        amazon_mask = create_region_mask(grid, amazon_bounds[:2], amazon_bounds[2:])
        amazon_precip = _regional_mean(precip, amazon_mask)
        amazon_baseline = _regional_mean(baseline.precipitation, amazon_mask)
        features.append(amazon_precip - amazon_baseline)

        # Sahel precipitation
        sahel_bounds = TELECONNECTION_REGIONS['sahel']
        sahel_mask = create_region_mask(grid, sahel_bounds[:2], sahel_bounds[2:])
        sahel_precip = _regional_mean(precip, sahel_mask)
        sahel_baseline = _regional_mean(baseline.precipitation, sahel_mask)
        features.append(sahel_precip - sahel_baseline)

    # --- Time-of-rollout feature ---
    if config.include_time:
        features.append(jnp.asarray(time_fraction, dtype=jnp.float64))

    # --- Absolute SST features (IC/season signal; appended at END so that
    # existing 11-feature checkpoints can be expanded by zero-padding) ---
    if config.include_absolute_sst:
        sst = coupled_carry["ocn"]["state"].sea_surface_temperature

        # Global-mean SST offset from a 288 K reference. Subtract the reference
        # BEFORE the area-weighted sum: summing ~288 K values and then
        # subtracting 288 is catastrophic float32 cancellation (the ~0.5 K
        # result inherits the ~2e-4 rounding of a 288 K sum), which the
        # non-uniform area weights amplify. Summing the ~O(1 K) anomaly is
        # mathematically identical (weights sum to 1) and float32-stable.
        features.append(_global_mean(sst - 288.0))

        # NH minus SH hemispheric mean SST (aquaplanet season signal)
        nh_mask = create_latitude_band_mask(coords, 0.0, 90.0)
        sh_mask = create_latitude_band_mask(coords, -90.0, 0.0)
        nh_sst = _regional_mean(sst, nh_mask)
        sh_sst = _regional_mean(sst, sh_mask)
        features.append(nh_sst - sh_sst)

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
    if config.include_time:
        dim += 1  # normalized time-of-rollout
    if config.include_absolute_sst:
        dim += 2  # global-mean SST offset + NH-SH hemispheric difference
    return dim


def create_coupled_feature_extractor(
    baseline: CoupledBaseline,
    coords,
    config: CoupledFeatureConfig = CoupledFeatureConfig(),
    ocean_mask: jnp.ndarray | None = None,
):
    """Create a curried feature extraction function.

    Useful for creating feature extractors with pre-bound parameters.

    Args:
        baseline: CoupledBaseline for anomaly computation.
        coords: Model coordinates.
        config: Feature extraction configuration.
        ocean_mask: Optional ocean mask (1.0 ocean, 0.0 land); when provided
            the area-weighted means are restricted to ocean cells.

    Returns:
        Function (coupled_carry) -> features array.

    """
    def extractor(coupled_carry: dict, time_fraction: float = 0.0) -> jnp.ndarray:
        return extract_coupled_features(
            coupled_carry, baseline, coords, config,
            time_fraction=time_fraction, ocean_mask=ocean_mask,
        )
    return extractor
