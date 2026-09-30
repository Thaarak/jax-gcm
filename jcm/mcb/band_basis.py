"""Low-dimensional controls and objectives for the long-horizon gradient study.

Controls
--------
K Gaussian latitude bands over the ocean, applied through the cloud-albedo
actuator: ``perturbation = sum_k a_k * band_k``. The default layout (centres
45N, 20N, 0, 20S, 45S; Gaussian width 10 deg) is the one Dubey, Abbot &
Chattopadhyay (2026, arXiv:2609.12528) used for prescribed-SST forcing in
JAX-GCM, so gradient-quality results are directly comparable with theirs.
Five controls keep brute-force finite differences affordable (2K+1 forward
runs per realization), which is what makes a ground-truth check possible.

Objectives
----------
Linear functionals of a field, reported by the caller as changes against a
paired baseline:

* ``T0``   area-weighted ocean-mean SST.
* ``T1``   interhemispheric contrast: projection onto sin(lat).
* ``T2``   equator-to-pole contrast: projection onto P2(sin lat)
           = (3 sin^2(lat) - 1) / 2.
* ``LAND`` area-weighted mean slab-land surface temperature over land.

T0/T1/T2 follow the Legendre moments used to specify targets in
stratospheric-aerosol feedback studies (Kravitz et al. 2017), computed here
over the ocean. The T1 and T2 basis functions are centred over the ocean
domain, so a spatially uniform change projects onto T0 only. The actuator can
reach LAND only through the atmosphere, which makes LAND the case where an
atmosphere-truncated gradient is expected to fail — the boundary of the
method's domain, and the objective class Dubey et al. studied.
"""

from typing import Dict, Sequence

import jax.numpy as jnp

DEFAULT_BAND_CENTERS_DEG = (45.0, 20.0, 0.0, -20.0, -45.0)
DEFAULT_BAND_WIDTH_DEG = 10.0
OBJECTIVE_NAMES = ("T0", "T1", "T2", "LAND")
OCEAN_OBJECTIVES = ("T0", "T1", "T2")


def _lat_grid(latitudes_rad, shape):
    """Broadcast (il,) latitudes in radians to the (ix, il) grid."""
    return jnp.broadcast_to(jnp.asarray(latitudes_rad)[None, :], shape)


def gaussian_band_patterns(
    latitudes_rad,
    ocean_mask,
    centers_deg: Sequence[float] = DEFAULT_BAND_CENTERS_DEG,
    width_deg: float = DEFAULT_BAND_WIDTH_DEG,
) -> jnp.ndarray:
    """Unit-peak Gaussian latitude bands restricted to ocean cells.

    Args:
        latitudes_rad: (il,) grid latitudes in RADIANS (dinosaur convention).
        ocean_mask: (ix, il) ocean mask (1 = ocean), e.g. from
            ``ocean_mask_from_coupler``.
        centers_deg: Band centres in degrees.
        width_deg: Gaussian standard deviation in degrees.

    Returns:
        (K, ix, il) array; band k equals 1 at its centre latitude over ocean.

    """
    ocean_mask = jnp.asarray(ocean_mask)
    lat_deg = jnp.rad2deg(_lat_grid(latitudes_rad, ocean_mask.shape))
    bands = [jnp.exp(-0.5 * ((lat_deg - c) / width_deg) ** 2) * ocean_mask
             for c in centers_deg]
    return jnp.stack(bands, axis=0)


def band_perturbation(amplitudes, patterns) -> jnp.ndarray:
    """Albedo perturbation field ``sum_k a_k * pattern_k`` (ix, il)."""
    return jnp.tensordot(jnp.asarray(amplitudes), patterns, axes=1)


def objective_weights(latitudes_rad, ocean_mask, land_mask) -> Dict[str, jnp.ndarray]:
    """Weight fields turning a grid field into each objective (dot product).

    Args:
        latitudes_rad: (il,) grid latitudes in radians.
        ocean_mask: (ix, il) ocean mask (1 = ocean).
        land_mask: (ix, il) land mask (1 = land), e.g. the slab land model's
            ``bmask_l``.

    Returns:
        Dict with keys ``OBJECTIVE_NAMES``. ``sum(W["T0"] * sst)`` is the
        ocean-mean SST; ``T1``/``T2`` weights sum to zero; ``LAND`` weights
        (applied to the land-temperature field) sum to one.

    """
    ocean_mask = jnp.asarray(ocean_mask)
    land_mask = jnp.asarray(land_mask)
    lat = _lat_grid(latitudes_rad, ocean_mask.shape)
    area = jnp.cos(lat)
    w_ocean = area * ocean_mask
    w_ocean = w_ocean / jnp.sum(w_ocean)
    w_land = area * land_mask
    w_land = w_land / jnp.sum(w_land)
    sin_lat = jnp.sin(lat)
    p2 = 0.5 * (3.0 * sin_lat ** 2 - 1.0)
    l1 = sin_lat - jnp.sum(w_ocean * sin_lat)
    l2 = p2 - jnp.sum(w_ocean * p2)
    return {"T0": w_ocean, "T1": w_ocean * l1, "T2": w_ocean * l2,
            "LAND": w_land}


def stack_objective_weights(weights: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Stack weight fields in ``OBJECTIVE_NAMES`` order: (n_obj, ix, il)."""
    return jnp.stack([weights[name] for name in OBJECTIVE_NAMES], axis=0)


def objective_values(sst, land_temperature, weight_stack,
                     reference_k: float = 288.0) -> jnp.ndarray:
    """Evaluate every objective: ocean ones on SST, LAND on land temperature.

    The ``reference_k`` offset is removed BEFORE the weighted sums: summing
    ~290 K values in float32 and differencing afterwards loses ~1e-4 K
    (the cancellation already met in ``coupled_features``), which is not
    negligible next to finite-difference signals of a few hundredths of a
    kelvin. T0 and LAND are therefore reported relative to ``reference_k``
    (their weights sum to one); T1 and T2 are unchanged mathematically (their
    weights sum to zero). Every use in this study takes differences.

    Returns:
        (n_obj,) vector in ``OBJECTIVE_NAMES`` order.

    """
    n_ocean = len(OCEAN_OBJECTIVES)
    ocean_part = jnp.tensordot(weight_stack[:n_ocean], sst - reference_k,
                               axes=([1, 2], [0, 1]))
    land_part = jnp.sum(weight_stack[n_ocean]
                        * (land_temperature - reference_k))
    return jnp.concatenate([ocean_part, land_part[None]])
