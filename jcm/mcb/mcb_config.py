"""MCB (Marine Cloud Brightening) configuration data structure.

This module defines the MCBConfig struct for specifying spatial and temporal
patterns of MCB radiative forcing. The struct is fully differentiable and
designed for gradient-based optimization of MCB deployment strategies.
"""

import jax.numpy as jnp
import tree_math
from jax import tree_util


@tree_math.struct
class MCBConfig:
    """Configuration for Marine Cloud Brightening forcing.

    This struct is fully differentiable and designed for gradient-based
    optimization of MCB deployment strategies.

    Attributes:
        albedo_perturbation: Grid-cell level albedo enhancement (ix, il).
            Values represent the increase in sea surface albedo due to MCB.
            Typical range: 0.0 to 0.15 (0 = no MCB, 0.15 = strong MCB).
        active_mask: Binary mask indicating where MCB is applied (ix, il).
            Values: 0.0 (inactive) or 1.0 (active). Used to restrict MCB
            to ocean regions and specific deployment zones.
        temporal_weights: Time-varying weights (365,) for seasonal modulation.
            Values typically in [0, 1]. Applied multiplicatively to the
            albedo_perturbation field.

    """

    albedo_perturbation: jnp.ndarray
    active_mask: jnp.ndarray
    temporal_weights: jnp.ndarray

    @classmethod
    def zeros(cls, nodal_shape, temporal_weights=None):
        """Create MCBConfig with no forcing applied.

        Args:
            nodal_shape: Tuple (ix, il) specifying the horizontal grid shape.
            temporal_weights: Optional (365,) array of daily weights.
                Defaults to all ones (year-round uniform forcing).

        Returns:
            MCBConfig with zero perturbation and zero active mask.

        """
        return cls(
            albedo_perturbation=jnp.zeros(nodal_shape),
            active_mask=jnp.zeros(nodal_shape),
            temporal_weights=temporal_weights if temporal_weights is not None
                             else jnp.ones(365),
        )

    @classmethod
    def uniform(cls, nodal_shape, perturbation=0.05, region_mask=None,
                temporal_weights=None):
        """Create MCBConfig with uniform perturbation over a region.

        Args:
            nodal_shape: Tuple (ix, il) specifying the horizontal grid shape.
            perturbation: Uniform albedo increase to apply. Default 0.05.
            region_mask: Optional (ix, il) mask specifying where MCB is active.
                Defaults to all ones (everywhere active).
            temporal_weights: Optional (365,) array of daily weights.
                Defaults to all ones (year-round uniform forcing).

        Returns:
            MCBConfig with uniform perturbation over the masked region.

        """
        mask = region_mask if region_mask is not None else jnp.ones(nodal_shape)
        return cls(
            albedo_perturbation=jnp.full(nodal_shape, perturbation),
            active_mask=mask,
            temporal_weights=temporal_weights if temporal_weights is not None
                             else jnp.ones(365),
        )

    @classmethod
    def from_spatial_field(cls, perturbation_field, active_mask,
                           temporal_weights=None):
        """Create MCBConfig from a pre-computed spatial perturbation field.

        Args:
            perturbation_field: (ix, il) array of albedo perturbations.
            active_mask: (ix, il) binary mask for MCB regions.
            temporal_weights: Optional (365,) array of daily weights.
                Defaults to all ones (year-round uniform forcing).

        Returns:
            MCBConfig with the specified perturbation field.

        """
        return cls(
            albedo_perturbation=perturbation_field,
            active_mask=active_mask,
            temporal_weights=temporal_weights if temporal_weights is not None
                             else jnp.ones(365),
        )

    def copy(self, albedo_perturbation=None, active_mask=None,
             temporal_weights=None):
        """Return a copy with optionally updated fields.

        Args:
            albedo_perturbation: New perturbation field, or None to keep current.
            active_mask: New active mask, or None to keep current.
            temporal_weights: New temporal weights, or None to keep current.

        Returns:
            New MCBConfig with updated fields.

        """
        return MCBConfig(
            albedo_perturbation=albedo_perturbation if albedo_perturbation is not None
                               else self.albedo_perturbation,
            active_mask=active_mask if active_mask is not None
                       else self.active_mask,
            temporal_weights=temporal_weights if temporal_weights is not None
                            else self.temporal_weights,
        )

    def isnan(self):
        """Check for NaN values in all fields."""
        return tree_util.tree_map(jnp.isnan, self)

    def any_true(self):
        """Check if any value is True across all fields."""
        return tree_util.tree_reduce(
            lambda x, y: x | y,
            tree_util.tree_map(jnp.any, self),
        )
